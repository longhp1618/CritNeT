# `critnet` — API reference

This is the per-symbol reference for the `critnet` package. For installation, repository layout, and a runnable Quickstart, see the root [README.md](../README.md).

> **TL;DR.** `critnet` ranks every "neuron" (row / column of a linear layer, or scalar of a norm) by the first-order Taylor importance $|w \odot \nabla_w \mathcal{L}|$, then lets you act on the global top-$k$ set: **analyse**, **deactivate**, **freeze**, or **fine-tune only those neurons**. One config object (`CriticalNeuronConfig`) describes the targeted modules and the sparsity ratio; every other class consumes it.

**Dependencies:** `torch`, `transformers`, `tqdm`. `safetensors` is optional (adapter and deactivated-checkpoint I/O fall back to `torch.save` when it is missing).

---

## Contents

1. [The five public components](#the-five-public-components)
2. [How importance scoring works](#how-importance-scoring-works)
3. [`CriticalNeuronConfig`](#criticalneuronconfig)
4. [`NeuronDetector`](#neurondetector)
5. [`NeuronStatistician` and `StatisticsResult`](#neuronstatistician-and-statisticsresult)
6. [`NeuronDeactivator` and `DeactivationResult`](#neurondeactivator-and-deactivationresult)
7. [`get_neuron_model`, `CriticalNeuronModel`, delta wrappers](#get_neuron_model-criticalneuronmodel-delta-wrappers)
8. [`freeze_neurons` and `FrozenNeuronHandle`](#freeze_neurons-and-frozenneuronhandle)
9. [Saved-artefact layout](#saved-artefact-layout)
10. [Cookbook](#cookbook)

---

## The five public components

```python
from critnet import (
    CriticalNeuronConfig,    # shared config: targeted modules + sparsity_ratio + neuron_indices
    NeuronDetector,          # 1. detect critical neurons for a corpus
    NeuronStatistician,      # 2. set-algebra (union / intersection / exclusive) across runs
    NeuronDeactivator,       # 3. zero out a neuron set in place (ablation)
    get_neuron_model,        # 4. wrap a model so only critical-neuron deltas train (sparse PEFT)
    freeze_neurons,          # 5. freeze a neuron set during full fine-tuning
    # supporting symbols
    StatisticsResult, DeactivationResult, CriticalNeuronModel, FrozenNeuronHandle,
    DEFAULT_SKIP_MODULES,
    LinearDeltaSubspace, NormDeltaSubspace, EmbeddingDeltaSubspace,
)
```

Every workflow follows the same shape:

```
build CriticalNeuronConfig  →  NeuronDetector.detect(loader)  →  cfg.neuron_indices is filled
                                                              ↓
                  ┌───────────────────────────────────────────┼──────────────────────────────┐
                  ↓                                           ↓                              ↓
          NeuronStatistician                          NeuronDeactivator              get_neuron_model
          (compare across runs)                       (zero out, save HF ckpt)       (train only deltas)
                                                                                            +
                                                                                     freeze_neurons
                                                                                     (full FT, mask grads)
```

---

## How importance scoring works

### Per-neuron importance

For each targeted parameter tensor `W`, the detector computes the elementwise product $|W \odot \nabla_W \mathcal{L}|$ and **reduces it to one scalar per neuron** based on the module type:

| Module type | What is a "neuron"? | Reduction |
|-------------|--------------------|-----------|
| **row** (`q_proj`, `up_proj`, …) | one **row** of $W$ — an output channel | sum over the input dim |
| **column** (`o_proj`, `down_proj`) | one **column** of $W$ — an input channel | sum over the output dim |
| **norm** (`input_layernorm`, …) | one scalar of the 1-D weight | identity (`|w_i · grad_i|`) |
| **embedding** (opt-in) | one **row** of the embedding (a vocab item) | sum over the embed dim |

### Global top-$k$ selection

Scores from **all** targeted modules are pooled into a single list, then the top

```text
k = max(1, int(total_neuron_count * sparsity_ratio))
```

neurons survive. `sparsity_ratio` is therefore a **fraction of pooled neurons across the whole model**, not a per-layer fraction. This naturally lets attention QKV neurons compete with FFN gate/up/down neurons on the same scale.

### SwiGLU gate handling

LLaMA-style FFNs compute `down( gate(x) * up(x) )`. Ablating only `up_proj` row $i$ without ablating the same row of `gate_proj` would leave a stale gate signal. `gate_combines_with = {"gate_proj": "up_proj"}` (the default) handles this:

1. The `gate_proj` score vector is **added** to its partner's before top-$k$.
2. The gate entry is removed from the score dict.
3. After selection, `gate_proj` receives a **copy** of `up_proj`'s selected indices.

Result: gate and up rows are always selected as a pair.

### Gradients accumulate inside `detect`

`NeuronDetector.detect` calls `model.zero_grad()` **once** at the start, then loops `loss.backward()` per batch **without zeroing between batches**. The importance step uses these accumulated gradients, which is mathematically equivalent to computing the gradient of the summed loss over the whole calibration corpus.

Label masking is **not** done by `detect`. Your dataset / collator is responsible for producing the right `labels`: set them to `-100` on prompt positions for chat SFT (so only completion tokens contribute to the loss), or equal to `input_ids` for pre-training (next-token prediction on every token).

---

## `CriticalNeuronConfig`

The single shared configuration object. Lives in `config.py`. Stores **which modules to target**, the **sparsity ratio**, and (after detection) the selected **neuron indices**.

Module matching is by **leaf name**: `"q_proj"` matches every fully-qualified path that ends in `q_proj`, e.g. `model.layers.0.self_attn.q_proj`.

### Fields

| Field | Type | Default / behavior |
|-------|------|---------------------|
| `row_modules` | `list[str] \| None` | `None` → `["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"]`. |
| `column_modules` | `list[str] \| None` | `None` → `["o_proj", "down_proj"]`. |
| `norm_modules` | `list[str] \| None` | `None` → `["input_layernorm", "post_attention_layernorm"]`. |
| `embedding_modules` | `list[str] \| None` | `None` (off). Pass a list to opt in; `[]` keeps it off. |
| `sparsity_ratio` | `float` | `0.05`. Must be strictly in $(0, 1)$. |
| `gate_combines_with` | `dict[str, str] \| None` | `None` → `{"gate_proj": "up_proj"}` if both are in the linear lists. `{}` disables. |
| `neuron_indices` | `dict[str, list[int]] \| None` | Filled by `NeuronDetector` or `from_pretrained`. |
| `base_model_name_or_path` | `str \| None` | Optional metadata for downstream tools. |

### Smart defaults

- Passing `None` for a module list → fill with architecture defaults (LLaMA / Mistral / Qwen share the same leaf names).
- Passing `[]` → explicitly empty (e.g. `norm_modules=[]` skips all norms).
- `embedding_modules` stays `None` unless you set it.

For **Qwen-3 / 3.5** add the QK norms:

```python
CriticalNeuronConfig(
    norm_modules=["input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"],
)
```

### Validation (`__post_init__`)

- `sparsity_ratio` must be in $(0, 1)$.
- A module suffix may appear in **at most one** of the four category lists.
- Every key and value of `gate_combines_with` must be in `row_modules` or `column_modules`.

### Methods

| Method | Purpose |
|--------|---------|
| `target_modules` (property) | Flat union of all four category lists. |
| `linear_modules` (property) | `row_modules + column_modules`. |
| `matches_target(path) -> bool` | Does this leaf name belong to any category? |
| `get_module_type(path) -> str` | Returns `"row" / "column" / "norm" / "embedding"`; raises `ValueError` otherwise. |
| `summary() -> str` | Human-readable dump. |
| `save_pretrained(dir)` | Writes `critical_neuron_config.json` (+ `neuron_indices.json` if present). |
| `from_pretrained(dir)` | Class method; restores config and (optionally) neuron indices. |

> `embed_tokens` and `lm_head` are **not** excluded by the config itself. They are skipped by `get_neuron_model` via `DEFAULT_SKIP_MODULES` (see below) to avoid breaking weight-tying and fused-kernel paths.

---

## `NeuronDetector`

Computes per-neuron importance scores from a calibration loader and picks the global top-$k$.

```python
NeuronDetector(model: nn.Module | None, config: CriticalNeuronConfig)
```

- `model` is required for `.detect(...)`. It is **optional** when you only call `.select_from_importance_cache(...)` on a previously saved scores blob.

### `detect(dataloader, save_importance_cache_path=None)`

Runs the calibration pass and returns `dict[module_path, sorted list[int]]`. Also stores the result on `self.config.neuron_indices` so you can immediately `config.save_pretrained(...)`.

| Argument | Description |
|----------|-------------|
| `dataloader` | Each batch must be a `dict` containing at least `input_ids`, `attention_mask`, and `labels`. See [Gradients accumulate inside `detect`](#gradients-accumulate-inside-detect) for how `labels` should be prepared by your dataset / collator. |
| `save_importance_cache_path` | If set, saves the **post–gate-combined** per-module score tensors so you can re-select with a different `sparsity_ratio` without rerunning the backward pass. |

### `save(save_path)`

Convenience wrapper around `config.save_pretrained(save_path)`.

### `select_from_importance_cache(path)`

Loads a cached scores blob and reapplies global top-$k$ using **the current `config.sparsity_ratio`**. Useful for sweeping ratios cheaply.

```python
cfg = CriticalNeuronConfig(sparsity_ratio=0.05)
det = NeuronDetector(model, cfg)
det.detect(loader, save_importance_cache_path="./cache.pt")  # expensive once

for r in [0.01, 0.02, 0.05, 0.10]:
    cfg.sparsity_ratio = r
    det.select_from_importance_cache("./cache.pt")           # cheap
    cfg.save_pretrained(f"./neurons/r{r}")
```

### `save_importance_cache(combined_scores, path)` (low-level)

Normally invoked indirectly via `save_importance_cache_path` on `.detect(...)`. See [Saved-artefact layout](#saved-artefact-layout) for the on-disk format.

---

## `NeuronStatistician` and `StatisticsResult`

Set-algebra across two or more detection runs. The classic use is comparing neuron sets for different tasks or languages.

```python
NeuronStatistician(model: nn.Module | None = None, config: CriticalNeuronConfig | None = None)
```

- Pass `model` so neuron totals and `param_coverage(...)` can use real weight shapes.
- Pass `config` so `get_module_type` resolves row vs column vs norm during counting.
- Both `None` is allowed (CPU-only stats), but `total_neurons_per_module`, `params_per_neuron`, and `total_model_params` will be empty / zero.

### `analyze(task_indices) -> StatisticsResult`

```python
task_indices: dict[
    str,                       # task / language name, e.g. "utility", "refusal", "en", "zh"
    dict[str, list[int]],      # the inner neuron_indices dict for that task
]
```

Per module, computes:

- **`union`** — neurons that appear in **any** task.
- **`intersection`** — neurons that appear in **all** tasks ("shared").
- **`exclusive[task]`** — `task`'s set minus the intersection. *(Note: this is **task − shared**, not pairwise set-difference. For two tasks `A` and `B`, `exclusive["A"] = A \ B`. For three or more, `exclusive["A"]` is everything in `A` that is not in `A ∩ B ∩ C`.)*
- **`non_shared`** — `union ∖ intersection`.

### `save_report(save_directory)`

After `analyze`, writes (see [Saved-artefact layout](#saved-artefact-layout)):
`union_neurons.json`, `shared_neurons.json`, `non_shared_neurons.json`, `exclusive_<task>_neurons.json` (one per task), and `statistics.csv`.

### `StatisticsResult` fields

| Field | Type | Meaning |
|-------|------|---------|
| `union` | `dict[str, list[int]]` | Per-module union. |
| `intersection` | `dict[str, list[int]]` | Per-module intersection. |
| `exclusive` | `dict[str, dict[str, list[int]]]` | Outer key = task name. |
| `non_shared` | `dict[str, list[int]]` | Union minus intersection. |
| `total_neurons_per_module` | `dict[str, int]` | Total neurons per module (needs `model`). |
| `params_per_neuron` | `dict[str, int]` | Scalar weights touched by one neuron index. |
| `total_model_params` | `int` | Full parameter count (needs `model`). |
| `task_names` | `list[str]` | Insertion order of tasks. |

Properties: `union_count`, `intersection_count`, `non_shared_count`, `total_neurons`.

Methods:

- `exclusive_count(task) -> int`
- `task_count(task, task_indices) -> int`
- `param_coverage(indices) -> float` — **percentage** (0–100) of total model parameters covered by `indices`.
- `summary(task_indices=None) -> str` — text report; pass `task_indices` for per-task lines.

---

## `NeuronDeactivator` and `DeactivationResult`

Zero out a neuron set **in place** in the underlying weights. Useful for ablation studies — the model architecture is unchanged, so a deactivated model is just a normal HuggingFace checkpoint that can be saved and reloaded as usual.

```python
NeuronDeactivator(model: nn.Module, config: CriticalNeuronConfig)
```

The model must be a plain HF-style stack (not wrapped by `CriticalNeuronModel`). The config supplies module categories so the deactivator knows **which axis** to zero.

### `deactivate(neuron_indices=None) -> DeactivationResult`

Uses `neuron_indices` if passed, else `config.neuron_indices`. The effect per module type:

| Module type | Effect |
|-------------|--------|
| **row** | `W[idx, :] = 0` |
| **column** | `W[:, idx] = 0` |
| **norm** | `w[idx] = 0`; `bias[idx] = 0` if present |
| **embedding** | `E[idx, :] = 0` |

Unknown modules are skipped with a warning.

### `save_pretrained(save_directory, tokenizer=None)`

Calls `model.save_pretrained`, optionally `tokenizer.save_pretrained`, plus `config.save_pretrained`. The result is loadable with `AutoModelForCausalLM.from_pretrained` just like any other checkpoint.

### `DeactivationResult` fields

| Field | Meaning |
|-------|---------|
| `modules_affected` | Count of modules touched. |
| `neurons_zeroed` | Count of neuron indices processed. |
| `total_weights_zeroed` | Scalar weight elements set to zero. |
| `per_module[path]` | `{"neurons", "weights", "module_type"}`. |

`.summary() -> str` prints aggregate counts.

---

## `get_neuron_model`, `CriticalNeuronModel`, delta wrappers

Sparse PEFT: keep the base weights frozen and learn a **small additive delta** restricted to selected neuron indices. Drop-in alternative to LoRA when you already have a `neuron_indices` dict.

### `get_neuron_model(model, config, modules_to_skip=None) -> CriticalNeuronModel`

| Argument | Description |
|----------|-------------|
| `model` | Any HF model with `nn.Linear` / norm targets (e.g. `AutoModelForCausalLM`). |
| `config` | Must have a non-`None` `neuron_indices`. |
| `modules_to_skip` | Set of **leaf** names skipped from wrapping. Default `DEFAULT_SKIP_MODULES = {"lm_head", "embed_tokens"}` to avoid breaking fused kernels and weight-tying. Pass `set()` to override. |

Wrapping rules: a module is replaced iff it (1) matches the config's targets, (2) is **not** in `modules_to_skip`, (3) appears in `neuron_indices`, and (4) has a non-empty index list. Replacement uses `LinearDeltaSubspace`, `NormDeltaSubspace`, or `EmbeddingDeltaSubspace` based on `get_module_type`. After wrapping:

- All parameters get `requires_grad_(False)`.
- Each wrapper's `dW` gets `requires_grad_(True)`.
- `model.enable_input_require_grads()` is called when available (so gradient checkpointing still works).

### `LinearDeltaSubspace(base_linear, indices, mode="row", train_bias=False)`

Behaves like an `nn.Linear` from the outside (`weight`, `bias`, `in_features`, `out_features` are exposed via properties).

| `mode` | `dW` shape | Forward |
|--------|-----------|---------|
| `"row"` | `[k, in_features]` | `y = base(x); y.index_add_(-1, idx, x @ dW.T)` |
| `"column"` | `[out_features, k]` | `y = base(x) + (x[..., idx] @ dW.T)` |

- `train_bias=False` freezes the base bias when present.
- `merge_to_linear_()` adds `dW` into the base weight in place and returns a plain `nn.Linear`.

### `NormDeltaSubspace(base_norm, indices)`

Forward applies a multiplicative-style correction on selected positions of the 1-D norm weight. `merge_to_norm_()` adds `dW` into the base weight.

### `EmbeddingDeltaSubspace(base_embedding, indices)`

Trainable delta restricted to selected vocabulary rows. `merge_to_embedding_()` merges into `base_embedding.weight`.

### `CriticalNeuronModel`

Thin `nn.Module` around the wrapped inner model. Compatible with `transformers.Trainer` thanks to `__getattr__` proxying.

| Attribute | Description |
|-----------|-------------|
| `model` | The inner HF model. |
| `peft_config` | The `CriticalNeuronConfig` used for wrapping. |

| Method | Description |
|--------|-------------|
| `forward`, `generate` | Delegated to `model`. |
| `get_adapter_state_dict()` | OrderedDict of `*.dW` and `*.idx` from every delta module. |
| `save_pretrained(save_directory, **kwargs)` | Writes the adapter + config JSONs (see on-disk layout). |
| `from_pretrained(adapter_path, base_model_name_or_path=None, model_kwargs=None, modules_to_skip=None)` | Class method. Loads the base model, calls `get_neuron_model`, then loads `dW` from the adapter file. `base_model_name_or_path` falls back to the value recorded in `adapter_path/critical_neuron_config.json` when omitted. |
| `merge_and_unload()` | In-place merge of every delta wrapper; returns the inner plain `nn.Module`. |
| `print_trainable_parameters()` | Prints trainable / total parameter counts. |

---

## `freeze_neurons` and `FrozenNeuronHandle`

Opposite of sparse PEFT: do **full** fine-tuning but block gradient flow into a chosen neuron set (typically the safety-critical ones).

```python
freeze_neurons(
    model: nn.Module,
    neuron_indices: dict[str, list[int]],
    config: CriticalNeuronConfig | None = None,
) -> FrozenNeuronHandle
```

| Argument | Description |
|----------|-------------|
| `model` | A trainable model (typically every parameter has `requires_grad=True`). |
| `neuron_indices` | Mapping from module path to indices to **freeze** (no gradient updates). |
| `config` | Optional; used for row/column/norm typing. A default config is constructed if omitted, and any module not covered by it falls back to heuristics on the leaf name. |

The implementation registers **backward hooks** that zero the gradient slices of frozen neurons. The forward pass is unchanged, so there is zero runtime overhead.

> **Supported module types:** `nn.Linear` (row/column) and 1-D norm weights. Embedding-row freezing is **not** wired into the current loop — do not pass embedding indices into `freeze_neurons` until this is fixed.

### `FrozenNeuronHandle`

| Member | Description |
|--------|-------------|
| `n_frozen` | Count of frozen **scalar** weight elements (for logging). |
| `n_total` | Total model parameter count. |
| `restore_frozen_weights()` | Restores saved slices of frozen neurons. Call **after every optimizer step** when `weight_decay > 0`; not needed otherwise. |
| `remove()` | Detach all hooks (effectively un-freezes the neurons). |
| `make_trainer_callback()` | Returns a HuggingFace `TrainerCallback` that calls `restore_frozen_weights` on `on_step_end`. |
| `print_frozen_summary()` | Prints frozen vs trainable counts. |

---

## Saved-artefact layout

All save / load methods are versioned-by-filename so directories can be passed around freely.

### `CriticalNeuronConfig.save_pretrained` / `from_pretrained`

A config directory may contain:

| File | Content |
|------|---------|
| `critical_neuron_config.json` | All category lists + `sparsity_ratio` + `gate_combines_with` + `base_model_name_or_path`. **Does not** include neuron indices. |
| `neuron_indices.json` | Optional. `dict[module_path, list[int]]` — keys are full module paths. |

`NeuronDetector.save(path)` is shorthand for `config.save_pretrained(path)`.

### Adapter checkpoint (`CriticalNeuronModel.save_pretrained`)

| File | Content |
|------|---------|
| `adapter_model.safetensors` | Preferred. Keys: `{module_path}.dW` and `{module_path}.idx`. |
| `adapter_model.pt` | Used when `safetensors` is not installed. |
| `critical_neuron_config.json`, `neuron_indices.json` | As above. |

### Importance cache (`NeuronDetector.save_importance_cache`)

A `torch.save` blob with:

- `version` (int)
- `scores`: `dict[str, Tensor]` — per-module **post–gate-combined** CPU float vectors
- `gate_to_partner_path`: `dict[gate_module_path, partner_module_path]` for mirroring indices after top-$k$

Load with `NeuronDetector.select_from_importance_cache(path)`. Use the **same** module lists and `gate_combines_with` as the run that produced the cache. A raw `dict[str, Tensor]` (no metadata) is still accepted, but the code logs a warning that gate modules may miss mirrored indices.

### `NeuronStatistician.save_report`

After `analyze`, writes:

- `union_neurons.json`, `shared_neurons.json`, `non_shared_neurons.json`
- `exclusive_<task>_neurons.json` — one per task name
- `statistics.csv`

---

## Cookbook

### Compare critical neurons across languages

```python
from critnet import CriticalNeuronConfig, NeuronStatistician

task_indices = {}
for lang in ["en", "zh", "ar", "sw"]:
    cfg = CriticalNeuronConfig.from_pretrained(f"./detected_neurons/{lang}")
    if cfg.neuron_indices is None:
        raise ValueError(f"Missing neuron_indices for {lang}")
    task_indices[lang] = cfg.neuron_indices

stat_config = CriticalNeuronConfig.from_pretrained("./detected_neurons/en")
statistician = NeuronStatistician(model=model, config=stat_config)
result = statistician.analyze(task_indices)
print(result.summary(task_indices))
statistician.save_report("./neuron_analysis")
```

`result.intersection` is the cross-lingual **foundational** neuron set; `result.exclusive["zh"]` is the part of Chinese-critical neurons that is **not** shared with the other three languages.

### Detect → sparse train → merge to a flat checkpoint (PEFT, base model)

Sparse-PEFT demonstrations use `Qwen/Qwen3-4B-Base` — fine-tuning a *base* (non-instruct) model is the realistic setting where a sparse delta needs to learn the most.

```python
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from critnet import CriticalNeuronConfig, NeuronDetector, get_neuron_model, CriticalNeuronModel

MODEL = "Qwen/Qwen3-4B-Base"

# 1. Detect
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
config = CriticalNeuronConfig(
    sparsity_ratio=0.05,
    norm_modules=["input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"],  # Qwen-3
    base_model_name_or_path=MODEL,
)
NeuronDetector(model, config).detect(calibration_loader)
config.save_pretrained("./neurons")
del model

# 2. Sparse PEFT
config = CriticalNeuronConfig.from_pretrained("./neurons")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
wrapped = get_neuron_model(model, config)
wrapped.print_trainable_parameters()
Trainer(
    model=wrapped,
    args=TrainingArguments(output_dir="./ckpts", num_train_epochs=1),
    train_dataset=train_dataset,
    data_collator=collator,
).train()
wrapped.save_pretrained("./my_adapter")

# 3. Reload + merge into a vanilla HF checkpoint
#    (base_model_name_or_path is read from ./my_adapter/critical_neuron_config.json
#     because we set it on `config` during step 1.)
merged = CriticalNeuronModel.from_pretrained(
    "./my_adapter", model_kwargs={"torch_dtype": torch.bfloat16},
)
plain = merged.merge_and_unload()
plain.save_pretrained("./my_merged_model")
```

### Freeze safety neurons during full fine-tuning

```python
import json
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from critnet import freeze_neurons

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(MODEL)
with open("./neurons/safety_indices.json") as f:
    safety_indices = json.load(f)

# `config` is optional: freeze_neurons falls back to a duck-typed norm
# check for module paths the default config does not enumerate, so
# Qwen-3 `q_norm` / `k_norm` entries in `safety_indices` are recognised
# automatically.
handle = freeze_neurons(model, safety_indices)
handle.print_frozen_summary()

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./ft_ckpts", num_train_epochs=3, weight_decay=0.01),
    train_dataset=train_dataset,
    callbacks=[handle.make_trainer_callback()],   # restores frozen weights after each step
)
trainer.train()
```

### Sweep `sparsity_ratio` without rerunning the backward pass

```python
import torch
from transformers import AutoModelForCausalLM
from critnet import CriticalNeuronConfig, NeuronDetector

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()

cfg = CriticalNeuronConfig(
    sparsity_ratio=0.05,
    norm_modules=["input_layernorm", "post_attention_layernorm", "q_norm", "k_norm"],
    base_model_name_or_path=MODEL,
)
det = NeuronDetector(model, cfg)
det.detect(loader, save_importance_cache_path="./cache.pt")  # expensive once

for r in [0.01, 0.02, 0.05, 0.10, 0.20]:
    cfg.sparsity_ratio = r
    det.select_from_importance_cache("./cache.pt")           # cheap
    cfg.save_pretrained(f"./neurons/r{r}")
```
