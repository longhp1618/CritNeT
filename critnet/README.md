# CritNet: A Critical Neurons Toolkit — API reference

This document is the **detailed** reference for the `critnet` package: public symbols, arguments, return types, on-disk formats, and behavior notes tied to the implementation.

For repository layout, installation, and a **high-level tutorial** (detect → analyse → train), see the root [README.md](../README.md).

**Dependencies:** `torch`, `transformers`, `tqdm`. `safetensors` is optional (adapter save/load and `NeuronDeactivator` checkpoints still work with fallbacks where applicable).

---

<details>
<summary><strong>Public API index</strong></summary>

Every name below is exported from `critnet` (see `__all__` in `__init__.py`).

| Symbol | Role |
|--------|------|
| `CriticalNeuronConfig` | Target modules, sparsity ratio, gate pairing, neuron indices, save/load. |
| `NeuronDetector` | First-order importance scores and global top-k selection. |
| `NeuronStatistician` | Union / intersection / exclusive sets across tasks or languages. |
| `StatisticsResult` | Structured output of `NeuronStatistician.analyze`. |
| `NeuronDeactivator` | Zero-out selected neurons in-place (ablation). |
| `DeactivationResult` | Summary of `NeuronDeactivator.deactivate`. |
| `get_neuron_model` | Wrap a HF model so only critical-neuron deltas train. |
| `CriticalNeuronModel` | Thin wrapper: save/load adapter, merge, Trainer-friendly. |
| `freeze_neurons` | Freeze selected neurons during **full** fine-tuning (gradient hooks). |
| `FrozenNeuronHandle` | Hook lifecycle + Trainer callback for `freeze_neurons`. |
| `DEFAULT_SKIP_MODULES` | Default `{"lm_head", "embed_tokens"}` for `get_neuron_model`. |
| `LinearDeltaSubspace` | Row/column sparse delta on `nn.Linear`. |
| `NormDeltaSubspace` | Sparse delta on 1-D norm weights. |
| `EmbeddingDeltaSubspace` | Sparse delta on selected embedding rows. |

```python
from critnet import (
    CriticalNeuronConfig,
    NeuronDetector,
    NeuronStatistician,
    StatisticsResult,
    NeuronDeactivator,
    DeactivationResult,
    CriticalNeuronModel,
    get_neuron_model,
    freeze_neurons,
    FrozenNeuronHandle,
    DEFAULT_SKIP_MODULES,
    LinearDeltaSubspace,
    NormDeltaSubspace,
    EmbeddingDeltaSubspace,
)
```

</details>

<details>
<summary><strong>Conceptual background</strong></summary>

### Importance score

For each targeted parameter tensor, the detector uses the elementwise product `|W ⊙ ∇W|` (with the gradient from the loss). That tensor is **reduced to one scalar per neuron** according to module type:

- **Row** (`q_proj`, `up_proj`, …): sum over the input dimension → one score per output row.
- **Column** (`o_proj`, `down_proj`, …): sum over the output dimension → one score per input column.
- **Norm**: one score per element of the 1-D weight.
- **Embedding** (if configured): same idea as row-like reduction when the weight is 2-D.

### Global top-k

Scores from **all** targeted modules are pooled into one list. The library keeps the top `k` neurons with

`k = max(1, int(total_neuron_count * sparsity_ratio))`.

So `sparsity_ratio` is a **fraction of all pooled neurons**, not per layer.

### SwiGLU gate combination

If `gate_combines_with` maps e.g. `gate_proj` → `up_proj`, the gate module’s score vector is **added** to the partner’s (same shape) before top-k; the gate entry is removed from the score dict, and after selection the gate module receives a **copy** of the partner’s index list.

### `NeuronDetector.detect` and gradients

`detect` calls `model.zero_grad()` once, then iterates the dataloader and calls `loss.backward()` **per batch without zeroing between batches**. Gradients therefore **accumulate** across batches; the importance step uses those accumulated gradients. The `mode` argument (`"chat"` vs `"pre-train"`) is only documentation for how you should build `labels`; masking is your dataset/collator’s job.

</details>

<details>
<summary><strong>On-disk layout</strong></summary>

### `CriticalNeuronConfig.save_pretrained` / `from_pretrained`

Directory may contain:

| File | Content |
|------|---------|
| `critical_neuron_config.json` | `row_modules`, `column_modules`, `norm_modules`, `embedding_modules`, `sparsity_ratio`, `gate_combines_with`, `base_model_name_or_path`. **No** `neuron_indices` inside this file. |
| `neuron_indices.json` | Optional. `dict[module_path, list[int]]` — full module paths as keys. |

`NeuronDetector.save(path)` delegates to `config.save_pretrained(path)`.

### Adapter checkpoint (`CriticalNeuronModel.save_pretrained`)

| File | Content |
|------|---------|
| `adapter_model.safetensors` | Preferred. Keys like `{module_path}.dW` and `{module_path}.idx`. |
| `adapter_model.pt` | Written if `safetensors` is not installed. |
| `critical_neuron_config.json`, `neuron_indices.json` | Same as above. |

### Importance cache (`NeuronDetector.save_importance_cache`)

A `torch.save` blob with:

- `version` (int)
- `scores`: `dict[str, Tensor]` — per-module **post–gate-combined** CPU float vectors
- `gate_to_partner_path`: `dict[gate_module_path, partner_module_path]` for mirroring indices after top-k

Load with `NeuronDetector.select_from_importance_cache(path)` using a `CriticalNeuronConfig` whose module lists and `gate_combines_with` match the run that produced the cache. If the file is a raw `dict[str, Tensor]` without metadata, the code logs a warning that gate modules may miss mirrored indices.

### `NeuronStatistician.save_report`

After `analyze`, writes:

- `union_neurons.json`, `shared_neurons.json` (intersection), `non_shared_neurons.json`
- `exclusive_{task}_neurons.json` per task name
- `statistics.csv`

</details>

<details>
<summary><strong><code>CriticalNeuronConfig</code></strong></summary>

Dataclass in `config.py`. Module **suffixes** (leaf names) are matched against `named_modules()` paths.

### Fields

| Field | Type | Default / behavior |
|-------|------|---------------------|
| `row_modules` | `list[str] \| None` | If `None`, defaults to `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`. |
| `column_modules` | `list[str] \| None` | If `None`, `o_proj`, `down_proj`. |
| `norm_modules` | `list[str] \| None` | If `None`, `input_layernorm`, `post_attention_layernorm`. |
| `embedding_modules` | `list[str] \| None` | Default `None` (not used). Pass a list to enable; `[]` disables. |
| `sparsity_ratio` | `float` | Default `0.05`. Must satisfy `0 < sparsity_ratio < 1`. |
| `gate_combines_with` | `dict[str, str] \| None` | If `None`, built from `DEFAULT_GATE_COMBINES_WITH` (`gate_proj` → `up_proj`) only when both suffixes appear in row/column lists. `{}` disables combination. |
| `neuron_indices` | `dict[str, list[int]] \| None` | Filled by `NeuronDetector` or loaded from disk. |
| `base_model_name_or_path` | `str \| None` | Optional metadata. |

### Validation (`__post_init__`)

- `sparsity_ratio` in `(0, 1)`.
- No module suffix may appear in more than one of `row_modules`, `column_modules`, `norm_modules`, `embedding_modules`.
- Every key and value of `gate_combines_with` must be a suffix in `row_modules` or `column_modules`.

### Smart defaults

- `None` for a module list → fill with architecture defaults.
- Explicit `[]` → truly empty category (e.g. `norm_modules=[]` skips norms).
- `embedding_modules` stays `None` unless you set it.

### Properties and methods

- **`target_modules`** — flat union of all configured suffixes.
- **`linear_modules`** — `row_modules + column_modules`.
- **`get_module_type(module_name: str) -> str`** — `"row"`, `"column"`, `"norm"`, or `"embedding"` from the **leaf** name; raises `ValueError` if unknown.
- **`matches_target(module_name: str) -> bool`** — whether the leaf is in `target_modules`.
- **`summary() -> str`** — human-readable dump.
- **`save_pretrained(save_directory: str)`** — writes JSON files (see above).
- **`from_pretrained(load_directory: str) -> CriticalNeuronConfig`** — class method; loads optional `neuron_indices.json`.

`embed_tokens` and `lm_head` are **not** excluded by the config itself; they are skipped by **`get_neuron_model`** via `DEFAULT_SKIP_MODULES` unless you override `modules_to_skip`.

</details>

<details>
<summary><strong><code>NeuronDetector</code></strong></summary>

### Constructor

`NeuronDetector(model: nn.Module | None, config: CriticalNeuronConfig)`

- **`detect`** requires `model` not `None`.
- For **`select_from_importance_cache`** only, `model` may be `None`.

### `detect(dataloader, mode="chat", save_importance_cache_path=None) -> dict[str, list[int]]`

| Argument | Description |
|----------|-------------|
| `dataloader` | Yields dict batches with at least `input_ids`, `attention_mask`, `labels`. |
| `mode` | `"chat"` or `"pre-train"` (validated); informs expected label layout only. |
| `save_importance_cache_path` | If set, after gate combination saves the combined score dict to this path (see cache format). |

Returns sorted indices per module path and sets `config.neuron_indices`.

### `save(save_path: str)`

Calls `config.save_pretrained(save_path)`.

### `save_importance_cache(combined_scores, path: str)`

Low-level: persists `combined_scores` plus `gate_to_partner_path` from the last `detect` run. Normally you pass `save_importance_cache_path` into `detect` instead.

### `select_from_importance_cache(path: str) -> dict[str, list[int]]`

Loads cache, runs `_global_topk` with **`config.sparsity_ratio`**, sets `config.neuron_indices`, returns indices.

</details>

<details>
<summary><strong><code>NeuronStatistician</code> and <code>StatisticsResult</code></strong></summary>

### Constructor

`NeuronStatistician(model: nn.Module | None = None, config: CriticalNeuronConfig | None = None)`

- With **`model`**, neuron totals and parameter coverage use real weight shapes.
- With **`config`**, `get_module_type` resolves row vs column vs norm for neuron counts.
- If `model` is `None`, `total_neurons_per_module`, `params_per_neuron`, and `total_model_params` are empty/zero; percentage breakdowns in `summary` / CSV are still consistent but param coverage is limited.

### `analyze(task_indices: dict[str, dict[str, list[int]]]) -> StatisticsResult`

| Key level | Meaning |
|-----------|---------|
| Outer | Task or language name (`"en"`, `"lima_s1"`, …). |
| Inner | Full module path → list of neuron indices (same structure as `neuron_indices`). |

Raises `ValueError` if `task_indices` is empty.

Computes per module: **union**, **intersection** (shared), per-task **exclusive** (task set minus intersection), **non_shared** (union minus intersection).

### `save_report(save_directory: str)`

Requires **`analyze`** to have been called. Writes the JSON/CSV files listed in [On-disk layout](#on-disk-layout).

### `StatisticsResult` attributes

| Attribute | Type | Meaning |
|-----------|------|---------|
| `union` | `dict[str, list[int]]` | Union across tasks. |
| `intersection` | `dict[str, list[int]]` | Shared across all tasks. |
| `exclusive` | `dict[str, dict[str, list[int]]]` | Per-task exclusive maps. |
| `non_shared` | `dict[str, list[int]]` | Union minus intersection. |
| `total_neurons_per_module` | `dict[str, int]` | Total neurons per module (from model if available). |
| `params_per_neuron` | `dict[str, int]` | Scalar weights per neuron index in that module. |
| `total_model_params` | `int` | Full model parameter count. |
| `task_names` | `list[str]` | Order of tasks. |

**Properties:** `union_count`, `intersection_count`, `non_shared_count`, `total_neurons`.

**Methods:**

- **`exclusive_count(task: str) -> int`**
- **`task_count(task, task_indices) -> int`**
- **`param_coverage(indices: dict[str, list[int]]) -> float`** — percent of total model parameters touched.
- **`summary(task_indices=None) -> str`** — pass `task_indices` for per-task lines in the text report.

</details>

<details>
<summary><strong><code>NeuronDeactivator</code> and <code>DeactivationResult</code></strong></summary>

### Constructor

`NeuronDeactivator(model: nn.Module, config: CriticalNeuronConfig)`

Model must be a plain HF-style module stack (not `CriticalNeuronModel`). `config` supplies module categories for zeroing axes.

### `deactivate(neuron_indices: dict[str, list[int]] | None = None) -> DeactivationResult`

Uses `neuron_indices` or `config.neuron_indices`. **In-place** zeroing:

| Module type | Effect |
|-------------|--------|
| row | `W[idx, :] = 0` (2-D linear) |
| column | `W[:, idx] = 0` |
| norm | weight `[idx] = 0`; bias `[idx] = 0` if present |
| embedding | `E[idx, :] = 0` (2-D) |

Skips unknown modules with warnings.

### `save_pretrained(save_directory: str, tokenizer=None)`

`model.save_pretrained`, optional `tokenizer.save_pretrained`, and `config.save_pretrained`.

### `DeactivationResult`

| Field | Meaning |
|-------|---------|
| `modules_affected` | Count of modules touched. |
| `neurons_zeroed` | Count of neuron indices processed. |
| `total_weights_zeroed` | Scalar weight elements set to zero. |
| `per_module[path]` | `{"neurons", "weights", "module_type"}`. |

**`summary() -> str`** prints aggregate counts.

</details>

<details>
<summary><strong><code>get_neuron_model</code> and delta wrappers</strong></summary>

### `get_neuron_model(model, config, modules_to_skip=None) -> CriticalNeuronModel`

| Argument | Description |
|----------|-------------|
| `model` | e.g. `AutoModelForCausalLM`. |
| `config` | `neuron_indices` must be non-`None`. |
| `modules_to_skip` | `set` of **leaf** names skipped for wrapping. Default `DEFAULT_SKIP_MODULES` = `frozenset({"lm_head", "embed_tokens"})` so fused kernels / weight-tying paths that bypass submodule `forward` still see real `nn.Linear` weights. Pass `set()` to attempt wrapping all targets. |

Only modules that **`matches_target`**, are **not** skipped, appear in **`neuron_indices`**, and have **non-empty** indices are replaced.

Wrapping uses `LinearDeltaSubspace`, `NormDeltaSubspace`, or `EmbeddingDeltaSubspace` depending on `get_module_type`. Then all parameters are frozen and only each wrapper’s `dW` is trainable; `enable_input_require_grads()` is called when present.

### `LinearDeltaSubspace(base_linear, indices, mode="row", train_bias=False)`

- **`mode`**: `"row"` → `dW` shape `[k, in_features]`; forward adds row deltas via `index_add`.
- **`mode`**: `"column"` → `dW` shape `[out_features, k]`; forward adds `x[..., idx] @ dW.T`.
- **`train_bias`**: if `False`, base bias is frozen when present.
- **`merge_to_linear_() -> nn.Linear`** — add `dW` into base weight in-place.

### `NormDeltaSubspace(base_norm, indices)`

Forward applies a multiplicative-style correction on selected positions; **`merge_to_norm_()`** adds `dW` into base weight.

### `EmbeddingDeltaSubspace(base_embedding, indices)`

Only selected vocabulary rows have a trainable delta; **`merge_to_embedding_()`** merges into `base_embedding.weight`.

</details>

<details>
<summary><strong><code>freeze_neurons</code> and <code>FrozenNeuronHandle</code></strong></summary>

### `freeze_neurons(model, neuron_indices, config=None) -> FrozenNeuronHandle`

| Argument | Description |
|----------|-------------|
| `model` | Full trainable model (typically all `requires_grad=True`). |
| `neuron_indices` | `dict[module_path, list[int]]` — neurons to **freeze** (no gradient updates). |
| `config` | Optional `CriticalNeuronConfig` for row/column/norm typing; default constructed if omitted. |

Registers **backward hooks** on weights to zero frozen components’ gradients. Forward pass is unchanged.

**Supported in implementation:** `nn.Linear` (row/column) and modules passing **`_is_norm`** (1-D weight and name hints). **Embedding layers are not handled** by the current loop; do not rely on `freeze_neurons` for embedding neuron indices until supported.

For modules not in `config` targets, the code falls back to heuristics (`_is_norm`, or linear with column vs row guess from leaf name).

### `FrozenNeuronHandle`

| Member | Description |
|--------|-------------|
| `n_frozen` | Count of frozen **scalar** weight elements used for logging. |
| `n_total` | Total model parameter count. |
| `restore_frozen_weights()` | After each optimizer step if **weight decay** would move frozen weights; restores saved slices. |
| `remove()` | Detach hooks. |
| `make_trainer_callback()` | HF `TrainerCallback` that calls `restore_frozen_weights` on `on_step_end`. |
| `print_frozen_summary()` | Prints frozen vs trainable counts. |

</details>

<details>
<summary><strong><code>CriticalNeuronModel</code></strong></summary>

Thin `nn.Module` around the inner HF model.

| Attribute | Description |
|-----------|-------------|
| `model` | Inner causal LM (or whatever was wrapped). |
| `peft_config` | The `CriticalNeuronConfig` used for wrapping. |

| Method | Description |
|--------|-------------|
| `forward`, `generate` | Delegated to `model`. |
| `__getattr__` | Proxies unknown attributes to `model` (Trainer compatibility). |
| `get_adapter_state_dict()` | OrderedDict of `*.dW` and `*.idx` from delta modules. |
| `save_pretrained(save_directory, **kwargs)` | Adapter + config JSONs (see on-disk layout). |
| `from_pretrained(base_model_name_or_path, adapter_path, model_kwargs=None, modules_to_skip=None)` | Class method: load base, `get_neuron_model`, load safetensors or `.pt` into `dW`. |
| `merge_and_unload()` | In-place merge of all delta wrappers; returns plain `nn.Module`. |
| `print_trainable_parameters()` | Prints trainable vs total counts. |

</details>

<details>
<summary><strong>Cookbook snippets</strong></summary>

### Statistician after per-language detection

Use a config consistent with detection (e.g. same defaults as saved under each run). **`task_indices` values must be the inner `neuron_indices` dicts, not the whole config.**

```python
from critnet import CriticalNeuronConfig, NeuronStatistician

task_indices = {}
for lang in ["en", "zh", "th", "bn"]:
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

### Detect → sparse train → merge

```python
import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from critnet import CriticalNeuronConfig, NeuronDetector, get_neuron_model, CriticalNeuronModel

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
config = CriticalNeuronConfig(sparsity_ratio=0.05)
NeuronDetector(model, config).detect(dataloader, mode="chat")
config.save_pretrained("./neurons")

del model
config = CriticalNeuronConfig.from_pretrained("./neurons")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
wrapped = get_neuron_model(model, config)
Trainer(model=wrapped, args=TrainingArguments(output_dir="./ckpts", num_train_epochs=1), ...).train()
wrapped.save_pretrained("./my_adapter")

merged = CriticalNeuronModel.from_pretrained(MODEL, "./my_adapter", model_kwargs={"torch_dtype": torch.bfloat16})
plain = merged.merge_and_unload()
```

Paper-aligned **freeze** fine-tuning and eval live under `lima_s1_exp` (see root [README.md](../README.md)).

</details>
