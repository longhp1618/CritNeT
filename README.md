# CritNet: A Critical Neurons Toolkit

This repository is organized into two primary top-level areas:

- **`critnet`**: reusable core library (detector, statistician, deactivator, freeze-tuning, PEFT-style critical-neuron training).
- **`lima_s1_exp`**: optional manuscript experiment suite (LIMA-S1 paper reproductions and benchmarks).

## Install

```bash
pip install -e .
```

This makes both packages importable:

- `critnet`
- `lima_s1_exp`

---

## Using `critnet`

The following is a **general** tutorial for the core API: it does not assume any particular benchmark or paper setup. For full API details (arguments, on-disk formats, edge cases), see [`critnet/README.md`](critnet/README.md).

<details>
<summary><strong>Overview — how it works, imports, workflow</strong></summary>

### How it works

Critical neurons are ranked with a first-order Taylor-style score: for each weight (or neuron slice), importance is proportional to the magnitude of the **elementwise** product **$|w \odot \nabla_w L|$**, reduced to one scalar per neuron depending on layer type. The top fraction **`sparsity_ratio`** of neurons **globally** (across all targeted modules) is kept as “critical.” You can **detect** them, **compare** sets across tasks or conditions, **deactivate** them for ablation, **freeze** them during full fine-tuning, or **train only** those neurons as a sparse PEFT adapter.

**Typical dependencies for the library:** `torch`, `transformers`, `tqdm`; `safetensors` is optional for some save paths.

```python
from critnet import (
    CriticalNeuronConfig,
    NeuronDetector,
    NeuronStatistician,
    NeuronDeactivator,
    CriticalNeuronModel,
    get_neuron_model,
    freeze_neurons,
)
```

### Workflow overview

```
Phase 1: Detect    -->  Phase 2: Analyse    -->  Phase 3: Train / freeze    -->  Phase 4: Load
NeuronDetector          NeuronStatistician       get_neuron_model or          CriticalNeuronModel
                                                 freeze_neurons + Trainer      .from_pretrained / merge
```

You can start at any phase if you already have saved neuron indices.

</details>

<details>
<summary><strong>Phase 1: Detect critical neurons</strong></summary>

**Config** (defaults suit common LLaMA / Mistral / Qwen-style stacks):

```python
import torch
from critnet import CriticalNeuronConfig

config = CriticalNeuronConfig(sparsity_ratio=0.05)
```

**Run detection** on your own dataset: build a `DataLoader` whose batches are **`dict`s** passed straight to `model(**batch)` (see [`NeuronDetector.detect`](critnet/detector.py)). Each batch must include at least:

- **`input_ids`**: `LongTensor`, shape `[batch, seq]`.
- **`attention_mask`**: `LongTensor`, same sequence length (typically `1` for real tokens, `0` for padding).
- **`labels`**: `LongTensor`, same shape as `input_ids`. With **`mode="chat"`**, set prompt / non-target tokens to **`-100`** so the loss is only on the spans you care about (e.g. assistant completion). With **`mode="pre-train"`**, use standard causal LM labels (usually **`labels == input_ids`**). Any other keys your model’s `forward` accepts (e.g. `position_ids`) can be present.

In practice you often use a Hugging Face **dataset + collator** (e.g. language-modeling or SFT collators) so each step yields that dict.

```python
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from critnet import NeuronDetector

model_name = "meta-llama/Llama-3.2-1B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model.to("cuda")

dataloader = DataLoader(dataset, batch_size=1)

detector = NeuronDetector(model, config)
detector.detect(dataloader, mode="chat")
detector.save("./detected_neurons/run_a")
```

Run detection multiple times with different data (e.g. per language, domain, or task) if you want separate index sets to compare. Each `ds_*` must be a PyTorch `Dataset` (or iterable) whose `DataLoader` batches are dicts with **`input_ids`**, **`attention_mask`**, and **`labels`** as above (same contract as a single run):

```python
# ds_a / ds_b: each __getitem__ (or collate_fn output) contributes to batches like:
#   {"input_ids": ..., "attention_mask": ..., "labels": ...}
# labels[..., prompt_positions] == -100 when mode="chat"

for name, dataset in [("domain_a", ds_a), ("domain_b", ds_b)]:
    # Use a collate_fn (e.g. from a HF DataCollator) if samples are not already dicts of tensors.
    dataloader = DataLoader(dataset, batch_size=1)
    detector = NeuronDetector(model, CriticalNeuronConfig(sparsity_ratio=0.05))
    detector.detect(dataloader, mode="chat")
    detector.save(f"./detected_neurons/{name}")
```

</details>

<details>
<summary><strong>Phase 2: Analyse across tasks or conditions</strong></summary>

Compare saved index sets (outer key = arbitrary task name; inner dict = module path → indices, as produced by the detector):

```python
from critnet import CriticalNeuronConfig, NeuronStatistician

task_indices = {}
for task in ["domain_a", "domain_b"]:
    cfg = CriticalNeuronConfig.from_pretrained(f"./detected_neurons/{task}")
    if cfg.neuron_indices is None:
        raise ValueError(f"No neuron_indices in ./detected_neurons/{task}")
    task_indices[task] = cfg.neuron_indices

stat_config = CriticalNeuronConfig.from_pretrained("./detected_neurons/domain_a")
statistician = NeuronStatistician(model=model, config=stat_config)
result = statistician.analyze(task_indices)
print(result.summary(task_indices))
statistician.save_report("./neuron_analysis")
```

Typical outputs: `union_neurons.json`, `shared_neurons.json`, `exclusive_<task>_neurons.json`, `non_shared_neurons.json`, `statistics.csv`.

</details>

<details>
<summary><strong>Phase 2b: Deactivate neurons (ablation)</strong></summary>

Use a `CriticalNeuronConfig` whose **module lists match detection** and whose **`neuron_indices`** are the set to zero out. If you saved a full detection directory, `from_pretrained` loads both config and indices. If you only have a JSON map (e.g. `union_neurons.json` from `save_report`), load it into a config with the same `row_modules` / `column_modules` / `norm_modules` as detection:

```python
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from critnet import CriticalNeuronConfig, NeuronDeactivator

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

config = CriticalNeuronConfig.from_pretrained("./detected_neurons/run_a")
# Or: config = CriticalNeuronConfig(sparsity_ratio=0.05)
#     with open("./neuron_analysis/union_neurons.json") as f:
#         config.neuron_indices = json.load(f)

deactivator = NeuronDeactivator(model, config)
result = deactivator.deactivate()  # or deactivate(neuron_indices={...})
print(result.summary())
deactivator.save_pretrained("./deactivated_model", tokenizer=tokenizer)
```

</details>

<details>
<summary><strong>Phase 2c: Freeze neurons during full fine-tuning</strong></summary>

Keep the full model trainable but **block gradient updates** on a chosen index set (any JSON mapping module paths → index lists). Gradient hooks plus an optional Trainer callback counteract weight decay on frozen slices.

```python
import json
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from critnet import CriticalNeuronConfig, freeze_neurons

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
with open("path/to/frozen_neuron_indices.json") as f:
    frozen_indices = json.load(f)

config = CriticalNeuronConfig()
handle = freeze_neurons(model, frozen_indices, config)
handle.print_frozen_summary()

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="./ckpts", num_train_epochs=3),
    train_dataset=train_dataset,
    callbacks=[handle.make_trainer_callback()],
)
trainer.train()
model.save_pretrained("./ft_checkpoint")
```

Reproducible **paper** freeze-FT scripts, Accelerate YAMLs, and evaluation drivers live under **`lima_s1_exp`** (see [`lima_s1_exp/README.md`](lima_s1_exp/README.md)).

</details>

<details>
<summary><strong>Phase 3: Fine-tune only critical neurons (sparse PEFT)</strong></summary>

```python
from critnet import CriticalNeuronConfig, get_neuron_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch

base_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_id)

config = CriticalNeuronConfig.from_pretrained("./neuron_analysis")
model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16)
wrapped = get_neuron_model(model, config)
wrapped.print_trainable_parameters()

# train_dataset + data_collator are yours: batches must include input_ids, attention_mask,
# and labels (same idea as detection — labels with -100 where you do not want loss).
training_args = TrainingArguments(
    output_dir="./cn_checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    learning_rate=2e-4,
    logging_steps=10,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=wrapped,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
trainer.train()
wrapped.save_pretrained("./my_adapter")
```

</details>

<details>
<summary><strong>Phase 4: Load adapter or merge to a full checkpoint</strong></summary>

```python
from critnet import CriticalNeuronModel
import torch

model = CriticalNeuronModel.from_pretrained(
    base_model_name_or_path="meta-llama/Llama-3.2-1B-Instruct",
    adapter_path="./my_adapter",
    model_kwargs={"torch_dtype": torch.bfloat16},
)
# merged = model.merge_and_unload()
# merged.save_pretrained("./merged_model")
```

</details>

---

<details>
<summary><strong>Experiment layout (<code>lima_s1_exp</code>)</strong></summary>

Optional code paths that pair the toolkit with fixed datasets, configs, and shell entrypoints for the LIMA-S1 manuscript:

- `exp1_utility_neuron_ablation_en` — utility-neuron ablation (English MMLU-style utility).
- `exp2_refusal_utility_grid_search` — refusal vs. utility neuron ratio grid.
- `exp3_safety_neuron_deactivation` — safety-neuron deactivation + eval.
- `exp4_freeze_ft_attack` — full fine-tuning with frozen neuron subsets + eval.
- `exp5_multilingual_critical_foundational` — multilingual detection, statistics, tuning.
- `eval`, `configs`, `ft_datasets`, `scripts` — shared utilities and runners.

See [`lima_s1_exp/README.md`](lima_s1_exp/README.md) for **experiment-level** goals, scripts, and expected artifacts.

</details>
