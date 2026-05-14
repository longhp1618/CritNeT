# CritNet: A Critical Neurons Toolkit

**CritNet** is a PyTorch toolkit for **identifying, analysing, ablating, freezing, and selectively fine-tuning the small set of neurons that carry most of a transformer language model's behaviour** — utility neurons (general reasoning), refusal / safety neurons, language-specific neurons, and any other functional sub-network you can define with a calibration corpus.

It ranks individual neurons — rows or columns of linear projections and elements of layer-norm vectors — by the first-order Taylor importance score $|w \odot \nabla_w \mathcal{L}|$, then lets you:

- **Detect** the global top-$k$ most critical neurons for any dataset.
- **Analyse** how index sets from different datasets overlap (union, intersection, set-difference).
- **Deactivate** a neuron set in-place to study its causal effect (ablation).
- **Freeze** a neuron set during full fine-tuning so its weights never update.
- **Train only** that neuron set as a sparse PEFT adapter — a drop-in alternative to LoRA.

The defaults work out of the box on LLaMA-, Mistral-, and Qwen-style architectures.

---

## Installation

Requires Python ≥ 3.9, PyTorch, and a recent `transformers`. From the repository root:

```bash
pip install -e .
```

The [Quickstart](#quickstart-isolate-the-safety-neurons-of-qwen3-4b-instruct) additionally uses `datasets` and `trl` for loading and tokenizing the public calibration corpora:

```bash
pip install datasets trl
```

For the test suite:

```bash
pip install -e ".[test]"
pytest
```

A single GPU with ≥ 40 GB of memory is enough to run the Quickstart on Qwen3-4B-Instruct in bfloat16.

---

## Toolkit structure

The library is one self-contained Python package:

```
critnet/
├── config.py         CriticalNeuronConfig         target modules + on-disk format
│                     save_neuron_indices,         module-level I/O helpers for the
│                     load_neuron_indices            companion neuron_indices.json
├── detector.py       NeuronDetector               first-order Taylor importance + global top-k
│                     DetectionResult              immutable record returned by detect()
│                     select_neurons_from_cache    cache-only ratio sweep (no model needed)
├── statistician.py   NeuronStatistician           union / intersection / per-task exclusive sets
│                     StatisticsResult             save_report() + summary() + param_coverage()
├── deactivator.py    NeuronDeactivator            zero out selected neurons (ablation)
│                     DeactivationResult
└── model.py          get_neuron_model             sparse-PEFT wrapper (LinearDeltaSubspace, ...)
                      CriticalNeuronModel          save / load / merge adapters
                      freeze_neurons               freeze neurons during full fine-tuning
                      FrozenNeuronHandle
```

All public symbols live on the top-level package:

```python
from critnet import (
    CriticalNeuronConfig,
    NeuronDetector,
    DetectionResult,
    select_neurons_from_cache,
    NeuronStatistician,
    NeuronDeactivator,
    get_neuron_model,
    freeze_neurons,
    CriticalNeuronModel,
    load_neuron_indices,
    save_neuron_indices,
)
```

A full API reference (arguments, on-disk formats, edge cases) lives in [`critnet/README.md`](critnet/README.md).

---

## Design at a glance

The toolkit deliberately separates three concerns that early prototypes conflated:

| Concern | Lives on |
| --- | --- |
| **Architecture** — which modules carry neurons, how they pair, where embeddings live | `CriticalNeuronConfig` |
| **Detection-time hyperparameter** — what fraction of neurons to keep this run | argument to `NeuronDetector.detect(..., sparsity_ratio=...)` |
| **Detection output** — the indices that came out of that run | `DetectionResult` (a plain dataclass) |

The config is therefore reusable across runs at different ratios, and the result is a pure value type that the user is free to ignore, save, mutate, or pass around. No method on the toolkit mutates the config silently.

---

## Quickstart: isolate the safety neurons of `Qwen3-4B-Instruct`

A single runnable script that reproduces one concrete experiment end-to-end: identifying the small subset of `Qwen3-4B-Instruct` neurons that **selectively enforce refusal behaviour**. Zeroing only this ~1.64 % of parameters breaks safety guardrails while leaving core utility benchmarks (ARC-E, PolyMath, MMLU) largely intact.

The safety set is defined as $\mathcal{N}_{s} = \mathcal{N}^{q}_{r} \setminus \mathcal{N}^{p}_{u}$, where $\mathcal{N}^{p}_{u}$ are the top-$p$ critical neurons on a **utility** corpus (1,000 LIMA + 630 s1K-1.1 conversations from [`iNLP-Lab/multilingual-lima`](https://huggingface.co/datasets/iNLP-Lab/multilingual-lima) and [`iNLP-Lab/multilingual-s1`](https://huggingface.co/datasets/iNLP-Lab/multilingual-s1)) and $\mathcal{N}^{q}_{r}$ are the top-$q$ critical neurons on a **refusal** corpus (1,000 harmful-prompt / refusal-response pairs from [`iNLP-Lab/multilingual-safety`](https://huggingface.co/datasets/iNLP-Lab/multilingual-safety)). We use $p = 0.13$ and $q = 0.06$. To replicate the experiment in another language, change every `"en"` below to `"zh"`, `"ar"`, `"sw"`, etc.

Save the following as `quickstart_safety_neurons.py` and run it on a single ≥ 40 GB GPU:

<details>
<summary>DataLoader helper code (click to expand)</summary>

```python
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LANG = "en"
P_UTILITY, Q_REFUSAL = 0.13, 0.06

tokenizer = AutoTokenizer.from_pretrained(MODEL)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

DEFAULT_SYS = "You are a helpful assistant."
REASONING_SYS = "You are a helpful assistant. Please reason step by step, and put your final answer within \\boxed{}."

def to_chat(ds):
    return [[{"role": "system", "content": REASONING_SYS if "\\boxed{" in r["output"][-100:] else DEFAULT_SYS},
             {"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": r["output"]}] for r in ds]

def build_loader(convs, max_len=2048):
    def tok(ex):
        p = tokenizer.apply_chat_template(ex["messages"][:-1], add_generation_prompt=True, tokenize=True)
        f = tokenizer.apply_chat_template(ex["messages"],     add_generation_prompt=False, tokenize=True)
        return {"input_ids": f, "completion_mask": [0]*len(p) + [1]*(len(f)-len(p))}
    ds = Dataset.from_dict({"messages": convs}).map(tok, remove_columns=["messages"])
    ds = ds.filter(lambda r: 0 < len(r["input_ids"]) <= max_len)
    collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id, completion_only_loss=True)
    return DataLoader(ds, batch_size=1, collate_fn=collator, shuffle=False)

utility_convs = (
    to_chat(load_dataset("iNLP-Lab/multilingual-lima", LANG, split="train"))
    + to_chat(load_dataset("iNLP-Lab/multilingual-s1", LANG, split="train"))
)
refusal_convs = to_chat(load_dataset("iNLP-Lab/multilingual-safety", LANG, split="train"))
```

</details>

```python
from critnet import (
    CriticalNeuronConfig, NeuronDetector, NeuronStatistician, NeuronDeactivator,
)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda()
config = CriticalNeuronConfig(base_model_name_or_path=MODEL)

utility = NeuronDetector(model, config).detect(build_loader(utility_convs), sparsity_ratio=P_UTILITY)
refusal = NeuronDetector(model, config).detect(build_loader(refusal_convs), sparsity_ratio=Q_REFUSAL)

stats = NeuronStatistician(model=model, config=config).analyze(
    {"utility": utility.indices, "refusal": refusal.indices},
)
safety_indices = stats.exclusive["refusal"]  # N_s = N^q_r \ N^p_u
print(f"|N_s| = {sum(len(v) for v in safety_indices.values()):,} neurons "
      f"(~{stats.param_coverage(safety_indices):.2f}% of model parameters)")

print(NeuronDeactivator(model, config).deactivate(safety_indices).summary())

prompt = "a harmful question."
inputs = tokenizer.apply_chat_template(
    [{"role": "system", "content": DEFAULT_SYS},
    {"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
).cuda()
model.eval()
print("=== Safety-deactivated ===")
print(tokenizer.decode(model.generate(inputs, max_new_tokens=512)[0], skip_special_tokens=True))
print("\n\n" + "-"*100 + "\n\n")
print("=== Original ===")
orig = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
print(tokenizer.decode(orig.generate(inputs, max_new_tokens=256)[0], skip_special_tokens=True))
print("\n\n" + "-"*100 + "\n\n")
```

You should see roughly **1.64 %** of the parameters flagged as safety-specific, the original model refusing the harmful prompt, and the deactivated model complying — empirical confirmation that this small subset causally implements Qwen3-4B-Instruct's safety behaviour.

---

## Workflow at a glance

```
Phase 1: Detect            Phase 2: Analyse / Ablate / Freeze         Phase 3: Sparse PEFT          Phase 4: Load or merge
NeuronDetector       -->   NeuronStatistician.analyze              -->  get_neuron_model       -->   CriticalNeuronModel
   .detect()                  -> StatisticsResult                       + HF Trainer                 .from_pretrained
DetectionResult            NeuronDeactivator.deactivate                                              .merge_and_unload
select_neurons_from_cache  freeze_neurons -> FrozenNeuronHandle
```

Every phase consumes the same `CriticalNeuronConfig` + `neuron_indices.json` pair on disk, so you can enter at any step once detection is cached. The Quickstart above uses Phases 1 → 2 (detect, analyse, deactivate). For **sparse PEFT** (Phase 3, sparse delta-tuning as a LoRA alternative) and **freeze-tuning** (Phase 2c), plus full argument-level documentation and on-disk formats, see [`critnet/README.md`](critnet/README.md).
