---
pretty_name: Multilingual s1 Dataset
language:
- en
- zh
- it
- bn
- ko
- th
- vi
- ar
- jv
- sw
---

# Multilingual s1 Dataset

Each language is stored as a separate Hugging Face **config**, translated by `google/gemini-2.0-flash-001`.

## Usage

```python
from datasets import load_dataset

ds_en = load_dataset("longhp1618/multilingual-s1", "en")
ds_zh = load_dataset("longhp1618/multilingual-s1", "zh")
```