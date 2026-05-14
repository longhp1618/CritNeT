---
pretty_name: Multilingual LIMA Dataset
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

# Multilingual LIMA Dataset

Each language is stored as a separate Hugging Face **config**, translated by `google/gemini-2.0-flash-001`.

## Usage

```python
from datasets import load_dataset

ds_en = load_dataset("longhp1618/multilingual-lima", "en")
ds_zh = load_dataset("longhp1618/multilingual-lima", "zh")
```