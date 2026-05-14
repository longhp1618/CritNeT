"""
LM data pipeline built on HuggingFace `datasets` + TRL collator / truncation.

- **Pre-training**: `raw` is a list of strings → `text` → batched `tokenizer` → `input_ids`.
- **Chat SFT**: `raw` is a list of conversations `[{role, content}, ...]` with **last** turn `assistant`
  → `input_ids` + `completion_mask` (same idea as TRL `SFTTrainer`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from trl import truncate_dataset
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

Conversation = List[Dict[str, Any]]
RawData = Union[Sequence[str], Sequence[Conversation]]


def _pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    raise ValueError("pad_token_id is not set")
    # if tokenizer.eos_token_id is not None:
    #     return int(tokenizer.eos_token_id)
    # return 0


def _normalize_chat_token_ids(ids) -> List[int]:
    """Align with TRL SFTTrainer (LLM flat list vs VLM nested)."""
    if torch.is_tensor(ids):
        return ids.flatten().tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        return ids[0]
    return list(ids)


def _tokenize_chat_example(example: dict, tokenizer) -> dict:
    messages: Conversation = example["messages"]
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError('Chat data: each row["messages"] must end with role "assistant".')
    prompt_msgs, completion_msgs = messages[:-1], [messages[-1]]
    prompt_ids = _normalize_chat_token_ids(
        tokenizer.apply_chat_template(
            prompt_msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            enable_thinking=False,
        )
    )
    full_ids = _normalize_chat_token_ids(
        tokenizer.apply_chat_template(
            prompt_msgs + completion_msgs,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            enable_thinking=False,
        )["input_ids"]
    )
    completion_mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
    return {"input_ids": full_ids, "completion_mask": completion_mask}


def _tokenize_pt_batch(batch: dict, tokenizer) -> dict:
    out = tokenizer(batch["text"], add_special_tokens=True)
    return {"input_ids": out["input_ids"]}


def build_lm_dataset(
    raw: RawData,
    tokenizer,
    max_length: int,
) -> Dataset:
    """
    Return a tokenized [`datasets.Dataset`] (columns for [`DataCollatorForLanguageModeling`]).
    """
    if len(raw) == 0:
        raise ValueError("raw must be non-empty.")

    first = raw[0]
    if isinstance(first, str):
        ds = Dataset.from_dict({"text": list(raw)})
        ds = ds.map(
            lambda b: _tokenize_pt_batch(b, tokenizer),
            batched=True,
            remove_columns=["text"],
        )
    elif isinstance(first, (list, tuple)):
        ds = Dataset.from_dict({"messages": list(raw)})
        ds = ds.map(
            _tokenize_chat_example,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=["messages"],
        )
    else:
        raise TypeError("Each item must be str (pretrain) or list of {role, content} (chat SFT).")

    return truncate_dataset(ds, max_length)


def make_collator(tokenizer) -> DataCollatorForLanguageModeling:
    return DataCollatorForLanguageModeling(pad_token_id=_pad_token_id(tokenizer), completion_only_loss=True)


def collate_fn(batch, MAX_LEN, tokenizer):
    del MAX_LEN
    pad_id = _pad_token_id(tokenizer)
    collator = make_collator(tokenizer)
    cleaned = []
    for item in batch:
        if item is None:
            continue
        row = {}
        for k in ("input_ids", "labels", "completion_mask"):
            if k not in item:
                continue
            v = item[k]
            if torch.is_tensor(v):
                v = v.flatten().tolist()
            row[k] = list(v)
        if "input_ids" in row:
            cleaned.append(row)
    if not cleaned:
        return {
            "input_ids": torch.tensor([[pad_id]], dtype=torch.long),
            "attention_mask": torch.tensor([[0]], dtype=torch.long),
            "labels": torch.tensor([[-100]], dtype=torch.long),
        }
    return collator(cleaned)


def create_dataloader(
    raw: RawData,
    tokenizer,
    bs: int,
    MAX_LEN: int,
    lang_lst: Optional[Sequence[Any]] = None,
    shuffle: bool = True,
    **kwargs,
):
    del lang_lst, kwargs
    ds = build_lm_dataset(raw, tokenizer, MAX_LEN)
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, MAX_LEN, tokenizer),
    )
