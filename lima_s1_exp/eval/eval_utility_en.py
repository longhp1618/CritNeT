import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.expanduser("~/.cache/torchinductor")
os.environ["TRITON_CACHE_DIR"] = os.path.expanduser("~/.cache/triton")
import re
import time

import pandas as pd
import swifter
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from lima_s1_exp.eval.grader import math_equal
from lima_s1_exp.eval.parser import extract_answer
from lima_s1_exp.utils import ID_LANGS, INTERSECTION_LANGS, marker_answer, marker_question, sys_boxed, sys_role, thinking_prefix


def _grade_math_pair(pair):
    """Helper for parallel math grading (must be top-level for ProcessPoolExecutor pickling)."""
    return math_equal(pair["pred"], pair["gold"], timeout=True)


def _grade_task_worker(path, task, langs):
    """Run grading for one task (used in a subprocess with timeout; must be top-level for pickling)."""
    acc_id = []
    acc_ood = []
    acc_dct = {}
    if task == "math" or task == "math_easy":
        lang_data = []
        all_pairs = []
        for lang in langs:
            df = pd.read_csv(f"{path}/{task}_{lang}.csv")
            golds = df["gold"]
            data_name = "math"
            df["final_pred"] = df["prediction"].swifter.apply(lambda x: extract_answer(str(x) if pd.notna(x) else "", data_name))
            pairs = [{"gold": str(g), "pred": str(p)} for g, p in zip(golds, df["final_pred"])]
            df["pair"] = pairs
            lang_data.append((lang, df))
            all_pairs.extend(pairs)
        n_workers = min(64, len(all_pairs), (os.cpu_count() or 8) * 2)
        pair_timeout = 5
        desc = "Grading math_easy" if task == "math_easy" else "Grading math"
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_grade_math_pair, p) for p in all_pairs]
            all_checks = []
            for f in tqdm(futures, desc=desc, unit="pair"):
                try:
                    all_checks.append(f.result(timeout=pair_timeout))
                except FuturesTimeoutError:
                    all_checks.append(False)
        offset = 0
        for lang, df in lang_data:
            n = len(df)
            df["check"] = all_checks[offset : offset + n]
            offset += n
            acc = df["check"].mean() * 100
            df.to_csv(f"{path}/{task}_{lang}.csv", index=False)
            acc_dct[lang] = acc
            if lang in ID_LANGS:
                acc_id.append(acc)
            else:
                acc_ood.append(acc)
    else:
        for lang in langs:
            df = pd.read_csv(f"{path}/{task}_{lang}.csv")
            golds = df["gold"]
            data_name = "mmlu" if task == "mmlu" else "mmlu_stem"
            df["final_pred"] = df["prediction"].swifter.apply(lambda x: extract_answer(str(x) if pd.notna(x) else "", data_name))
            df["pair"] = [{"gold": str(g), "pred": str(p)} for g, p in zip(golds, df["final_pred"])]
            df["check"] = df["pair"].swifter.apply(lambda x: str(x["pred"]).lower().strip() == str(x["gold"]).lower().strip())
            acc = df["check"].mean() * 100
            df.to_csv(f"{path}/{task}_{lang}.csv", index=False)
            acc_dct[lang] = acc
            if lang in ID_LANGS:
                acc_id.append(acc)
            else:
                acc_ood.append(acc)
    return (acc_dct, acc_id, acc_ood)


answer_math = {
    "ar": "\nالإجابة النهائية هي $\\boxed{",
    "bn": "\nচূড়ান্ত উত্তর হলো $\\boxed{",
    "de": "\nDie endgültige Antwort ist $\\boxed{",
    "en": "\nThe final answer is $\\boxed{",
    "es": "\nLa respuesta final es $\\boxed{",
    "fr": "\nLa réponse finale est $\\boxed{",
    "id": "\nJawaban akhir adalah $\\boxed{",
    "it": "\nLa risposta finale è $\\boxed{",
    "ja": "\n最終的な答えは $\\boxed{",
    "ko": "\n최종 답은 $\\boxed{",
    "pt": "\nA resposta final é $\\boxed{",
    "sw": "\nJibu la mwisho ni $\\boxed{",
    "zh": "\n最终答案是$\\boxed{",
}

answer_mqc4 = {
    "ar": "\nالإجابة النهائية (حرف واحد: A, B, C, D) هي $\\boxed{",
    "bn": "\nচূড়ান্ত উত্তর (একটি একক অক্ষর: A, B, C, D) $\\boxed{",
    "de": "\nDie endgültige Antwort (ein einzelner Buchstabe: A, B, C, D) ist $\\boxed{",
    "en": "\nThe final answer (a single character: A, B, C, D) is $\\boxed{",
    "es": "\nLa respuesta final (un solo carácter: A, B, C, D) está en $\\boxed{",
    "fr": "\nLa réponse finale (un seul caractère : A, B, C, D) est $\\boxed{",
    "id": "\nJawaban akhir (satu karakter: A, B, C, D) adalah $\\boxed{",
    "it": "\nLa risposta finale (un singolo carattere: A, B, C, D) è $\\boxed{",
    "ja": "\n最終的な答え（1文字：A、B、C、D）は$\\boxed{",
    "ko": "\n최종 답(한 문자: A, B, C, D)은 $\\boxed{",
    "pt": "\nA resposta final (um único caractere: A, B, C, D) é $\\boxed{",
    "sw": "\nJibu la mwisho (herufi moja: A, B, C, D) limebandikwa $\\boxed{",
    "zh": "\n最终答案（单个字符：A、B、C、D）是$\\boxed{",
}


OOD_LANGS = [l for l in INTERSECTION_LANGS if l not in ID_LANGS]  # out-of-distribution
OOD_LANGS = [l for l in INTERSECTION_LANGS if l not in ID_LANGS]  # out-of-distribution

# Benchmark-specific configs — English-only MMLU for this script (Global-MMLU-Lite).
mmlu_langs = ["en"]
math_langs = INTERSECTION_LANGS  # Qwen/PolyMath
belebele_lang_map = {  # facebook/belebele FLORES config names
    "ar": "arb_Arab",
    "bn": "ben_Beng",
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "id": "ind_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "pt": "por_Latn",
    "es": "spa_Latn",
    "sw": "swh_Latn",
}

gpqa_langs = INTERSECTION_LANGS
arc_challenge_langs = INTERSECTION_LANGS
arc_easy_langs = INTERSECTION_LANGS
truthfulqa_langs = INTERSECTION_LANGS

langs_dct_full = {
    "mmlu": mmlu_langs,
    "math": math_langs,
    "math_easy": math_langs,
    "belebele": INTERSECTION_LANGS,
    "gpqa": gpqa_langs,
    "arc_challenge": arc_challenge_langs,
    "arc_easy": arc_easy_langs,
    "truthfulqa": truthfulqa_langs,
}

# This module is English-only utility eval: every supported task uses ``en`` only.
_UTILITY_EN_LANGS = ["en"]
langs_dct = {k: _UTILITY_EN_LANGS for k in langs_dct_full}


def create_option_text(lst):
    choices = ["A", "B", "C", "D"]
    text = ""
    for i, j in zip(choices, lst):
        text += f"\n{i}: {j}"
    return text


def prepare_dataset(tokenizer, lang="en", task="math", zeroshot=False, max_context_length=8192):
    def apply_chat_template(tokenizer, question, sys_prompt):
        chat = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    if task == "math_easy":
        ds = load_dataset("Qwen/PolyMath", lang)
        # Only use low and medium difficulty splits
        easy_splits = concatenate_datasets([ds[split] for split in ["low", "medium"] if split in ds.keys()])
        df = easy_splits.to_pandas()
        df["gold"] = df["answer"]
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    elif task == "math":
        ds = load_dataset("Qwen/PolyMath", lang)
        all_splits = concatenate_datasets([ds[split] for split in ds.keys()])
        df = all_splits.to_pandas()
        df["gold"] = df["answer"]
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    elif task == "mmlu":
        ds = load_dataset("CohereLabs/Global-MMLU-Lite", lang)
        df = ds["test"].to_pandas()
        opt_cols = [c for c in df.columns if c.startswith("option_")]
        df["gold"] = df["answer"]

        def build_question(row):
            q = row["question"]
            for i, col in enumerate(opt_cols):
                if col in row.index and pd.notna(row.get(col)) and str(row[col]).strip():
                    q += f"\n{chr(65 + i)}: {row[col]}"
            return q

        df["question"] = df.apply(build_question, axis=1)
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    elif task == "belebele":
        config = belebele_lang_map[lang]
        ds = load_dataset("facebook/belebele", config)
        df = ds["test"].to_pandas()
        # correct_answer_num is 1-indexed (1,2,3,4); coerce to int in case parquet gives strings
        num_to_letter = {1: "A", 2: "B", 3: "C", 4: "D"}
        df["gold"] = df["correct_answer_num"].astype(int).map(num_to_letter)
        df["question"] = (
            df["flores_passage"]
            + f"\n\n{marker_question[lang]} "
            + df["question"]
            + "\nA: "
            + df["mc_answer1"]
            + "\nB: "
            + df["mc_answer2"]
            + "\nC: "
            + df["mc_answer3"]
            + "\nD: "
            + df["mc_answer4"]
        )
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    elif task == "gpqa":
        config = f"GPQADataset_lighteval_{lang}"
        ds = load_dataset("aialt/MuBench", config)
        df = ds["test"].to_pandas()
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        df["gold"] = df["label"].astype(int).map(idx_to_letter)
        df["question"] = df.apply(lambda row: row["prompt"] + create_option_text(row["choices"]), axis=1)
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    elif task == "arc_challenge":
        config = f"ARCChallengeDataset_lighteval_{lang}"
        ds = load_dataset("aialt/MuBench", config)
        df = ds["test"].to_pandas()
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        df["gold"] = df["label"].astype(int).map(idx_to_letter)
        df["question"] = df.apply(lambda row: row["prompt"] + create_option_text(row["choices"]), axis=1)
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    elif task == "arc_easy":
        config = f"ARCEasyDataset_lighteval_{lang}"
        ds = load_dataset("aialt/MuBench", config)
        df = ds["test"].to_pandas()
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        df["gold"] = df["label"].astype(int).map(idx_to_letter)
        df["question"] = df.apply(lambda row: row["prompt"] + create_option_text(row["choices"]), axis=1)
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    elif task == "truthfulqa":
        config = f"TruthfulQADataset_lighteval_{lang}"
        ds = load_dataset("aialt/MuBench", config)
        df = ds["test"].to_pandas()
        idx_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
        df["gold"] = df["label"].astype(int).map(idx_to_letter)
        df["question"] = df.apply(lambda row: row["prompt"] + create_option_text(row["choices"]), axis=1)
        test_input = df["question"].tolist()
        golds = df["gold"].tolist()
        df = df[["question", "gold"]]
    else:
        raise ValueError(f"Unknown task: {task}")

    thinking_prompt = sys_boxed[lang]

    # Truncate long inputs so prompt + max_new_tokens fits within model context
    task_max_new = 768 + 128 + 32 if zeroshot else 1536 + 128 + 32
    max_prompt_tokens = max_context_length - task_max_new
    if task in ("belebele") and zeroshot:
        suffix_str = f"\n{thinking_prompt}\n\n{marker_answer[lang]}"
        overhead = len(tokenizer.encode(suffix_str, add_special_tokens=False))
    elif task in ("math", "math_easy", "mmlu", "gpqa", "arc_challenge", "arc_easy", "truthfulqa") and zeroshot:
        prefix_str = f"{thinking_prompt}\n{marker_question[lang]} "
        suffix_str = f"\n\n{marker_answer[lang]} {thinking_prefix[lang]}"
        overhead = len(tokenizer.encode(prefix_str + suffix_str, add_special_tokens=False))
    else:
        # chat template for math, mmlu, belebele
        _sys = sys_role[lang] + "\n" + thinking_prompt
        _template_only = apply_chat_template(tokenizer, "", _sys)
        overhead = len(tokenizer.encode(_template_only, add_special_tokens=False))
    max_content_tokens = max(0, max_prompt_tokens - overhead)
    truncated = []
    for i in test_input:
        tok = tokenizer.encode(i, add_special_tokens=False)
        if len(tok) <= max_content_tokens:
            truncated.append(i)
        else:
            truncated.append(tokenizer.decode(tok[:max_content_tokens]))
    test_input = truncated
    df["question"] = test_input  # saved CSV reflects truncated input

    if zeroshot:
        print("Using zeroshot CoT.")
        if task == 'belebele':
            prompts = [f"{i}\n{thinking_prompt}\n\n{marker_answer[lang]}" for i in test_input]
        else:
            prompts = [f"{thinking_prompt}\n{marker_question[lang]} {i}\n\n{marker_answer[lang]} {thinking_prefix[lang]}" for i in test_input]
    else:
        print("Using chat template")
        sys_prompt = sys_role[lang] + "\n" + thinking_prompt
        prompts = [apply_chat_template(tokenizer, f"{i}", sys_prompt) for i in test_input]

    print("Num questions:", len(prompts))
    return prompts, golds, df


def inference(llm, prompts, lang, zeroshot, task, tokenizer=None, max_context_length=8192):
    if task == "math" or task == "math_easy":
        answer_begin = answer_math[lang]
    else:
        answer_begin = answer_mqc4[lang]

    if zeroshot:
        stop_tokens = [marker_question[lang], marker_answer[lang], thinking_prefix[lang]]
        sampling_params = {"temperature": 0.0, "max_new_tokens": 768, "stop": stop_tokens}
    else:
        sampling_params = {"temperature": 0.0, "max_new_tokens": 1536, "stop": ["<|im_end|>", "<|eot_id|>"]}

    outputs = llm.generate(prompts, sampling_params)
    pred_texts = [out["text"] for out in outputs]
    initial_completions = pred_texts

    final_completions = [None] * len(initial_completions)
    modified_questions = []
    indices_to_modify = []
    first_completions = []

    for idx, comp in enumerate(initial_completions):
        if task == "math" or task == "math_easy":
            check = bool(re.search(r"\\boxed\{.*?\}", comp))
        elif task == "mmlu":
            targets = [f"\\boxed{{{c}}}" for c in "ABCDEFGHIJ"]
            check = any(t in comp for t in targets)
        else:
            targets = ["\\boxed{A}", "\\boxed{B}", "\\boxed{C}", "\\boxed{D}"]
            check = any(t in comp for t in targets)
        if check:
            final_completions[idx] = comp
        else:
            modified_question = prompts[idx] + comp + answer_begin
            # Retry input must fit in context; truncate prompt part if needed
            if tokenizer is not None:
                max_retry_tokens = max_context_length - 8  # leave room for second completion
                tok = tokenizer.encode(modified_question, add_special_tokens=False)
                if len(tok) > max_retry_tokens:
                    suffix = comp + answer_begin
                    suffix_tok = tokenizer.encode(suffix, add_special_tokens=False)
                    max_prompt_tok = max(0, max_retry_tokens - len(suffix_tok))
                    prompt_tok = tokenizer.encode(prompts[idx], add_special_tokens=False)
                    trunc_prompt = tokenizer.decode(prompt_tok[:max_prompt_tok]) if len(prompt_tok) > max_prompt_tok else prompts[idx]
                    modified_question = trunc_prompt + comp + answer_begin
            modified_questions.append(modified_question)
            indices_to_modify.append(idx)
            first_completions.append(comp + answer_begin)

    if len(modified_questions) > 0:
        sampling_params2 = {"temperature": 0.0, "max_new_tokens": 32, "stop": ["<|im_end|>", "<|eot_id|>"]}
        new_outputs = llm.generate(modified_questions, sampling_params2)
        new_completions = [o["text"] for o in new_outputs]
        for index, new_comp, first_com in zip(indices_to_modify, new_completions, first_completions):
            final_completions[index] = first_com + new_comp
    return final_completions


if __name__ == "__main__":
    # Example (English utility: math_easy, mmlu, belebele, arc_easy by default):
    #   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lima_s1_exp.eval.eval_utility_en \
    #     --model_name saved_models/freeze_ft/lima_s1/Qwen/Qwen3-4B-Instruct-2507 --dp_size 4
    # Override tasks:
    #   --tasks mmlu,arc_easy
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name, e.g., Qwen2.5-7B",
    )
    parser.add_argument(
        "--zero_shot",
        action="store_true",
        help="Use zeroshot CoT instead of chat template",
    )
    parser.add_argument(
        "--dp_size",
        type=int,
        default=4,
        help="Data parallel size",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: only run inference on a small number of samples per language/task",
    )
    parser.add_argument(
        "--debug_samples",
        type=int,
        default=8,
        help="Number of samples per language/task in debug mode (default: 10)",
    )
    parser.add_argument(
        "--max_context_length",
        type=int,
        default=None,
        help="Override model max context length (for xlsum truncation). If unset, read from model config.",
    )
    parser.add_argument(
        "--engine_init_timeout",
        type=int,
        default=300,
        help="Timeout (seconds) for Engine initialization; interrupt and retry if exceeded (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--retry_on_timeout",
        action="store_true",
        default=True,
        help="Retry automatically when Engine init times out (default: True)",
    )

    parser.add_argument(
        "--no_retry_wrapper",
        action="store_true",
        help=argparse.SUPPRESS,  # Internal: avoid subprocess recursion
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="math_easy,mmlu,belebele,arc_easy",
        help="Comma-separated tasks to run (English only in this script). "
        "Choices: math, math_easy, mmlu, belebele, gpqa, arc_challenge, arc_easy, truthfulqa.",
    )

    import sglang as sgl  # <-- NEW: use SGLang offline engine

    args = parser.parse_args()
    model_name = args.model_name
    if args.max_context_length is None:
        config = AutoConfig.from_pretrained(model_name)
        args.max_context_length = getattr(config, "max_position_embeddings", None) or getattr(config, "max_sequence_length", None) or 8192
        print(f"Using max_context_length from config: {args.max_context_length}")
    args.zeroshot = args.zero_shot

    _allowed_tasks = frozenset(langs_dct.keys())
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in _allowed_tasks]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {unknown}. Allowed: {sorted(_allowed_tasks)}"
        )
    if not tasks:
        raise ValueError("No tasks after parsing --tasks")

    # ---- Retry wrapper: run in subprocess and retry on timeout ----
    if args.retry_on_timeout and not args.no_retry_wrapper:
        retry_args = [a for a in sys.argv[1:] if a != "--retry_on_timeout"] + ["--no_retry_wrapper"]
        max_retries = 5
        for attempt in range(max_retries):
            proc = subprocess.run(
                [sys.executable, "-m", "lima_s1_exp.eval.eval_utility_en"] + retry_args,
                env=os.environ.copy(),
            )
            if proc.returncode == 0:
                sys.exit(0)
            if proc.returncode == 124:  # timeout
                print(f"\n[Attempt {attempt + 1}/{max_retries}] Engine init timed out after {args.engine_init_timeout}s. Retrying in 5s...")
                if attempt < max_retries - 1:
                    time.sleep(5)
                else:
                    print("Max retries reached. Exiting.")
                    sys.exit(124)
            else:
                sys.exit(proc.returncode)

    # ---- SGLang offline Engine with timeout ----
    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Engine initialization exceeded {args.engine_init_timeout} seconds")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(args.engine_init_timeout)
    try:
        llm = sgl.Engine(model_path=model_name, tp_size=1, dp_size=args.dp_size)
    except TimeoutError as e:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        print(f"\n{e}. Interrupting.")
        # Kill child processes (Engine schedulers) to free GPUs before retry
        try:
            subprocess.run(
                ["pkill", "-9", "-P", str(os.getpid())],
                timeout=5,
                capture_output=True,
            )
        except Exception:
            pass
        time.sleep(2)
        sys.exit(124)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    if model_name.count("/") == 1:
        path = f"predictions/base/{model_name}"
    else:
        path = f"predictions/{model_name.replace('saved_models', '')}"

    path = f"lima_s1_exp/eval/{path}"
    if not os.path.exists(path):
        os.makedirs(path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Main eval loop
    for task in tasks:
        langs = langs_dct[task]
        for lang in langs:
            # if os.path.exists(f"{path}/{task}_{lang}.csv"):
            #     print(f"Have done {task}_{lang}, skipping...")
            #     continue
            prompts, golds, df = prepare_dataset(
                tokenizer,
                lang,
                task=task,
                zeroshot=args.zeroshot,
                max_context_length=args.max_context_length,
            )
            if args.debug:
                n = min(args.debug_samples, len(prompts))
                prompts = prompts[:n]
                golds = golds[:n]
                df = df.head(n).reset_index(drop=True)
                print(f"[debug] {task}/{lang}: using {n} samples")
            preds = inference(
                llm,
                prompts,
                lang,
                task=task,
                zeroshot=args.zeroshot,
                tokenizer=tokenizer,
                max_context_length=args.max_context_length,
            )
            df["prediction"] = preds

            df.to_csv(f"{path}/{task}_{lang}.csv", index=False)

    # Clean up engine if desired
    llm.shutdown()  # optional, but supported in offline API

    del llm
    del tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    GRADING_STEP_TIMEOUT = 300  # 5 minutes

    summary = {}
    for task in tasks:
        acc_id = []
        acc_ood = []
        acc_dct = {}
        langs = langs_dct[task]

        while True:
            ex = ProcessPoolExecutor(max_workers=1)
            fut = ex.submit(_grade_task_worker, path, task, langs)
            try:
                acc_dct, acc_id, acc_ood = fut.result(timeout=GRADING_STEP_TIMEOUT)
                ex.shutdown(wait=True, cancel_futures=True)
                break
            except FuturesTimeoutError:
                print(f"Grading step for task '{task}' stuck for more than 10 minutes, re-running...")
                # Don't block waiting for the stuck worker; just abandon this executor
                ex.shutdown(wait=False, cancel_futures=True)
        id_acc = sum(acc_id) / len(acc_id) if acc_id else 0.0
        ood_acc = sum(acc_ood) / len(acc_ood) if acc_ood else None
        summary[f"{task}_id"] = id_acc
        summary[f"{task}_ood"] = ood_acc

        with open(f"{path}/{task}.txt", "w") as f:
            for lang in ID_LANGS:
                if lang in langs:
                    print(f"The accuracy at the {lang} is {acc_dct[lang]:.2f}", file=f)
                    print(f"The accuracy at the {lang} is {acc_dct[lang]:.2f}")
            print(f"The ID accuracy at the task {task} is {id_acc:.2f}", file=f)
            print(f"The ID accuracy at the task {task} is {id_acc:.2f}")
            for lang in langs:
                if lang not in ID_LANGS:
                    print(f"The accuracy at the {lang} is {acc_dct[lang]:.2f}", file=f)
                    print(f"The accuracy at the {lang} is {acc_dct[lang]:.2f}")
            if acc_ood:
                print(f"The OOD accuracy at the task {task} is {ood_acc:.2f}", file=f)
                print(f"The OOD accuracy at the task {task} is {ood_acc:.2f}")

        print(f"Results are saved to {path}")

    summary_path = os.path.join(path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")

    # time.sleep(120)