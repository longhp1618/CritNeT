import os
import re

import pandas as pd
import ast
import swifter

from datasets import load_dataset
from transformers import AutoTokenizer


ID_LANGS = ["en", "zh", "ar", "sw"]  # in-distribution
# Intersection of languages supported by Global-MMLU-Lite, PolyMath, Belebele, and MuBench
INTERSECTION_LANGS = ["ar", "bn", "de", "en", "es", "fr", "id", "it", "ja", "ko", "pt", "sw", "zh"]
SEA_LANGS = ["vi", "my", "fil", "id", "ms", "th", "ta"]
marker_question = {
    "ar": "السؤال:",
    "bn": "প্রশ্ন:",
    "de": "Frage:",
    "en": "Question:",
    "es": "Pregunta:",
    "fil": "Tanong:",
    "fr": "Question:",
    "id": "Pertanyaan:",
    "it": "Domanda:",
    "ja": "質問：",
    "ko": "질문:",
    "ms": "Soalan:",
    "my": "မေးခွန်း:",
    "pt": "Pergunta:",
    "sw": "Swali:",
    "ta": "கேள்வி:",
    "th": "คำถาม:",
    "vi": "Câu hỏi:",
    "zh": "问题：",
}

marker_answer = {
    "ar": "الإجابة:",
    "bn": "উত্তর:",
    "de": "Antwort:",
    "en": "Answer:",
    "es": "Respuesta:",
    "fil": "Sagot:",
    "fr": "Réponse:",
    "id": "Jawaban:",
    "it": "Risposta:",
    "ja": "回答：",
    "ko": "답변:",
    "ms": "Jawapan:",
    "my": "အဖြေ:",
    "pt": "Resposta:",
    "sw": "Jibu:",
    "ta": "பதில்:",
    "th": "คำตอบ:",
    "vi": "Câu trả lời:",
    "zh": "回答：",
}

marker_passage = {
    "ar": "فقرة:",      
    "bn": "অনুচ্ছেদ:",   
    "de": "Passage:",    
    "en": "Passage:",    
    "es": "Pasaje:",     
    "fil": "Talata:",
    "fr": "Passage :",   
    "id": "Kutipan:",    
    "it": "Brano:",      
    "ja": "文章:",       
    "ko": "지문:",       
    "ms": "Petikan:",
    "my": "စာပိုဒ်:",
    "pt": "Trecho:",     
    "sw": "Kifungu:",    
    "ta": "பத்தி:",
    "th": "ข้อความ:",
    "vi": "Đoạn văn:",
    "zh": "段落:"        
}

marker_final_answer = {
    "ar": "الإجابة النهائية هي $\\boxed{",
    "bn": "চূড়ান্ত উত্তর হলো $\\boxed{",
    "de": "Die endgültige Antwort ist $\\boxed{",
    "en": "The final answer is $\\boxed{",
    "es": "La respuesta final es $\\boxed{",
    "fil": "Ang huling sagot ay $\\boxed{",
    "fr": "La réponse finale est $\\boxed{",
    "id": "Jawaban akhir adalah $\\boxed{",
    "it": "La risposta finale è $\\boxed{",
    "ja": "最終的な答えは $\\boxed{",
    "ko": "최종 답은 $\\boxed{",
    "ms": "Jawapan akhir ialah $\\boxed{",
    "my": "နောက်ဆုံးအဖြေမှာ $\\boxed{",
    "pt": "A resposta final é $\\boxed{",
    "sw": "Jibu la mwisho ni $\\boxed{",
    "ta": "இறுதிப் பதில் $\\boxed{",
    "th": "คำตอบสุดท้ายคือ $\\boxed{",
    "vi": "Câu trả lời cuối cùng là $\\boxed{",
    "zh": "最终答案是$\\boxed{",
}

sys_role = {
    "ar": "أنت مساعد مفيد.",
    "bn": "আপনি একজন সহায়ক সহকারী।",
    "de": "Du bist ein hilfreicher Assistent.",
    "en": "You are a helpful assistant.",
    "es": "Eres un asistente servicial.",
    "fil": "Isa kang matulunging assistant.",
    "fr": "Vous êtes un assistant utile.",
    "id": "Anda adalah asisten yang membantu.",
    "it": "Sei un assistente utile.",
    "ja": "あなたは役に立つアシスタントです。",
    "ko": "당신은 유용한 어시스턴트입니다.",
    "ms": "Anda adalah pembantu yang berguna.",
    "my": "သင်သည် အသုံးဝင်သော အကူအညီပေးသူဖြစ်သည်။",
    "pt": "Você é um assistente prestativo.",
    "sw": "Wewe ni msaidizi mwenye msaada.",
    "ta": "நீங்கள் ஒரு பயனுள்ள உதவியாளர்.",
    "th": "คุณเป็นผู้ช่วยเหลือที่ดี",
    "vi": "Bạn là một trợ lý hữu ích.",
    "zh": "你是一个乐于助人的助手。",
}

thinking_prefix = {
    "ar": "دعنا نفكر خطوة بخطوة.",
    "bn": "চলুন ধাপে ধাপে ভাবি।",
    "de": "Lass uns Schritt für Schritt nachdenken.",
    "en": "Let's think step by step.",
    "es": "Pensemos paso a paso.",
    "fil": "Mag-isip tayo nang hakbang-hakbang.",
    "fr": "Réfléchissons étape par étape.",
    "id": "Mari kita berpikir langkah demi langkah.",
    "it": "Pensiamo passo dopo passo.",
    "ja": "一歩一歩考えていきましょう。",
    "ko": "단계별로 생각해 봅시다.",
    "ms": "Mari kita fikirkan langkah demi langkah.",
    "my": "တစ်ဆင့်ချင်း စဉ်းစားကြည့်ရအောင်။",
    "pt": "Vamos pensar passo a passo.",
    "sw": "Hebu tufikiri hatua kwa hatua.",
    "ta": "படிப்படியாக சிந்திப்போம்.",
    "th": "มาคิดไปทีละขั้นตอนกันเถอะ",
    "vi": "Hãy cùng suy nghĩ từng bước một.",
    "zh": "让我们一步一步地思考。",
}

sys_boxed = {
    "ar": "يرجى التفكير خطوة بخطوة، وضع إجابتك النهائية داخل \\boxed{}.",
    "bn": "অনুগ্রহ করে ধাপে ধাপে যুক্তি করুন এবং আপনার চূড়ান্ত উত্তরটি \\boxed{} এর মধ্যে লিখুন।",
    "de": "Bitte denke Schritt für Schritt und schreibe deine endgültige Antwort in \\boxed{}.",
    "en": "Please reason step by step, and put your final answer within \\boxed{}.",
    "es": "Por favor, razona paso a paso y coloca tu respuesta final dentro de \\boxed{}.",
    "fil": "Mangyaring mag-isip nang hakbang-hakbang, at ilagay ang iyong pinal na sagot sa loob ng \\boxed{}.",
    "fr": "Veuillez raisonner étape par étape et inscrire votre réponse finale dans \\boxed{}.",
    "id": "Silakan pikirkan langkah demi langkah, dan letakkan jawaban akhir Anda di dalam \\boxed{}.",
    "it": "Per favore, ragiona passo dopo passo e inserisci la tua risposta finale in \\boxed{}.",
    "ja": "段階的に推論し、最終的な答えを \\boxed{} の中に書いてください。",
    "ko": "단계별로 추론하고 최종 답을 \\boxed{} 안에 넣어 주세요.",
    "ms": "Sila berikan alasan langkah demi langkah, dan letakkan jawapan akhir anda dalam \\boxed{}.",
    "my": "ကျေးဇူးပြု၍ တစ်ဆင့်ချင်း စဉ်းစားပြီး သင့်နောက်ဆုံးအဖြေကို \\boxed{} ထဲတွင် ထည့်ပါ။",
    "pt": "Por favor, raciocine passo a passo e coloque sua resposta final dentro de \\boxed{}.",
    "sw": "Tafadhali tafakari hatua kwa hatua na weka jibu lako la mwisho ndani ya \\boxed{}.",
    "ta": "தயவுசெய்து படிப்படியாக சிந்தித்து, உங்கள் இறுதிப் பதிலை \\boxed{} க்குள் வைக்கவும்.",
    "th": "กรุณาคิดขั้นตอนทีละขั้นตอน และใส่คำตอบสุดท้ายลงใน \\boxed{} ให้ถูกต้อง",
    "vi": "Vui lòng lập luận từng bước một và đặt câu trả lời cuối cùng của bạn trong \\boxed{}.",
    "zh": "请一步一步地推理，并把你的最终答案写在 \\boxed{} 中。",
}


# Chat templates and assistant markers per model family (markers still used in eval prompts)
CHAT_TEMPLATES = {
    "llama3": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n' + message['content'] | trim + eos_token }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] | trim + eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] | trim + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    ),
    "qwen": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] + eos_token + '\\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|im_start|>user\n' + message['content'] + eos_token + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' + message['content'] + eos_token + '\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    ),
    "mistral": (
        "{% if messages[0]['role'] == 'system' %}"
        "{% set system_message = messages[0]['content'] %}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{{ bos_token }}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{% if loop.first and system_message is defined %}"
        "{{ ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}"
        "{% else %}"
        "{{ ' [INST] ' + message['content'] + ' [/INST]' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ ' ' + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ ' ' }}"
        "{% endif %}"
    ),
}

ASSISTANT_MARKERS = {
    "llama3": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "qwen": "<|im_start|>assistant\n",
    "mistral": " [/INST]",
}


def define_chat_template(tokenizer, model_family):
    """Set chat_template and return tokenizer. model_family: 'llama3', 'qwen', or 'mistral'."""
    tokenizer.chat_template = CHAT_TEMPLATES[model_family]
    return tokenizer


def build_language_qa_chat_template(lang: str) -> str:
    """
    Jinja2 string for `tokenizer.apply_chat_template` with language-specific Q/A markers.

    Rendered layout (newlines as shown)::

        {system_prompt}
        {marker_question[lang]} {user_prompt}
        {marker_answer[lang]} {completion_prompt}   # if assistant turn present

    If there is no assistant message and ``add_generation_prompt=True``, ends with::

        {marker_answer[lang]}\\n

    Message roles: ``system``, ``user``, and optionally ``assistant`` (one block each; last wins if repeated).
    """
    if lang not in marker_question or lang not in marker_answer:
        raise KeyError(
            f"Unknown lang {lang!r}; add entries to marker_question and marker_answer (see ID_LANGS / INTERSECTION_LANGS)."
        )
    mq = marker_question[lang]
    ma = marker_answer[lang]
    return (
        "{%- set ns = namespace(system='', user='', assistant='') -%}"
        "{%- for message in messages -%}"
        "{%- if message['role'] == 'system' -%}{%- set ns.system = message['content'] | trim -%}"
        "{%- elif message['role'] == 'user' -%}{%- set ns.user = message['content'] | trim -%}"
        "{%- elif message['role'] == 'assistant' -%}{%- set ns.assistant = message['content'] | trim -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
        "{% if ns.system %}{{ ns.system }}\n{% endif %}"
        + mq
        + " {{ ns.user }}\n"
        "{% if ns.assistant %}"
        + ma
        + " {{ ns.assistant }}"
        "{% elif add_generation_prompt %}"
        + ma
        + "\n"
        "{% endif %}"
    )


def define_chat_template_base(tokenizer, lang: str):
    """Set ``tokenizer.chat_template`` to the language-specific Q/A format (see [`build_language_qa_chat_template`])."""
    tokenizer.chat_template = build_language_qa_chat_template(lang)
    return tokenizer


def prepare_tokenizer(model_name, chat_template=True):
    if model_name.find("Llama-3") != -1:
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        tokenizer.pad_token = "<|reserved_special_token_5|>"
        model_family = "llama3"
    elif model_name.find("Qwen") != -1:  # Qwen2, Qwen2.5, Qwen3
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        tokenizer.pad_token = "<|fim_pad|>"
        model_family = "qwen"
    elif model_name.find("Mistral") != -1:
        tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
        # tokenizer.pad_token = "[PAD]" # tokenizer.eos_token
        model_family = "mistral"
    else:
        raise ValueError("This model is not supported")
    if chat_template:
        tokenizer = define_chat_template(tokenizer, model_family)

    marker = ASSISTANT_MARKERS[model_family]
    return tokenizer, marker


def _lima_row_as_messages(lang: str, prompt, completion, tokenizer=None) -> list:
    """Structured turns for [`dataloader.build_lm_dataset`] (completion-only loss via chat template).

    Always system / user / assistant with raw ``prompt`` and ``completion`` (no embedded Q/A markers).
    Markers belong in ``define_chat_template_base`` / model chat templates only.
    """
    del tokenizer  # retained for call-site compatibility
    system_prompt = sys_role[lang]
    if str(completion)[-100:].find("\\boxed{") != -1:
        system_prompt = sys_role[lang] + "\n" + sys_boxed[lang]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(completion)},
    ]


def process_data(stype):
    if stype == "curated_sea_instruct":
        lst = SEA_LANGS
    else:
        lst = ID_LANGS

    text_dct = dict()
    for lang in lst:
        if stype in ["lima_s1", "wiki_limas1", "fineweb_limas1"]:
            lima_ds = load_dataset("iNLP-Lab/multilingual-lima", lang, split="train")
            s1_ds = load_dataset("iNLP-Lab/multilingual-s1", lang, split="train")
            prompts = list(lima_ds["prompt"]) + list(s1_ds["prompt"])
            completions = list(lima_ds["output"]) + list(s1_ds["output"])
            text_dct[lang] = [
                _lima_row_as_messages(lang, p, c) for p, c in zip(prompts, completions)
            ]
        elif stype == "safety":
            file_path = os.path.join(f"lima_s1_exp/ft_datasets/safety/{lang}.csv")
            df = pd.read_csv(file_path)
            prompts = df["prompt"].tolist()
            completions = df["output"].tolist()
            text_dct[lang] = [_lima_row_as_messages(lang, p, c) for p, c in zip(prompts, completions)]
        elif stype == "curated_sea_instruct":
            file_path = os.path.join(f"lima_s1_exp/ft_datasets/curated_sea_instruct/{lang}.csv")
            df = pd.read_csv(file_path)
            text_dct[lang] = df["conversations"].swifter.apply(lambda x: ast.literal_eval(x)).tolist()
            continue
        else:
            raise ValueError("Now only applicable for lima_s1 or wiki")

    return text_dct


def get_layer_group(param_name: str) -> str:
    """Map a parameter name to its semantic layer group.

    All parameters with the same layer index are grouped together.
    """
    # Extract layer number if present (e.g. model.layers.0, layers.15)
    layer_match = re.search(r"layers\.(\d+)\.", param_name)
    layer_num = int(layer_match.group(1)) if layer_match else None

    # Embedding layer
    if "embed_tokens" in param_name or "wte" in param_name or "word_embedding" in param_name:
        return "embedding"

    # LM head
    if "lm_head" in param_name:
        return "lm_head"

    # Transformer layers - group ALL parameters with same layer index together
    if layer_num is not None:
        return f"layer_{layer_num:02d}"

    # Final layer norm (not inside any layer)
    if "norm" in param_name.lower() or "ln_f" in param_name:
        return "final_norm"

    # Other/unknown
    return "other"

def resize_pad_embeddings(model, tokenizer):  # only for alpaca-trained
    pad_token = "[PAD]"
    special_tokens_dict = dict(pad_token=pad_token)
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.pad_token = pad_token  # ensure pad_token_id is set (needed for DataLoader workers)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
