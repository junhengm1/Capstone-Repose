import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from peft import PeftModel
import json
import pandas as pd
import re
import numpy as np
import argparse
import gc
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# 配置参数
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = os.environ.get("CAPSTONE_BASE_MODEL", str(PROJECT_ROOT / "Qwen3-4B"))
LORA_PATH = os.environ.get(
    "CAPSTONE_LORA_PATH",
    str(PROJECT_ROOT / "checkpoints" / "train_2026-05-23-20-15-30"),
)
TEST_DATA_PATH = os.environ.get("CAPSTONE_EVAL_DATA", str(PROJECT_ROOT / "data" / "sft_eval.jsonl"))
MAX_NEW_TOKENS = 1024
BATCH_SIZE = 4
NUM_SAMPLES = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
OUTPUT_DIR = os.environ.get("CAPSTONE_EVAL_OUTPUT_DIR", str(PROJECT_ROOT / "outputs" / "eval"))
SEMANTIC_MODEL_PATH = os.environ.get(
    "SEMANTIC_MODEL_PATH",
    str(PROJECT_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2")
)
SEMANTIC_DEVICE = os.environ.get("SEMANTIC_DEVICE", "cpu")

semantic_model = None

# 忽略警告
warnings.filterwarnings("ignore")

# 加载模型和tokenizer
def load_models(use_lora=False):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="flash_attention_2"
    )
    if use_lora:
        model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    
    return tokenizer, model, model.device


class TransformersEmbeddingModel:
    def __init__(self, model_path, device="cpu"):
        self.device = torch.device(device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        except Exception:
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_file=os.path.join(model_path, "tokenizer.json"),
                model_max_length=128,
                unk_token="<unk>",
                sep_token="</s>",
                pad_token="<pad>",
                cls_token="<s>",
                mask_token="<mask>"
            )
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(self.device)
        self.model.eval()

    def encode(self, texts, convert_to_tensor=True):
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * attention_mask).sum(dim=1)
        embeddings = embeddings / attention_mask.sum(dim=1).clamp(min=1)
        return embeddings if convert_to_tensor else embeddings.cpu().numpy()


def load_semantic_model():
    global semantic_model
    if semantic_model is not None:
        return semantic_model

    try:
        semantic_model = SentenceTransformer(
            SEMANTIC_MODEL_PATH,
            device=SEMANTIC_DEVICE,
            local_files_only=True
        )
    except Exception as exc:
        print(f"\nWarning: SentenceTransformer load failed, fallback to transformers. ({exc})")
        try:
            semantic_model = TransformersEmbeddingModel(SEMANTIC_MODEL_PATH, device=SEMANTIC_DEVICE)
        except Exception as fallback_exc:
            print(f"Warning: semantic model not available, skip semantic metrics. ({fallback_exc})")
            semantic_model = None

    return semantic_model

# 数据加载和预处理
def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith(".jsonl"):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
    return data

# 改进的文本清理函数（处理思考模式输出的<think>标签）
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # 移除思考模式标签
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_end\|>|<\|endoftext\|>', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 计算指标（返回整数百分比）
def calculate_token_metrics(preds, refs):
    preds = [p for p in preds if p]
    refs = [r for r in refs if r]
    
    if not preds or not refs or len(preds) != len(refs):
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0}
    
    precisions, recalls, f1s, accuracies = [], [], [], []
    
    for pred, ref in zip(preds, refs):
        pred_set = set(pred.split())
        ref_set = set(ref.split())
        common = pred_set & ref_set
        precision = 100 * len(common) / len(pred_set) if pred_set else 0
        recall = 100 * len(common) / len(ref_set) if ref_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = 100 * len(common) / len(ref_set) if ref_set else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
    
    return {
        "precision": int(np.round(np.mean(precisions))),
        "recall": int(np.round(np.mean(recalls))),
        "f1": int(np.round(np.mean(f1s))),
        "accuracy": int(np.round(np.mean(accuracies)))
    }

def calculate_semantic_metrics(preds, refs):
    if semantic_model is None:
        return None

    preds = [p if p else " " for p in preds]
    refs = [r if r else " " for r in refs]
    
    if not preds or not refs or len(preds) != len(refs):
        return {
            "semantic_cosine_mean": 0,
            "semantic_cosine_std": 0,
            "semantic_cosine_min": 0,
            "semantic_cosine_max": 0
        }
    
    batch_size = 32
    cos_sims = []
    for i in range(0, len(preds), batch_size):
        batch_preds = preds[i:i+batch_size]
        batch_refs = refs[i:i+batch_size]
        with torch.no_grad():
            pred_embs = semantic_model.encode(batch_preds, convert_to_tensor=True)
            ref_embs = semantic_model.encode(batch_refs, convert_to_tensor=True)
            batch_cos_sims = torch.nn.functional.cosine_similarity(pred_embs, ref_embs)
            cos_sims.extend(batch_cos_sims.cpu().tolist())
    
    cos_sims = [100 * x for x in cos_sims]  # 转换为百分比
    return {
        "semantic_cosine_mean": int(np.round(np.mean(cos_sims))),
        "semantic_cosine_std": int(np.round(np.std(cos_sims))),
        "semantic_cosine_min": int(np.round(np.min(cos_sims))),
        "semantic_cosine_max": int(np.round(np.max(cos_sims)))
    }

def calculate_classic_metrics(preds, refs):
    preds = [p if p else " " for p in preds]
    refs = [r if r else " " for r in refs]
    
    if not preds or not refs or len(preds) != len(refs):
        return {
            "rouge-1": 0, "rouge-2": 0, "rouge-l": 0, "bleu": 0,
            "precision": 0, "recall": 0, "f1": 0, "accuracy": 0
        }
    
    # ROUGE (转换为百分比)
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(preds, refs)
        rouge_metrics = {
            "rouge-1": int(round(100 * sum(s['rouge-1']['f'] for s in rouge_scores) / len(rouge_scores))),
            "rouge-2": int(round(100 * sum(s['rouge-2']['f'] for s in rouge_scores) / len(rouge_scores))),
            "rouge-l": int(round(100 * sum(s['rouge-l']['f'] for s in rouge_scores) / len(rouge_scores)))
        }
    except:
        rouge_metrics = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 0}
    
    # BLEU (转换为百分比)
    smoothie = SmoothingFunction().method4
    bleu_scores = []
    for p, r in zip(preds, refs):
        try:
            bleu_scores.append(100 * sentence_bleu([r.split()], p.split(), smoothing_function=smoothie))
        except:
            bleu_scores.append(0)
    rouge_metrics["bleu"] = int(round(sum(bleu_scores) / len(bleu_scores)))
    
    # Token级指标
    token_metrics = calculate_token_metrics(preds, refs)
    rouge_metrics.update(token_metrics)
    
    return rouge_metrics

def batch_generate(tokenizer, model, questions, question_types):
    all_responses = []
    
    for i in tqdm(range(0, len(questions), BATCH_SIZE), desc="Generating"):
        batch = questions[i:i+BATCH_SIZE]
        batch_types = question_types[i:i+BATCH_SIZE]
        
        # 为每个问题准备输入（根据type决定enable_thinking）
        inputs = []
        for q, q_type in zip(batch, batch_types):
            messages = [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": q}
            ]
            inputs.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=(q_type == "reason")  # 关键修改：根据type启用思考模式
                )
            )
        
        # Tokenize并生成
        inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        device = next(model.parameters()).device
        # 将所有张量移动到指定的设备上
        for key, value in inputs.items():
            inputs[key] = value.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码并清理
        decoded = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=False
        )
        cleaned = [clean_text(d) for d in decoded]
        all_responses.extend(cleaned)
        
        # 清理内存
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return all_responses

def get_question(sample):
    if "question" in sample:
        return sample["question"]

    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    if instruction and input_text:
        return f"{instruction}\n\n{input_text}"
    return instruction or input_text


def get_reference(sample):
    if "answer" in sample:
        return sample["answer"]
    return sample.get("output", "")


def evaluate(model_label, tokenizer, model, test_data):
    questions = [get_question(d) for d in test_data]
    references = [get_reference(d) for d in test_data]
    cots = [d.get("cot", "") for d in test_data]
    question_types = [d.get("type", "normal") for d in test_data]
    
    # 生成回答
    answers = batch_generate(tokenizer, model, questions, question_types)
    load_semantic_model()
    
    # 准备评估数据
    eval_data = []
    for i, (q, ref, cot, q_type) in enumerate(zip(questions, references, cots, question_types)):
        eval_data.append({
            "question": q,
            "generated": answers[i] if i < len(answers) else "",
            "reference": ref,
            "cot_reference": cot,
            "type": q_type
        })
    
    # 计算指标
    metrics = {}
    
    # 1. Answer部分评估（所有样本）
    gen_answers = [d["generated"] for d in eval_data]
    ref_answers = [d["reference"] for d in eval_data]
    
    metrics["answer"] = {
        "classic": calculate_classic_metrics(gen_answers, ref_answers),
        "semantic": calculate_semantic_metrics(gen_answers, ref_answers)
    }
    
    # 2. CoT部分评估（仅reason类型且有cot_reference的样本）
    cot_data = [d for d in eval_data if d["type"] == "reason" and d["cot_reference"]]
    if cot_data:
        gen_cots = [d["generated"] for d in cot_data]
        ref_cots = [d["cot_reference"] for d in cot_data]
        
        metrics["cot"] = {
            "classic": calculate_classic_metrics(gen_cots, ref_cots),
            "semantic": calculate_semantic_metrics(gen_cots, ref_cots)
        }
        print(f"\nEvaluated {len(cot_data)} CoT samples")
    else:
        print("\nNo CoT samples to evaluate")
        metrics["cot"] = None
    
    # 3. Combined评估（所有样本，reason类型拼接cot）
    combined_gen = []
    combined_ref = []
    for d in eval_data:
        if d["type"] == "reason" and d["cot_reference"]:
            # 对于有CoT的样本，拼接生成内容和参考CoT
            combined_gen.append(f"{d['generated']} {d['cot_reference']}")
            combined_ref.append(f"{d['reference']} {d['cot_reference']}")
        else:
            # 对于没有CoT的样本，只使用回答部分
            combined_gen.append(d["generated"])
            combined_ref.append(d["reference"])
    
    metrics["combined"] = {
        "classic": calculate_classic_metrics(combined_gen, combined_ref),
        "semantic": calculate_semantic_metrics(combined_gen, combined_ref)
    }
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result_path = os.path.join(OUTPUT_DIR, f"{model_label}_results.csv")
    metrics_path = os.path.join(OUTPUT_DIR, f"{model_label}_metrics.json")
    pd.DataFrame(eval_data).to_csv(result_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    return metrics, eval_data, result_path, metrics_path

def print_metrics(name, data):
    if data is None:
        print(f"\n[{name.upper()}] No data to display")
        return
    
    print(f"\n[{name.upper()}]")
    print("Classic Metrics:")
    print(f"  ROUGE-1: {data['classic']['rouge-1']}%")
    print(f"  ROUGE-2: {data['classic']['rouge-2']}%")
    print(f"  ROUGE-L: {data['classic']['rouge-l']}%")
    print(f"  BLEU:    {data['classic']['bleu']}%")
    print(f"  Precision: {data['classic']['precision']}%")
    print(f"  Recall:    {data['classic']['recall']}%")
    print(f"  F1:        {data['classic']['f1']}%")
    print(f"  Accuracy:  {data['classic']['accuracy']}%")
    if data["semantic"] is None:
        print("\nSemantic Metrics: skipped")
    else:
        print("\nSemantic Metrics:")
        print(f"  Cosine Mean: {data['semantic']['semantic_cosine_mean']}%")
        print(f"  Cosine Std:  ±{data['semantic']['semantic_cosine_std']}%")
        print(f"  Cosine Min:  {data['semantic']['semantic_cosine_min']}%")
        print(f"  Cosine Max:  {data['semantic']['semantic_cosine_max']}%")

def run_one(model_label, use_lora, test_data):
    print(f"\n=== Evaluating {model_label.upper()} model ===")
    tokenizer, model, _ = load_models(use_lora=use_lora)
    metrics, results, result_path, metrics_path = evaluate(model_label, tokenizer, model, test_data)

    print("\n=== Evaluation Results ===")
    print(f"\nEvaluated {len(results)} samples")

    print_metrics("Answer", metrics["answer"])
    if "cot" in metrics and metrics["cot"] is not None:
        print_metrics("Chain-of-Thought", metrics["cot"])
    print_metrics("Combined", metrics["combined"])
    print(f"\nSaved results to: {result_path}")
    print(f"Saved metrics to: {metrics_path}")

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_variant",
        choices=["base", "lora", "both"],
        default="both",
        help="选择评估原模型、LoRA微调模型，或两者都评估"
    )
    parser.add_argument("--eval-data", default=TEST_DATA_PATH, help="Path to the SFT eval JSONL file")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory for evaluation CSV and JSON outputs")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base model path or Hugging Face model ID")
    parser.add_argument("--lora-path", default=LORA_PATH, help="LoRA checkpoint path")
    parser.add_argument(
        "--semantic-model-path",
        default=SEMANTIC_MODEL_PATH,
        help="Local MiniLM/SentenceTransformer path for semantic similarity metrics",
    )
    parser.add_argument("--semantic-device", default=SEMANTIC_DEVICE, help="Device for the semantic model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Generation batch size")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS, help="Maximum generated tokens")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    MODEL_NAME = args.model_name
    LORA_PATH = args.lora_path
    TEST_DATA_PATH = args.eval_data
    OUTPUT_DIR = args.output_dir
    SEMANTIC_MODEL_PATH = args.semantic_model_path
    SEMANTIC_DEVICE = args.semantic_device
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens

    test_data = load_test_data(TEST_DATA_PATH)

    if args.model_variant in ["base", "both"]:
        run_one("base", use_lora=False, test_data=test_data)
    if args.model_variant in ["lora", "both"]:
        run_one("lora", use_lora=True, test_data=test_data)
