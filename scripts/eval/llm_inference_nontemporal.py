import json
import torch
from datasets import Dataset
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, LlamaTokenizer, AutoTokenizer,AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import os
from tqdm import tqdm
import sys
import time
import numpy as np
from rouge_score import rouge_scorer
from evaluate import load
from bert_score.utils import model2layers

bertscore = evaluate.load("models/bertscore.py")
rouge = evaluate.load("models/rouge.py")


def calculate_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)
    
    avg_scores = {key: sum(value) / len(value) for key, value in scores.items()}
    return avg_scores

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NO_ANSWER_STR = "No Answer"

input_file = "data/non-temporal/mmlu_abstain.json"
model_name = 'HARD_COT_SFT_GRPO(RougeReward+BertScore(False))_Qwen/Qwen2.5-1.5b-Instruct_train_Epoch=3_BS=8_Beta=0.01_LR=1e-05_CompLength=256_FormatRD=0.5_CompletionRD=0_Context_results'
log_file = "results/mmlu_Inf_qwen1.5b-Easy-RL.log"

MAX_TOKEN_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = 0
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.to(device)
model.eval()

def load_json_data(json_file_path):
    data_question_answer = []
    with open(json_file_path, 'r') as f:
        f = json.load(f)['data']
        for line in f:
            data = line
            question = ""
            question += data['question'] + " "
            choices = data['choices']
            for key in choices.keys():
                question += str(key) + ": " + choices[key] + " "
            answer_str = data['answer']
            
            data_process = {'question': question, 'answer': answer_str}
            data_question_answer.append(data_process)
    
    return data_question_answer
    

def call_llm(prompt, tokenizer, model):
    model.eval()
    encoded_prompt = tokenizer(prompt, return_tensors='pt')
    
    generate_input = {
        "input_ids": encoded_prompt['input_ids'].to(device),
        "attention_mask": encoded_prompt['attention_mask'].to(device),
        "max_new_tokens": MAX_TOKEN_LENGTH,
        "early_stopping": True,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 7,
        "temperature": 0.6,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
    }
    generate_ids = model.generate(**generate_input)
    text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    
    return text

instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it based on its own knowledge. The assistant should only output the options 'A', 'B', or 'C' if there exists a correct answer. If there is no correct answer to the question, print D' only. \n\
Only give the option, do not print more words!!!\n\
The reasoning  process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer> \n"



def add_prompt(data):
    for entry in data:
        question = entry['question']
        prompt = question
        entry['question_with_prompt'] = prompt
    return data

data = load_json_data(input_file)
data = add_prompt(data)

dataset = Dataset.from_dict({
    'question': [entry['question'] for entry in data],
    'answer': [entry['answer'] for entry in data],
    'question_with_prompt': [entry['question_with_prompt'] for entry in data],
})

refs = []
preds = []

true_positive = 0
false_positive = 0
false_negative = 0
precision, recall, f_score = 0, 0, 0

abstain_llm = 0
abstain_total = 0

exact_match_llm = 0
exact_match_total = 0

with open(log_file, 'w', encoding='utf-8') as log:
    for i in tqdm(range(len(data))):
        prompt = data[i]['question_with_prompt']
        target = data[i]['answer'].lower()

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 

        llm_answer = call_llm(prompt, tokenizer, model)
        llm_answer = llm_answer.split("assistant")[-1].strip()

        log_entry = f"LLM Answer: {llm_answer}\nCorrect Answer: {target}\n=======================================================\n"
        refs.append(target)

        # Extract answer from <answer> tags
        if '<answer>\n' in llm_answer and '\n</answer' in llm_answer:
            llm_answer_final = llm_answer.split('<answer>\n')[1].split('\n</answer')[0].strip()
        elif '<answer>' in llm_answer and '</answer' in llm_answer:
            llm_answer_final = llm_answer.split('<answer>')[1].split('</answer')[0].strip()
        else:
            llm_answer_final = llm_answer.strip()
        
        llm_answer_final = llm_answer_final.lower()
        preds.append(llm_answer_final)

        print(f"LLM Answer: {llm_answer_final}\nCorrect Answer: {target}  ------------")
        rouge_scores = calculate_rouge(preds, refs)

        exact_match_llm += 1 if llm_answer_final == target else 0
        exact_match_total += 1

        compute_fg = True
        if target.lower() == "d":
            abstain_total += 1

        if "d" ==  llm_answer_final.lower() and target.lower() == "d":
            true_positive += 1
            abstain_llm += 1
            
        elif "d" == llm_answer_final.lower() and target.lower() != "d":
            false_positive += 1
        elif "d" != llm_answer_final.lower() and target.lower() == "d":
            false_negative += 1
        else:
            compute_fg = False
            pass
        
        if compute_fg and (true_positive + false_positive) != 0 and (true_positive + false_negative) != 0:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        print(rouge_scores, f"Prec={precision}, Recall={recall}, f1={f_score}, TP={true_positive}, FP={false_positive}, FN={false_negative}, Accuracy={abstain_llm/abstain_total if abstain_total > 0 else 0:.4f}, Total={abstain_total}, EM={exact_match_llm / exact_match_total if exact_match_total > 0 else 0:.4f}")

bert_score = bertscore.compute(
    predictions=preds, references=refs, 
    model_type="models/bert-base-uncased",
    num_layers=model2layers["bert-base-uncased"],
)

bert_f1 = np.mean(bert_score['f1'])
print(f"Bert F1: {bert_f1:.4f}, rouge_score: {rouge_scores}, Prec={precision}, Recall={recall}, f1={f_score}, TP={true_positive}, FP={false_positive}, FN={false_negative}, Accuracy={abstain_llm/abstain_total if abstain_total > 0 else 0:.4f}, Total={abstain_total}, EM={exact_match_llm / exact_match_total if exact_match_total > 0 else 0:.4f}")
