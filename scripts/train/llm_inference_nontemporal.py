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

# Load and prepare datasetH
input_file =  "data/non-temporal/mmlu_abstain.json"
# "data/non-temporal/squad_valid_samples_3000.json"
#"data/non-temporal/mmlu_abstain.json"
#"data/non-temporal/mmlu_abstain.json"
# "data/non-temporal/hellaswag_abstain.json"
model_name = 'HARD_COT_SFT_GRPO(RougeReward+BertScore(False))_Qwen/Qwen2.5-1.5b-Instruct_train_Epoch=3_BS=8_Beta=0.01_LR=1e-05_CompLength=256_FormatRD=0.5_CompletionRD=0_Context_results'

#'HARD_COT_SFT_GRPO(RougeReward+BertScore(False))_Qwen/Qwen2.5-1.5b-Instruct_train_Epoch=3_BS=8_Beta=0.01_LR=1e-05_CompLength=256_FormatRD=0.5_CompletionRD=0_Context_results'

#'SFT_Context_Easy_models/qwen2.5-1.5b-instruct_train_Epoch=2_BS=4'


# 'HARD_COT_SFT_GRPO(RougeReward+BertScore(False))_Qwen/Qwen2.5-1.5b-Instruct_train_Epoch=3_BS=8_Beta=0.01_LR=1e-05_CompLength=256_FormatRD=0.5_CompletionRD=0_Context_results'
#'EASY_COT_SFT(Epo1)_GRPO_NoAssist(RougeReward+BertScore(False))_Qwen/Qwen2.5-1.5b-Instruct_train_Epoch=3_BS=8_Beta=0.01_LR=1e-05_CompLength=256_FormatRD=0.5_CompletionRD=0_Context_results'

#'SFT_Context_Hard_models/qwen2.5-1.5b-instruct_train_Epoch=2_BS=4'
# 'EASY_COT_SFT(Epo1)_GRPO_NoAssist(RougeReward+BertScore(False))_Qwen/Qwen2.5-1.5b-Instruct_train_Epoch=3_BS=8_Beta=0.01_LR=1e-05_CompLength=256_FormatRD=0.5_CompletionRD=0_Context_results'
# 'SFT_TimeSents_Hard_models/qwen2.5-1.5b-instruct_train_Epoch=2_BS=4'
# "SFT_COT_Hard_models/qwen2.5-7b-instruct_train_Epoch=1_BS=4"
# "SFT_COT_EASY_models/qwen2.5-0.5b-instruct_train_Epoch=1_BS=4"
# original_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
log_file = "results/mmlu_Inf_qwen1.5b-Easy-RL.log"

MAX_TOKEN_LENGTH = 128

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model.to(device)
model.eval()

# Load JSON data
def load_json_data(json_file_path):
    data_question_answer=[]
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

            # question = data['question']
            # context = data['context']
            # targets = data['targets'][0]

            # if targets == "":
            #     answer_str="No Answer"
            # else:
            #     answer_str=targets

            


         
            
            data_process = {'question': question, 'answer': answer_str,
                            # 'context': context
                            }
            data_question_answer.append(data_process)


    return data_question_answer
    

def call_llm(prompt, tokenizer, model):
    model.eval()
    encoded_prompt = tokenizer(
        prompt,
        return_tensors='pt'
    )
    # print('encoded prompt len: ', len(tokenizer.tokenize(prompt)))
    
    generate_input = {
        "input_ids": encoded_prompt['input_ids'].to(device),
        "attention_mask": encoded_prompt['attention_mask'].to(device),
        "max_new_tokens":MAX_TOKEN_LENGTH,
        "early_stopping": True,  # 启用提前停止
        "num_return_sequences": 1,  # 限制返回的序列数量
        "no_repeat_ngram_size": 7,  # 避免生成重复的n-gram
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
    
    # end_index = text.find(prompt) + len(prompt)
    # unique_text = text[end_index:]
    
    return text


# instruction = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\
# You are a helpful assistant.\n\
# Give the correct answer of the following question without any other explanation.\n\
# If there is no correct answer to the question, print No Answer.\n\
# <|eot_id|><|start_header_id|>user<|end_header_id|>"

# # # ### Context instruction
# instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
#      Think and give the correct answer of the following question without any other explanation based on the given context.\n\
#     If there is no correct answer to the question, print No Answer.\n"

# ## CONTRASTIVE context:
# instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
# Think and give the correct answer of the following question without any other explanation based on the given examples and context. \n\
# Examples:\n\
# Question: Which team did George Moorhouse play for from 1921 to 1923?  Answer: Tranmere Rovers \n\
# Question: What was the place of detention for Josep Rull from Jun 2019 to Jun 2020? Answer: No Answer \n\
# If there is no correct answer to the question, print No Answer.\n"

# ## Positive context:
# instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
# Think and give the correct answer of the following question without any other explanation based on the given examples and context. \n\
# Examples:\n\
# Question: Which team did George Moorhouse play for from 1921 to 1923?  Answer: Tranmere Rovers \n\
# If there is no correct answer to the question, print No Answer.\n"

## Negative context:
# instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
# Think and give the correct answer of the following question without any other explanation based on the given examples and context. \n\
# Examples:\n\
# Question: What was the place of detention for Josep Rull from Jun 2019 to Jun 2020? Answer: No Answer \n\
# If there is no correct answer to the question, print No Answer.\n"



# instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
# A conversation between User and Assistant. The user asks a question, and the Assistant solves it based on its own knowledge. The assistant should output the answer if there exists a correct answer. If there is no correct answer to the question, print 'No Answer' only. \n"
# #The reasoning  process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer> \n"

instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it based on its own knowledge. The assistant should only output the options 'A', 'B', or 'C' if there exists a correct answer. If there is no correct answer to the question, print D' only. \n\
Only give the option, do not print more words!!!\n\
The reasoning  process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer> \n"



def add_prompt(data):
    for entry in data:
        question = entry['question']
        prompt = question
        

        # qwen prompt ##
        # ## only question
        # prompt = "Question: " + question + "\n"

        # # ## with context
        # context = entry['context']
        # prompt = "Question: " + question + "\n" + "Context: " + context[:] + "\n"

        ## with time_sents
        # prompt = "Question: " + question + "\n" + "Context: " + "\n"
        # for i in range(len(entry['time_sents'])):
        #     prompt += entry['time_sents'][i] + "\n"

        # ## with Kgs
        # prompt  =  "Question: " + question + "\n" + "Knowledge graphs: " 

        # KG_NUM = 7
        # for i in range(min(KG_NUM, len(entry["kgs"]))):
        #     ## concate the head and tail
        #     kg = entry["kgs"][i]['original_kg']
        #     # print(kg)

        #     prompt += kg["head"] if kg["head"] is not None else "" 
        #     prompt += " " 
        #     prompt += kg["relation"] if kg["relation"] is not None else "" 
        #     prompt += " " 
        #     prompt += kg["tail"] if kg["tail"] is not None else ""
        #     if kg['timestamp'] is not None:
        #         prompt += ' on ' + kg['timestamp'] + '.\n'  # + KG

            # ## use the rephrased_sentence
            # kg = entry["kgs"][i]
            # prompt += kg["rephrased_sentence"] if kg["rephrased_sentence"] is not None else ""
            # prompt += "\n"




     


        entry['question_with_prompt'] = prompt


        # ## llama prompt ##
        # # ## with KGs
        # # entry['question_with_prompt'] = instruction + "Question: " +  question # + "\n" + "Knowledge graphs: "  #+ entry['answer']

        # # prompt = entry['question_with_prompt']

    
        # # KG_NUM = 7
        # # for i in range(min(KG_NUM, len(entry["kgs"]))):

        # #     # ## concate the KGs
        # #     # kg = entry["kgs"][i]
        # #     # prompt += kg["head"] if kg["head"] is not None else "" 
        # #     # prompt += " " 
        # #     # prompt += kg["relation"] if kg["relation"] is not None else "" 
        # #     # prompt += " " 
        # #     # prompt += kg["tail"] if kg["tail"] is not None else ""
        # #     # if kg['timestamp'] is not None:
        # #     #     prompt += ' on ' + kg['timestamp'] + '.\n'  # + KG

        # #     ## use the rephrased_sentence
        # #     kg = entry["kgs"][i]
        # #     prompt += kg["rephrased_sentence"] if kg["rephrased_sentence"] is not None else ""
        # #     prompt += "\n"


        # ## with Contexts
        # entry['question_with_prompt'] = instruction + "Question: " +  question + "\n" + "Context: "  + context[:4096] + "\n"
        # prompt = entry['question_with_prompt']
        

        # entry['question_with_prompt'] = prompt + llama_prompt


        # entry['verification_prompt'] = f"Is there an answer for the question: {question}"
        # entry['answerable'] = 0 if "No Answer" in entry['answer'] else 1
    
    return data

data = load_json_data(input_file)
data = add_prompt(data)

dataset = Dataset.from_dict({
    'question': [entry['question'] for entry in data],
    'answer': [entry['answer'] for entry in data],
    # 'context': [entry['context'] for entry in data],
    'question_with_prompt': [entry['question_with_prompt'] for entry in data],
})

refs = []
preds = []

true_positive = 0
false_positive = 0
false_negative = 0
precision, recall, f_score = 0, 0, 0

abstain_llm=0
abstain_total=0

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
        ## only keep the part after "assistant"
        llm_answer = llm_answer.split("assistant")[-1].strip()

        # print(llm_answer)
        # sys.exit(0)
        log_entry = f"LLM Answer: {llm_answer}\nCorrect Answer: {target}\n=======================================================\n"
        refs.append(target)

        # llm_answer_final = llm_answer.lower()

        #### FOR THINK MODE
       ## extract the answer between <answer> </answer>
        if '<answer>\n' in llm_answer and '\n</answer' in llm_answer:
            # print(llm_answer.split('<answer>')[1])
            llm_answer_final = llm_answer.split('<answer>\n')[1].split('\n</answer')[0].strip()
        elif '<answer>' in llm_answer and '</answer' in llm_answer:
            # print(llm_answer.split('<answer>')[1])
            llm_answer_final = llm_answer.split('<answer>')[1].split('</answer')[0].strip()
        else:
            llm_answer_final = llm_answer.strip()
        #### FOR THINK MODE
        
        llm_answer_final = llm_answer_final.lower()
        preds.append(llm_answer_final)

        print(f"LLM Answer: {llm_answer_final}\nCorrect Answer: {target}  ------------")
        # sys.exit(0)
        # time.sleep(5)
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
            # print(f"true positive: {true_positive}, false positive: {false_positive}, false negative: {false_negative}")
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0


        print(rouge_scores, f"Prec={precision}, Recall={recall}, f1={f_score}, TP={true_positive}, FP={false_positive}, FN={false_negative}, Accuracy={abstain_llm/abstain_total if abstain_total > 0 else 0:.4f}, Total={abstain_total}, EM={exact_match_llm / exact_match_total if exact_match_total > 0 else 0:.4f}")
        

bert_score = bertscore.compute(
    predictions=preds, references=refs, 
    model_type="models/bert-base-uncased",
    num_layers=model2layers["bert-base-uncased"],)

bert_f1 = np.mean(bert_score['f1'])
print(f"Bert F1: {bert_f1:.4f}, rouge_score: {rouge_scores}, Prec={precision}, Recall={recall}, f1={f_score}, TP={true_positive}, FP={false_positive}, FN={false_negative}, Accuracy={abstain_llm/abstain_total if abstain_total > 0 else 0:.4f}, Total={abstain_total}, EM={exact_match_llm / exact_match_total if exact_match_total > 0 else 0:.4f}")
        

        


       
