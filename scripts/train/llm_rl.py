import re
import evaluate
import json
import torch
import torch.nn as nn
import sys
from datasets import Dataset
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, LlamaTokenizer, AutoTokenizer,AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer
from bert_score.utils import model2layers


## Use bfloat16
#torch.set_default_dtype(torch.bfloat16)
#torch.set_default_device("cuda")
bertscore = evaluate.load("models/bertscore.py")
rouge = evaluate.load("models/rouge.py")

from  accelerate  import PartialState
device_string = PartialState().process_index


## Load JSON data
def load_json_data(json_file_path):
    data_question_answer=[]
    with open(json_file_path, 'r') as f:
        f = json.load(f)
        for line in f:
            data = line
            answer_str = ''

          

            ## check if data has "targets"
            if 'targets' in data:
                pass
            else:
                print(data)
                sys.exit()

            

            if data['targets'][0] == '':
                #continue
                answer_str = 'No Answer'
            else:
                for i in range(len(data['targets'])):
                    answer_str += f"{data['targets'][i]}" if i == 0 else f", {data['targets'][i]}"
                    
            data_process = {'question': data['question'], 'answer': answer_str,
                                'context': data['context']}
            data_question_answer.append(data_process)


    return data_question_answer

# Load and prepare dataset
train_or_test = 'train'

json_file_path = 'data/easy/train_question_KG_gpt-4o-mini_extract_sentence_with_time.json' ## NOTE: Easy-Version Training

######################################


test_file_path = 'data/test_question_KG_gpt-4o-mini_extract_sentence_with_time.json' ## NOTE: Easy-Version Test



model_name = "Qwen/Qwen2.5-1.5b-Instruct"# "Qwen/Qwen2.5-7B-Instruct" # "qwen2.5-7b-instruct"
ckpt = "CoT_SFT_Easy_models/qwen2.5-1.5b-instruct_train_Epoch=1_BS=4"



############### Experiment and model settings
LORA_RANK = 32 ##NOTE: 64 or 256
EPOCHS = 3
BATCH_SIZE = 8#  12
NUM_ROLLOUT = 4 # 4
LR = 1e-5
BETA = 0.01 
USE_BERTSCORE=False
USE_EM=True
USE_ROUGE=True


### instruction with Think in Qwen
instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it based on the given information. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer within 80 words. If there is no correct answer to the question, print No Answer \n\
   The reasoning  process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer> \n"



############ LOAD DATA ################
data = load_json_data(json_file_path)
dataset = Dataset.from_dict({
    'input': [item['question'] for item in data],
    'context': [item['context'] for item in data],
    'output': [item['answer'] for item in data]
})

test_data = load_json_data(test_file_path)
test_dataset = Dataset.from_dict({
    'input': [item['question'] for item in test_data],
    'context': [item['context'] for item in test_data],
    'output': [ item['answer'] for item in test_data]
})

def make_conversion(example):

    target = example["output"]
    context = example["context"]
    question = example["input"]

    # # with context NOTE: Context
    prompt = "Question: " + question + "\n" + "Context: " + context[:] + "\n" 

   


    return {
        "prompt":[
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt},
        ],
    }



print("**********************PREPROCESS DATASETS*********************************")
train_dataset = dataset.map(make_conversion)
train_dataset = train_dataset.remove_columns(["context"])
test_dataset = test_dataset.map(make_conversion)
test_dataset = test_dataset.remove_columns(["context"])
print(test_dataset[0]) ## features: 'input', 'output', 'prompt'
print("**********************FINISHED*******************************")

# sys.exit()


### Define reward
COMPLETION_REWARD = 0
FORMAT_REWARD = 0.5
MAX_COMPLETION_LENGTH = 256
NO_ANSWER_REWARD=1.0
FP_FN_REWARD=0


## Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(ckpt)
tokenizer.pad_token_id = (0)

model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    torch_dtype=torch.bfloat16,

)
# lora_config = LoraConfig(
#     task_type="CAUSAL_LM",
#     r=LORA_RANK,
#     lora_alpha=2*LORA_RANK,
#     target_modules=['q_proj', 'v_proj'],
#     use_rslora=True,
#     lora_dropout=0.1,
# )
# model = get_peft_model(model, lora_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

OUTPUT_DIR = f"./EASY_COT_SFT(Epo1)_GRPO_RougeReward-{USE_ROUGE}-BertScore-{USE_BERTSCORE}-EM-{USE_EM}-{model_name}_{train_or_test}_Epoch={EPOCHS}_BS={BATCH_SIZE}_Beta={BETA}_LR={LR}_CompLength={MAX_COMPLETION_LENGTH}_FormatRD={FORMAT_REWARD}_NoAnswerRD={NO_ANSWER_REWARD}_FP_FNRD{FP_FN_REWARD}_CompletionRD={COMPLETION_REWARD}_Context_results" ## NOTE: TimeSents/Context

training_args = GRPOConfig(
    output_dir= OUTPUT_DIR,
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    save_strategy='epoch',
    save_safetensors=False,
    loss_type='dr_grpo',
    scale_rewards=False,
    bf16=True,
    max_completion_length=MAX_COMPLETION_LENGTH,  #  previous experiments: 128
    num_generations=NUM_ROLLOUT,  # default: 8
    max_prompt_length=2048,
    beta=BETA,

    dataloader_num_workers=32,
    dataloader_pin_memory=True,
)




def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n(.*?)\n</think>\n\n<answer>\n(.*?)\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    return [FORMAT_REWARD if match else 0.0 for match in matches]


def ans_reward_func(completions, **kwargs):
    """Reward function that checks if the completion matches the reference answer."""
    ref_answers = kwargs['output']
    NoExtract = False

    completion_contents = [None if re.search(r"<answer>\n(.*?)\n</answer>", completion[0]["content"], re.DOTALL) is None else re.search(r"<answer>\n(.*?)\n</answer>", completion[0]["content"], re.DOTALL).group(1) for completion in completions]
    print(completion_contents)
    ## if ref != "No Answer" then check if the completion is equal to the reference answer
    ## otherwise use the rouge score to compute the reward
    rewards = []
    for ref, completion in zip(ref_answers, completion_contents):
        if completion is None:
            rewards.append(COMPLETION_REWARD)
        else:
            if ref == "No Answer":
                rewards.append(NO_ANSWER_REWARD if ref == completion else FP_FN_REWARD)
            elif completion.lower() == 'no answer' and ref.lower()!='no answer':
                rewards.append(FP_FN_REWARD)
            else:
                rouge_reward = 0.0
                ## use rouge score to compute the reward
                if USE_ROUGE:
                    rouge_reward = rouge.compute(predictions=[completion], references=[ref])["rougeL"]

                bertscore_reward = 0.0
                em_reward = 0.0
                if USE_BERTSCORE:
                    bertscore_reward = bertscore.compute(
                                        predictions=[completion],
                                        references=[ref],
                                        model_type="models/bert-base-uncased",
                                        num_layers=model2layers["bert-base-uncased"],
                                    )["f1"][0]  
                if USE_EM:
                    if ref.lower() == completion.lower():
                        em_reward = 1.0
                    else:
                        em_reward = 0.0
                
                rewards.append(rouge_reward + bertscore_reward + em_reward)
        
    return rewards


trainer = GRPOTrainer(
    model=model, 
    reward_funcs=[format_reward_func, ans_reward_func], 
    args=training_args, 
    train_dataset=train_dataset,
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
