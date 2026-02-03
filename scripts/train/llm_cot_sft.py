import json
import torch
import torch.nn as nn
import sys
from datasets import Dataset
from transformers import LlamaForCausalLM, Trainer, TrainingArguments, LlamaTokenizer, AutoTokenizer,AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
# ## Use bfloat16
# torch.set_default_dtype(torch.bfloat16)

from  accelerate  import PartialState
device_string = PartialState().process_index


# Load JSON data
def load_json_data(json_file_path):
    data_question_answer=[]
    with open(json_file_path, 'r') as f:
        f = json.load(f)
        for line in f:
            data = line
            answer_str = ''

          
            # Cot data
            answer_str += data['reasoning']

            data_process = {'question': data['question'], 'answer': answer_str,
                                'reasoning': data['reasoning'],
                                'context': data['context']}
            data_question_answer.append(data_process)


    return data_question_answer

# Load and prepare dataset
train_or_test = 'train'

## cot data
json_file_path = 'data/hard/cot_generate_valid_answers_hard.json'
#'data/easy/cot_generate_valid_answers_easy.json'


model_name = "models/qwen2.5-0.5b-instruct"

LORA_RANK=64
EPOCHS=1
BATCH_SIZE=4



## instruction with Think in Qwen
instruction = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\
A conversation between User and Assistant. The user asks a question, and the Assistant solves it based on the given information. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer within 80 words. If there is no correct answer to the question, print No Answer \n\
   The reasoning  process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer> \n"






data = load_json_data(json_file_path)
dataset = Dataset.from_dict({
    'input': [item['question'] for item in data],
    'context': [item['context'] for item in data],
    'output': [ item['answer'] for item in data]
})




# Tokenization
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )


def tokenize_function(example):

    target =  example['output']
    context = example['context']
    question = example['input']

    ## qwen prompt ##
    ## with context
    prompt = "Question: " + question + "\n" + "Context: " + context[:] + "\n"




    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": prompt}
    ]
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(full_text)
    # sys.exit()

    full_text_tgt = full_text + target
   


    
    # print(full_text)
    max_length = 1024

    inputs = tokenizer(full_text_tgt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    attn_mask = inputs['input_ids'].ne(tokenizer.pad_token_id)
    
    # labels = outputs["input_ids"]
    
    labels = inputs['input_ids'].clone()
    ## MASK THE PROMPTS
    output_start_index = len(tokenizer(full_text)['input_ids'])
    labels[:, :output_start_index] = -100
    labels[labels == tokenizer.pad_token_id] = -100

   
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": attn_mask.squeeze(),
        "labels": labels.squeeze(),
     
    }

print("**********************PREPROCESS DATASETS*********************************")
tokenized_dataset = [tokenize_function(example) for example in dataset]
tokenized_dataset = Dataset.from_list(tokenized_dataset)

print("**********************FINISHED*******************************")


## load model from local path
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    local_files_only=True
    # output_hidden_states=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


output_dir = f"./CoT_SFT_Hard_{model_name}_{train_or_test}_Epoch={EPOCHS}_BS={BATCH_SIZE}"

# Training
training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=20,
    save_steps=2000,
    save_safetensors=False,
    bf16=True,
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_dataset, 
    )
trainer.train()
model.save_pretrained(output_dir)
