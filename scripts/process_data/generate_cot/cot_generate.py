import json
import openai
from openai import OpenAI
from tqdm import tqdm
import re
import random
import tiktoken

client = OpenAI(api_key="")

model_name = "o1"
max_tokens = 16385
reserved_tokens = 1000

with open("system_message.txt", "r", encoding="utf-8") as f:
    system_message = f.read()
    
input_file = "../train.hard.json"
output_file = "cot_gpt_generate_hard.json"

encoding = tiktoken.encoding_for_model(model_name)

def construct_prompt(question, context):
    if len(encoding.encode(context)) > max_tokens:
        context_tokens = encoding.encode(context)
        truncated_tokens = context_tokens[:(max_tokens - reserved_tokens)]
        context = encoding.decode(truncated_tokens)
    return "question: " + question + "\n" + "context: " + context + "\n"

def ask_gpt(prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
    )
    return response.choices[0].message.content.strip()


def select_lines(input_file, not_first=False):
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            lines.append(entry)

    # Randomly select 100 entries from lines
    selected_lines = random.sample(lines, min(200, len(lines)))
    lines = selected_lines
    
    if not_first:
        # Load existing entries from cot_gpt_generate_1.json to check IDs
        existing_ids = set()
        try:
            with open('cot_gpt_generate_1.json', 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_ids = {entry['id'] for entry in existing_data}
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Keep selecting until we have enough unique entries
        final_lines = []
        while len(final_lines) < min(200, len(lines)):
            # Get remaining lines that don't have duplicate IDs
            remaining_lines = [line for line in lines if line['idx'] not in existing_ids]
            
            if not remaining_lines:
                break
                
            # Select a random entry from remaining lines
            selected = random.choice(remaining_lines)
            final_lines.append(selected)
            existing_ids.add(selected['idx'])

            return final_lines
    else:
        return lines


lines = select_lines(input_file, False)
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('[\n')
    
for line_idx, entry in enumerate(tqdm(lines[:], desc="Processing questions")):
    question = entry["question"]
    context = entry["context"]
    id = entry["idx"]
    target = entry["targets"]
    
    answer = ask_gpt(construct_prompt(context, question))
    
    answer_json = {
        "id": id,
        "question": question,
        "answer": answer,
        "target": target
    }
    
    result_json = json.dumps(answer_json, ensure_ascii=False, indent=4)
    indented_result_json = "\n".join("    " + line for line in result_json.splitlines())

    with open(output_file, 'a', encoding='utf-8') as f:
        if line_idx > 0:
            f.write(',\n')
        f.write(indented_result_json)

# write closing bracket
with open(output_file, 'a', encoding='utf-8') as f:
    f.write('\n]\n')

print(f"All answers has been saved to {output_file}.")