import json
from openai import OpenAI
from tqdm import tqdm
import tiktoken

client = OpenAI(api_key="")

model_name = "o4-mini"
system_message = "You are a top expert at answering questions about time. \
You have to think step by step according to the context and the question. \
Give your think process in <think></think> tags. \
Finally, give your answer in <answer></answer> tags. \
If there is no answer towards the question in the context, please respond \"no answer\" in <answer></answer> tags. \
Do not search the website."

max_tokens = 16385
reserved_tokens = 1000

input_file = "../test.easy.json"
output_file = "openAI_test_easy_o4-mini.json"

encoding = tiktoken.encoding_for_model(model_name)

def construct_prompt(context, question):
    if len(encoding.encode(context)) > max_tokens:
        context = context[:max_tokens - reserved_tokens]
    return "question: " + question + "\n" + "context: " + context + "\n" + "answer: "


def ask_gpt(prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response.choices[0].message.content.strip()
    return answer


lines = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line.strip())
        lines.append(entry)

# Initialize output file with opening bracket
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
        "context": context,
        "answer": answer,
        "target": target
    }
    
    result_json = json.dumps(answer_json, ensure_ascii=False, indent=4)
    indented_result_json = "\n".join("    " + line for line in result_json.splitlines())

    with open(output_file, 'a', encoding='utf-8') as f:
        if line_idx > 0:
            f.write(',\n')
        f.write(indented_result_json)

# Write closing bracket
with open(output_file, 'a', encoding='utf-8') as f:
    f.write('\n]\n')

print(f"Results saved to {output_file}")