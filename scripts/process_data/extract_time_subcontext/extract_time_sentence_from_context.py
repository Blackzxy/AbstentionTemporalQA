import json
from openai import OpenAI
from tqdm import tqdm
import tiktoken

client = OpenAI(api_key="")

MAX_TOKENS = 16385
SAFE_MARGIN = 1000

model_name = "gpt-4o-mini"
encoding = tiktoken.encoding_for_model(model_name)

input_file = '../test.hard.json'
output_file = f'test_question_KG_{model_name}_extract_sentence_with_time.hard.json'

system_message =  'You are a top-tier algorithm designed for extracting sentences with time information in the text. \
Try to capture as much time information from the text as possible without \
sacrificing accuracy. Do not add any information that is not explicitly \
mentioned in the text. \
Your task is to identify the complete sentences with time information requested with the user prompt from a given \
text related to the given query. You must generate the output in a JSON format containing a list of complete sentences with time information from the given text. \
IMPORTANT NOTES:\n- Don\'t add any explanation and text. Don\'t change the original sentence. \
Identify all the events or actions that have time-related details.'

def count_tokens(text):
    return len(encoding.encode(text))

def truncate_to_max_tokens(context, max_tokens):
    tokens = encoding.encode(context)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Initialize output file with opening bracket
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('[\n')
  
results = []

for line_idx, line in enumerate(tqdm(lines[:], desc="Processing items")):
    line = line.strip()
    if not line:
        continue
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON line: {line}")
        continue
    
    context = item['context']
    idx = item['idx']
    
    context = f"Query: {item['question']}\nContext: {context}"
    
    if count_tokens(context) + count_tokens(system_message) > MAX_TOKENS - SAFE_MARGIN:
        context = truncate_to_max_tokens(context, MAX_TOKENS - SAFE_MARGIN - count_tokens(system_message))
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": context
            }
        ],
        temperature=0,
        top_p=1
    )
    response_text = response.choices[0].message.content
    
    # Parse JSON response with fallback for malformed output
    try:
        t_sentences = json.loads(response_text)
    except json.JSONDecodeError:
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        if start_idx != -1 and end_idx != -1:
            response_text = response_text[start_idx:end_idx+1]
            try:
                t_sentences = json.loads(response_text)
            except json.JSONDecodeError:
                t_sentences = []
        else:
            t_sentences = []
            
    result = {
        'id': idx,
        'question': item['question'],
        'context': item['context'],
        'time_sentences': t_sentences,
        'targets': item['targets']
    }
    
    # Write result to output file with proper indentation
    result_json = json.dumps(result, ensure_ascii=False, indent=4)
    indented_result_json = "\n".join("    " + line for line in result_json.splitlines())

    with open(output_file, 'a', encoding='utf-8') as f:
        if line_idx > 0:
            f.write(',\n')
        f.write(indented_result_json)

    results.append(result)

# Write closing bracket
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(']\n')

print(f"Results saved to {output_file}")

