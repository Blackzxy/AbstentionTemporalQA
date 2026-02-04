import json
import openai
from openai import OpenAI
from tqdm import tqdm
import tiktoken

model_name = "gpt-4o"
client = OpenAI(api_key="") # your API key
encoding = tiktoken.encoding_for_model(model_name)

MAX_TOKENS = 16385
SAFE_MARGIN = 1000

# the input file
input_file = '../test.hard.json'
output_file = f'test_question_KG_{model_name}_whole_context.hard.json' # the output file name


system_message =  'You are a top-tier algorithm designed for extracting information in \
structured formats to build a knowledge graph. \
Try to capture as much information from the text as possible without \
sacrificing accuracy. Do not add any information that is not explicitly \
mentioned in the text. \
Your task is to identify the entities and relations and timestamps requested with the user prompt from a given \
text. You must generate the output in a JSON format containing a list \
with JSON objects. Each object should have the keys: "head", \
"head_type", "relation", "tail", "tail_type" and "timestamp". The "head" \
key must contain the text of the extracted entity. \
The "head_type" key must contain the type of the extracted head entity, \
The "relation" key must contain the type of relation between the "head" and the "tail". \
The "tail" key must represent the text of an extracted entity which is \
the tail of the relation, and the "tail_type" key must contain the type \
of the tail entity. The "timestamp" key must contain the timestamp of the event \
if it is present in the text. If the timestamp is not present, the value \
of the "timestamp" key must be null. \
Your task is to extract relationships from text strictly adhering \
to the provided schema. The relationships can only appear \
between specific node types are presented in the schema format \
like: (Entity1Type, RELATIONSHIP_TYPE, Entity2Type, TIME) /n\
Attempt to extract as many entities and relations as you can. Maintain \
Entity Consistency: When extracting entities, it\'s vital to ensure \
consistency. If an entity, such as "John Doe", is mentioned multiple \
times in the text but is referred to by different names or pronouns \
(e.g., "Joe", "he"), always use the most complete identifier for \
that entity. The knowledge graph should be coherent and easily \
understandable, so maintaining consistency in entity references is crucial. \
Identify all the events or actions that have time-related details. \
IMPORTANT NOTES:\n- Don\'t add any explanation and text.'


def count_tokens(text):
    return len(encoding.encode(text))

def truncate_to_max_tokens(context, max_tokens):
    tokens = encoding.encode(context)
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# output file
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('[\n')
  
results = []
kg_cache = {} # cache the KG for the same context with different questions

for line_idx, line in enumerate(tqdm(lines[:], desc="Processing items")):
    line = line.strip()
    if not line:
        continue  # skip empty line
    try:
        item = json.loads(line)
    except json.JSONDecodeError:
        print(f"Skipping invalid JSON line: {line}")
        continue  # skip invalid JSON data
    
    context = item['context']
    idx = item['idx']
    
    parts = idx.split('#')
    idx_context = '#'.join(parts[:2])
    
    # truncate the context if it is too long
    if count_tokens(context) + count_tokens(system_message) > MAX_TOKENS - SAFE_MARGIN:
        context = truncate_to_max_tokens(context, MAX_TOKENS - SAFE_MARGIN - count_tokens(system_message))
    
    if idx_context in kg_cache:
        kg = kg_cache[idx_context]
    else:
        response = client.chat.completions.create(
			model = model_name,
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
		
		# remove the part that is not JSON format
        try:
            kg = json.loads(response_text)
        except json.JSONDecodeError:
			# try to remove the part that is not JSON format
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            if start_idx != -1 and end_idx != -1:
                response_text = response_text[start_idx:end_idx+1]
                try:
                    kg = json.loads(response_text)
                except json.JSONDecodeError:
                    kg = []
            else:
                kg = []
        kg_cache[idx_context] = kg
            
    result = {
        'id': idx,
        'question': item['question'],
        'context': context,
        'KG': kg
    }
    # print('id: \n', result['id'])
    # print('KG: \n', result['KG'])
    
    # append the result to the output file
    result_json = json.dumps(result, ensure_ascii=False, indent=4)
    indented_result_json = "\n".join("    " + line for line in result_json.splitlines())  # add 4 spaces for each line

    with open(output_file, 'a', encoding='utf-8') as f:
        if line_idx > 0:
            f.write(',\n')  # add comma to separate the lines
        f.write(indented_result_json)  # write the adjusted JSON content

    results.append(result)
    
    
# write the end of the output file
with open(output_file, 'a', encoding='utf-8') as f:
    f.write(']\n')
    
print("All KGs have been saved to the output file.")
