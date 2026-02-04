from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer,AutoModelForCausalLM, LlamaConfig
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import torch
import faiss
import numpy as np
from tqdm import tqdm

import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

input_KG_file = 'test_question_KG_gpt-4o-mini_whole_context.hard.json'
input_original_file = '../test.hard.json'


def load_KG_data(json_file_path):
    questions_kgs = []

    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in tqdm(data, desc="Processing items"):
        idx = item.get('id')
        question = item.get('question')
        kg = item.get('KG')
        context = item.get('context')
        
        if question and kg:
            questions_kgs.append({'idx': idx, 'question': question, 'KG': kg, 'context': context})
        else:
            print(f"idx {idx} has no question or KG")

    print('len question_kgs:', len(questions_kgs))
    return questions_kgs


def load_target_data(target_file_path):
    targets_dict = {}
    with open(target_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Processing targets"):
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")
            continue  # Skip invalid JSON data

        idx = item.get('idx')
        targets = item.get('targets')
        
        if idx and targets:
            targets_dict[idx] = targets

    return targets_dict

def merge_data(questions_kgs, targets):
    for qkg in questions_kgs:
        idx = qkg['idx']
        if idx in targets:
            qkg['target'] = targets[idx]
        else:
            print(f"Target not found for question {idx}")
    return questions_kgs


model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def embed_text(text):
    embeddings = model.encode(text, convert_to_tensor=True)
    return embeddings.cpu().numpy()

def generate_similar_kgs(data, k=10, output_file="similar_kgs.json"):
    all_results = []
    print(len(data))

    for data_example in tqdm(data):
        question = data_example['question']
        kg_sentences = []
        for item in data_example['KG']:
            kg_sentence = ""
            if item.get('head'):
                kg_sentence += f" {item['head']}"
            if item.get('relation'):
                kg_sentence += f" {item['relation']}"
            if item.get('tail'):
                kg_sentence += f" {item['tail']}"
            if item.get('timestamp'):
                kg_sentence += f" on {item['timestamp']}"
            kg_sentences.append(kg_sentence)
        
        # Calculate KG embeddings
        triple_embeddings = [embed_text(kg_sentence) for kg_sentence in kg_sentences]
        triple_embeddings_np = np.array(triple_embeddings).astype("float32")
        
        # Calculate question embedding
        question_embedding = embed_text(question).astype("float32")
        
        # Create FAISS index
        index = faiss.IndexFlatL2(triple_embeddings_np.shape[1])  # Use L2 distance
        index.add(triple_embeddings_np)  # Add vectors to index
        
        # Search for most similar KGs
        distances, indices = index.search(np.array([question_embedding]), k)
        similar_kgs = [data_example['KG'][i] for i in indices[0]]
        
        # Save results
        all_results.append({
            "idx": data_example['idx'],
            "question": question,
            'target': data_example['target'],
            'context': data_example['context'],
            "similar_kgs": similar_kgs
        })
    
    # Save to file
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)

output_file = "test_question_faiss_similar_KG_kgnum10_gpt-4o-mini_whole_context.hard.json"
question_kgs = merge_data(load_KG_data(input_KG_file), load_target_data(input_original_file))
generate_similar_kgs(question_kgs, k=10, output_file=output_file)
