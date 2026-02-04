import json
from keybert import KeyBERT
from tqdm import tqdm
from difflib import SequenceMatcher

keybert_model = KeyBERT()
keyword_number = 5
kg_number = 10

# Define file paths
input_file = "test_question_KG_gpt-4o-mini_windows0.3_step0.15.hard.json"
output_file = f"test_question_keybert_similar_KG_keywordnum{keyword_number}_kgnum{kg_number}_gpt-4o-mini_windows0.3_step0.15.hard.json"


def calculate_similarity(str1, str2):
    return SequenceMatcher(None, str1, str2).ratio()


with open(input_file, "r", encoding="utf-8") as f:
    entries = json.load(f)
    
final_entries = []
for entry in tqdm(entries, desc="Processing questions"):
    question = entry["question"]
    kg = entry["KG"]
    
    # Extract top keywords from question
    question_keywords = keybert_model.extract_keywords(question, top_n=keyword_number)
    question_keywords = [kw[0] for kw in question_keywords]

    # Calculate similarity between question keywords and KG entities
    for i in range(len(kg)):
        kg_item = kg[i]
        head = kg_item.get("head")
        tail = kg_item.get("tail")
        relation = kg_item.get("relation")
        timestamp = kg_item.get("timestamp")
        
        # Calculate similarity scores for each component
        head_similarity = max([calculate_similarity(keyword, head) for keyword in question_keywords]) if head else 0
        tail_similarity = max([calculate_similarity(keyword, tail) for keyword in question_keywords]) if tail else 0
        relation_similarity = max([calculate_similarity(keyword, relation) for keyword in question_keywords]) if relation else 0
        timestamp_similarity = max([calculate_similarity(keyword, str(timestamp)) for keyword in question_keywords]) if timestamp else 0
        
        # Combine similarity scores
        kg[i]['similarity'] = head_similarity + tail_similarity + relation_similarity + timestamp_similarity

    # Sort KGs by similarity and select top N
    kg.sort(key=lambda x: x["similarity"], reverse=True)
    top_n_kg = kg[:kg_number]
    
    final_entries.append({
        "id": entry["id"],
        "question": question,
        "context": entry["context"],
        "KG": top_n_kg
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_entries, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")
