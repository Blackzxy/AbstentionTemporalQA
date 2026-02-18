# AbstentionTemporalQA

Official repository for the paper [**"When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?"** (ICLR 2026)](https://arxiv.org/abs/2602.04755).

## Overview

This repository provides code and data for improving language models on temporal question answering, along with a systematic analysis of their selective abstention behavior on temporal QA tasks and general QA tasks.

## Table of Contents

- [Data](#data)
- [Project Structure](#project-structure)
- [Data Processing](#data-processing)
- [Training](#training)

## Data

We provide curated Chain-of-Thought (CoT) data for SFT cold-start training in the `data` folder:

### Temporal QA Datasets
- **`cot_generate_valid_answers_easy.json`**: CoT data for TimeQA-Easy
- **`cot_generate_valid_answers_hard.json`**: CoT data for TimeQA-Hard

Each entry contains:
- `id`: Question ID
- `context`: Background information
- `question`: The temporal question
- `target`: Ground truth answer(s)
- `reasoning`: Step-by-step reasoning process enclosed in `<think>` tags
- `answer`: Final answer enclosed in `<answer>` tags

### Non-Temporal Datasets
We also provide abstention versions of standard benchmarks:
- **`squad_valid_samples_3000.json`**: SQuAD samples for testing generalization
- **`mmlu_abstain.json`**: MMLU with abstention options
- **`hellaswag_abstain.json`**: HellaSwag with abstention options

## Project Structure

```
AbstentionTemporalQA/
├── data/                           # Training and evaluation data
│   ├── cot_generate_valid_answers_easy.json
│   ├── cot_generate_valid_answers_hard.json
│   ├── squad_valid_samples_3000.json
│   ├── mmlu_abstain.json
│   └── hellaswag_abstain.json
├── scripts/
│   ├── process_data/                   # Data preprocessing scripts
│   │   ├── extract_kg/                 # Knowledge graph extraction
│   │   │   ├── extract_KG_from_whole_context.py
│   │   │   ├── faiss_similarity_select_kgs.py
│   │   │   └── keybert_similarity_select_kgs.py
│   │   ├── extract_time_subcontext/    # time sub-context extraction 
│   │   │   └── extract_time_sentence_from_context.py
│   │   └── generate_cot/               # Chain-of-Thought generation
│   │       ├── cot_generate.py
│   │       └── system_message.txt
│   ├── train/                          # Training scripts
│       ├── llm_cot_sft.py              # Supervised fine-tuning
│       ├── llm_rl.py                   # Reinforcement learning (GRPO)
│       ├── train.sh                    # Training launcher
│       └── ds_config                   # DeepSpeed configuration
│   
├── models/                             # Evaluation metrics
│   ├── bertscore.py
│   └── rouge.py
├── LICENSE
└── README.md
```

## Data Processing

### 1. Knowledge Graph Extraction

Extract structured knowledge graphs from contexts:

```bash
cd scripts/process_data/extract_kg
python extract_KG_from_whole_context.py
```

This script:
- Uses API to extract entities, relations, and timestamps
- Caches KG for contexts with multiple questions
- Outputs structured JSON with head, tail, relation, and timestamp fields

### 2. Knowledge Graph Similarity Selection

Select most relevant KG triples using semantic similarity:

**FAISS-based Selection:**
```bash
python faiss_similarity_select_kgs.py
```
- Uses `SentenceTransformer` embeddings
- Selects top-k KG triples most similar to the question
- Efficient for large-scale KG selection

**KeyBERT-based Selection:**
```bash
python keybert_similarity_select_kgs.py
```
- Extracts keywords from questions using KeyBERT
- Calculates string similarity between keywords and KG entities
- Ranks KG triples by combined similarity scores

### 3. Time-Relevant Sentence Extraction

Extract sentences containing temporal information:

```bash
cd scripts/process_data/extract_time_subcontext
python extract_time_sentence_from_context.py
```

- Uses LLM API to identify time-related sentences
- Reduces context length while preserving temporal information
- Useful for focused temporal reasoning

### 4. Chain-of-Thought Data Generation

Generate reasoning traces for training:

```bash
cd scripts/process_data/generate_cot
python cot_generate.py
```

Configuration:
- System prompt: Defined in `system_message.txt`
- Output format: `<think>reasoning</think><answer>answer</answer>`

## Training

### Stage 1: Supervised Fine-Tuning (SFT)

Train the model on curated CoT data:

```bash
cd scripts/train
accelerate launch --config_file ds_config llm_cot_sft.py
```

### Stage 2: Reinforcement Learning (GRPO)

Refine abstention behavior using Group Relative Policy Optimization:

```bash
accelerate launch --config_file ds_config llm_rl.py
```

**Reward Components:**

1. **Format Reward**: Encourages structured output with `<think>` and `<answer>` tags
2. **Answer Reward**:
   - **No Answer Case**: for correctly abstaining
   - **Has Answer Case**: 
     - ROUGE-L similarity (optional)
     - BERTScore F1 (optional)
     - Exact Match (binary)
   - **False Positive/Negative**: Penalty reward for incorrect abstention decisions

### Training Script

```bash
# Run SFT/RL
bash train.sh
```

The script uses DeepSpeed for distributed training with the configuration in `ds_config`.

## Citation
If you find this repository is useful, please star🌟 this repo and cite🔗 our paper.
```
@misc{zhou2026silencegoldenllmslearn,
      title={When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?}, 
      author={Xinyu Zhou and Chang Jin and Carsten Eickhoff and Zhijiang Guo and Seyed Ali Bahrainian},
      year={2026},
      eprint={2602.04755},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.04755}, 
}
```