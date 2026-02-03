# AbstentionTemporalQA
Repo for paper "When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?" in ICLR2026


## Data
We provide the curated CoT data in the `data` folder for SFT cold-start training, named `cot_generate_valid_answers_easy.json` for TimeQA-Easy, and `cot_generate_valid_answers_hard.json` for TimeQA-Hard, which contain the `context`, `question`, `target`, and `reasoning`.

## Code
We provide the main code in the `scripts` folder, and we utilize the DeepSpeed as well. For training, please refer to `scripts/train`; and for evaluation, please refer to `scripts/eval`.