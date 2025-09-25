# FastThink

FastThink implements codebook-based adapter models for fast mathematical reasoning.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Generate CoT Training Data
```bash
python create_cot_data.py --task <task_type> --input-file <input.jsonl> --output-file <output.jsonl>
```

### Training
```bash
python train.py -source_file <training_data.jsonl> -save_path <output_dir>
```

### Evaluation
```bash
python evaluate.py -c <checkpoint_path> -f <test_data.jsonl> -o <output.jsonl> -t <task_type>
```


Task types: `math_reasoning`, `programming`, `olympiad`