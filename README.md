# Hallucination Hunter
Dual-Model Auditing: Using Natural Language Inference (NLI) to Detect Hallucinations in High-Stakes LLM Outputs

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/dmkhang1101/hallucination_hunter.git
cd hallucination_hunter

# pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

**2. Configure API key**
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Running the Pipeline

| Command | Description |
|---|---|
| `python main.py` | Full run — 150 questions, real GPT-4o calls |
| `python main.py --n 100` | Custom question count |
| `python main.py --dry-run` | No API calls, placeholder answers (for testing) |
| `python main.py --skip-generate` | Skip generation, re-run claim extraction only |

## Output

All outputs are saved to `data/`:

| File | Description |
|---|---|
| `sampled_questions.csv` | Stratified sample of TruthfulQA questions across all categories |
| `primary_answers.csv` | Questions + GPT-4o generated answers |
| `atomic_claims.csv` | Each answer split into individual sentences/claims |
