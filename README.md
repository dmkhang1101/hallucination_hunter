# Hallucination Hunter
Dual-Model Auditing: Using Natural Language Inference (NLI) to Detect Hallucinations in High-Stakes LLM Outputs

## Setup

**1. Clone and install dependencies**
```bash
git clone <repo-url>
cd hallucination_hunter

# pip
pip install -r requirements.txt

# conda
conda install -c huggingface datasets
pip install openai spacy python-dotenv tqdm pandas

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

`atomic_claims.csv` schema:

| Column | Description |
|---|---|
| `question_id` | Stable hash ID linking claim back to its question |
| `question` | Original TruthfulQA question |
| `category` | TruthfulQA category (e.g. Health, Law, History) |
| `primary_answer` | Full GPT-4o answer |
| `claim_index` | 0-based position of claim within the answer |
| `claim` | Individual atomic fact/sentence |

## Project Structure

```
hallucination_hunter/
├── config.py          # Central config (paths, model names, defaults)
├── generate.py        # Step 1-2: dataset sampling + GPT-4o answer generation
├── extract.py         # Step 3: atomic claim extraction via spaCy
├── main.py            # CLI entry point
├── requirements.txt
├── .env.example       # API key template
└── data/              # Generated outputs (git-ignored)
```
