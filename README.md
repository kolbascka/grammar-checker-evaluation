# Spell Checker Evaluation Project

### Instructions for Reproducing Results Locally

#### **1. Environment Setup**

a. Clone the repository: 
```bash
git clone https://github.com/kolbascka/grammar-checker-evaluation/
```

b. Set up the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
```
c. Install dependencies:
```bash
pip install -r requirements.txt
```

#### **2. API Key Setup**

Note: This project uses the OpenAI API, so you will need an API key to run the evaluation on the ChatGPT-3.5 model evaluation. For testing on my own machine, and in the github workflow, I used my own secret API key.

a. Add your OpenAI API key to a .env file in the root directory of the project.
```bash
OPENAI_API_KEY=your_openai_api_key
```

#### **3. Running the Scripts**

a. Data Preparation: Run `prepare_datasets.py` to load and generate the datasets.

```python
python src/prepare_datasets.py
```

b. Spell Checker Evaluation: Run `evaluate_spell_checkers.py` to evaluate the models and log the metrics.

```python
python src/evaluate_spell_checkers.py
```

#### **4. Results**

The output will be saved in `evaluation_metrics_summary.csv` in the root directory of the project, containing all evaluation metrics for each model.
