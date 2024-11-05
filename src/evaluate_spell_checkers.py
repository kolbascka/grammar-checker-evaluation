import os
import pandas as pd
from dotenv import load_dotenv
from spellchecker import SpellChecker
from textblob import TextBlob
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import openai
from openai import OpenAI, OpenAIError

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def load_combined_dataset(filepath='datasets/combined_spelling_dataset.csv'):
    return pd.read_csv(filepath)

spellchecker = SpellChecker()

def check_spelling_pyspellchecker(word):
    corrected = spellchecker.correction(word)
    return corrected if corrected else word  # Return original word if correction is None

def check_spelling_textblob(word):
    blob = TextBlob(word)
    corrected = blob.correct()
    return str(corrected) if corrected else word  # Return original word if correction is None

def check_spelling_finetuned_model(word):
    tokenizer = AutoTokenizer.from_pretrained("pszemraj/grammar-synthesis-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("pszemraj/grammar-synthesis-small")

    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def check_spelling_gpt3(word):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Correct the following word for spelling errors: {word}"}
            ]
        )

        corrected = response.choices[0].message.content.strip()
        return corrected

    except OpenAIError as e:
        print("OpenAI Error:", e)
        return word  # Return the original word if there's an error


def evaluate_spell_checker(spell_checker_func, dataset, output_file="evaluation_metrics_summary.csv"):
    results = []
    y_true = []
    y_pred = []

    for _, row in dataset.iterrows():
        misspelled = row['Misspelled']
        correct = row['Correct']
        corrected = spell_checker_func(misspelled)
        
        # Calculate basic metrics
        accuracy = 1 if corrected == correct else 0
        levenshtein = levenshtein_distance(corrected, correct) if corrected and correct else float('inf')
        
        # Append true and predicted labels for later evaluation
        y_true.append(correct == misspelled)
        y_pred.append(correct == corrected)
        
        results.append((misspelled, correct, corrected, accuracy, levenshtein))

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results, columns=['Misspelled', 'Correct', 'Corrected', 'Accuracy', 'Levenshtein'])

    # Calculate overall metrics
    avg_accuracy = results_df['Accuracy'].mean()
    avg_levenshtein = results_df['Levenshtein'].mean()
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Save metrics to dictionary and then to CSV
    metrics = {
        "Model": spell_checker_func.__name__,
        "Accuracy": avg_accuracy,
        "Average Levenshtein Distance": avg_levenshtein,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    # Append metrics to CSV file
    metrics_df = pd.DataFrame([metrics])
    with open(output_file, 'a') as f:
        metrics_df.to_csv(f, header=f.tell()==0, index=False)

    print(f"Metrics for {spell_checker_func.__name__} saved to {output_file}")
    return results_df

def main():
    dataset = load_combined_dataset().sample(n=70)
    output_file = "evaluation_metrics_summary.csv"

    # Clear the output file at the beginning of each run
    with open(output_file, 'w') as f:
        f.write('')  # Empty the file before appending new metrics

    print("Evaluating with PySpellChecker...")
    evaluate_spell_checker(check_spelling_pyspellchecker, dataset, output_file)

    print("Evaluating with TextBlob...")
    evaluate_spell_checker(check_spelling_textblob, dataset, output_file)

    print("Evaluating with Fine-tuned model...")
    evaluate_spell_checker(check_spelling_finetuned_model, dataset, output_file)

    print("Evaluating with OpenAI GPT-3.5...")
    evaluate_spell_checker(check_spelling_gpt3, dataset, output_file)

if __name__ == "__main__":
    main()
