import pandas as pd
from spellchecker import SpellChecker
from textblob import TextBlob
from Levenshtein import distance as levenshtein_distance
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import openai
from openai import OpenAI, OpenAIError
import os
from dotenv import load_dotenv

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


def evaluate_spell_checker(spell_checker_func, dataset):
    results = []
    for _, row in dataset.iterrows():
        misspelled = row['Misspelled']
        correct = row['Correct']
        corrected = spell_checker_func(misspelled)
        
        # Calculate metrics, with None handling
        accuracy = 1 if corrected == correct else 0
        levenshtein = levenshtein_distance(corrected, correct) if corrected and correct else float('inf')
        results.append((misspelled, correct, corrected, accuracy, levenshtein))

    return pd.DataFrame(results, columns=['Misspelled', 'Correct', 'Corrected', 'Accuracy', 'Levenshtein'])

# Main function to run evaluations and calculate metrics
def main():
    dataset = load_combined_dataset().sample(n=70)
    
    # Evaluate using each of the libraries/models
    print("Evaluating with PySpellChecker...")
    pyspell_results = evaluate_spell_checker(check_spelling_pyspellchecker, dataset)
    print(pyspell_results[['Accuracy', 'Levenshtein']].mean())

    print("Evaluating with TextBlob...")
    textblob_results = evaluate_spell_checker(check_spelling_textblob, dataset)
    print(textblob_results[['Accuracy', 'Levenshtein']].mean())

    print("Evaluating with Fine-tuned model...")
    fine_tuned_results = evaluate_spell_checker(check_spelling_finetuned_model, dataset)
    print(fine_tuned_results[['Accuracy', 'Levenshtein']].mean())

    print("Evaluating with OpenAI GPT-3...")
    gpt3_results = evaluate_spell_checker(check_spelling_gpt3, dataset)
    print(gpt3_results[['Accuracy', 'Levenshtein']].mean())

    for results, name in zip([pyspell_results, textblob_results, fine_tuned_results, gpt3_results],
                             ["PySpellChecker", "TextBlob", "Fine tuned model", "GPT-3"]):
        print(f"{name} Accuracy: {results['Accuracy'].mean()}")
        print(f"{name} Average Levenshtein Distance: {results['Levenshtein'].mean()}")

if __name__ == "__main__":
    main()
