import pandas as pd
from spellchecker import SpellChecker
from textblob import TextBlob
from Levenshtein import distance as levenshtein_distance
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file (for local testing)
load_dotenv()

# Retrieve OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load the combined dataset
def load_combined_dataset(filepath='combined_spelling_dataset.csv'):
    return pd.read_csv(filepath)

# Initialize spell checkers
spellchecker = SpellChecker()

# Spell-checking function using PySpellChecker
def check_spelling_pyspellchecker(word):
    corrected = spellchecker.correction(word)
    return corrected if corrected else word  # Return original word if correction is None

# Spell-checking function using TextBlob (alternative to Hunspell)
def check_spelling_textblob(word):
    blob = TextBlob(word)
    corrected = blob.correct()
    return str(corrected) if corrected else word  # Return original word if correction is None

# Spell-checking function using fine-tuned BERT model
def check_spelling_finetuned_bert(word):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Replace with fine-tuned model if available
    model = AutoModelForSequenceClassification.from_pretrained("path_to_finetuned_model")
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    prediction = torch.argmax(logits, dim=-1)
    return tokenizer.decode(prediction[0])

# Spell-checking function using OpenAI's GPT-3
def check_spelling_gpt3(word):
    prompt = f"Correct the following word for spelling errors: {word}"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=5,
        temperature=0
    )
    corrected = response.choices[0].text.strip()
    return corrected if corrected else word  # Return original word if correction is None

# Evaluate a spell checker on the dataset with None handling
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
    
    # Convert to DataFrame for analysis
    return pd.DataFrame(results, columns=['Misspelled', 'Correct', 'Corrected', 'Accuracy', 'Levenshtein'])

# Main function to run evaluations and calculate metrics
def main():
    dataset = load_combined_dataset()
    
    # Evaluate using PySpellChecker
    print("Evaluating with PySpellChecker...")
    pyspell_results = evaluate_spell_checker(check_spelling_pyspellchecker, dataset)
    print(pyspell_results[['Accuracy', 'Levenshtein']].mean())

    # Evaluate using TextBlob (alternative to Hunspell)
    print("Evaluating with TextBlob...")
    textblob_results = evaluate_spell_checker(check_spelling_textblob, dataset)
    print(textblob_results[['Accuracy', 'Levenshtein']].mean())
    
    # Evaluate using Fine-tuned BERT model
    print("Evaluating with Fine-tuned BERT...")
    bert_results = evaluate_spell_checker(check_spelling_finetuned_bert, dataset)
    print(bert_results[['Accuracy', 'Levenshtein']].mean())

    # Evaluate using OpenAI GPT-3
    print("Evaluating with OpenAI GPT-3...")
    gpt3_results = evaluate_spell_checker(check_spelling_gpt3, dataset)
    print(gpt3_results[['Accuracy', 'Levenshtein']].mean())

    # Calculate additional metrics if needed
    for results, name in zip([pyspell_results, textblob_results, bert_results, gpt3_results],
                             ["PySpellChecker", "TextBlob", "BERT", "GPT-3"]):
        print(f"{name} Accuracy: {results['Accuracy'].mean()}")
        print(f"{name} Average Levenshtein Distance: {results['Levenshtein'].mean()}")

if __name__ == "__main__":
    main()
