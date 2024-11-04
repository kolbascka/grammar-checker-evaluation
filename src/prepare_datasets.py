import random
import pandas as pd

# Load misspelled dataset with multiple misspelled forms per correct word
def load_misspelled_dataset(filepath='datasets/misspelled_dataset.txt'):
    data = []
    with open(filepath, 'r') as file:
        for line in file:
            # Split line by ':' to separate correct word from misspelled words
            parts = line.strip().split(':')
            if len(parts) == 2:
                correct = parts[0].strip()
                misspelled_variants = parts[1].split(',')
                
                # Process each misspelled variant
                for misspelled in misspelled_variants:
                    # Clean up whitespace and any extra characters
                    misspelled = misspelled.strip()
                    if misspelled:
                        data.append((misspelled, correct))
            else:
                print(f"Skipping malformed line: {line.strip()}")
    
    # Return as DataFrame
    return pd.DataFrame(data, columns=["Misspelled", "Correct"])

# Function to introduce synthetic typos into a clean sentence
def introduce_typos(text, error_rate=0.1):
    words = text.split()
    typo_data = []
    for word in words:
        if random.random() < error_rate:
            misspelled_word = typo_maker(word)
            if misspelled_word != word:
                typo_data.append((misspelled_word, word))
    return typo_data

def typo_maker(word):
    if len(word) > 2:
        i = random.randint(0, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word

# Load clean text for synthetic data generation
def load_clean_text(filepath='datasets/correctly_spelled_dataset'):
    with open(filepath, 'r') as file:
        clean_sentences = file.readlines()
    return clean_sentences

# Generate synthetic dataset
def generate_synthetic_dataset(clean_text_filepath='datasets/correctly_spelled_dataset', error_rate=0.1):
    clean_sentences = load_clean_text(clean_text_filepath)
    synthetic_data = []
    for sentence in clean_sentences:
        synthetic_data.extend(introduce_typos(sentence, error_rate))
    return pd.DataFrame(synthetic_data, columns=["Misspelled", "Correct"])

# Combine misspelled dataset with synthetic data
def combine_datasets(misspelled_filepath='datasets/misspelled_dataset.txt', clean_text_filepath='correctly_spelled_dataset', error_rate=0.1):
    misspelled_df = load_misspelled_dataset(misspelled_filepath)
    synthetic_df = generate_synthetic_dataset(clean_text_filepath, error_rate)
    combined_df = pd.concat([misspelled_df, synthetic_df], ignore_index=True)
    return combined_df

# Save combined dataset
def save_combined_dataset(output_filepath='datasets/combined_spelling_dataset.csv'):
    combined_df = combine_datasets()
    combined_df.to_csv(output_filepath, index=False)
    print(f"Combined dataset saved to {output_filepath}")

# Evaluation function placeholder (update with chosen spell checkers later)
def evaluate_spell_checkers(combined_df):
    # Placeholder for spell-checker evaluations on the combined dataset
    # Each spell checker would evaluate corrections based on "Misspelled" and "Correct" columns
    # Implement different evaluation metrics here (e.g., Accuracy, F1, Levenshtein Distance)
    pass

# Main execution
if __name__ == "__main__":
    save_combined_dataset()
