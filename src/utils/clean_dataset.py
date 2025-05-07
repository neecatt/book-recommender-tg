import pandas as pd
from langdetect import detect, LangDetectException
import re
from tqdm import tqdm

def detect_language_safe(text):
    """Safely detect language of text with error handling."""
    try:
        text = str(text)  # Convert to string in case of non-string input
        text = re.sub(r'[0-9]', '', text)  # Remove numbers
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = text.strip()
        
        if not text:
            return 'unknown'
        return detect(text)
    except LangDetectException:
        return 'unknown'

def remove_non_english_books(input_path, output_path):
    """
    Remove non-English books based on title language detection.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save cleaned CSV file
    """
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    
    print("\nDetecting languages in titles...")
    df['title_lang'] = [detect_language_safe(title) for title in tqdm(df['title'], desc="Processing")]
    
    # Keep only English titles
    df_clean = df[df['title_lang'] == 'en'].copy()
    
    # Remove the temporary language detection column
    df_clean = df_clean.drop('title_lang', axis=1)
    
    # Save the cleaned dataset
    df_clean.to_csv(output_path, index=False)
    
    # Print statistics
    final_count = len(df_clean)
    removed_count = initial_count - final_count
    
    print("\nCleaning completed!")
    print(f"Original number of books: {initial_count}")
    print(f"Number of English books: {final_count}")
    print(f"Removed {removed_count} non-English books ({(removed_count/initial_count*100):.2f}%)")
    print(f"Cleaned dataset saved to: {output_path}")

if __name__ == "__main__":
    input_file = "/Users/neecat/Desktop/Projects/book-recommender-tg/src/data/books.csv"
    output_file = "/Users/neecat/Desktop/Projects/book-recommender-tg/src/data/books_english.csv"
    remove_non_english_books(input_file, output_file) 