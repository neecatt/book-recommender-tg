import time
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.data_processor import DataProcessor
from src.data.book_finder import BookFinder
from src.models.recommender import BookRecommender

def main():
    """Initialize and run the book recommender system with example recommendations."""
    print("Initializing book recommender system...")
    start_time = time.time()
    
    data_processor = DataProcessor()
    data_processor.load_and_clean_data()
    print(f"Data processing time: {time.time() - start_time:.2f} seconds")
    
    book_finder = BookFinder(data_processor)
    
    start_time = time.time()
    recommender = BookRecommender(data_processor, book_finder)
    recommender.train()
    print(f"Model training time: {time.time() - start_time:.2f} seconds")
    
    recommender.get_recommendations(
        title="Harry Potter and the Philosopher's Stone",
        max_pages=400,
    )

if __name__ == "__main__":
    main() 