import time
from data_processor import DataProcessor
from book_finder import BookFinder
from recommender import BookRecommender

def main():
    """Initialize and run the book recommender system with example recommendations."""
    print("Initializing book recommender system...")
    start_time = time.time()
    
    # Create data processor and load data
    data_processor = DataProcessor()
    data_processor.load_and_clean_data()
    print(f"Data processing time: {time.time() - start_time:.2f} seconds")
    
    # Create book finder
    book_finder = BookFinder(data_processor)
    
    # Create and train recommender
    start_time = time.time()
    recommender = BookRecommender(data_processor, book_finder)
    recommender.train()
    print(f"Model training time: {time.time() - start_time:.2f} seconds")
    
    # Example usage
    print("\nExample recommendations:")
    recommendations = recommender.get_recommendations(
        title="A brief history of time",
        max_pages=400,
    )
    
    if not recommendations.empty:
        print("\nRecommended books:")
        for _, book in recommendations.iterrows():
            print(f"\nTitle: {book['title']}")
            print(f"Author: {book['author']}")
            print(f"Genres: {', '.join(book['genres'])}")
            print(f"Rating: {book['rating']:.2f}")
            print(f"Pages: {book['pages']}")
    
    # # Evaluate model
    # print("\nEvaluating model...")
    # metrics = recommender.evaluate(num_samples=100, max_pages=400)
    # print("\nEvaluation metrics:")
    # for metric, value in metrics.items():
    #     print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main() 