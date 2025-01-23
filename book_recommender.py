import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

class BookRecommender:
    """Book recommendation system using content-based filtering with TF-IDF and cosine similarity."""
    
    def __init__(self):
        """Initialize the recommender with data loading and preprocessing."""
        self.books_df = pd.read_csv('data/books_1.Best_Books_Ever.csv')
        self.books_df = self.books_df.fillna('')
        
        self.tfidf = TfidfVectorizer(stop_words='english')
        
        self.books_df['features'] = self.books_df['title'].astype(str) + ' ' + \
                                  self.books_df['author'].astype(str) + ' ' + \
                                  self.books_df['genres'].astype(str)
        
        self.tfidf_matrix = self.tfidf.fit_transform(self.books_df['features'])
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
    def get_recommendations(self, book_title, n_recommendations=5):
        """
        Get book recommendations based on a given book title.
        
        Args:
            book_title: Title of the book to base recommendations on
            n_recommendations: Number of recommendations to return
            
        Returns:
            DataFrame containing recommended books with their details
        """
        try:
            idx = self.books_df[self.books_df['title'].str.lower() == book_title.lower()].index[0]
        except IndexError:
            return "Book not found in the database. Please check the spelling or try another book."
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        book_indices = [i[0] for i in sim_scores]
        recommendations = self.books_df.iloc[book_indices][['title', 'author', 'rating', 'genres']]
        return recommendations

def main():
    """Run an interactive book recommendation session."""
    recommender = BookRecommender()
    
    print("Welcome to the Book Recommendation System!")
    print("----------------------------------------")
    
    while True:
        book_title = input("\nEnter a book title (or 'quit' to exit): ")
        
        if book_title.lower() == 'quit':
            break
        
        recommendations = recommender.get_recommendations(book_title)
        
        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print("\nBased on your input, here are some book recommendations:")
            print("\nRecommended Books:")
            for idx, row in recommendations.iterrows():
                print(f"\nTitle: {row['title']}")
                print(f"Author: {row['author']}")
                print(f"Rating: {row['rating']}")
                print(f"Genres: {row['genres']}")
                print("-" * 50)

if __name__ == "__main__":
    main() 