import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from collections import defaultdict
from exceptions import BookNotFoundError, InvalidParameterError, BookRecommenderError
from config import BookRecommenderConfig as cfg

class BookRecommender:
    def __init__(self, data_processor, book_finder):
        self.data_processor = data_processor
        self.book_finder = book_finder
        self.model = None
        self.train_books = None
        self.test_books = None
        
    def train(self):
        """Train the recommendation model"""
        # Create feature matrix
        feature_matrix = self.data_processor.create_feature_matrix()
        
        # Split data
        train_indices, test_indices = train_test_split(
            np.arange(len(self.data_processor.books_cleaned)), 
            test_size=cfg.TEST_SIZE, 
            random_state=cfg.RANDOM_STATE
        )
        
        train_matrix = feature_matrix[train_indices]
        self.train_books = self.data_processor.books_cleaned.iloc[train_indices]
        self.test_books = self.data_processor.books_cleaned.iloc[test_indices]
        
        # Train model
        self.model = NearestNeighbors(
            algorithm='auto',
            metric='cosine',
            n_neighbors=cfg.N_NEIGHBORS
        )
        self.model.fit(train_matrix)
        
    def recommend_books(self, input_title, input_author=None, max_pages=None, 
                       language=None, num_recommendations=None, max_books_per_author=None):
        """
        Recommend books based on input book and filters.
        
        Args:
            input_title (str): Title of the input book
            input_author (str, optional): Author of the input book
            max_pages (int, optional): Maximum number of pages
            language (str, optional): Language filter
            num_recommendations (int): Number of recommendations to return
            max_books_per_author (int): Maximum books per author in recommendations
            
        Returns:
            pandas.DataFrame: Recommended books
            
        Raises:
            InvalidParameterError: If input parameters are invalid
            BookNotFoundError: If the input book cannot be found
            BookRecommenderError: For other recommendation-related errors
        """
        # Set default values
        num_recommendations = num_recommendations or cfg.DEFAULT_NUM_RECOMMENDATIONS
        max_books_per_author = max_books_per_author or cfg.DEFAULT_MAX_BOOKS_PER_AUTHOR
        
        # Validate parameters
        if not isinstance(num_recommendations, int) or num_recommendations <= 0:
            raise InvalidParameterError("num_recommendations must be a positive integer")
        
        if not isinstance(max_books_per_author, int) or max_books_per_author <= 0:
            raise InvalidParameterError("max_books_per_author must be a positive integer")
            
        if max_pages is not None and (not isinstance(max_pages, (int, float)) or max_pages <= 0):
            raise InvalidParameterError("max_pages must be a positive number")
            
        try:
            # Find the book
            book_match = self.book_finder.find_book_name(input_title, input_author)
            if not book_match:
                return pd.DataFrame(columns=['title', 'author', 'genres', 'rating', 'pages', 'weighted_rating'])
            
            matched_title, matched_author = book_match
            input_book = self.data_processor.books_cleaned[
                (self.data_processor.books_cleaned['title'] == matched_title) & 
                (self.data_processor.books_cleaned['author'] == matched_author)
            ]
            
            if input_book.empty:
                raise BookNotFoundError(f"Book '{matched_title}' by {matched_author} not found in database")
                
            input_book = input_book.iloc[0]
            input_vector = csr_matrix(
                self.data_processor.encode_book_vectorized(pd.DataFrame([input_book]))
            )
            
            # Get recommendations
            n_neighbors = min(num_recommendations * 5, len(self.train_books))
            distances, indices = self.model.kneighbors(input_vector, n_neighbors=n_neighbors)
            
            # Get recommendations and apply filters
            recommendations = self.train_books.iloc[indices[0]].copy()
            recommendations['distance'] = distances[0]
            
            # Remove input book
            recommendations = recommendations[
                ~((recommendations['title'] == matched_title) & 
                  (recommendations['author'] == matched_author))
            ]
            
            # Apply filters
            if max_pages is not None:
                recommendations = recommendations[recommendations['pages'] <= max_pages]
            if language is not None:
                recommendations = recommendations[recommendations['language'] == language]
            
            if len(recommendations) == 0:
                return pd.DataFrame(columns=['title', 'author', 'genres', 'rating', 'pages', 'weighted_rating'])
            
            # Ensure author diversity
            author_counts = defaultdict(int)
            diverse_recommendations = []
            
            for _, book in recommendations.iterrows():
                if author_counts[book['author']] < max_books_per_author:
                    diverse_recommendations.append({
                        'title': book['title'],
                        'author': book['author'],
                        'genres': book['genres'],
                        'rating': book['rating'],
                        'pages': book['pages'],
                        'weighted_rating': book['weighted_rating'],
                        'distance': book['distance']
                    })
                    author_counts[book['author']] += 1
                    
                if len(diverse_recommendations) >= num_recommendations:
                    break
            
            if not diverse_recommendations:
                return pd.DataFrame(columns=['title', 'author', 'genres', 'rating', 'pages', 'weighted_rating'])
            
            # Create final recommendations DataFrame
            recommendations = pd.DataFrame(diverse_recommendations)
            
            # Calculate final score
            recommendations['final_score'] = (
                recommendations['weighted_rating'] * cfg.WEIGHTED_RATING_SCORE + 
                (1 - recommendations['distance']) * cfg.DISTANCE_SCORE
            )
            recommendations = recommendations.sort_values('final_score', ascending=False)
            
            return recommendations[['title', 'author', 'genres', 'rating', 'pages', 'weighted_rating']].head(num_recommendations)
            
        except BookRecommenderError:
            raise
        except Exception as e:
            raise BookRecommenderError(f"An error occurred while generating recommendations: {str(e)}")
            
    def evaluate(self, num_samples=100, max_pages=None):
        """
        Evaluate the recommendation system.
        
        Args:
            num_samples (int): Number of test samples
            max_pages (int, optional): Maximum pages filter for evaluation
            
        Returns:
            dict: Evaluation metrics
            
        Raises:
            InvalidParameterError: If input parameters are invalid
        """
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise InvalidParameterError("num_samples must be a positive integer")
            
        if max_pages is not None and (not isinstance(max_pages, (int, float)) or max_pages <= 0):
            raise InvalidParameterError("max_pages must be a positive number")
        
        try:
            np.random.seed(cfg.RANDOM_STATE)
            if num_samples > len(self.test_books):
                raise InvalidParameterError(
                    f"num_samples ({num_samples}) cannot be greater than the number of test books ({len(self.test_books)})"
                )
                
            test_indices = np.random.choice(len(self.test_books), num_samples, replace=False)
            test_sample = self.test_books.iloc[test_indices]
            
            metrics = {
                'genre_precision': 0,
                'author_diversity': 0,
                'page_compliance': 0
            }
            
            successful_samples = 0
            
            for _, test_row in test_sample.iterrows():
                try:
                    recommendations = self.recommend_books(
                        test_row['title'],
                        max_pages=max_pages if max_pages else test_row['pages'] + 100
                    )
                    
                    if recommendations.empty:
                        continue
                        
                    successful_samples += 1
                    
                    # Genre precision
                    test_genres = set(test_row['genres'])
                    if test_genres:  # Only calculate if the test book has genres
                        recommended_genres = set([genre for rec in recommendations['genres'] for genre in rec])
                        metrics['genre_precision'] += len(test_genres.intersection(recommended_genres)) / len(test_genres)
                    
                    # Author diversity
                    unique_authors = len(recommendations['author'].unique())
                    metrics['author_diversity'] += unique_authors / len(recommendations)
                    
                    # Page compliance
                    if max_pages:
                        metrics['page_compliance'] += (recommendations['pages'] <= max_pages).all()
                        
                except BookRecommenderError:
                    continue  # Skip failed recommendations
            
            if successful_samples == 0:
                raise BookRecommenderError("No successful recommendations during evaluation")
                
            return {k: v/successful_samples for k, v in metrics.items()}
            
        except Exception as e:
            if isinstance(e, BookRecommenderError):
                raise
            raise BookRecommenderError(f"An error occurred during evaluation: {str(e)}")
            
    def get_recommendations(self, title, author=None, max_pages=None, language=None):
        """
        Wrapper function for getting recommendations with error handling.
        
        Args:
            title (str): Book title
            author (str, optional): Book author
            max_pages (int, optional): Maximum pages
            language (str, optional): Language filter
            
        Returns:
            pandas.DataFrame: Recommended books
        """
        try:
            recommendations = self.recommend_books(
                title,
                input_author=author,
                max_pages=max_pages,
                language=language
            )
            
            if recommendations.empty:
                print("No recommendations found with the given criteria.")
            return recommendations
            
        except BookRecommenderError as e:
            print(f"Error: {str(e)}")
            return pd.DataFrame(columns=['title', 'author', 'genres', 'rating', 'pages', 'weighted_rating']) 