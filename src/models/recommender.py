import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from annoy import AnnoyIndex
from scipy.sparse import csr_matrix
from collections import defaultdict
from src.utils.exceptions import BookNotFoundError, InvalidParameterError, BookRecommenderError
from src.utils.config import BookRecommenderConfig as cfg

class BookRecommender:
    def __init__(self, data_processor, book_finder):
        self.data_processor = data_processor
        self.book_finder = book_finder
        self.model = None
        self.train_books = None
        self.test_books = None
        self.feature_matrix = None
        self.n_features = None
        
        # Set up checkpoint directory
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self):
        """Save the trained model, feature matrices and book splits to checkpoint file."""
        checkpoint = {
            'train_books': self.train_books,
            'test_books': self.test_books,
            'feature_matrix': self.feature_matrix,
            'n_features': self.n_features
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, 'recommender_data.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        if self.model is not None:
            annoy_path = os.path.join(self.checkpoint_dir, 'annoy_index.ann')
            self.model.save(annoy_path)
            
    def load_checkpoint(self):
        """Load trained model and feature matrices from checkpoint file if it exists."""
        try:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'recommender_data.pkl')
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                
            self.train_books = checkpoint['train_books']
            self.test_books = checkpoint['test_books']
            self.feature_matrix = checkpoint['feature_matrix']
            self.n_features = checkpoint['n_features']
            
            annoy_path = os.path.join(self.checkpoint_dir, 'annoy_index.ann')
            if os.path.exists(annoy_path):
                self.model = AnnoyIndex(self.n_features, 'angular')
                self.model.load(annoy_path)
                print("Loaded trained model and features from checkpoint")
                return True
            return False
        except (FileNotFoundError, EOFError, KeyError):
            return False
        
    def train(self):
        """Train the recommendation model with checkpointing using Approximate Nearest Neighbors."""
        if self.load_checkpoint():
            return
            
        print("Training model from scratch...")
        self.feature_matrix = self.data_processor.create_feature_matrix()
        
        train_indices, test_indices = train_test_split(
            np.arange(len(self.data_processor.books_cleaned)), 
            test_size=cfg.TEST_SIZE, 
            random_state=cfg.RANDOM_STATE
        )
        
        train_matrix = self.feature_matrix[train_indices]
        self.train_books = self.data_processor.books_cleaned.iloc[train_indices]
        self.test_books = self.data_processor.books_cleaned.iloc[test_indices]
        
        train_matrix_dense = train_matrix.toarray()
        self.n_features = train_matrix_dense.shape[1]
        
        self.model = AnnoyIndex(self.n_features, 'angular')
        for i in range(len(train_matrix_dense)):
            self.model.add_item(i, train_matrix_dense[i])
            
        self.model.build(50)
        
        self.save_checkpoint()
        print("Saved trained model and features to checkpoint")
        
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
            pandas.DataFrame: Recommended books with rank numbers
        """
        num_recommendations = num_recommendations or cfg.DEFAULT_NUM_RECOMMENDATIONS
        max_books_per_author = max_books_per_author or cfg.DEFAULT_MAX_BOOKS_PER_AUTHOR
        
        if not isinstance(num_recommendations, int) or num_recommendations <= 0:
            raise InvalidParameterError("num_recommendations must be a positive integer")
        
        if not isinstance(max_books_per_author, int) or max_books_per_author <= 0:
            raise InvalidParameterError("max_books_per_author must be a positive integer")
            
        if max_pages is not None and (not isinstance(max_pages, (int, float)) or max_pages <= 0):
            raise InvalidParameterError("max_pages must be a positive number")
            
        try:
            book_match = self.book_finder.find_book_name(input_title, input_author)
            if not book_match:
                return pd.DataFrame(columns=['rank', 'title', 'author', 'genres', 'rating', 'pages', 'weighted_rating'])
            
            matched_title, matched_author = book_match
            input_book = self.data_processor.books_cleaned[
                (self.data_processor.books_cleaned['title'] == matched_title) & 
                (self.data_processor.books_cleaned['author'] == matched_author)
            ]
            
            if input_book.empty:
                raise BookNotFoundError(f"Book '{matched_title}' by {matched_author} not found in database")
                
            input_book = input_book.iloc[0]
            input_vector = self.data_processor.encode_book_vectorized(pd.DataFrame([input_book])).flatten()
            
            n_neighbors = min(num_recommendations * 10, len(self.train_books))
            indices, distances = self.model.get_nns_by_vector(
                input_vector, 
                n_neighbors,
                include_distances=True
            )
            
            recommendations = self.train_books.iloc[indices].copy()
            recommendations['similarity'] = 1 - np.array(distances)
            
            recommendations = recommendations[
                ~((recommendations['title'] == matched_title) & 
                  (recommendations['author'] == matched_author))
            ]
            
            if max_pages is not None:
                recommendations = recommendations[recommendations['pages'] <= max_pages]
            if language is not None:
                recommendations = recommendations[recommendations['language'] == language]
            
            if len(recommendations) == 0:
                return pd.DataFrame(columns=['rank', 'title', 'author', 'genres', 'rating', 'pages', 'weighted_rating'])
            
            max_weighted_rating = recommendations['weighted_rating'].max()
            min_weighted_rating = recommendations['weighted_rating'].min()
            recommendations['weighted_rating_norm'] = (recommendations['weighted_rating'] - min_weighted_rating) / (max_weighted_rating - min_weighted_rating)
            
            recommendations['ranking_score'] = (
                recommendations['weighted_rating_norm'] * 0.4 +
                recommendations['similarity'] * 0.4 +
                (recommendations['rating'] / 5.0) * 0.2
            )
            
            recommendations = recommendations.sort_values('ranking_score', ascending=False)
            
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
                        'similarity': book['similarity'],
                        'ranking_score': book['ranking_score']
                    })
                    author_counts[book['author']] += 1
                    
                if len(diverse_recommendations) >= num_recommendations:
                    break
            
            if not diverse_recommendations:
                return pd.DataFrame(columns=['rank', 'title', 'author', 'genres', 'rating', 'pages', 'weighted_rating'])
            
            recommendations = pd.DataFrame(diverse_recommendations)
            final_recommendations = recommendations[['title', 'author', 'genres', 'rating', 'pages', 'weighted_rating']].head(num_recommendations)
            
            # Add rank numbers starting from 1
            final_recommendations.insert(0, 'rank', range(1, len(final_recommendations) + 1))
            return final_recommendations
            
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
                    
                    test_genres = set(test_row['genres'])
                    if test_genres:
                        recommended_genres = set([genre for rec in recommendations['genres'] for genre in rec])
                        metrics['genre_precision'] += len(test_genres.intersection(recommended_genres)) / len(test_genres)
                    
                    unique_authors = len(recommendations['author'].unique())
                    metrics['author_diversity'] += unique_authors / len(recommendations)
                    
                    if max_pages:
                        metrics['page_compliance'] += (recommendations['pages'] <= max_pages).all()
                        
                except BookRecommenderError:
                    continue
            
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
            pandas.DataFrame: Recommended books with rank numbers
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
            else:
                print("\nTop Recommendations:")
                for _, book in recommendations.iterrows():
                    print(f"\n{book['rank']}. Title: {book['title']}")
                    print(f"   Author: {book['author']}")
                    print(f"   Genres: {', '.join(book['genres'])}")
                    print(f"   User Rating: {book['rating']:.2f}/5.0")
                    print(f"   Weighted Rating: {book['weighted_rating']:.2f}")
                    print(f"   Pages: {book['pages']}")
                    print("   " + "-" * 50)
            return recommendations
            
        except BookRecommenderError as e:
            print(f"Error: {str(e)}")
            return pd.DataFrame(columns=['rank', 'title', 'author', 'genres', 'rating', 'pages', 'weighted_rating']) 