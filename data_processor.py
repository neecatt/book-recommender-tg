import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config import BookRecommenderConfig as cfg

class DataProcessor:
    """Handles data loading, cleaning, and feature engineering for book recommendations."""
    
    def __init__(self):
        """Initialize data processor with necessary attributes."""
        self.scaler = MinMaxScaler()
        self.books_cleaned = None
        self.book_titles = None
        self.book_authors = None
        self.title_author_pairs = None
        self.title_author_lookup = None
        self.unique_genres = None
        self.genre_to_idx = None
        
    def load_and_clean_data(self):
        """Load and preprocess the books dataset, including data cleaning and feature engineering."""
        books_dataset = pd.read_csv(cfg.DATASET_PATH)
        
        books_dataset['genres'].fillna('[]', inplace=True)
        books_dataset['pages'] = pd.to_numeric(books_dataset['pages'], errors='coerce')
        books_dataset['pages'].fillna(books_dataset['pages'].median(), inplace=True)
        books_dataset['genres'] = books_dataset['genres'].apply(
            lambda x: eval(x) if isinstance(x, str) else []
        )
        
        selected_columns = ['title', 'author', 'genres', 'rating', 'numRatings', 'pages', 'language']
        self.books_cleaned = books_dataset[selected_columns]
        
        # Calculate weighted rating using IMDB formula
        C = self.books_cleaned['numRatings'].mean()
        m = self.books_cleaned['numRatings'].quantile(0.75)
        self.books_cleaned['weighted_rating'] = (
            (self.books_cleaned['numRatings'] / (self.books_cleaned['numRatings'] + m)) * 
            self.books_cleaned['rating'] +
            (m / (self.books_cleaned['numRatings'] + m)) * C
        )
        
        self.book_titles = self.books_cleaned['title'].tolist()
        self.book_authors = self.books_cleaned['author'].tolist()
        self.title_author_pairs = list(zip(self.book_titles, self.book_authors))
        self.title_author_lookup = {
            (title.lower(), author.lower()): (title, author) 
            for title, author in self.title_author_pairs
        }
        
        self.unique_genres = list(set([
            genre for sublist in self.books_cleaned['genres'] for genre in sublist
        ]))
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.unique_genres)}
        
        numeric_features = self.books_cleaned[['rating', 'weighted_rating', 'pages']].values
        self.scaler.fit(numeric_features)
        
        return self.books_cleaned
    
    def encode_book_vectorized(self, books_batch):
        """
        Encode books into feature vectors.
        
        Args:
            books_batch: DataFrame containing batch of books to encode
            
        Returns:
            numpy array of encoded book features
        """
        genre_matrix = np.zeros((len(books_batch), len(self.unique_genres)))
        for i, genres in enumerate(books_batch['genres']):
            for genre in genres:
                if genre in self.genre_to_idx:
                    genre_matrix[i, self.genre_to_idx[genre]] = 1
        
        numeric = self.scaler.transform(
            books_batch[['rating', 'weighted_rating', 'pages']].values
        )
        
        genre_features = genre_matrix * cfg.GENRE_WEIGHT
        rating_features = numeric[:, 0].reshape(-1, 1) * cfg.RATING_WEIGHT
        weighted_rating_features = numeric[:, 1].reshape(-1, 1) * cfg.WEIGHTED_RATING_WEIGHT
        pages_features = numeric[:, 2].reshape(-1, 1) * cfg.PAGES_WEIGHT
        
        return np.hstack([
            genre_features, 
            rating_features, 
            weighted_rating_features, 
            pages_features
        ])
    
    def create_feature_matrix(self):
        """Create sparse feature matrix for all books in batches."""
        from scipy.sparse import csr_matrix, vstack
        
        sparse_feature_matrix_batches = []
        for start in range(0, len(self.books_cleaned), cfg.BATCH_SIZE):
            batch = self.books_cleaned.iloc[start:start + cfg.BATCH_SIZE]
            batch_features = self.encode_book_vectorized(batch)
            sparse_feature_matrix_batches.append(csr_matrix(batch_features))
            
        return vstack(sparse_feature_matrix_batches) 