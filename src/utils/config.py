import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable with default."""
    value = os.getenv(key)
    return float(value) if value is not None else default

def get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable with default."""
    value = os.getenv(key)
    return int(value) if value is not None else default

class BookRecommenderConfig:
    """Configuration for the book recommender system."""
    
    # Model parameters
    BATCH_SIZE = get_env_int('BATCH_SIZE', 1000)
    TEST_SIZE = get_env_float('TEST_SIZE', 0.2)
    RANDOM_STATE = get_env_int('RANDOM_STATE', 42)
    N_NEIGHBORS = get_env_int('N_NEIGHBORS', 5)
    DEFAULT_NUM_RECOMMENDATIONS = get_env_int('DEFAULT_NUM_RECOMMENDATIONS', 5)
    DEFAULT_MAX_BOOKS_PER_AUTHOR = get_env_int('DEFAULT_MAX_BOOKS_PER_AUTHOR', 2)
    FUZZY_MATCH_THRESHOLD = get_env_int('FUZZY_MATCH_THRESHOLD', 80)
    
    # Feature weights
    GENRE_WEIGHT = get_env_float('GENRE_WEIGHT', 0.4)
    RATING_WEIGHT = get_env_float('RATING_WEIGHT', 0.2)
    WEIGHTED_RATING_WEIGHT = get_env_float('WEIGHTED_RATING_WEIGHT', 0.3)
    PAGES_WEIGHT = get_env_float('PAGES_WEIGHT', 0.1)
    
    # Data configuration
    DATASET_PATH = os.getenv('DATASET_PATH', 'data/books_1.Best_Books_Ever.csv')
    
    # Scoring weights
    WEIGHTED_RATING_SCORE = 0.7
    DISTANCE_SCORE = 0.3
    
    # Recommendation parameters
    DEFAULT_MAX_BOOKS_PER_AUTHOR = 1
    
    # Scoring weights
    WEIGHTED_RATING_SCORE = 0.7
    DISTANCE_SCORE = 0.3 