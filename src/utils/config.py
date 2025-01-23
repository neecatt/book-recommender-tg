class BookRecommenderConfig:
    # Data paths
    DATASET_PATH = '/Users/neecat/Desktop/Projects/book-recommender-tg/src/data/books.csv'
    
    # Model parameters
    BATCH_SIZE = 5000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_NEIGHBORS = 10
    
    # Feature weights
    GENRE_WEIGHT = 0.5
    RATING_WEIGHT = 0.2
    WEIGHTED_RATING_WEIGHT = 0.2
    PAGES_WEIGHT = 0.1
    
    # Recommendation parameters
    DEFAULT_NUM_RECOMMENDATIONS = 5
    DEFAULT_MAX_BOOKS_PER_AUTHOR = 1
    
    # Scoring weights
    WEIGHTED_RATING_SCORE = 0.7
    DISTANCE_SCORE = 0.3
    
    # Fuzzy matching parameters
    FUZZY_MATCH_THRESHOLD = 70 