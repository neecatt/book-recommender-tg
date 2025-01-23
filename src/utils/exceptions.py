class BookRecommenderError(Exception):
    """Base exception class for book recommender errors"""
    pass

class BookNotFoundError(BookRecommenderError):
    """Raised when a book cannot be found"""
    pass

class AmbiguousBookError(BookRecommenderError):
    """Raised when multiple books match the query"""
    pass

class InvalidParameterError(BookRecommenderError):
    """Raised when invalid parameters are provided"""
    pass 