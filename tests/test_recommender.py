import pytest
import pandas as pd
from src.models.recommender import BookRecommender
from src.data.data_processor import DataProcessor
from src.data.book_finder import BookFinder
from src.utils.exceptions import BookNotFoundError, InvalidParameterError, BookRecommenderError
from src.utils.config import BookRecommenderConfig as cfg
import numpy as np
from latest_test import (
    recommend_books_ann,
    find_book_name,
    books_cleaned,
    evaluate_recommendations,
    AmbiguousBookError,
)

class TestBookRecommender(pytest.mark.usefixtures("setup_test_cases")):
    def setUp(self):
        """Set up test cases"""
        self.valid_book = {
            'title': 'The Hunger Games',
            'author': 'Suzanne Collins'
        }
        self.invalid_book = {
            'title': 'NonexistentBook123',
            'author': 'NonexistentAuthor123'
        }
        # Find a title that actually exists multiple times in our dataset
        title_counts = books_cleaned['title'].value_counts()
        self.ambiguous_title = title_counts[title_counts > 1].index[0]

    def test_find_book_name(self):
        """Test book finding functionality"""
        # Test valid book with author
        result = find_book_name(self.valid_book['title'], self.valid_book['author'])
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], self.valid_book['title'])
        self.assertEqual(result[1], self.valid_book['author'])

        # Test invalid book
        with self.assertRaises(BookNotFoundError):
            find_book_name(self.invalid_book['title'], self.invalid_book['author'])

        # Test book without author (should raise AmbiguousBookError for a title that exists multiple times)
        print(f"Testing ambiguous title: {self.ambiguous_title}")  # Debug print
        with self.assertRaises(AmbiguousBookError):
            find_book_name(self.ambiguous_title)

        # Test case insensitive matching
        result = find_book_name(self.valid_book['title'].lower(), self.valid_book['author'].upper())
        self.assertIsNotNone(result)

        # Test empty title
        with self.assertRaises(InvalidParameterError):
            find_book_name("")

        # Test invalid input types
        with self.assertRaises(InvalidParameterError):
            find_book_name(123)
        with self.assertRaises(InvalidParameterError):
            find_book_name("title", 123)

    def test_recommend_books_ann(self):
        """Test recommendation functionality"""
        # Test valid recommendation
        recommendations = recommend_books_ann(
            self.valid_book['title'],
            self.valid_book['author'],
            max_pages=400,
            num_recommendations=5
        )
        self.assertIsInstance(recommendations, pd.DataFrame)
        self.assertFalse(recommendations.empty)
        self.assertEqual(len(recommendations), 5)
        
        # Check required columns
        required_columns = ['title', 'author', 'genres', 'rating', 'pages', 'weighted_rating']
        for col in required_columns:
            self.assertIn(col, recommendations.columns)

        # Test invalid book
        with self.assertRaises(BookNotFoundError):
            recommend_books_ann(
                self.invalid_book['title'],
                self.invalid_book['author']
            )

        # Test invalid parameters
        with self.assertRaises(InvalidParameterError):
            recommend_books_ann(
                self.valid_book['title'],
                self.valid_book['author'],
                max_pages=-1
            )
        with self.assertRaises(InvalidParameterError):
            recommend_books_ann(
                self.valid_book['title'],
                self.valid_book['author'],
                num_recommendations=0
            )

        # Test page limit
        max_pages = 200
        recommendations = recommend_books_ann(
            self.valid_book['title'],
            self.valid_book['author'],
            max_pages=max_pages
        )
        self.assertTrue(all(recommendations['pages'] <= max_pages))

        # Test language filter
        language = 'eng'
        recommendations = recommend_books_ann(
            self.valid_book['title'],
            self.valid_book['author'],
            language=language
        )
        if not recommendations.empty:
            self.assertTrue(all(recommendations['language'] == language))

        # Test author diversity
        max_books_per_author = 1
        recommendations = recommend_books_ann(
            self.valid_book['title'],
            self.valid_book['author'],
            max_books_per_author=max_books_per_author
        )
        if not recommendations.empty:
            author_counts = recommendations['author'].value_counts()
            self.assertTrue(all(count <= max_books_per_author for count in author_counts))

    def test_input_book_exclusion(self):
        """Test that input book is not in recommendations"""
        recommendations = recommend_books_ann(
            self.valid_book['title'],
            self.valid_book['author']
        )
        if not recommendations.empty:
            self.assertFalse(
                any((recommendations['title'] == self.valid_book['title']) & 
                    (recommendations['author'] == self.valid_book['author']))
            )

    def test_evaluation_metrics(self):
        """Test evaluation functionality"""
        # Test invalid parameters
        with self.assertRaises(InvalidParameterError):
            evaluate_recommendations(num_samples=0)
        with self.assertRaises(InvalidParameterError):
            evaluate_recommendations(num_samples=-1)
        with self.assertRaises(InvalidParameterError):
            evaluate_recommendations(max_pages=-1)
        with self.assertRaises(InvalidParameterError):
            evaluate_recommendations(max_pages=0)
            
        # Test with valid parameters
        metrics = evaluate_recommendations(num_samples=1, max_pages=400)
        
        # Check if all metrics exist and are valid
        required_metrics = ['genre_precision', 'author_diversity', 'page_compliance']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], float)
            self.assertTrue(0 <= metrics[metric] <= 1)
        
        # Test with too many samples
        with self.assertRaises(InvalidParameterError):
            evaluate_recommendations(num_samples=len(books_cleaned) + 1)

if __name__ == '__main__':
    pytest.main() 