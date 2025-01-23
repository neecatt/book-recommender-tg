# Book Recommender System

A machine learning-based book recommendation system that uses content-based filtering and approximate nearest neighbors to suggest books based on user preferences.

## Features

- Content-based book recommendations
- Approximate Nearest Neighbors for fast similarity search
- IMDB-style weighted rating system
- Support for filtering by pages and language
- Author diversity in recommendations
- Checkpoint system for faster loading

## Project Structure

```
book-recommender-tg/
├── src/                      # Source code directory
│   ├── data/                 # Data handling
│   ├── models/              # Model implementations
│   ├── utils/               # Utility functions
│   └── main.py
├── data/                    # Data files
├── tests/                   # Test files
├── checkpoints/             # Model checkpoints
├── notebooks/              # Jupyter notebooks
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/book-recommender-tg.git
cd book-recommender-tg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the recommender system:
```bash
python src/main.py
```

## Testing

Run the tests:
```bash
python -m pytest tests/
``` 