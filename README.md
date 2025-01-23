# Book Recommender System

A machine learning-based book recommendation system that uses content-based filtering and approximate nearest neighbors to suggest books based on user preferences. Available as both a command-line tool and a Telegram bot.

## Features

- Content-based book recommendations
- Approximate Nearest Neighbors for fast similarity search
- IMDB-style weighted rating system
- Support for filtering by pages and language
- Author diversity in recommendations
- Checkpoint system for faster loading
- Telegram bot interface
- Configurable via environment variables

## Project Structure

```
book-recommender-tg/
├── src/                      # Source code directory
│   ├── data/                 # Data handling
│   ├── models/              # Model implementations
│   ├── utils/               # Utility functions
│   ├── bot/                 # Telegram bot
│   └── main.py
├── data/                    # Data files
├── tests/                   # Test files
├── checkpoints/             # Model checkpoints
├── notebooks/              # Jupyter notebooks
├── .env                    # Environment configuration
├── .env.example            # Environment template
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

3. Set up environment:
```bash
cp .env.example .env
```
Then edit `.env` with your settings. At minimum, you need to set:
- `TELEGRAM_BOT_TOKEN`: Your bot token from [@BotFather](https://t.me/botfather)

Other settings in `.env` can be adjusted to tune the recommender system:
- Model parameters (batch size, test size, etc.)
- Feature weights
- Dataset path
- Recommendation settings

## Usage

### Command Line Interface
Run the recommender system:
```bash
python src/main.py
```

### Telegram Bot
1. Create a new bot and get the token from [@BotFather](https://t.me/botfather)

2. Add your bot token to `.env`:
```bash
TELEGRAM_BOT_TOKEN=your_bot_token_here
```

3. Run the bot:
```bash
python src/bot/run_bot.py
```

4. Start chatting with your bot:
   - Send /start to begin
   - Enter a book title when prompted
   - Enter the author name (or '-' if unknown)
   - Get personalized book recommendations!

## Configuration

All configuration is done through environment variables in the `.env` file:

### Model Parameters
- `BATCH_SIZE`: Size of batches for processing (default: 1000)
- `TEST_SIZE`: Fraction of data used for testing (default: 0.2)
- `N_NEIGHBORS`: Number of neighbors for recommendations (default: 5)
- `DEFAULT_NUM_RECOMMENDATIONS`: Default number of recommendations (default: 5)

### Feature Weights
- `GENRE_WEIGHT`: Weight for genre matching (default: 0.4)
- `RATING_WEIGHT`: Weight for user ratings (default: 0.2)
- `WEIGHTED_RATING_WEIGHT`: Weight for weighted ratings (default: 0.3)
- `PAGES_WEIGHT`: Weight for page count similarity (default: 0.1)

### Other Settings
- `FUZZY_MATCH_THRESHOLD`: Threshold for fuzzy title matching (default: 80)
- `DATASET_PATH`: Path to the books dataset

## Testing

Run the tests:
```bash
python -m pytest tests/
``` 