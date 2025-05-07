import os
import sys
from dotenv import load_dotenv

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.bot.telegram_bot import BookRecommenderBot

def main():
    # Load environment variables
    load_dotenv()
    
    # Get the token from environment variable
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in .env file")
        print("Please create a .env file based on .env.example")
        sys.exit(1)
        
    # Create and run the bot
    bot = BookRecommenderBot(token)
    print("Starting bot...")
    bot.run()

if __name__ == "__main__":
    main() 