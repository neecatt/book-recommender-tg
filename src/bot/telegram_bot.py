import logging
import os
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from telegram import __version__ as TG_VER

try:
    from telegram import __version_info__
except ImportError:
    __version_info__ = (0, 0, 0, 0, 0)  # type: ignore[assignment]

if __version_info__ < (20, 0, 0, "alpha", 1):
    raise RuntimeError(
        f"This bot is only compatible with python-telegram-bot version 20.0a1 or higher, but your version is {TG_VER}"
    )

from telegram import Update
from src.data.data_processor import DataProcessor
from src.data.book_finder import BookFinder
from src.models.recommender import BookRecommender
from src.utils.exceptions import BookNotFoundError, AmbiguousBookError

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
TITLE, AUTHOR = range(2)

class BookRecommenderBot:
    def __init__(self, token):
        """Initialize the bot with the given token."""
        self.token = token
        
        # Initialize recommender system
        print("Initializing book recommender system...")
        self.data_processor = DataProcessor()
        self.data_processor.load_and_clean_data()
        
        self.book_finder = BookFinder(self.data_processor)
        self.recommender = BookRecommender(self.data_processor, self.book_finder)
        self.recommender.train()
        print("Book recommender system initialized!")
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start the conversation and ask for the book title."""
        await update.message.reply_text(
            "👋 Hi! I'm your Book Recommender Bot!\n\n"
            "I can help you find book recommendations based on a book you like.\n"
            "Please tell me the title of a book you enjoyed:"
        )
        return TITLE
    
    async def title(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Store the title and ask for the author."""
        context.user_data['title'] = update.message.text
        await update.message.reply_text(
            "Great! Now, please tell me the author of the book.\n"
            "If you don't know the author, just send '-'"
        )
        return AUTHOR
    
    async def recommend(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Process the author and provide recommendations."""
        title = context.user_data['title']
        author = update.message.text if update.message.text != '-' else None
        
        try:
            recommendations = self.recommender.get_recommendations(
                title=title,
                author=author,
                max_pages=None
            )
            
            if recommendations.empty:
                await update.message.reply_text(
                    "😕 Sorry, I couldn't find any recommendations for that book.\n"
                )
            else:
                response = "📚 Here are your recommendations:\n\n"
                for _, book in recommendations.iterrows():
                    response += f"{book['rank']}. {book['title']}\n"
                    response += f"   by {book['author']}\n"
                    response += f"   Rating: {book['rating']:.2f}/5.0\n"
                    response += f"   Genres: {', '.join(book['genres'])}\n"
                    response += f"   Pages: {int(book['pages'])}\n\n"
                
                await update.message.reply_text(response)
                
        except BookNotFoundError as e:
            await update.message.reply_text(
                f"😕 {str(e)}\n\nLet's try again! Please tell me another book title:"
            )
            return TITLE
            
        except AmbiguousBookError as e:
            await update.message.reply_text(
                f"❗ {str(e)}\n\nLet's try again! Please tell me another book title:"
            )
            return TITLE
            
        await update.message.reply_text(
            "Would you like more recommendations? Just tell me another book title!"            
        )
        return TITLE
    
    async def cancel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Cancel the conversation."""
        await update.message.reply_text(
            "Bye! Hope you found some interesting books to read! 👋"
        )
        return ConversationHandler.END
    
    def run(self):
        """Run the bot."""
        # Create the Application
        application = Application.builder().token(self.token).build()
        
        # Add conversation handler
        conv_handler = ConversationHandler(
            entry_points=[CommandHandler('start', self.start)],
            states={
                TITLE: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.title)],
                AUTHOR: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.recommend)],
            },
            fallbacks=[CommandHandler('cancel', self.cancel)],
        )
        
        application.add_handler(conv_handler)
        
        # Start the Bot
        application.run_polling(allowed_updates=Update.ALL_TYPES) 



        #TODO: add recognition for example recognize sapiens without full title
        #TODO: add pagination for recommendations
        #TODO: add changeable parameters
        