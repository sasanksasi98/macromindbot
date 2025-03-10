from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import attrs
from typing import List, Optional, Dict, Any
import groq
import os
import logging
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from telegram import Update
from telegram.constants import ParseMode
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, 
    CommandHandler, 
    MessageHandler, 
    filters, 
    ContextTypes,
    CallbackQueryHandler
)
from contextlib import asynccontextmanager
import aiohttp

# Load environment variables and setup
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Configure logging
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "macromind.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add these constants at the top of the file after imports
BASE_URL = "http://localhost:8000"  # Development
# BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")  # Production

# Initialize FastAPI and Groq client
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    try:
        # Initialize HTTP client
        app.state.http_client = aiohttp.ClientSession()
        
        # Initialize Telegram bot
        bot_app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Delete webhook to ensure clean polling
        await bot_app.bot.delete_webhook(drop_pending_updates=True)
        
        # Set commands description
        await bot_app.bot.set_my_commands([
            ('start', 'Start the bot'),
            ('analyze', 'Analyze nutrition of a meal'),
            ('recipe', 'Get recipe suggestions - Add ingredients after command'),
            ('help', 'Show help guide')
        ])
        
        # Add handlers with non-blocking configuration
        bot_app.add_handler(CommandHandler("start", start, block=False))
        bot_app.add_handler(CommandHandler("help", help_command, block=False))
        bot_app.add_handler(CommandHandler("analyze", analyze_meal, block=False))
        bot_app.add_handler(CommandHandler(
            "recipe", 
            recipe_command,
            filters=filters.ChatType.PRIVATE,
            block=False
        ))
        bot_app.add_handler(CallbackQueryHandler(button_callback, block=False))
        bot_app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            analyze_meal,
            block=False
        ))
        
        # Initialize bot
        await bot_app.initialize()
        
        # Store bot instance and task
        app.state.bot = bot_app
        app.state.bot_running = True
        app.state.bot_task = asyncio.create_task(
            bot_app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
                read_timeout=30,
                write_timeout=30,
                connect_timeout=30,
                pool_timeout=30
            )
        )
        
        logger.info("Bot services started successfully")
        yield
        
    finally:
        logger.info("Starting bot services cleanup...")
        
        # Stop polling first
        if hasattr(app.state, 'bot_running'):
            app.state.bot_running = False
        
        # Cancel polling task
        if hasattr(app.state, 'bot_task'):
            logger.debug("Cancelling polling task...")
            app.state.bot_task.cancel()
            try:
                await app.state.bot_task
            except asyncio.CancelledError:
                logger.info("Polling task cancelled successfully")
            except Exception as e:
                logger.error(f"Error cancelling polling task: {e}")
        
        # Stop bot
        if hasattr(app.state, 'bot'):
            logger.debug("Stopping bot application...")
            try:
                await app.state.bot.updater.stop()
                await app.state.bot.stop()
                await app.state.bot.shutdown()
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
        
        # Close HTTP client
        if hasattr(app.state, 'http_client'):
            logger.debug("Closing HTTP client...")
            await app.state.http_client.close()
        
        logger.info("Bot services cleanup completed")

app = FastAPI(
    title="MacroMind API",
    lifespan=lifespan
)
groq_client = groq.Client(api_key=GROQ_API_KEY)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Replace Pydantic models with attrs models
@attrs.define
class MealRequest:
    meal_description: str

@attrs.define
class NutritionInfo:
    calories: int
    protein: str
    carbs: str
    fats: str

@attrs.define
class QuickMeal:
    name: str
    cooking_time: int
    difficulty: str
    ingredients_needed: List[str]
    instructions: List[str]
    nutrition_info: NutritionInfo

@attrs.define
class MealPrepIdea:
    name: str
    servings: int
    storage_time: int
    ingredients_needed: List[str]
    instructions: List[str]

@attrs.define
class RecipeSuggestions:
    quick_meals: List[QuickMeal]
    meal_prep_ideas: List[MealPrepIdea]

@attrs.define
class RecipeRequest:
    ingredients: List[str]

@attrs.define
class AnalysisResponse:
    analysis: Dict
    metadata: Dict

@attrs.define
class RecipeResponse:
    suggestions: RecipeSuggestions
    metadata: Dict

# Update the start command to include command description
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message with inline buttons"""
    keyboard = [
        [
            InlineKeyboardButton("🔍 Analyze Meal", callback_data="analyze"),
            InlineKeyboardButton("👩‍🍳 Find Recipes", callback_data="recipe")
        ],
        [InlineKeyboardButton("❓ Help", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"👋 *Welcome to MacroMind!*\n\n"
        "*Choose an option:*\n"
        "🔍 `/analyze` - Analyze nutrition of a meal\n"
        "👩‍🍳 `/recipe` - Get recipe suggestions\n"
        "❓ `/help` - Show help guide",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show help message"""
    await update.message.reply_text(
        "*🤖 MacroMind Help Guide*\n\n"
        "*1. Meal Analysis*\n"
        "• Use `/analyze [meal]`\n"
        "• Example: `/analyze 2 eggs with toast`\n\n"
        "*2. Recipe Suggestions*\n"
        "• Use `/recipe [ingredients]`\n"
        "• Example: `/recipe chicken, rice, tomatoes`\n\n"
        "*Tips:*\n"
        "• Be specific with portions\n"
        "• Include cooking methods\n"
        "• List all ingredients",
        parse_mode=ParseMode.MARKDOWN
    )

# Update the analyze_meal function
async def analyze_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle meal analysis"""
    try:
        meal_description = ' '.join(context.args) if context.args else update.message.text
        if meal_description.startswith('/analyze'):
            meal_description = meal_description.replace('/analyze', '').strip()
        
        if not meal_description:
            await update.message.reply_text(
                "Please describe your meal.\n"
                "Example: `/analyze 2 eggs with toast`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        processing_msg = await update.message.reply_text(
            "🔍 Analyzing your meal... Please wait.",
            parse_mode=ParseMode.MARKDOWN
        )

        # Use full URL for API endpoint
        async with app.state.http_client.post(
            f"{BASE_URL}/api/v1/analyze-meal",
            json={"meal_description": meal_description},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                
                # Format the response
                analysis = data.get('analysis', {})
                macros = analysis.get('macronutrients', {})
                insights = analysis.get('health_insights', {})
                
                reply = [
                    "*🍽️ Nutrition Analysis*",
                    f"*Calories:* {analysis.get('basic_info', {}).get('total_calories', 'N/A')} kcal\n",
                    "*📊 Macronutrients*",
                    f"• Protein: {macros.get('protein', {}).get('grams', 'N/A')}g",
                    f"• Carbs: {macros.get('carbohydrates', {}).get('grams', 'N/A')}g",
                    f"• Fats: {macros.get('fats', {}).get('total_grams', 'N/A')}g\n"
                ]

                if insights.get('benefits'):
                    reply.extend([
                        "*💪 Benefits*",
                        *[f"• {benefit}" for benefit in insights['benefits'][:3]]
                    ])

                await processing_msg.edit_text(
                    '\n'.join(reply),
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                error_text = await response.text()
                logger.error(f"API Error: {response.status} - {error_text}")
                await processing_msg.edit_text(
                    "❌ Error analyzing meal. Please try again.",
                    parse_mode=ParseMode.MARKDOWN
                )

    except Exception as e:
        logger.error(f"Error in analyze_meal: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "❌ Service unavailable. Please try again later.",
            parse_mode=ParseMode.MARKDOWN
        )

# Update the recipe_command handler
async def recipe_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle recipe suggestions"""
    try:
        # Check if command was sent without arguments
        if not context.args:
            await update.message.reply_text(
                "*👩‍🍳 Recipe Search*\n\n"
                "Please add your ingredients after the /recipe command, separated by commas.\n"
                "Example: `/recipe chicken, rice, tomatoes`",
                parse_mode=ParseMode.MARKDOWN
            )
            return
            
        ingredients_text = ' '.join(context.args)
        
        if not ingredients_text:
            await update.message.reply_text(
                "Please list your ingredients, separated by commas.\n"
                "Example: `/recipe chicken, rice, tomatoes`",
                parse_mode=ParseMode.MARKDOWN
            )
            return

        processing_msg = await update.message.reply_text(
            "👩‍🍳 Finding recipes... Please wait.",
            parse_mode=ParseMode.MARKDOWN
        )

        ingredients = [i.strip() for i in ingredients_text.split(',') if i.strip()]
        
        # Use full URL for API endpoint
        async with app.state.http_client.post(
            f"{BASE_URL}/api/v1/suggest-recipes",
            json={"ingredients": ingredients},
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                data = await response.json()
                suggestions = data['suggestions']
                
                # Format recipe response
                reply = await format_recipe_response(suggestions)
                
                await processing_msg.edit_text(
                    '\n'.join(reply),
                    parse_mode=ParseMode.MARKDOWN
                )
            else:
                await processing_msg.edit_text(
                    "❌ Error finding recipes. Please try again.",
                    parse_mode=ParseMode.MARKDOWN
                )

    except Exception as e:
        logger.error(f"Error in recipe_command: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "❌ Service unavailable. Please try again later.",
            parse_mode=ParseMode.MARKDOWN
        )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses"""
    query = update.callback_query
    await query.answer()  # Remove loading state

    if query.data == "analyze":
        await query.message.reply_text(
            "*🔍 Meal Analysis*\n\n"
            "Please describe your meal.\n"
            "Example: `2 eggs with toast`\n\n"
            "_Just type your meal description or use the /analyze command_",
            parse_mode=ParseMode.MARKDOWN
        )
    elif query.data == "recipe":
        await query.message.reply_text(
            "*👩‍🍳 Recipe Search*\n\n"
            "Please list your ingredients, separated by commas.\n"
            "Example: `chicken, rice, tomatoes`\n\n"
            "_Use the /recipe command followed by your ingredients_",
            parse_mode=ParseMode.MARKDOWN
        )
    elif query.data == "help":
        await help_command(update, context)

# FastAPI endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "bot": "running" if app.state.bot else "stopped"
        }
    }

# Update FastAPI endpoint to use cattrs for conversion
import cattrs

# Configure cattrs for FastAPI integration
converter = cattrs.Converter()

# Update the endpoint handlers with proper response models
@attrs.define
class APIResponse:
    data: Any
    metadata: Dict[str, Any]

@app.post("/api/v1/analyze-meal")
async def analyze_meal_endpoint(request: Dict[str, Any]):
    """Analyze meal nutrition"""
    try:
        meal_request = MealRequest(meal_description=request["meal_description"])
        request_id = str(uuid.uuid4())
        logger.info(f"Analyzing meal request {request_id}: {meal_request.meal_description}")

        analysis_data = await create_meal_analysis(meal_request.meal_description)
        
        response = APIResponse(
            data=analysis_data,
            metadata={
                "request_id": request_id,
                "model": "mixtral-8x7b-32768",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return converter.unstructure(response)

    except Exception as e:
        logger.error(f"Error analyzing meal: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Internal server error during analysis"
        )

@app.post("/api/v1/suggest-recipes")
async def suggest_recipes_endpoint(request: Dict[str, Any]):
    """Suggest recipes based on ingredients"""
    try:
        recipe_request = RecipeRequest(ingredients=request["ingredients"])
        request_id = str(uuid.uuid4())
        
        suggestions = await create_recipe_suggestions(recipe_request.ingredients)
        
        response = APIResponse(
            data=suggestions,
            metadata={
                "request_id": request_id,
                "ingredients": recipe_request.ingredients,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        return converter.unstructure(response)
        
    except Exception as e:
        logger.error(f"Error suggesting recipes: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def create_meal_analysis(meal_description: str):
    """Create meal analysis using Groq"""
    try:
        prompt = f"""Analyze this meal: {meal_description}
        Provide nutritional analysis in this exact JSON format:
        {{
            "basic_info": {{
                "total_calories": 300,
                "serving_size": "1 serving"
            }},
            "macronutrients": {{
                "protein": {{"grams": 12}},
                "carbohydrates": {{"grams": 30}},
                "fats": {{"total_grams": 14}}
            }},
            "health_insights": {{
                "benefits": [
                    "Good source of protein",
                    "Contains healthy fats",
                    "Provides energy"
                ],
                "considerations": [
                    "Monitor portion size",
                    "Consider whole grain options"
                ]
            }}
        }}"""

        # Run Groq completion in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                temperature=0.7,
                max_tokens=1000,
                response_format={ "type": "json_object" }
            )
        )

        # Extract and validate JSON response
        response_text = completion.choices[0].message.content
        try:
            json_response = json.loads(response_text)
            # Validate required fields
            assert "basic_info" in json_response
            assert "macronutrients" in json_response
            assert "health_insights" in json_response
            return json_response
        except (json.JSONDecodeError, AssertionError) as e:
            logger.error(f"Invalid JSON response: {response_text[:200]}...")
            raise ValueError("Invalid response format from model")

    except Exception as e:
        logger.error(f"Error in create_meal_analysis: {str(e)}", exc_info=True)
        raise

async def create_recipe_suggestions(ingredients: List[str]):
    """Create recipe suggestions using Groq"""
    try:
        prompt = f"""Given these ingredients: {', '.join(ingredients)}
        Suggest recipes in JSON format:
        {{
            "quick_meals": [{{
                "name": "string",
                "cooking_time": "number",
                "difficulty": "easy/medium/hard",
                "ingredients_needed": ["list"],
                "instructions": ["steps"],
                "nutrition_info": {{
                    "calories": "number",
                    "protein": "string",
                    "carbs": "string",
                    "fats": "string"
                }}
            }}],
            "meal_prep_ideas": [{{
                "name": "string",
                "servings": "number",
                "storage_time": "number",
                "ingredients_needed": ["list"],
                "instructions": ["steps"]
            }}]
        }}"""

        # Run Groq completion in a thread pool
        loop = asyncio.get_event_loop()
        completion = await loop.run_in_executor(
            None,
            lambda: groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
                response_format={ "type": "json_object" }
            )
        )

        # Extract and validate JSON response
        response_text = completion.choices[0].message.content
        try:
            json_response = json.loads(response_text)
            assert "quick_meals" in json_response
            return json_response
        except (json.JSONDecodeError, AssertionError) as e:
            logger.error(f"Invalid JSON response: {response_text[:200]}...")
            raise ValueError("Invalid response format from model")

    except Exception as e:
        logger.error(f"Error in create_recipe_suggestions: {str(e)}", exc_info=True)
        raise

async def format_recipe_response(suggestions: dict) -> List[str]:
    """Format recipe suggestions for Telegram"""
    reply = ["*🍳 Recipe Suggestions*\n"]
    
    if quick_meals := suggestions.get('quick_meals', []):
        for idx, meal in enumerate(quick_meals[:2], 1):
            reply.extend([
                f"*{idx}. {meal['name']}*",
                f"⏱️ Time: {meal['cooking_time']} minutes",
                f"📊 Difficulty: {meal['difficulty']}\n",
                "*Ingredients:*"
            ])
            reply.extend(f"• {ing}" for ing in meal['ingredients_needed'])
            
            reply.append("\n*Steps:*")
            reply.extend(f"{i}. {step}" for i, step in enumerate(meal['instructions'][:4], 1))
            
            if nutrition := meal.get('nutrition_info'):
                reply.extend([
                    "\n*Nutrition per Serving:*",
                    f"• Calories: {nutrition['calories']}",
                    f"• Protein: {nutrition['protein']}",
                    f"• Carbs: {nutrition['carbs']}",
                    f"• Fats: {nutrition['fats']}\n"
                ])
            reply.append("")

    return reply

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("\nShutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Kill any existing processes
    os.system("lsof -t -i:8000 | xargs kill -9 2>/dev/null")
    
    # Clear webhook
    import requests
    requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/deleteWebhook",
        json={"drop_pending_updates": True}
    )
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="debug",
        workers=1,
        timeout_keep_alive=30
    )
