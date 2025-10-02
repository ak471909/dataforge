"""
FastAPI server for Training Data Bot.

Provides REST API endpoints for document processing and dataset management.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training_data_bot import TrainingDataBot
from training_data_bot.core import get_logger
from api.dependencies import set_bot
from api.middleware.auth import AuthMiddleware
from api.middleware.rate_limit import limiter

logger = get_logger("api.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Training Data Bot API server...")
    bot_instance = TrainingDataBot()
    
    # Configure AI client if API key is available
    import os
    openai_key = os.getenv("TDB_OPENAI_API_KEY")
    if openai_key:
        bot_instance.set_ai_client(
            provider="openai",
            api_key=openai_key,
            model="gpt-3.5-turbo"
        )
        logger.info("AI client configured with OpenAI")
    else:
        logger.warning("No OpenAI API key found - AI generation disabled")
    
    set_bot(bot_instance)
    logger.info("Bot initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")
    await bot_instance.cleanup()
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Training Data Bot API",
    description="Enterprise-grade training data curation bot for LLM fine-tuning",
    version="0.1.0",
    lifespan=lifespan
)


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add authentication middleware
app.add_middleware(AuthMiddleware)

from monitoring.metrics import metrics
import time

@app.middleware("http")
async def track_metrics(request, call_next):
    """Track request metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    metrics.record_request(
        endpoint=request.url.path,
        duration=duration,
        status_code=response.status_code
    )
    
    return response


# Add rate limiting
app.state.limiter = limiter


# Import and include routers AFTER app is created
from api.routes import documents, processing, datasets, monitoring


app.include_router(documents.router, prefix="/api/v1/documents", tags=["Documents"])
app.include_router(processing.router, prefix="/api/v1/processing", tags=["Processing"])
app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["Datasets"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Training Data Bot API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from api.dependencies import get_bot
    try:
        bot = get_bot()
        bot_ready = True
    except:
        bot_ready = False
    
    return {
        "status": "healthy",
        "bot_initialized": bot_ready
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )