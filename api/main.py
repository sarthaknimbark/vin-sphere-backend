import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.config import API_TITLE, API_VERSION
from api.services.model_router import load_all_predict_modules
from api.services.valuation_service import load_valuation_assets
from api.routers.decode import router as decode_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan context manager to handle model and dataset loading on startup."""
    logger.info("Initializing models and loading datasets...")
    try:
        load_all_predict_modules()
        load_valuation_assets()
        logger.info("API preloading completed successfully!")
    except Exception as e:
        logger.error(f"Failed to preload assets during API startup: {e}", exc_info=True)
    yield
    logger.info("Shutting down API...")

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="Scalable HTTP API for standard 17-char VIN decoding and short 9/10/11/12/13-char chassis prediction & valuation.",
    lifespan=lifespan
)

# CORS Middleware Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(decode_router)

@app.get("/")
def read_root():
    return {
        "app": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
def health():
    return {"status": "ok"}
