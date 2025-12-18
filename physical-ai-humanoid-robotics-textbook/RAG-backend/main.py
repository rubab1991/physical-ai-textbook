from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging

# Import API routes
from api.chat import router as chat_router
from api.auth import router as auth_router
from api.chapters import router as chapters_router
from api.user import router as user_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Physical AI & Humanoid Robotics Textbook RAG API",
    description="API for the Physical AI & Humanoid Robotics Textbook with RAG capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router, prefix="/api", tags=["chat"])
app.include_router(auth_router, prefix="/api", tags=["authentication"])
app.include_router(chapters_router, prefix="/api", tags=["chapters"])
app.include_router(user_router, prefix="/api", tags=["user"])

@app.get("/")
async def root():
    """
    Root endpoint for the API
    """
    return {
        "message": "Welcome to the Physical AI & Humanoid Robotics Textbook RAG API",
        "version": "1.0.0",
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "RAG API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # This allows running the app directly with python main.py
    uvicorn.run(app, host="0.0.0.0", port=8000)