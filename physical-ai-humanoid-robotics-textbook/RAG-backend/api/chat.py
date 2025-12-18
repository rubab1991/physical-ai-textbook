from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from ..config.qdrant_config import get_qdrant_client, get_collection_name
from ..services.chunking_service import get_chunker_service
import uuid
import asyncio
from openai import AsyncOpenAI

router = APIRouter()

# Initialize services
qdrant_client = get_qdrant_client()
collection_name = get_collection_name()
chunker_service = get_chunker_service(qdrant_client, collection_name)

# Initialize OpenAI client (in production, use your actual API key)
openai_client = AsyncOpenAI(api_key="sk-your-openai-api-key-here")

# Request models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources: list = []
    confidence: float = 1.0

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint that processes user messages and returns relevant responses
    based on the textbook content using RAG (Retrieval Augmented Generation)
    """
    # Log the incoming request
    logging.info(f"Received chat request: {request.message[:50]}...")

    # Generate a session ID if none provided
    session_id = request.session_id or str(uuid.uuid4())

    try:
        # Search for relevant content in the vector database
        search_results = chunker_service.search_chunks(request.message, limit=5)

        if not search_results:
            # If no relevant content found, return a default response
            return ChatResponse(
                response="I couldn't find specific information about that topic in the textbook. Please try rephrasing your question or ask about concepts related to Physical AI, ROS, simulation, AI integration, or Vision-Language-Action systems.",
                session_id=session_id,
                sources=[],
                confidence=0.5
            )

        # Format the retrieved content for context
        context_parts = []
        sources = []

        for result in search_results:
            context_parts.append(result["content"])

            # Add source information
            sources.append({
                "chapter_id": result["chapter_id"],
                "relevance_score": result["relevance_score"],
                "content_preview": result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
            })

        # Create context for the LLM
        context = "\n\n".join(context_parts)

        # Prepare the prompt for the language model
        system_prompt = """You are an AI assistant for the Physical AI & Humanoid Robotics textbook.
        Answer questions based on the provided textbook content.
        Be accurate, concise, and cite the relevant chapters when possible.
        If the information is not in the provided context, say so clearly."""

        user_prompt = f"""
        Context from textbook:
        {context}

        User question: {request.message}

        Please provide a helpful answer based on the context above:
        """

        # Call the language model
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # In production, you might want to use gpt-4 or another model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # Lower temperature for more factual responses
            max_tokens=500
        )

        # Extract the response
        llm_response = response.choices[0].message.content

        # Calculate confidence based on the number of sources and their relevance scores
        avg_relevance = sum(src["relevance_score"] for src in sources) / len(sources) if sources else 0.0
        confidence = min(avg_relevance, 1.0)  # Cap at 1.0

        return ChatResponse(
            response=llm_response,
            session_id=session_id,
            sources=sources,
            confidence=confidence
        )

    except Exception as e:
        logging.error(f"Error processing chat request: {str(e)}")

        # Return a safe fallback response
        return ChatResponse(
            response="I encountered an issue processing your request. Please try again or ask about a different topic related to the textbook content.",
            session_id=session_id,
            sources=[],
            confidence=0.0
        )

# Additional endpoints can be added here as needed