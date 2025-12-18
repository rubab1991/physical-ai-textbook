"""
Configuration for Qdrant vector database
"""
import os
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)

class QdrantConfig:
    def __init__(self):
        # Qdrant connection parameters
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "textbook_chunks")

        # Vector parameters
        self.vector_size = int(os.getenv("VECTOR_SIZE", 384))  # Default for sentence-transformers/all-MiniLM-L6-v2
        self.distance_metric = Distance.COSINE

        # Initialize client
        self.client = self.initialize_client()

    def initialize_client(self) -> QdrantClient:
        """Initialize Qdrant client with configuration"""
        try:
            if self.api_key:
                client = QdrantClient(
                    url=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    prefer_grpc=True
                )
            else:
                client = QdrantClient(
                    host=self.host,
                    port=self.port
                )

            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            return client

        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    def create_collection(self):
        """Create the textbook chunks collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)

            if not collection_exists:
                # Create collection for textbook content chunks
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=self.distance_metric
                    )
                )

                # Create payload index for chapter_id to improve filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="chapter_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                # Create payload index for module_id to improve filtering
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="module_id",
                    field_schema=models.PayloadSchemaType.KEYWORD
                )

                logger.info(f"Created collection '{self.collection_name}' with vector size {self.vector_size}")
            else:
                logger.info(f"Collection '{self.collection_name}' already exists")

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def get_client(self) -> QdrantClient:
        """Get the initialized Qdrant client"""
        return self.client

    def get_collection_name(self) -> str:
        """Get the collection name"""
        return self.collection_name

# Global configuration instance
qdrant_config = QdrantConfig()

def get_qdrant_client() -> QdrantClient:
    """Get the Qdrant client instance"""
    return qdrant_config.get_client()

def get_collection_name() -> str:
    """Get the collection name"""
    return qdrant_config.get_collection_name()

# Initialize collection on import
try:
    qdrant_config.create_collection()
except Exception as e:
    logger.error(f"Failed to initialize Qdrant collection: {e}")
    # Don't raise here as the service should still start, but collection creation failed