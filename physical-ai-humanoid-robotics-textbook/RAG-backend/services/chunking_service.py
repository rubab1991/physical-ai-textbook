"""
Document chunking service for the Physical AI & Humanoid Robotics textbook
"""
import os
import re
from typing import List, Dict, Any
from pathlib import Path
import hashlib
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import markdown
from bs4 import BeautifulSoup
import tiktoken

logger = logging.getLogger(__name__)

class DocumentChunker:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name

        # Initialize sentence tokenizer
        self.enc = tiktoken.get_encoding("cl100k_base")  # Good for code and text

        # Load embedding model (using a lightweight model for demonstration)
        # In production, you might want to use a different model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        except:
            # Fallback to a simple approach if model loading fails
            logger.warning("Could not load embedding model, using simple approach")
            self.tokenizer = None
            self.model = None

    def chunk_document(self, content: str, chapter_id: str, module_id: str,
                      max_chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Chunk document content into smaller pieces for vector storage
        """
        # Clean and prepare content
        clean_content = self.clean_markdown(content)

        # Split content into chunks
        chunks = self.split_into_chunks(clean_content, max_chunk_size, overlap)

        # Create chunk documents
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_doc = {
                "id": f"{chapter_id}_chunk_{i}",
                "chapter_id": chapter_id,
                "module_id": module_id,
                "content": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            chunk_docs.append(chunk_doc)

        return chunk_docs

    def clean_markdown(self, content: str) -> str:
        """
        Clean markdown content by removing code blocks and converting to plain text
        """
        # Remove code blocks
        content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        content = re.sub(r'`[^`]*`', '', content)  # Inline code

        # Convert markdown to plain text
        try:
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            clean_text = soup.get_text(separator=' ')
        except:
            # If markdown conversion fails, use the original content
            clean_text = content

        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        return clean_text

    def split_into_chunks(self, text: str, max_chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks based on token count
        """
        # Split text into sentences
        sentences = re.split(r'[.!?]+\s+', text)

        chunks = []
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            # Estimate token count
            sentence_size = len(self.enc.encode(sentence))

            if current_size + sentence_size > max_chunk_size and current_chunk:
                # Add current chunk to results
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                if overlap > 0:
                    # Add overlap by taking sentences from the end of current chunk
                    overlap_sentences = self.get_overlap_sentences(current_chunk, overlap)
                    current_chunk = overlap_sentences + " " + sentence
                    current_size = len(self.enc.encode(current_chunk))
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def get_overlap_sentences(self, text: str, token_count: int) -> str:
        """
        Get the last few sentences from text that fit within the token count
        """
        sentences = re.split(r'[.!?]+\s+', text)
        overlap_text = ""

        for sentence in reversed(sentences):
            test_text = sentence + " " + overlap_text
            if len(self.enc.encode(test_text)) <= token_count:
                overlap_text = test_text
            else:
                break

        return overlap_text.strip()

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for the text chunks
        """
        if self.model is None:
            # Fallback: return simple numeric representations
            embeddings = []
            for text in texts:
                # Create a simple hash-based embedding as fallback
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                embedding = [(hash_val >> i) % 256 / 128.0 - 1.0 for i in range(384)]
                embeddings.append(embedding)
            return embeddings

        # Use transformer model for embeddings
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling to get sentence embedding
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

            embeddings.append(embedding)

        return embeddings

    def store_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Store chunks in Qdrant vector database
        """
        if not chunks:
            return

        # Extract content for embedding generation
        texts = [chunk["content"] for chunk in chunks]

        # Generate embeddings
        embeddings = self.generate_embeddings(texts)

        # Prepare points for insertion
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=chunk["id"],
                vector=embedding,
                payload={
                    "chapter_id": chunk["chapter_id"],
                    "module_id": chunk["module_id"],
                    "content": chunk["content"],
                    "chunk_index": chunk["chunk_index"],
                    "total_chunks": chunk["total_chunks"],
                    "hash": hashlib.md5(chunk["content"].encode()).hexdigest()
                }
            )
            points.append(point)

        # Upload to Qdrant
        try:
            self.qdrant_client.upload_points(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully uploaded {len(points)} chunks to Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.error(f"Failed to upload chunks to Qdrant: {e}")
            raise

    def process_textbook_content(self, textbook_directory: str):
        """
        Process all textbook content and store in vector database
        """
        content_dir = Path(textbook_directory) / "docs" / "chapters"

        if not content_dir.exists():
            logger.error(f"Textbook content directory does not exist: {content_dir}")
            return

        # Process each chapter file
        for chapter_file in content_dir.glob("*.md"):
            logger.info(f"Processing chapter: {chapter_file.name}")

            try:
                # Read chapter content
                with open(chapter_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract chapter ID from filename
                chapter_id = chapter_file.stem  # Remove .md extension

                # Determine module based on chapter name
                module_id = self.determine_module_id(chapter_id)

                # Chunk the document
                chunks = self.chunk_document(content, chapter_id, module_id)

                # Store in Qdrant
                self.store_chunks(chunks)

                logger.info(f"Successfully processed {len(chunks)} chunks for chapter {chapter_id}")

            except Exception as e:
                logger.error(f"Failed to process chapter {chapter_file.name}: {e}")
                continue

    def determine_module_id(self, chapter_id: str) -> str:
        """
        Determine module ID based on chapter ID
        """
        # Simple heuristic based on chapter names
        if "introduction" in chapter_id.lower():
            return "module-0"
        elif "ros2" in chapter_id.lower():
            return "module-1"
        elif "simulation" in chapter_id.lower() or "digital" in chapter_id.lower() or "physics" in chapter_id.lower():
            return "module-2"
        elif "isaac" in chapter_id.lower():
            return "module-3"
        elif "vla" in chapter_id.lower():
            return "module-4"
        elif "conclusion" in chapter_id.lower():
            return "module-5"
        else:
            return "module-unknown"

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks based on query
        """
        if self.model is None:
            # Fallback search implementation
            logger.warning("Embedding model not available, returning empty results")
            return []

        try:
            # Generate embedding for query
            query_inputs = self.tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512)

            with torch.no_grad():
                query_outputs = self.model(**query_inputs)
                query_embedding = query_outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

            # Search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )

            # Format results
            results = []
            for hit in search_results:
                results.append({
                    "id": hit.id,
                    "content": hit.payload.get("content", ""),
                    "chapter_id": hit.payload.get("chapter_id", ""),
                    "module_id": hit.payload.get("module_id", ""),
                    "relevance_score": hit.score,
                    "chunk_index": hit.payload.get("chunk_index", 0)
                })

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []


# Global chunker instance (will be initialized with actual Qdrant client)
chunker_service = None

def get_chunker_service(qdrant_client: QdrantClient, collection_name: str) -> DocumentChunker:
    """
    Get or create the chunker service instance
    """
    global chunker_service
    if chunker_service is None:
        chunker_service = DocumentChunker(qdrant_client, collection_name)
    return chunker_service