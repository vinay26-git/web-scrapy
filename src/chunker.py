"""
Advanced text chunking module with hybrid approach.
Combines recursive chunking with content-aware techniques.
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

from .config import ChunkingConfig
from .utils import extract_sentences, extract_paragraphs, clean_text, is_meaningful_text


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    start_index: int
    end_index: int
    chunk_id: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class HybridChunker:
    """
    Advanced text chunker that combines recursive chunking with content-aware techniques.
    """
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )
        
        # Content-aware parameters
        self.min_chunk_size = config.min_chunk_size
        self.max_chunk_size = config.max_chunk_size
        self.preserve_paragraphs = config.preserve_paragraphs
        self.preserve_sentences = config.preserve_sentences
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Main method to chunk text using hybrid approach.
        
        Args:
            text: Input text to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not is_meaningful_text(text):
            return []
        
        # Clean the text
        text = clean_text(text)
        
        # Step 1: Initial recursive chunking
        initial_chunks = self._recursive_chunk(text)
        
        # Step 2: Content-aware refinement
        refined_chunks = self._content_aware_refinement(initial_chunks)
        
        # Step 3: Final optimization
        final_chunks = self._optimize_chunks(refined_chunks)
        
        # Step 4: Create Chunk objects with metadata
        chunks = []
        for i, chunk_text in enumerate(final_chunks):
            chunk = Chunk(
                text=chunk_text,
                start_index=text.find(chunk_text),
                end_index=text.find(chunk_text) + len(chunk_text),
                chunk_id=f"chunk_{i}_{hash(chunk_text) % 10000}",
                metadata=metadata or {}
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Perform initial recursive chunking.
        
        Args:
            text: Input text
            
        Returns:
            List of initial chunks
        """
        try:
            # Use LangChain's recursive splitter
            documents = [Document(page_content=text)]
            split_docs = self.recursive_splitter.split_documents(documents)
            
            chunks = [doc.page_content for doc in split_docs]
            
            # Filter out empty or very short chunks
            chunks = [chunk for chunk in chunks if len(chunk.strip()) >= self.min_chunk_size]
            
            logger.debug(f"Recursive chunking created {len(chunks)} initial chunks")
            return chunks
            
        except Exception as e:
            logger.warning(f"Recursive chunking failed, falling back to simple chunking: {e}")
            return self._simple_chunk(text)
    
    def _simple_chunk(self, text: str) -> List[str]:
        """
        Simple chunking fallback method.
        
        Args:
            text: Input text
            
        Returns:
            List of chunks
        """
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = extract_paragraphs(text)
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= self.config.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _content_aware_refinement(self, chunks: List[str]) -> List[str]:
        """
        Apply content-aware refinement to chunks.
        
        Args:
            chunks: Initial chunks from recursive splitting
            
        Returns:
            Refined chunks
        """
        refined_chunks = []
        
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_size:
                # Chunk is already good size
                refined_chunks.append(chunk)
            else:
                # Need to split further using content-aware approach
                sub_chunks = self._content_aware_split(chunk)
                refined_chunks.extend(sub_chunks)
        
        return refined_chunks
    
    def _content_aware_split(self, chunk: str) -> List[str]:
        """
        Split a chunk using content-aware techniques.
        
        Args:
            chunk: Input chunk to split
            
        Returns:
            List of sub-chunks
        """
        if len(chunk) <= self.max_chunk_size:
            return [chunk]
        
        # Try to split by semantic boundaries
        sub_chunks = self._split_by_semantic_boundaries(chunk)
        
        # If semantic splitting didn't work well, try structural splitting
        if not sub_chunks or any(len(c) > self.max_chunk_size for c in sub_chunks):
            sub_chunks = self._split_by_structure(chunk)
        
        return sub_chunks
    
    def _split_by_semantic_boundaries(self, chunk: str) -> List[str]:
        """
        Split chunk by semantic boundaries (paragraphs, sentences).
        
        Args:
            chunk: Input chunk
            
        Returns:
            List of sub-chunks
        """
        sub_chunks = []
        
        if self.preserve_paragraphs:
            # Split by paragraphs first
            paragraphs = extract_paragraphs(chunk)
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) <= self.max_chunk_size:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        sub_chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
            
            if current_chunk:
                sub_chunks.append(current_chunk.strip())
        
        elif self.preserve_sentences:
            # Split by sentences
            sentences = extract_sentences(chunk)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        sub_chunks.append(current_chunk.strip())
                    current_chunk = sentence + " "
            
            if current_chunk:
                sub_chunks.append(current_chunk.strip())
        
        return sub_chunks
    
    def _split_by_structure(self, chunk: str) -> List[str]:
        """
        Split chunk by structural elements.
        
        Args:
            chunk: Input chunk
            
        Returns:
            List of sub-chunks
        """
        # Find natural break points
        break_points = []
        
        # Look for section headers, bullet points, etc.
        section_patterns = [
            r'\n\s*[A-Z][A-Z\s]+\n',  # ALL CAPS headers
            r'\n\s*\d+\.\s+',         # Numbered lists
            r'\n\s*[-*]\s+',          # Bullet points
            r'\n\s*[A-Z][^.!?]*:\n',  # Headers ending with colon
        ]
        
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, chunk))
            for match in matches:
                break_points.append(match.start())
        
        # Sort break points
        break_points.sort()
        
        # Split at break points
        sub_chunks = []
        start = 0
        
        for break_point in break_points:
            if break_point - start >= self.min_chunk_size:
                sub_chunk = chunk[start:break_point].strip()
                if sub_chunk:
                    sub_chunks.append(sub_chunk)
                start = break_point
        
        # Add remaining content
        if start < len(chunk):
            remaining = chunk[start:].strip()
            if remaining and len(remaining) >= self.min_chunk_size:
                sub_chunks.append(remaining)
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _optimize_chunks(self, chunks: List[str]) -> List[str]:
        """
        Optimize chunks for better quality and consistency.
        
        Args:
            chunks: Input chunks
            
        Returns:
            Optimized chunks
        """
        optimized_chunks = []
        
        for chunk in chunks:
            # Clean the chunk
            cleaned_chunk = clean_text(chunk)
            
            # Skip if too short or not meaningful
            if not is_meaningful_text(cleaned_chunk, self.min_chunk_size):
                continue
            
            # Ensure chunk doesn't exceed max size
            if len(cleaned_chunk) > self.max_chunk_size:
                # Split the oversized chunk
                sub_chunks = self._split_oversized_chunk(cleaned_chunk)
                optimized_chunks.extend(sub_chunks)
            else:
                optimized_chunks.append(cleaned_chunk)
        
        return optimized_chunks
    
    def _split_oversized_chunk(self, chunk: str) -> List[str]:
        """
        Split a chunk that exceeds the maximum size.
        
        Args:
            chunk: Oversized chunk
            
        Returns:
            List of smaller chunks
        """
        # Try to split at sentence boundaries
        sentences = extract_sentences(chunk)
        sub_chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks
    
    def merge_similar_chunks(self, chunks: List[Chunk], similarity_threshold: float = 0.8) -> List[Chunk]:
        """
        Merge chunks that are semantically similar.
        
        Args:
            chunks: List of chunks
            similarity_threshold: Threshold for merging
            
        Returns:
            List of merged chunks
        """
        if len(chunks) <= 1:
            return chunks
        
        # Extract text from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Calculate TF-IDF vectors
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(tfidf_matrix)
            
            # Find chunks to merge
            merged_indices = set()
            merged_chunks = []
            
            for i in range(len(chunks)):
                if i in merged_indices:
                    continue
                
                current_chunk = chunks[i]
                similar_chunks = [current_chunk]
                
                for j in range(i + 1, len(chunks)):
                    if j in merged_indices:
                        continue
                    
                    if similarities[i, j] > similarity_threshold:
                        similar_chunks.append(chunks[j])
                        merged_indices.add(j)
                
                if len(similar_chunks) > 1:
                    # Merge similar chunks
                    merged_text = "\n\n".join([c.text for c in similar_chunks])
                    merged_chunk = Chunk(
                        text=merged_text,
                        start_index=min(c.start_index for c in similar_chunks),
                        end_index=max(c.end_index for c in similar_chunks),
                        chunk_id=f"merged_{len(merged_chunks)}",
                        metadata=current_chunk.metadata
                    )
                    merged_chunks.append(merged_chunk)
                else:
                    merged_chunks.append(current_chunk)
            
            logger.info(f"Merged {len(chunks)} chunks into {len(merged_chunks)} chunks")
            return merged_chunks
            
        except Exception as e:
            logger.warning(f"Failed to merge chunks: {e}")
            return chunks
    
    def get_chunk_statistics(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """
        Get statistics about the chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'min_chunk_size': 0,
                'max_chunk_size': 0,
                'total_text_length': 0
            }
        
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        total_length = sum(chunk_sizes)
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': np.mean(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_text_length': total_length,
            'std_chunk_size': np.std(chunk_sizes)
        }