"""
Hybrid text chunking system combining recursive and content-aware approaches.
Implements enterprise-grade chunking with semantic awareness and optimization.
"""
import re
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from loguru import logger

from .preprocessor import ProcessedContent
from .config import Config


@dataclass
class TextChunk:
    """Container for text chunks with metadata."""
    text: str
    start_index: int
    end_index: int
    chunk_id: str
    source_url: str
    chunk_method: str
    token_count: int
    sentence_count: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, **kwargs) -> List[TextChunk]:
        """Chunk text into smaller pieces."""
        pass


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive text chunker that splits text hierarchically using separators.
    Based on LangChain's RecursiveCharacterTextSplitter but enhanced.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.separators = config.chunking.separators
        self.chunk_size = config.chunking.chunk_size
        self.chunk_overlap = config.chunking.chunk_overlap
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception:
            self.tokenizer = None
            logger.warning("Failed to load tiktoken, using character-based counting")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: approximate token count
            return len(text.split()) * 1.3  # Rough approximation
    
    def _split_text_with_separator(self, text: str, separator: str) -> List[str]:
        """Split text using a specific separator."""
        if separator == "":
            # Character-level splitting
            return list(text)
        else:
            return text.split(separator)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits back into chunks of appropriate size."""
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if not split.strip():
                continue
                
            # Calculate potential new chunk size
            if current_chunk:
                potential_chunk = current_chunk + separator + split
            else:
                potential_chunk = split
            
            token_count = self._count_tokens(potential_chunk)
            
            if token_count <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # Add current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with current split
                if self._count_tokens(split) <= self.chunk_size:
                    current_chunk = split
                else:
                    # Split is too large, needs further splitting
                    chunks.append(split)
                    current_chunk = ""
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue
            
            # Get overlap from previous chunk
            prev_chunk = chunks[i - 1]
            overlap_tokens = self.chunk_overlap
            
            # Find appropriate overlap boundary (prefer sentence boundaries)
            sentences = prev_chunk.split('. ')
            overlap_text = ""
            current_tokens = 0
            
            for j in range(len(sentences) - 1, -1, -1):
                sentence = sentences[j] + '. ' if j < len(sentences) - 1 else sentences[j]
                sentence_tokens = self._count_tokens(sentence)
                
                if current_tokens + sentence_tokens <= overlap_tokens:
                    overlap_text = sentence + overlap_text
                    current_tokens += sentence_tokens
                else:
                    break
            
            # Add overlap to current chunk
            if overlap_text.strip():
                overlapped_chunk = overlap_text.strip() + " " + chunk
            else:
                overlapped_chunk = chunk
            
            overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks
    
    def chunk_text(self, text: str, source_url: str = "", **kwargs) -> List[TextChunk]:
        """
        Recursively chunk text using hierarchical separators.
        
        Args:
            text: Text to chunk
            source_url: Source URL for metadata
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        splits = [text]
        
        # Apply separators hierarchically
        for separator in self.separators:
            new_splits = []
            
            for split in splits:
                if self._count_tokens(split) <= self.chunk_size:
                    new_splits.append(split)
                else:
                    # Further split this piece
                    sub_splits = self._split_text_with_separator(split, separator)
                    merged_splits = self._merge_splits(sub_splits, separator)
                    new_splits.extend(merged_splits)
            
            splits = new_splits
            
            # Check if all splits are small enough
            if all(self._count_tokens(split) <= self.chunk_size for split in splits):
                break
        
        # Add overlap between chunks
        final_chunks = self._add_overlap(splits)
        
        # Create TextChunk objects
        current_index = 0
        for i, chunk_text in enumerate(final_chunks):
            if not chunk_text.strip():
                continue
            
            # Calculate indices
            start_index = text.find(chunk_text[:50], current_index) if len(chunk_text) > 50 else text.find(chunk_text, current_index)
            if start_index == -1:
                start_index = current_index
            end_index = start_index + len(chunk_text)
            current_index = end_index
            
            chunk = TextChunk(
                text=chunk_text.strip(),
                start_index=start_index,
                end_index=end_index,
                chunk_id=f"recursive_{i}",
                source_url=source_url,
                chunk_method="recursive",
                token_count=self._count_tokens(chunk_text),
                sentence_count=len([s for s in chunk_text.split('.') if s.strip()]),
                metadata={
                    "separator_used": "hierarchical",
                    "chunk_index": i,
                    "overlap_applied": self.chunk_overlap > 0
                }
            )
            chunks.append(chunk)
        
        return chunks


class ContentAwareChunker(ChunkingStrategy):
    """
    Content-aware chunker that uses semantic similarity and content structure.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.similarity_threshold = config.chunking.sentence_similarity_threshold
        
        # Initialize sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded sentence transformer model for semantic chunking")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Initialize TF-IDF vectorizer as fallback
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return len(text.split()) * 1.3
    
    def _detect_section_boundaries(self, sentences: List[str]) -> List[int]:
        """Detect natural section boundaries in text."""
        boundaries = [0]  # Always start with first sentence
        
        for i in range(1, len(sentences)):
            sentence = sentences[i].strip()
            
            # Detect headers (short sentences with title case)
            if (len(sentence.split()) <= 8 and 
                sentence.istitle() and 
                not sentence.endswith('.')):
                boundaries.append(i)
                continue
            
            # Detect topic transitions (using keywords)
            transition_keywords = [
                'however', 'moreover', 'furthermore', 'in addition',
                'on the other hand', 'in contrast', 'meanwhile',
                'subsequently', 'consequently', 'therefore'
            ]
            
            if any(sentence.lower().startswith(keyword) for keyword in transition_keywords):
                boundaries.append(i)
                continue
            
            # Detect paragraph breaks (sentences that don't connect well)
            if i > 0:
                prev_sentence = sentences[i-1].strip()
                if (len(prev_sentence.split()) <= 3 or
                    (sentence[0].isupper() and not prev_sentence.endswith('.'))):
                    boundaries.append(i)
        
        return boundaries
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text segments."""
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            except Exception as e:
                logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to TF-IDF similarity
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception:
            # Final fallback: word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union)
    
    def _group_similar_sentences(self, sentences: List[str]) -> List[List[int]]:
        """Group sentences by semantic similarity."""
        if len(sentences) <= 1:
            return [[0]] if sentences else []
        
        groups = []
        current_group = [0]
        
        for i in range(1, len(sentences)):
            # Calculate similarity with sentences in current group
            group_texts = [sentences[j] for j in current_group]
            current_text = sentences[i]
            
            # Compare with last few sentences in current group
            recent_texts = group_texts[-3:] if len(group_texts) >= 3 else group_texts
            similarities = [
                self._calculate_semantic_similarity(current_text, text)
                for text in recent_texts
            ]
            
            avg_similarity = np.mean(similarities)
            
            if avg_similarity >= self.similarity_threshold:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        
        # Add last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _merge_small_groups(self, groups: List[List[int]], sentences: List[str]) -> List[List[int]]:
        """Merge groups that are too small to form meaningful chunks."""
        min_chunk_size = self.config.chunking.chunk_size // 4  # Quarter of target size
        merged_groups = []
        
        i = 0
        while i < len(groups):
            current_group = groups[i]
            current_text = " ".join([sentences[j] for j in current_group])
            
            # If current group is too small, try to merge with next group
            if (self._count_tokens(current_text) < min_chunk_size and 
                i + 1 < len(groups)):
                
                next_group = groups[i + 1]
                merged_text = current_text + " " + " ".join([sentences[j] for j in next_group])
                
                # Only merge if combined size is reasonable
                if self._count_tokens(merged_text) <= self.config.chunking.chunk_size:
                    merged_groups.append(current_group + next_group)
                    i += 2  # Skip next group as it's merged
                else:
                    merged_groups.append(current_group)
                    i += 1
            else:
                merged_groups.append(current_group)
                i += 1
        
        return merged_groups
    
    def chunk_text(self, text: str, source_url: str = "", **kwargs) -> List[TextChunk]:
        """
        Chunk text using content-aware semantic grouping.
        
        Args:
            text: Text to chunk
            source_url: Source URL for metadata
            
        Returns:
            List of TextChunk objects
        """
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not sentences:
            return []
        
        # Detect natural section boundaries
        section_boundaries = self._detect_section_boundaries(sentences)
        
        # Group similar sentences
        similarity_groups = self._group_similar_sentences(sentences)
        
        # Merge small groups
        merged_groups = self._merge_small_groups(similarity_groups, sentences)
        
        # Create chunks
        chunks = []
        for i, group in enumerate(merged_groups):
            chunk_text = " ".join([sentences[j] for j in group])
            
            if not chunk_text.strip():
                continue
            
            # Calculate indices
            start_sentence_idx = min(group)
            end_sentence_idx = max(group)
            
            chunk = TextChunk(
                text=chunk_text.strip(),
                start_index=start_sentence_idx,
                end_index=end_sentence_idx,
                chunk_id=f"content_aware_{i}",
                source_url=source_url,
                chunk_method="content_aware",
                token_count=self._count_tokens(chunk_text),
                sentence_count=len(group),
                metadata={
                    "sentence_indices": group,
                    "semantic_grouping": True,
                    "avg_similarity": self.similarity_threshold
                }
            )
            chunks.append(chunk)
        
        return chunks


class HybridChunker:
    """
    Hybrid chunker that combines recursive and content-aware strategies.
    Provides enterprise-grade chunking with optimal balance of size and semantics.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.recursive_chunker = RecursiveChunker(config)
        self.content_aware_chunker = ContentAwareChunker(config)
        
        logger.info("Initialized hybrid chunker with recursive and content-aware strategies")
    
    def _evaluate_chunk_quality(self, chunk: TextChunk) -> float:
        """Evaluate the quality of a chunk based on various metrics."""
        score = 100.0
        
        # Size optimization (prefer chunks near target size)
        target_size = self.config.chunking.chunk_size
        size_ratio = chunk.token_count / target_size
        
        if size_ratio < 0.3:  # Too small
            score -= 30
        elif size_ratio > 1.5:  # Too large
            score -= 20
        elif 0.7 <= size_ratio <= 1.2:  # Optimal size
            score += 10
        
        # Sentence completeness (prefer complete sentences)
        text = chunk.text.strip()
        if text.endswith(('.', '!', '?')):
            score += 5
        else:
            score -= 10
        
        # Content coherence (prefer balanced sentence distribution)
        if chunk.sentence_count > 1:
            avg_sentence_length = len(chunk.text) / chunk.sentence_count
            if 50 <= avg_sentence_length <= 200:  # Reasonable sentence length
                score += 5
        
        return max(0, min(100, score))
    
    def _select_best_chunks(self, recursive_chunks: List[TextChunk], 
                           content_aware_chunks: List[TextChunk]) -> List[TextChunk]:
        """Select the best chunks from both strategies."""
        # Evaluate quality of both chunk sets
        recursive_scores = [self._evaluate_chunk_quality(chunk) for chunk in recursive_chunks]
        content_aware_scores = [self._evaluate_chunk_quality(chunk) for chunk in content_aware_chunks]
        
        recursive_avg = np.mean(recursive_scores) if recursive_scores else 0
        content_aware_avg = np.mean(content_aware_scores) if content_aware_scores else 0
        
        logger.info(f"Recursive chunking avg quality: {recursive_avg:.1f}")
        logger.info(f"Content-aware chunking avg quality: {content_aware_avg:.1f}")
        
        # Choose strategy based on quality and characteristics
        if content_aware_avg > recursive_avg + 10:  # Significant improvement
            logger.info("Selected content-aware chunks")
            return content_aware_chunks
        elif recursive_avg > content_aware_avg + 5:  # Recursive is better
            logger.info("Selected recursive chunks")
            return recursive_chunks
        else:
            # Hybrid approach: combine best of both
            logger.info("Using hybrid approach")
            return self._create_hybrid_chunks(recursive_chunks, content_aware_chunks)
    
    def _create_hybrid_chunks(self, recursive_chunks: List[TextChunk], 
                             content_aware_chunks: List[TextChunk]) -> List[TextChunk]:
        """Create hybrid chunks by combining the best aspects of both approaches."""
        # For now, prefer content-aware chunks but fall back to recursive for edge cases
        if not content_aware_chunks:
            return recursive_chunks
        if not recursive_chunks:
            return content_aware_chunks
        
        # Use content-aware as primary, but ensure no chunk is too large
        hybrid_chunks = []
        max_size = self.config.chunking.chunk_size * 1.5
        
        for chunk in content_aware_chunks:
            if chunk.token_count <= max_size:
                hybrid_chunks.append(chunk)
            else:
                # Split large content-aware chunks using recursive strategy
                sub_chunks = self.recursive_chunker.chunk_text(chunk.text, chunk.source_url)
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.chunk_id = f"{chunk.chunk_id}_sub_{i}"
                    sub_chunk.chunk_method = "hybrid"
                    hybrid_chunks.append(sub_chunk)
        
        return hybrid_chunks
    
    def chunk_content(self, processed_content: ProcessedContent) -> List[TextChunk]:
        """
        Chunk processed content using hybrid approach.
        
        Args:
            processed_content: ProcessedContent object to chunk
            
        Returns:
            List of TextChunk objects
        """
        text = processed_content.cleaned_text
        source_url = processed_content.original_content.url
        
        logger.info(f"Chunking content from {source_url} using hybrid approach")
        
        try:
            # Apply both chunking strategies
            recursive_chunks = self.recursive_chunker.chunk_text(text, source_url)
            content_aware_chunks = self.content_aware_chunker.chunk_text(text, source_url)
            
            # Select the best chunks
            final_chunks = self._select_best_chunks(recursive_chunks, content_aware_chunks)
            
            # Add metadata
            for chunk in final_chunks:
                chunk.metadata.update({
                    'original_title': processed_content.original_content.title,
                    'website_type': processed_content.original_content.website_type.value,
                    'language': processed_content.language,
                    'quality_score': processed_content.quality_metrics.get('overall_quality_score', 0)
                })
            
            logger.success(f"Created {len(final_chunks)} chunks from {source_url}")
            return final_chunks
            
        except Exception as e:
            logger.error(f"Failed to chunk content from {source_url}: {e}")
            return []
    
    def chunk_batch(self, processed_contents: List[ProcessedContent]) -> List[TextChunk]:
        """
        Chunk multiple processed contents in batch.
        
        Args:
            processed_contents: List of ProcessedContent objects
            
        Returns:
            List of all TextChunk objects
        """
        all_chunks = []
        
        for i, content in enumerate(processed_contents):
            logger.info(f"Chunking content {i+1}/{len(processed_contents)}")
            
            chunks = self.chunk_content(content)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} total chunks from {len(processed_contents)} contents")
        return all_chunks