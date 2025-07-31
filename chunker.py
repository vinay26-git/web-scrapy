"""
Hybrid Chunking System: Recursive + Content-Aware Chunking
Advanced text segmentation optimized for different content types.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from loguru import logger
from config import config


@dataclass
class TextChunk:
    """Data class for text chunks with metadata."""
    text: str
    chunk_id: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    embedding_ready: bool = False
    
    def __len__(self):
        return len(self.text)
    
    def __str__(self):
        return f"Chunk {self.chunk_id}: {len(self.text)} chars"


class BaseChunker(ABC):
    """Abstract base class for chunkers."""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """Chunk text into segments."""
        pass


class RecursiveChunker(BaseChunker):
    """Recursive character-based chunker using LangChain."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or config.chunking.chunk_size
        self.chunk_overlap = chunk_overlap or config.chunking.chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=config.chunking.separators,
            length_function=len,
            keep_separator=True
        )
    
    def chunk(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """Chunk text using recursive character splitting."""
        if not metadata:
            metadata = {}
        
        # Get chunks from LangChain splitter
        raw_chunks = self.splitter.split_text(text)
        
        # Convert to TextChunk objects with positions
        chunks = []
        current_position = 0
        
        for i, chunk_text in enumerate(raw_chunks):
            # Find the actual position in the original text
            start_pos = text.find(chunk_text, current_position)
            if start_pos == -1:
                # Fallback if exact match not found
                start_pos = current_position
            
            end_pos = start_pos + len(chunk_text)
            
            chunk_metadata = {
                **metadata,
                'chunking_method': 'recursive',
                'original_index': i,
                'chunk_type': 'recursive'
            }
            
            chunk = TextChunk(
                text=chunk_text.strip(),
                chunk_id=i,
                start_position=start_pos,
                end_position=end_pos,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
            current_position = end_pos - self.chunk_overlap
        
        return chunks


class ContentAwareChunker(BaseChunker):
    """Content-aware chunker that respects semantic boundaries."""
    
    def __init__(self, site_type: str = "general"):
        self.site_type = site_type
        self.min_chunk_size = config.chunking.min_chunk_size
        self.max_chunk_size = config.chunking.max_chunk_size
        
    def _identify_content_boundaries(self, text: str) -> List[int]:
        """Identify natural content boundaries in text."""
        boundaries = [0]  # Start of text
        
        # Different boundary patterns for different site types
        if self.site_type == "wikipedia":
            patterns = [
                r'\n==+\s+.*?\s+==+\n',  # Wikipedia section headers
                r'\n===+\s+.*?\s+===+\n',  # Wikipedia subsection headers
                r'\n\s*\n(?=[A-Z])',  # Paragraph breaks followed by capitalized text
            ]
        elif self.site_type == "news":
            patterns = [
                r'\n\s*\n(?=\w+\s*-\s*)',  # Dateline patterns
                r'\n\s*\n(?=[A-Z][A-Z\s]+[:])',  # All-caps sections
                r'\n\s*\n(?=In\s+\w+\s+news)',  # News section breaks
                r'\n\s*\n(?=Meanwhile)',  # Story transitions
            ]
        else:  # business or general
            patterns = [
                r'\n\s*\n(?=[•\-\*]\s+)',  # Bullet points
                r'\n\s*\n(?=\d+\.\s+)',  # Numbered lists
                r'\n\s*\n(?=[A-Z][a-z]+:)',  # Section headers with colons
                r'\n\s*\n(?=Key\s+\w+|Important|Summary)',  # Key sections
            ]
        
        # Find all boundary matches
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
        
        # Add sentence boundaries for fine-grained splitting
        sentences = sent_tokenize(text)
        current_pos = 0
        for sentence in sentences:
            sent_start = text.find(sentence, current_pos)
            if sent_start != -1:
                boundaries.append(sent_start)
                current_pos = sent_start + len(sentence)
        
        # Add end of text
        boundaries.append(len(text))
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        return boundaries
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two text segments."""
        try:
            # Use TF-IDF for similarity calculation
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            
            # Fit on both texts
            docs = [text1, text2]
            tfidf_matrix = vectorizer.fit_transform(docs)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback: simple word overlap
            words1 = set(word_tokenize(text1.lower()))
            words2 = set(word_tokenize(text2.lower()))
            
            if not words1 or not words2:
                return 0.0
            
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            
            return overlap / total if total > 0 else 0.0
    
    def _merge_similar_segments(self, segments: List[str]) -> List[str]:
        """Merge semantically similar adjacent segments."""
        if len(segments) <= 1:
            return segments
        
        merged = [segments[0]]
        
        for i in range(1, len(segments)):
            current_segment = segments[i]
            last_merged = merged[-1]
            
            # Check if we should merge
            should_merge = False
            
            # Size constraints
            combined_size = len(last_merged) + len(current_segment)
            if combined_size <= self.max_chunk_size:
                # Check semantic similarity
                similarity = self._calculate_semantic_similarity(last_merged, current_segment)
                
                if similarity >= config.chunking.similarity_threshold:
                    should_merge = True
                # Also merge very short segments
                elif len(current_segment) < self.min_chunk_size:
                    should_merge = True
            
            if should_merge:
                merged[-1] = last_merged + "\n\n" + current_segment
            else:
                merged.append(current_segment)
        
        return merged
    
    def chunk(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """Chunk text using content-aware boundaries."""
        if not metadata:
            metadata = {}
        
        # Step 1: Identify content boundaries
        boundaries = self._identify_content_boundaries(text)
        
        # Step 2: Create initial segments
        segments = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment = text[start:end].strip()
            
            if segment and len(segment) >= 10:  # Filter very short segments
                segments.append(segment)
        
        # Step 3: Merge similar adjacent segments
        merged_segments = self._merge_similar_segments(segments)
        
        # Step 4: Create TextChunk objects
        chunks = []
        current_position = 0
        
        for i, segment in enumerate(merged_segments):
            # Find position in original text
            start_pos = text.find(segment, current_position)
            if start_pos == -1:
                start_pos = current_position
            
            end_pos = start_pos + len(segment)
            
            chunk_metadata = {
                **metadata,
                'chunking_method': 'content_aware',
                'original_index': i,
                'chunk_type': 'semantic',
                'site_type': self.site_type
            }
            
            chunk = TextChunk(
                text=segment,
                chunk_id=i,
                start_position=start_pos,
                end_position=end_pos,
                metadata=chunk_metadata
            )
            
            chunks.append(chunk)
            current_position = end_pos
        
        return chunks


class HybridChunker:
    """Hybrid chunker combining recursive and content-aware approaches."""
    
    def __init__(self, site_type: str = "general"):
        self.site_type = site_type
        self.recursive_chunker = RecursiveChunker()
        self.content_aware_chunker = ContentAwareChunker(site_type)
        
    def _evaluate_chunk_quality(self, chunks: List[TextChunk]) -> Dict[str, float]:
        """Evaluate the quality of a chunking approach."""
        if not chunks:
            return {'score': 0.0}
        
        scores = {
            'size_consistency': 0.0,
            'content_coherence': 0.0,
            'boundary_quality': 0.0,
            'coverage': 0.0
        }
        
        # Size consistency (prefer chunks close to target size)
        target_size = config.chunking.chunk_size
        size_deviations = [abs(len(chunk) - target_size) / target_size for chunk in chunks]
        scores['size_consistency'] = 1.0 - (sum(size_deviations) / len(size_deviations))
        scores['size_consistency'] = max(0.0, scores['size_consistency'])
        
        # Content coherence (chunks should be semantically coherent)
        coherence_scores = []
        for chunk in chunks:
            # Simple coherence measure: sentence boundary respect
            sentences = sent_tokenize(chunk.text)
            if len(sentences) <= 1:
                coherence_scores.append(1.0)
            else:
                # Check if chunk ends with complete sentences
                last_char = chunk.text.rstrip()[-1] if chunk.text.rstrip() else ''
                if last_char in '.!?':
                    coherence_scores.append(1.0)
                else:
                    coherence_scores.append(0.7)
        
        scores['content_coherence'] = sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        
        # Boundary quality (prefer natural breaks)
        boundary_scores = []
        for chunk in chunks:
            text = chunk.text.strip()
            
            # Check for natural start/end
            natural_start = text[0].isupper() if text else False
            natural_end = text[-1] in '.!?' if text else False
            
            boundary_score = (natural_start + natural_end) / 2
            boundary_scores.append(boundary_score)
        
        scores['boundary_quality'] = sum(boundary_scores) / len(boundary_scores) if boundary_scores else 0.0
        
        # Coverage (chunks should cover the text well)
        total_chunk_chars = sum(len(chunk.text) for chunk in chunks)
        original_chars = sum(chunk.end_position - chunk.start_position for chunk in chunks)
        scores['coverage'] = min(1.0, total_chunk_chars / original_chars) if original_chars > 0 else 0.0
        
        # Calculate overall score
        weights = {
            'size_consistency': 0.3,
            'content_coherence': 0.3,
            'boundary_quality': 0.2,
            'coverage': 0.2
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores)
        scores['score'] = overall_score
        
        return scores
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-process chunks to improve quality."""
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            text = chunk.text.strip()
            
            # Skip very short chunks (merge with adjacent if possible)
            if len(text) < config.chunking.min_chunk_size:
                if processed_chunks:
                    # Merge with previous chunk
                    prev_chunk = processed_chunks[-1]
                    combined_text = prev_chunk.text + "\n\n" + text
                    
                    if len(combined_text) <= config.chunking.max_chunk_size:
                        # Update previous chunk
                        prev_chunk.text = combined_text
                        prev_chunk.end_position = chunk.end_position
                        prev_chunk.metadata['merged_chunks'] = prev_chunk.metadata.get('merged_chunks', 0) + 1
                        continue
                elif i < len(chunks) - 1:
                    # Merge with next chunk
                    next_chunk = chunks[i + 1]
                    combined_text = text + "\n\n" + next_chunk.text
                    
                    if len(combined_text) <= config.chunking.max_chunk_size:
                        next_chunk.text = combined_text
                        next_chunk.start_position = chunk.start_position
                        next_chunk.metadata['merged_chunks'] = next_chunk.metadata.get('merged_chunks', 0) + 1
                        continue
            
            # Split very large chunks
            if len(text) > config.chunking.max_chunk_size:
                # Use recursive splitter on this chunk
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.chunking.chunk_size,
                    chunk_overlap=config.chunking.chunk_overlap // 2,
                    separators=config.chunking.separators
                )
                
                sub_texts = sub_splitter.split_text(text)
                
                for j, sub_text in enumerate(sub_texts):
                    sub_chunk = TextChunk(
                        text=sub_text.strip(),
                        chunk_id=len(processed_chunks),
                        start_position=chunk.start_position,  # Approximate
                        end_position=chunk.end_position,      # Approximate
                        metadata={
                            **chunk.metadata,
                            'split_from_large': True,
                            'sub_chunk_index': j
                        }
                    )
                    processed_chunks.append(sub_chunk)
            else:
                # Update chunk ID and add to processed
                chunk.chunk_id = len(processed_chunks)
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def chunk(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        Perform hybrid chunking using both recursive and content-aware methods.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include in chunks
            
        Returns:
            List of optimized TextChunk objects
        """
        if not text or len(text.strip()) < config.chunking.min_chunk_size:
            return []
        
        logger.info(f"Starting hybrid chunking for {len(text)} characters of {self.site_type} content")
        
        # Method 1: Recursive chunking
        recursive_chunks = self.recursive_chunker.chunk(text, metadata)
        recursive_score = self._evaluate_chunk_quality(recursive_chunks)
        
        # Method 2: Content-aware chunking
        content_aware_chunks = self.content_aware_chunker.chunk(text, metadata)
        content_aware_score = self._evaluate_chunk_quality(content_aware_chunks)
        
        logger.info(f"Recursive chunking: {len(recursive_chunks)} chunks, score: {recursive_score['score']:.3f}")
        logger.info(f"Content-aware chunking: {len(content_aware_chunks)} chunks, score: {content_aware_score['score']:.3f}")
        
        # Choose the better approach or combine them
        if recursive_score['score'] > content_aware_score['score'] + 0.1:
            chosen_chunks = recursive_chunks
            method = "recursive"
        elif content_aware_score['score'] > recursive_score['score'] + 0.1:
            chosen_chunks = content_aware_chunks
            method = "content_aware"
        else:
            # Hybrid approach: use content-aware for structure, recursive for size
            if len(content_aware_chunks) > 0 and len(recursive_chunks) > 0:
                # Use content-aware chunks but apply recursive splitting to large chunks
                hybrid_chunks = []
                
                for chunk in content_aware_chunks:
                    if len(chunk.text) > config.chunking.max_chunk_size:
                        # Split large chunks recursively
                        sub_chunks = self.recursive_chunker.chunk(chunk.text, chunk.metadata)
                        for sub_chunk in sub_chunks:
                            sub_chunk.metadata['hybrid_method'] = 'content_aware_recursive'
                        hybrid_chunks.extend(sub_chunks)
                    else:
                        chunk.metadata['hybrid_method'] = 'content_aware_only'
                        hybrid_chunks.append(chunk)
                
                chosen_chunks = hybrid_chunks
                method = "hybrid"
            else:
                chosen_chunks = recursive_chunks if recursive_chunks else content_aware_chunks
                method = "fallback"
        
        # Post-process chunks
        final_chunks = self._post_process_chunks(chosen_chunks)
        
        # Update metadata
        for chunk in final_chunks:
            chunk.metadata.update({
                'final_chunking_method': method,
                'total_chunks': len(final_chunks),
                'site_type': self.site_type
            })
        
        logger.success(f"Hybrid chunking completed: {len(final_chunks)} final chunks using {method} method")
        
        return final_chunks


# Example usage and testing
if __name__ == "__main__":
    # Test with different content types
    sample_texts = {
        "business": """
        Our Company Overview
        
        We are a leading technology company specializing in artificial intelligence solutions.
        Our mission is to democratize AI and make it accessible to businesses of all sizes.
        
        Our Services
        
        Machine Learning Consulting: We help businesses implement ML solutions.
        Data Analytics: Transform your data into actionable insights.
        AI Development: Custom AI applications for your specific needs.
        
        Why Choose Us
        
        • 10+ years of experience
        • Expert team of data scientists
        • Proven track record
        • 24/7 support
        """,
        
        "wikipedia": """
        Artificial Intelligence
        
        Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to 
        the natural intelligence displayed by humans and animals. Leading AI textbooks define 
        the field as the study of "intelligent agents".
        
        == History ==
        
        The field of AI research was born at a Dartmouth College workshop in 1956. 
        Attendees Allen Newell, Herbert Simon, John McCarthy, Marvin Minsky and Arthur Samuel 
        became the founders and leaders of AI research.
        
        === Early developments ===
        
        In the 1950s and 1960s, researchers developed programs that could solve algebra 
        word problems, prove geometric theorems, and learn to speak English.
        """,
        
        "news": """
        Breaking Tech News
        
        SAN FRANCISCO - A major breakthrough in artificial intelligence was announced today 
        by researchers at Stanford University.
        
        The new model, called GPT-Next, represents a significant advancement in natural 
        language processing capabilities.
        
        Meanwhile, industry experts are calling for increased regulation of AI systems 
        as they become more powerful and widespread.
        
        "This is a pivotal moment for AI development," said Dr. Jane Smith, a leading 
        AI researcher at MIT.
        """
    }
    
    for site_type, text in sample_texts.items():
        print(f"\n=== Testing {site_type.upper()} content ===")
        
        chunker = HybridChunker(site_type=site_type)
        chunks = chunker.chunk(text, metadata={'source': 'test'})
        
        print(f"Generated {len(chunks)} chunks:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1} ({len(chunk.text)} chars):")
            print(f"Method: {chunk.metadata.get('final_chunking_method', 'unknown')}")
            print(f"Preview: {chunk.text[:100]}...")