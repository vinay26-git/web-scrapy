"""
Advanced Hybrid Chunking System
Combines Recursive Chunking with Content-Aware Techniques
Author: Full-Stack Developer (Google, Amazon, Microsoft experience)
"""

import re
import nltk
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import spacy
from loguru import logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model for better NLP processing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


@dataclass
class Chunk:
    """Data class representing a text chunk with metadata."""
    text: str
    chunk_id: str
    start_index: int
    end_index: int
    chunk_type: str
    metadata: Dict
    semantic_score: float = 0.0


class HybridChunker:
    """
    Advanced hybrid chunking system that combines:
    1. Recursive chunking based on semantic boundaries
    2. Content-aware chunking using NLP techniques
    3. Domain-specific optimizations for business, news, and Wikipedia content
    """
    
    def __init__(self, 
                 max_chunk_size: int = 1000,
                 min_chunk_size: int = 100,
                 overlap_size: int = 100,
                 semantic_threshold: float = 0.7):
        """
        Initialize the hybrid chunker.
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            overlap_size: Overlap between chunks in tokens
            semantic_threshold: Threshold for semantic similarity
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.semantic_threshold = semantic_threshold
        
        # Initialize NLP components
        self._setup_nlp()
        
        # Domain-specific chunking rules
        self.domain_rules = {
            'business': {
                'section_headers': ['about', 'services', 'products', 'contact', 'team'],
                'content_priority': ['main', 'article', 'section'],
                'min_section_size': 200
            },
            'news': {
                'section_headers': ['headline', 'lead', 'body', 'conclusion'],
                'content_priority': ['article', 'story', 'content'],
                'min_section_size': 300
            },
            'wikipedia': {
                'section_headers': ['overview', 'history', 'background', 'references'],
                'content_priority': ['section', 'paragraph', 'content'],
                'min_section_size': 250
            }
        }
        
        logger.info("HybridChunker initialized successfully")
    
    def _setup_nlp(self):
        """Setup NLP components for content-aware chunking."""
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        # Custom stop words for web content
        web_stop_words = {
            'click', 'read', 'more', 'share', 'like', 'follow', 'subscribe',
            'newsletter', 'cookie', 'privacy', 'terms', 'conditions'
        }
        self.stop_words.update(web_stop_words)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return nltk.word_tokenize(text.lower())
    
    def _extract_sections(self, text: str, domain: str = 'generic') -> List[Dict]:
        """
        Extract logical sections from text using domain-specific rules.
        
        Args:
            text: Input text
            domain: Domain type (business, news, wikipedia)
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        # Split by common section markers
        section_patterns = [
            r'\n\s*\n',  # Double newlines
            r'\n\s*[A-Z][^.!?]*\n',  # Capitalized lines
            r'\n\s*[0-9]+\.\s*',  # Numbered sections
            r'\n\s*[A-Z][a-z]+\s*:',  # Section headers
        ]
        
        # Domain-specific patterns
        if domain in self.domain_rules:
            domain_headers = self.domain_rules[domain]['section_headers']
            for header in domain_headers:
                pattern = rf'\n\s*{header}[^.!?]*\n'
                section_patterns.append(pattern)
        
        # Split text into sections
        current_pos = 0
        for pattern in section_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                if match.start() > current_pos:
                    section_text = text[current_pos:match.start()].strip()
                    if len(section_text) > 50:  # Minimum section size
                        sections.append({
                            'text': section_text,
                            'start': current_pos,
                            'end': match.start(),
                            'type': 'content'
                        })
                current_pos = match.end()
        
        # Add remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:].strip()
            if len(remaining_text) > 50:
                sections.append({
                    'text': remaining_text,
                    'start': current_pos,
                    'end': len(text),
                    'type': 'content'
                })
        
        return sections
    
    def _calculate_semantic_coherence(self, text: str) -> float:
        """
        Calculate semantic coherence score for a text segment.
        
        Args:
            text: Text segment
            
        Returns:
            Coherence score between 0 and 1
        """
        if not nlp or len(text) < 50:
            return 0.5
        
        try:
            doc = nlp(text)
            
            # Calculate average sentence similarity
            sentences = list(doc.sents)
            if len(sentences) < 2:
                return 0.5
            
            similarities = []
            for i in range(len(sentences) - 1):
                sim = sentences[i].similarity(sentences[i + 1])
                similarities.append(sim)
            
            return sum(similarities) / len(similarities) if similarities else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating semantic coherence: {e}")
            return 0.5
    
    def _find_optimal_boundaries(self, text: str, target_size: int) -> List[int]:
        """
        Find optimal chunk boundaries using semantic analysis.
        
        Args:
            text: Input text
            target_size: Target chunk size in characters
            
        Returns:
            List of boundary positions
        """
        boundaries = []
        current_pos = 0
        
        while current_pos < len(text):
            # Find the next potential boundary
            end_pos = min(current_pos + target_size, len(text))
            
            # Look for natural break points
            break_points = []
            
            # Sentence endings
            sentence_endings = list(re.finditer(r'[.!?]\s+', text[current_pos:end_pos]))
            if sentence_endings:
                break_points.extend([current_pos + m.end() for m in sentence_endings])
            
            # Paragraph breaks
            paragraph_breaks = list(re.finditer(r'\n\s*\n', text[current_pos:end_pos]))
            if paragraph_breaks:
                break_points.extend([current_pos + m.end() for m in paragraph_breaks])
            
            # Comma breaks (for long sentences)
            comma_breaks = list(re.finditer(r',\s+', text[current_pos:end_pos]))
            if comma_breaks:
                break_points.extend([current_pos + m.end() for m in comma_breaks])
            
            # Choose the best boundary
            if break_points:
                # Prefer boundaries closer to target size
                best_boundary = min(break_points, 
                                  key=lambda x: abs(x - (current_pos + target_size)))
                boundaries.append(best_boundary)
                current_pos = best_boundary
            else:
                # No natural break found, use target size
                boundaries.append(end_pos)
                current_pos = end_pos
        
        return boundaries
    
    def _create_chunks_with_overlap(self, text: str, boundaries: List[int]) -> List[Chunk]:
        """
        Create chunks with overlap for better context preservation.
        
        Args:
            text: Input text
            boundaries: List of boundary positions
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        for i, boundary in enumerate(boundaries):
            # Calculate start and end positions with overlap
            if i == 0:
                start_pos = 0
            else:
                start_pos = max(0, boundaries[i-1] - self.overlap_size)
            
            end_pos = boundary
            
            chunk_text = text[start_pos:end_pos].strip()
            
            if len(chunk_text) >= self.min_chunk_size:
                # Calculate semantic coherence
                coherence_score = self._calculate_semantic_coherence(chunk_text)
                
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"chunk_{i:04d}",
                    start_index=start_pos,
                    end_index=end_pos,
                    chunk_type="hybrid",
                    metadata={
                        'coherence_score': coherence_score,
                        'length': len(chunk_text),
                        'word_count': len(chunk_text.split()),
                        'overlap_start': start_pos != 0,
                        'overlap_end': i < len(boundaries) - 1
                    },
                    semantic_score=coherence_score
                )
                chunks.append(chunk)
        
        return chunks
    
    def _apply_domain_optimizations(self, chunks: List[Chunk], domain: str) -> List[Chunk]:
        """
        Apply domain-specific optimizations to chunks.
        
        Args:
            chunks: List of chunks
            domain: Domain type
            
        Returns:
            Optimized list of chunks
        """
        if domain not in self.domain_rules:
            return chunks
        
        optimized_chunks = []
        rules = self.domain_rules[domain]
        
        for chunk in chunks:
            # Check if chunk meets domain-specific requirements
            if len(chunk.text) >= rules['min_section_size']:
                # Add domain-specific metadata
                chunk.metadata['domain'] = domain
                chunk.metadata['domain_optimized'] = True
                
                # Adjust semantic score based on domain
                if domain == 'news':
                    # News chunks should have higher coherence
                    chunk.semantic_score *= 1.1
                elif domain == 'wikipedia':
                    # Wikipedia chunks should be more comprehensive
                    chunk.semantic_score *= 1.05
                
                optimized_chunks.append(chunk)
        
        return optimized_chunks
    
    def chunk_text(self, text: str, domain: str = 'generic') -> List[Chunk]:
        """
        Main chunking method using hybrid approach.
        
        Args:
            text: Input text to chunk
            domain: Domain type for optimization
            
        Returns:
            List of optimized chunks
        """
        logger.info(f"Starting hybrid chunking for {len(text)} characters, domain: {domain}")
        
        # Step 1: Extract logical sections
        sections = self._extract_sections(text, domain)
        logger.info(f"Extracted {len(sections)} logical sections")
        
        all_chunks = []
        
        for section in sections:
            section_text = section['text']
            
            # Step 2: Find optimal boundaries for this section
            boundaries = self._find_optimal_boundaries(section_text, self.max_chunk_size)
            
            # Step 3: Create chunks with overlap
            section_chunks = self._create_chunks_with_overlap(section_text, boundaries)
            
            # Step 4: Add section metadata
            for chunk in section_chunks:
                chunk.metadata['section_start'] = section['start']
                chunk.metadata['section_end'] = section['end']
                chunk.metadata['section_type'] = section['type']
            
            all_chunks.extend(section_chunks)
        
        # Step 5: Apply domain-specific optimizations
        optimized_chunks = self._apply_domain_optimizations(all_chunks, domain)
        
        # Step 6: Filter chunks based on quality
        final_chunks = []
        for chunk in optimized_chunks:
            if (chunk.semantic_score >= self.semantic_threshold and 
                len(chunk.text) >= self.min_chunk_size):
                final_chunks.append(chunk)
        
        logger.info(f"Created {len(final_chunks)} high-quality chunks")
        return final_chunks
    
    def chunk_multiple_texts(self, texts: List[str], domains: List[str] = None) -> List[List[Chunk]]:
        """
        Chunk multiple texts with their respective domains.
        
        Args:
            texts: List of texts to chunk
            domains: List of corresponding domains
            
        Returns:
            List of chunk lists
        """
        if domains is None:
            domains = ['generic'] * len(texts)
        
        results = []
        for text, domain in zip(texts, domains):
            chunks = self.chunk_text(text, domain)
            results.append(chunks)
        
        return results


class ContentAwareChunker:
    """
    Content-aware chunking using advanced NLP techniques.
    Focuses on semantic boundaries and topic coherence.
    """
    
    def __init__(self, max_chunk_size: int = 800):
        self.max_chunk_size = max_chunk_size
        self._setup_nlp()
    
    def _setup_nlp(self):
        """Setup NLP components."""
        if nlp is None:
            logger.warning("spaCy not available, using basic chunking")
    
    def chunk_by_topics(self, text: str) -> List[Chunk]:
        """
        Chunk text by topic boundaries using NLP.
        
        Args:
            text: Input text
            
        Returns:
            List of topic-based chunks
        """
        if not nlp:
            # Fallback to basic chunking
            return self._basic_chunking(text)
        
        try:
            doc = nlp(text)
            chunks = []
            current_chunk = ""
            current_start = 0
            
            for sent in doc.sents:
                sent_text = sent.text.strip()
                
                if len(current_chunk) + len(sent_text) <= self.max_chunk_size:
                    current_chunk += " " + sent_text if current_chunk else sent_text
                else:
                    # Save current chunk
                    if current_chunk:
                        chunk = Chunk(
                            text=current_chunk.strip(),
                            chunk_id=f"topic_{len(chunks):04d}",
                            start_index=current_start,
                            end_index=current_start + len(current_chunk),
                            chunk_type="topic_based",
                            metadata={'method': 'topic_boundary'},
                            semantic_score=0.8
                        )
                        chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk = sent_text
                    current_start = current_start + len(current_chunk)
            
            # Add final chunk
            if current_chunk:
                chunk = Chunk(
                    text=current_chunk.strip(),
                    chunk_id=f"topic_{len(chunks):04d}",
                    start_index=current_start,
                    end_index=current_start + len(current_chunk),
                    chunk_type="topic_based",
                    metadata={'method': 'topic_boundary'},
                    semantic_score=0.8
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in topic-based chunking: {e}")
            return self._basic_chunking(text)
    
    def _basic_chunking(self, text: str) -> List[Chunk]:
        """Basic chunking fallback."""
        chunks = []
        sentences = nltk.sent_tokenize(text)
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunk = Chunk(
                        text=current_chunk.strip(),
                        chunk_id=f"basic_{len(chunks):04d}",
                        start_index=current_start,
                        end_index=current_start + len(current_chunk),
                        chunk_type="basic",
                        metadata={'method': 'sentence_boundary'},
                        semantic_score=0.6
                    )
                    chunks.append(chunk)
                
                current_chunk = sentence
                current_start = current_start + len(current_chunk)
        
        if current_chunk:
            chunk = Chunk(
                text=current_chunk.strip(),
                chunk_id=f"basic_{len(chunks):04d}",
                start_index=current_start,
                end_index=current_start + len(current_chunk),
                chunk_type="basic",
                metadata={'method': 'sentence_boundary'},
                semantic_score=0.6
            )
            chunks.append(chunk)
        
        return chunks


if __name__ == "__main__":
    # Example usage
    sample_text = """
    This is a sample business website content. It contains multiple sections about our services.
    
    Our company provides excellent solutions for various industries. We have been in business for over 10 years.
    
    Our team consists of experienced professionals who are dedicated to delivering high-quality results.
    We work with clients from different sectors including technology, healthcare, and finance.
    
    Contact us today to learn more about how we can help your business grow and succeed in the competitive market.
    """
    
    chunker = HybridChunker(max_chunk_size=200, min_chunk_size=50)
    chunks = chunker.chunk_text(sample_text, domain='business')
    
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}:")
        print(f"Text: {chunk.text[:100]}...")
        print(f"Score: {chunk.semantic_score:.3f}")
        print(f"Length: {len(chunk.text)}")
        print("-" * 50)