"""
Advanced Embedding Generation Pipeline
Optimized for different content types with batch processing and GPU acceleration.
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple, Any
from dataclasses import dataclass
import time
from pathlib import Path
import pickle

from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from tqdm import tqdm

from loguru import logger
from config import config
from chunker import TextChunk


@dataclass
class EmbeddingResult:
    """Data class for embedding results."""
    chunk_id: int
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    model_name: str
    embedding_time: float
    
    def __len__(self):
        return len(self.embedding)
    
    def __str__(self):
        return f"Embedding {self.chunk_id}: {len(self.embedding)} dimensions"


class EmbeddingGenerator:
    """Advanced embedding generator with optimization for different content types."""
    
    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.embedding.model_name
        self.device = device or config.embedding.device
        self.model = None
        self.model_loaded = False
        
        # Performance tracking
        self.embedding_cache = {}
        self.performance_stats = {
            'total_embeddings': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'batch_count': 0
        }
        
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        if self.model_loaded:
            return
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Initialize model
            self.model = SentenceTransformer(self.model_name)
            
            # Set device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to('cuda')
                logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                self.model = self.model.to('cpu')
                logger.info("Model loaded on CPU")
            
            # Set model to evaluation mode for better performance
            self.model.eval()
            
            # Configure model settings
            if hasattr(self.model, 'max_seq_length'):
                self.model.max_seq_length = config.embedding.max_seq_length
            
            self.model_loaded = True
            logger.success(f"Embedding model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def _preprocess_text(self, text: str, site_type: str = None) -> str:
        """Preprocess text for optimal embedding generation."""
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Site-specific preprocessing
        if site_type == "wikipedia":
            # For Wikipedia, keep structured content indicators
            text = text.replace("===", " ")
            text = text.replace("==", " ")
        elif site_type == "news":
            # For news, remove datelines at the beginning
            lines = text.split('\n')
            if lines and len(lines[0]) < 50 and any(city in lines[0].upper() 
                                                   for city in ['NEW YORK', 'LONDON', 'WASHINGTON', 'SAN FRANCISCO']):
                text = '\n'.join(lines[1:])
        
        # Truncate if too long for the model
        max_length = config.embedding.max_seq_length * 4  # Approximate character limit
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]  # Cut at word boundary
        
        return text
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def _batch_encode(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """Encode texts in batches for better GPU utilization."""
        if not texts:
            return np.array([])
        
        self._load_model()
        
        batch_size = batch_size or config.embedding.batch_size
        
        # Adjust batch size based on available memory
        if self.device == "cuda" and torch.cuda.is_available():
            # Get GPU memory info
            try:
                memory_free = torch.cuda.get_device_properties(0).total_memory
                memory_used = torch.cuda.memory_allocated(0)
                memory_available = memory_free - memory_used
                
                # Adjust batch size based on available memory
                memory_per_sample = 1024 * 1024  # Rough estimate: 1MB per sample
                max_batch_size = max(1, int(memory_available / memory_per_sample / 4))
                batch_size = min(batch_size, max_batch_size)
                
            except Exception:
                # Fallback to smaller batch size if memory check fails
                batch_size = min(batch_size, 16)
        
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            try:
                # Generate embeddings for batch
                with torch.no_grad():
                    batch_embeddings = self.model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=config.embedding.normalize_embeddings
                    )
                
                all_embeddings.append(batch_embeddings)
                
                # Clear cache periodically on GPU
                if self.device == "cuda" and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {str(e)}")
                # Create zero embeddings as fallback
                embedding_dim = self.model.get_sentence_embedding_dimension()
                fallback_embeddings = np.zeros((len(batch_texts), embedding_dim))
                all_embeddings.append(fallback_embeddings)
        
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])
    
    def _enhance_embeddings(self, embeddings: np.ndarray, chunks: List[TextChunk]) -> np.ndarray:
        """Apply post-processing enhancements to embeddings."""
        if len(embeddings) == 0:
            return embeddings
        
        enhanced = embeddings.copy()
        
        # Apply PCA for dimensionality reduction if requested
        # (This could be configurable for specific use cases)
        
        # Normalize embeddings if not already done
        if not config.embedding.normalize_embeddings:
            enhanced = normalize(enhanced, norm='l2', axis=1)
        
        # Content-type specific enhancements
        site_types = [chunk.metadata.get('site_type', 'general') for chunk in chunks]
        
        # Group by site type for type-specific processing
        for site_type in set(site_types):
            indices = [i for i, st in enumerate(site_types) if st == site_type]
            
            if site_type == "wikipedia":
                # For Wikipedia, slightly boost embeddings to account for structured content
                enhanced[indices] *= 1.02
            elif site_type == "news":
                # For news, no specific enhancement currently
                pass
            else:  # business or general
                # For business content, no specific enhancement currently
                pass
        
        return enhanced
    
    def generate_embeddings(self, chunks: List[TextChunk], use_cache: bool = True) -> List[EmbeddingResult]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of TextChunk objects
            use_cache: Whether to use caching for duplicate texts
            
        Returns:
            List of EmbeddingResult objects
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        start_time = time.time()
        
        # Prepare texts and track cache hits
        texts_to_encode = []
        chunk_to_text_map = {}
        cached_results = {}
        
        for i, chunk in enumerate(chunks):
            # Preprocess text
            site_type = chunk.metadata.get('site_type', 'general')
            processed_text = self._preprocess_text(chunk.text, site_type)
            
            # Check cache
            cache_key = self._get_cache_key(processed_text)
            
            if use_cache and cache_key in self.embedding_cache:
                cached_results[i] = self.embedding_cache[cache_key]
                self.performance_stats['cache_hits'] += 1
            else:
                texts_to_encode.append(processed_text)
                chunk_to_text_map[len(texts_to_encode) - 1] = i
        
        # Generate embeddings for new texts
        new_embeddings = []
        if texts_to_encode:
            new_embeddings = self._batch_encode(texts_to_encode)
            self.performance_stats['batch_count'] += 1
        
        # Apply enhancements
        if len(new_embeddings) > 0:
            relevant_chunks = [chunks[chunk_to_text_map[i]] for i in range(len(texts_to_encode))]
            new_embeddings = self._enhance_embeddings(new_embeddings, relevant_chunks)
        
        # Create results
        results = []
        new_embedding_idx = 0
        
        for i, chunk in enumerate(chunks):
            if i in cached_results:
                # Use cached result
                embedding = cached_results[i]
                embedding_time = 0.0  # Cached, so no time spent
            else:
                # Use newly generated embedding
                embedding = new_embeddings[new_embedding_idx]
                embedding_time = (time.time() - start_time) / len(texts_to_encode)
                new_embedding_idx += 1
                
                # Cache the result
                if use_cache:
                    site_type = chunk.metadata.get('site_type', 'general')
                    processed_text = self._preprocess_text(chunk.text, site_type)
                    cache_key = self._get_cache_key(processed_text)
                    self.embedding_cache[cache_key] = embedding
            
            # Create result object
            result = EmbeddingResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                metadata={
                    **chunk.metadata,
                    'embedding_model': self.model_name,
                    'embedding_dimension': len(embedding),
                    'device_used': self.device
                },
                model_name=self.model_name,
                embedding_time=embedding_time
            )
            
            results.append(result)
        
        # Update performance stats
        total_time = time.time() - start_time
        self.performance_stats['total_embeddings'] += len(chunks)
        self.performance_stats['total_time'] += total_time
        
        logger.success(f"Generated {len(results)} embeddings in {total_time:.2f}s "
                      f"({len(results)/total_time:.1f} embeddings/sec)")
        
        if cached_results:
            logger.info(f"Cache hits: {len(cached_results)}/{len(chunks)} "
                       f"({len(cached_results)/len(chunks)*100:.1f}%)")
        
        return results
    
    def save_embeddings(self, embeddings: List[EmbeddingResult], filepath: str) -> None:
        """Save embeddings to disk."""
        logger.info(f"Saving {len(embeddings)} embeddings to {filepath}")
        
        # Convert to a serializable format
        serializable_data = []
        for emb in embeddings:
            data = {
                'chunk_id': emb.chunk_id,
                'text': emb.text,
                'embedding': emb.embedding.tolist(),  # Convert numpy array to list
                'metadata': emb.metadata,
                'model_name': emb.model_name,
                'embedding_time': emb.embedding_time
            }
            serializable_data.append(data)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(serializable_data, f)
        
        logger.success(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str) -> List[EmbeddingResult]:
        """Load embeddings from disk."""
        logger.info(f"Loading embeddings from {filepath}")
        
        with open(filepath, 'rb') as f:
            serializable_data = pickle.load(f)
        
        # Convert back to EmbeddingResult objects
        embeddings = []
        for data in serializable_data:
            embedding = EmbeddingResult(
                chunk_id=data['chunk_id'],
                text=data['text'],
                embedding=np.array(data['embedding']),  # Convert list back to numpy array
                metadata=data['metadata'],
                model_name=data['model_name'],
                embedding_time=data['embedding_time']
            )
            embeddings.append(embedding)
        
        logger.success(f"Loaded {len(embeddings)} embeddings from {filepath}")
        return embeddings
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats['total_embeddings'] > 0:
            stats['avg_time_per_embedding'] = stats['total_time'] / stats['total_embeddings']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_embeddings']
        else:
            stats['avg_time_per_embedding'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        stats['cache_size'] = len(self.embedding_cache)
        stats['model_loaded'] = self.model_loaded
        stats['device'] = self.device
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class EmbeddingAnalyzer:
    """Analyzer for embedding quality and characteristics."""
    
    @staticmethod
    def analyze_embeddings(embeddings: List[EmbeddingResult]) -> Dict[str, Any]:
        """Analyze embedding quality and characteristics."""
        if not embeddings:
            return {}
        
        # Extract embedding vectors
        vectors = np.array([emb.embedding for emb in embeddings])
        
        analysis = {
            'count': len(embeddings),
            'dimension': vectors.shape[1] if len(vectors) > 0 else 0,
            'statistics': {},
            'quality_metrics': {},
            'content_analysis': {}
        }
        
        if len(vectors) > 0:
            # Basic statistics
            analysis['statistics'] = {
                'mean_norm': float(np.mean(np.linalg.norm(vectors, axis=1))),
                'std_norm': float(np.std(np.linalg.norm(vectors, axis=1))),
                'mean_values': vectors.mean(axis=0).tolist(),
                'std_values': vectors.std(axis=0).tolist()
            }
            
            # Quality metrics
            if len(vectors) > 1:
                # Compute pairwise similarities
                from sklearn.metrics.pairwise import cosine_similarity
                similarities = cosine_similarity(vectors)
                
                # Remove diagonal (self-similarity)
                mask = np.eye(similarities.shape[0], dtype=bool)
                off_diagonal = similarities[~mask]
                
                analysis['quality_metrics'] = {
                    'mean_similarity': float(np.mean(off_diagonal)),
                    'std_similarity': float(np.std(off_diagonal)),
                    'min_similarity': float(np.min(off_diagonal)),
                    'max_similarity': float(np.max(off_diagonal))
                }
            
            # Content analysis
            site_types = [emb.metadata.get('site_type', 'unknown') for emb in embeddings]
            chunking_methods = [emb.metadata.get('final_chunking_method', 'unknown') for emb in embeddings]
            
            analysis['content_analysis'] = {
                'site_type_distribution': {st: site_types.count(st) for st in set(site_types)},
                'chunking_method_distribution': {cm: chunking_methods.count(cm) for cm in set(chunking_methods)},
                'text_length_stats': {
                    'mean': float(np.mean([len(emb.text) for emb in embeddings])),
                    'std': float(np.std([len(emb.text) for emb in embeddings])),
                    'min': min(len(emb.text) for emb in embeddings),
                    'max': max(len(emb.text) for emb in embeddings)
                }
            }
        
        return analysis


# Example usage and testing
if __name__ == "__main__":
    from chunker import HybridChunker, TextChunk
    
    # Test with sample chunks
    sample_chunks = [
        TextChunk(
            text="Artificial intelligence is revolutionizing many industries including healthcare, finance, and transportation.",
            chunk_id=0,
            start_position=0,
            end_position=100,
            metadata={'site_type': 'business', 'final_chunking_method': 'hybrid'}
        ),
        TextChunk(
            text="Machine learning algorithms can process vast amounts of data to identify patterns and make predictions.",
            chunk_id=1,
            start_position=100,
            end_position=200,
            metadata={'site_type': 'wikipedia', 'final_chunking_method': 'content_aware'}
        ),
        TextChunk(
            text="Breaking news: A new AI breakthrough was announced today by researchers at Stanford University.",
            chunk_id=2,
            start_position=200,
            end_position=300,
            metadata={'site_type': 'news', 'final_chunking_method': 'recursive'}
        )
    ]
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    results = generator.generate_embeddings(sample_chunks)
    
    print(f"Generated {len(results)} embeddings")
    for result in results:
        print(f"Chunk {result.chunk_id}: {len(result.embedding)} dimensions")
        print(f"  Text: {result.text[:50]}...")
        print(f"  Model: {result.model_name}")
        print(f"  Time: {result.embedding_time:.4f}s")
    
    # Analyze embeddings
    analyzer = EmbeddingAnalyzer()
    analysis = analyzer.analyze_embeddings(results)
    
    print("\nEmbedding Analysis:")
    print(f"Count: {analysis['count']}")
    print(f"Dimension: {analysis['dimension']}")
    print(f"Quality metrics: {analysis.get('quality_metrics', {})}")
    
    # Performance stats
    stats = generator.get_performance_stats()
    print(f"\nPerformance Stats:")
    print(f"Total embeddings: {stats['total_embeddings']}")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Avg time per embedding: {stats['avg_time_per_embedding']:.4f}s")
    print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")