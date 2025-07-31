"""
Advanced embedding generation pipeline with support for multiple models and optimization.
Implements enterprise-grade vector generation with batching, caching, and quality control.
"""
import os
import pickle
import hashlib
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
from tqdm import tqdm
from loguru import logger

from .chunker import TextChunk
from .config import Config


@dataclass
class ChunkEmbedding:
    """Container for chunk embeddings with metadata."""
    chunk_id: str
    embedding: np.ndarray
    model_name: str
    embedding_dim: int
    source_url: str
    chunk_text: str  # Store for reference/debugging
    token_count: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert numpy array to list for JSON serialization
        data['embedding'] = self.embedding.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChunkEmbedding':
        """Create from dictionary."""
        # Convert list back to numpy array
        data['embedding'] = np.array(data['embedding'])
        return cls(**data)


class EmbeddingCache:
    """Simple file-based cache for embeddings."""
    
    def __init__(self, cache_dir: str = "cache/embeddings"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized embedding cache at {cache_dir}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for text and model combination."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get cache file path for a given key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Retrieve cached embedding."""
        cache_key = self._get_cache_key(text, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached embedding: {e}")
        
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """Cache embedding."""
        cache_key = self._get_cache_key(text, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def clear(self):
        """Clear all cached embeddings."""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info("Cleared embedding cache")


class EmbeddingGenerator:
    """
    Enterprise-grade embedding generator with support for multiple models.
    
    Features:
    - Multiple embedding model support
    - Batch processing for efficiency
    - GPU acceleration when available
    - Embedding caching
    - Quality validation
    - Normalization options
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_name = config.embedding.model_name
        self.device = self._get_device()
        self.cache = EmbeddingCache()
        
        # Load model
        self._load_model()
        
        logger.info(f"Initialized embedding generator with {self.model_name} on {self.device}")
    
    def _get_device(self) -> str:
        """Determine the best device for model inference."""
        if self.config.embedding.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("Using CPU for embedding generation")
        
        return device
    
    def _load_model(self):
        """Load the embedding model based on configuration."""
        try:
            if "sentence-transformers" in self.model_name:
                # Use SentenceTransformers
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded SentenceTransformer model: {self.model_name}")
            else:
                # Use Transformers library directly
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info(f"Loaded Transformers model: {self.model_name}")
                
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a smaller model
            try:
                self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.warning(f"Falling back to {self.model_name}")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise RuntimeError("Could not load any embedding model")
    
    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to model output."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _generate_embedding_transformers(self, text: str) -> np.ndarray:
        """Generate embedding using Transformers library."""
        # Tokenize
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.config.embedding.max_seq_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize if specified
            if self.config.embedding.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
            
            return embedding.cpu().numpy()[0]
    
    def _generate_embedding_sentence_transformers(self, text: str) -> np.ndarray:
        """Generate embedding using SentenceTransformers."""
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.config.embedding.normalize_embeddings,
            show_progress_bar=False
        )
        return embedding
    
    def _validate_embedding(self, embedding: np.ndarray, text: str) -> bool:
        """Validate embedding quality."""
        if embedding is None or len(embedding) == 0:
            logger.warning("Empty embedding generated")
            return False
        
        # Check for NaN or infinite values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            logger.warning("Embedding contains NaN or infinite values")
            return False
        
        # Check if embedding is all zeros (unusual)
        if np.allclose(embedding, 0):
            logger.warning("Embedding is all zeros")
            return False
        
        # Check embedding magnitude
        magnitude = np.linalg.norm(embedding)
        if magnitude < 0.1 or magnitude > 100:
            logger.warning(f"Unusual embedding magnitude: {magnitude}")
            return False
        
        return True
    
    def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if generation failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
        
        # Check cache first
        cached_embedding = self.cache.get(text, self.model_name)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            # Generate embedding based on model type
            if "sentence-transformers" in self.model_name:
                embedding = self._generate_embedding_sentence_transformers(text)
            else:
                embedding = self._generate_embedding_transformers(text)
            
            # Validate embedding
            if not self._validate_embedding(embedding, text):
                logger.error(f"Invalid embedding generated for text: {text[:100]}...")
                return None
            
            # Cache the embedding
            self.cache.set(text, self.model_name, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors (None for failed generations)
        """
        if not texts:
            return []
        
        embeddings = []
        batch_size = self.config.embedding.batch_size
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            
            # Check cache for each text in batch
            batch_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch_texts):
                if not text or not text.strip():
                    batch_embeddings.append(None)
                    continue
                
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    batch_embeddings.append(cached)
                else:
                    batch_embeddings.append(None)  # Placeholder
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                try:
                    if "sentence-transformers" in self.model_name:
                        # Batch generation with SentenceTransformers
                        batch_results = self.model.encode(
                            uncached_texts,
                            convert_to_numpy=True,
                            normalize_embeddings=self.config.embedding.normalize_embeddings,
                            show_progress_bar=False,
                            batch_size=min(len(uncached_texts), 32)  # Internal batch size
                        )
                    else:
                        # Generate one by one for Transformers (can be optimized further)
                        batch_results = []
                        for text in uncached_texts:
                            emb = self._generate_embedding_transformers(text)
                            batch_results.append(emb)
                        batch_results = np.array(batch_results)
                    
                    # Validate and cache results
                    for idx, (text, embedding) in enumerate(zip(uncached_texts, batch_results)):
                        if self._validate_embedding(embedding, text):
                            batch_embeddings[uncached_indices[idx]] = embedding
                            self.cache.set(text, self.model_name, embedding)
                        else:
                            batch_embeddings[uncached_indices[idx]] = None
                            
                except Exception as e:
                    logger.error(f"Batch embedding generation failed: {e}")
                    # Mark all uncached as failed
                    for idx in uncached_indices:
                        batch_embeddings[idx] = None
            
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_chunks(self, chunks: List[TextChunk]) -> List[ChunkEmbedding]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of ChunkEmbedding objects
        """
        if not chunks:
            return []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = self.generate_embeddings_batch(texts)
        
        # Create ChunkEmbedding objects
        chunk_embeddings = []
        successful_count = 0
        
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk.chunk_id,
                    embedding=embedding,
                    model_name=self.model_name,
                    embedding_dim=len(embedding),
                    source_url=chunk.source_url,
                    chunk_text=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                    token_count=chunk.token_count,
                    metadata={
                        **chunk.metadata,
                        'chunk_method': chunk.chunk_method,
                        'sentence_count': chunk.sentence_count,
                        'embedding_model': self.model_name,
                        'device_used': self.device
                    }
                )
                chunk_embeddings.append(chunk_embedding)
                successful_count += 1
            else:
                logger.warning(f"Failed to generate embedding for chunk {chunk.chunk_id}")
        
        logger.success(f"Successfully generated {successful_count}/{len(chunks)} embeddings")
        return chunk_embeddings
    
    def save_embeddings(self, embeddings: List[ChunkEmbedding], filepath: str):
        """
        Save embeddings to file.
        
        Args:
            embeddings: List of ChunkEmbedding objects
            filepath: Path to save file
        """
        try:
            # Convert to serializable format
            data = {
                'embeddings': [emb.to_dict() for emb in embeddings],
                'metadata': {
                    'model_name': self.model_name,
                    'total_count': len(embeddings),
                    'embedding_dim': embeddings[0].embedding_dim if embeddings else 0,
                    'config': {
                        'normalize_embeddings': self.config.embedding.normalize_embeddings,
                        'max_seq_length': self.config.embedding.max_seq_length,
                        'batch_size': self.config.embedding.batch_size
                    }
                }
            }
            
            # Save as JSON (for metadata) and numpy arrays separately
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save metadata as JSON
            json_path = filepath.replace('.pkl', '.json')
            with open(json_path, 'w') as f:
                # Exclude embeddings from JSON (too large)
                json_data = {k: v for k, v in data.items() if k != 'embeddings'}
                json_data['embeddings_count'] = len(embeddings)
                json.dump(json_data, f, indent=2)
            
            # Save full data as pickle
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.success(f"Saved {len(embeddings)} embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, filepath: str) -> List[ChunkEmbedding]:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            List of ChunkEmbedding objects
        """
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            embeddings = [ChunkEmbedding.from_dict(emb_data) for emb_data in data['embeddings']]
            
            logger.success(f"Loaded {len(embeddings)} embeddings from {filepath}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return []
    
    def get_embedding_stats(self, embeddings: List[ChunkEmbedding]) -> Dict:
        """
        Calculate statistics for a set of embeddings.
        
        Args:
            embeddings: List of ChunkEmbedding objects
            
        Returns:
            Dictionary with embedding statistics
        """
        if not embeddings:
            return {}
        
        embedding_matrix = np.array([emb.embedding for emb in embeddings])
        
        stats = {
            'count': len(embeddings),
            'dimension': embeddings[0].embedding_dim,
            'model_name': embeddings[0].model_name,
            'mean_magnitude': float(np.mean(np.linalg.norm(embedding_matrix, axis=1))),
            'std_magnitude': float(np.std(np.linalg.norm(embedding_matrix, axis=1))),
            'mean_embedding': np.mean(embedding_matrix, axis=0).tolist(),
            'source_urls': list(set(emb.source_url for emb in embeddings)),
            'chunk_methods': list(set(emb.metadata.get('chunk_method', 'unknown') for emb in embeddings)),
            'avg_token_count': np.mean([emb.token_count for emb in embeddings])
        }
        
        return stats