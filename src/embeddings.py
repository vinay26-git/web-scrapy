"""
Advanced embedding generation module with caching and optimization.
"""
import os
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import tiktoken
from loguru import logger
from tqdm import tqdm

from .config import EmbeddingConfig
from .chunker import Chunk


class EmbeddingGenerator:
    """
    Advanced embedding generator with caching, batching, and optimization.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._setup_device()
        self._load_model()
        self.cache = {}
        self._load_cache()
    
    def _setup_device(self) -> str:
        """Setup the device for model inference."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        else:
            device = self.config.device
        
        return device
    
    def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            
            if "sentence-transformers" in self.config.model_name:
                # Use SentenceTransformers
                self.model = SentenceTransformer(
                    self.config.model_name,
                    device=self.device
                )
                logger.info("SentenceTransformers model loaded successfully")
            else:
                # Use HuggingFace transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self.model = AutoModel.from_pretrained(self.config.model_name)
                self.model.to(self.device)
                self.model.eval()
                logger.info("HuggingFace model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.config.model_name}: {e}")
            # Fallback to default model
            logger.info("Falling back to default model")
            self.config.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.model = SentenceTransformer(self.config.model_name, device=self.device)
    
    def _load_cache(self):
        """Load embedding cache from disk."""
        cache_file = Path(self.config.cache_dir) / "embeddings_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        if not self.config.save_embeddings:
            return
        
        cache_file = Path(self.config.cache_dir) / "embeddings_cache.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text content."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        text_hash = self._get_text_hash(text)
        return self.cache.get(text_hash)
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding for future use."""
        if not self.config.save_embeddings:
            return
        
        text_hash = self._get_text_hash(text)
        self.cache[text_hash] = embedding
    
    def _encode_with_sentence_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode texts using SentenceTransformers."""
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=True,
                normalize_embeddings=self.config.normalize_embeddings,
                max_length=self.config.max_length
            )
            return embeddings
        except Exception as e:
            logger.error(f"Error encoding with SentenceTransformers: {e}")
            raise
    
    def _encode_with_transformers(self, texts: List[str]) -> np.ndarray:
        """Encode texts using HuggingFace transformers."""
        try:
            embeddings = []
            
            for i in tqdm(range(0, len(texts), self.config.batch_size), desc="Generating embeddings"):
                batch_texts = texts[i:i + self.config.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # Use mean pooling
                    attention_mask = inputs['attention_mask']
                    embeddings_batch = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)
                    
                    if self.config.normalize_embeddings:
                        embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                    
                    embeddings.append(embeddings_batch.cpu().numpy())
            
            return np.vstack(embeddings)
            
        except Exception as e:
            logger.error(f"Error encoding with transformers: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cached_emb = self._get_cached_embedding(text)
            if cached_emb is not None:
                cached_embeddings.append(cached_emb)
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} uncached texts")
            
            if "sentence-transformers" in self.config.model_name:
                new_embeddings = self._encode_with_sentence_transformers(uncached_texts)
            else:
                new_embeddings = self._encode_with_transformers(uncached_texts)
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                self._cache_embedding(text, embedding)
        else:
            new_embeddings = np.array([])
        
        # Combine cached and new embeddings
        if len(cached_embeddings) > 0 and len(new_embeddings) > 0:
            all_embeddings = np.zeros((len(texts), new_embeddings.shape[1]))
            all_embeddings[uncached_indices] = new_embeddings
            
            cached_idx = 0
            for i in range(len(texts)):
                if i not in uncached_indices:
                    all_embeddings[i] = cached_embeddings[cached_idx]
                    cached_idx += 1
            
            embeddings = all_embeddings
        elif len(cached_embeddings) > 0:
            embeddings = np.array(cached_embeddings)
        else:
            embeddings = new_embeddings
        
        # Save cache
        self._save_cache()
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def generate_chunk_embeddings(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generate embeddings for chunks and attach them to chunk objects.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of Chunk objects with embeddings attached
        """
        if not chunks:
            return chunks
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist()
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        if self.model is None:
            return 0
        
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            return self.model.get_sentence_embedding_dimension()
        else:
            # For HuggingFace models, we need to check the config
            return self.model.config.hidden_size
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        if self.config.normalize_embeddings:
            # If embeddings are already normalized, use dot product
            return np.dot(embedding1, embedding2)
        else:
            # Compute cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    def find_similar_chunks(self, query_embedding: np.ndarray, chunk_embeddings: List[np.ndarray], 
                           top_k: int = 5) -> List[tuple]:
        """
        Find most similar chunks to a query embedding.
        
        Args:
            query_embedding: Query embedding
            chunk_embeddings: List of chunk embeddings
            top_k: Number of top similar chunks to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        similarities = []
        
        for i, chunk_embedding in enumerate(chunk_embeddings):
            similarity = self.compute_similarity(query_embedding, chunk_embedding)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Get statistics about embeddings.
        
        Args:
            embeddings: Numpy array of embeddings
            
        Returns:
            Dictionary with embedding statistics
        """
        if len(embeddings) == 0:
            return {
                'count': 0,
                'dimension': 0,
                'mean_norm': 0,
                'std_norm': 0,
                'min_norm': 0,
                'max_norm': 0
            }
        
        # Compute norms
        norms = np.linalg.norm(embeddings, axis=1)
        
        return {
            'count': len(embeddings),
            'dimension': embeddings.shape[1],
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms))
        }
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to file.
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save embeddings
        """
        try:
            np.save(filepath, embeddings)
            logger.info(f"Saved embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save embeddings: {e}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Numpy array of embeddings
        """
        try:
            embeddings = np.load(filepath)
            logger.info(f"Loaded embeddings from {filepath}")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return np.array([])
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        cache_file = Path(self.config.cache_dir) / "embeddings_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Embedding cache cleared")