"""
Advanced Embedding Generation System
Supports multiple embedding models with domain-specific optimizations
Author: Full-Stack Developer (Google, Amazon, Microsoft experience)
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import pickle
import json
import os
from pathlib import Path
import time
from tqdm import tqdm
from loguru import logger

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

from chunking import Chunk


@dataclass
class Embedding:
    """Data class representing an embedding with metadata."""
    chunk_id: str
    embedding: np.ndarray
    model_name: str
    metadata: Dict
    generation_time: float
    embedding_dim: int


class EmbeddingGenerator:
    """
    Advanced embedding generator with support for multiple models
    and domain-specific optimizations.
    """
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "auto",
                 batch_size: int = 32,
                 max_length: int = 512):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model
            device: Device to use ('auto', 'cpu', 'cuda')
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._setup_device(device)
        
        # Initialize the model
        self.model = None
        self.tokenizer = None
        self._load_model()
        
        # Domain-specific embedding strategies
        self.domain_strategies = {
            'business': {
                'focus_keywords': ['service', 'product', 'company', 'business', 'solution'],
                'weight_multiplier': 1.1,
                'context_window': 200
            },
            'news': {
                'focus_keywords': ['news', 'report', 'announcement', 'update', 'breaking'],
                'weight_multiplier': 1.2,
                'context_window': 150
            },
            'wikipedia': {
                'focus_keywords': ['information', 'data', 'fact', 'reference', 'source'],
                'weight_multiplier': 1.0,
                'context_window': 300
            }
        }
        
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
    
    def _setup_device(self, device: str) -> str:
        """
        Setup the best available device.
        
        Args:
            device: Device specification
            
        Returns:
            Device string
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("CUDA available, using GPU")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        
        return device
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # Set model parameters
            self.model.max_seq_length = self.max_length
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            # Fallback to a simpler model
            try:
                logger.info("Trying fallback model: all-MiniLM-L6-v2")
                self.model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
                self.model_name = "all-MiniLM-L6-v2"
                logger.info("Fallback model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise
    
    def _preprocess_text(self, text: str, domain: str = "generic") -> str:
        """
        Preprocess text for better embedding quality.
        
        Args:
            text: Input text
            domain: Domain type for optimization
            
        Returns:
            Preprocessed text
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Domain-specific preprocessing
        if domain in self.domain_strategies:
            strategy = self.domain_strategies[domain]
            
            # Add domain context if text is too short
            if len(text) < 100:
                context_keywords = " ".join(strategy['focus_keywords'][:3])
                text = f"{context_keywords}: {text}"
        
        return text
    
    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embeddings
        """
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            embedding_dim = self.model.get_sentence_embedding_dimension()
            return np.zeros((len(texts), embedding_dim))
    
    def generate_embedding(self, chunk: Chunk, domain: str = "generic") -> Embedding:
        """
        Generate embedding for a single chunk.
        
        Args:
            chunk: Chunk object
            domain: Domain type for optimization
            
        Returns:
            Embedding object
        """
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess_text(chunk.text, domain)
        
        # Generate embedding
        embedding_array = self._generate_embeddings_batch([processed_text])
        embedding = embedding_array[0]
        
        # Create metadata
        metadata = {
            'original_length': len(chunk.text),
            'processed_length': len(processed_text),
            'domain': domain,
            'chunk_type': chunk.chunk_type,
            'semantic_score': chunk.semantic_score,
            'model_name': self.model_name,
            'device': self.device
        }
        
        # Add chunk metadata
        metadata.update(chunk.metadata)
        
        embedding_obj = Embedding(
            chunk_id=chunk.chunk_id,
            embedding=embedding,
            model_name=self.model_name,
            metadata=metadata,
            generation_time=time.time() - start_time,
            embedding_dim=len(embedding)
        )
        
        return embedding_obj
    
    def generate_embeddings_batch(self, 
                                chunks: List[Chunk], 
                                domains: List[str] = None) -> List[Embedding]:
        """
        Generate embeddings for multiple chunks efficiently.
        
        Args:
            chunks: List of chunks
            domains: List of corresponding domains
            
        Returns:
            List of embedding objects
        """
        if domains is None:
            domains = ["generic"] * len(chunks)
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        
        embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(chunks), self.batch_size), desc="Generating embeddings"):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_domains = domains[i:i + self.batch_size]
            
            # Preprocess texts
            processed_texts = []
            for chunk, domain in zip(batch_chunks, batch_domains):
                processed_text = self._preprocess_text(chunk.text, domain)
                processed_texts.append(processed_text)
            
            # Generate embeddings for batch
            batch_embeddings = self._generate_embeddings_batch(processed_texts)
            
            # Create embedding objects
            for j, (chunk, domain, embedding_array) in enumerate(zip(batch_chunks, batch_domains, batch_embeddings)):
                start_time = time.time()
                
                metadata = {
                    'original_length': len(chunk.text),
                    'processed_length': len(processed_texts[j]),
                    'domain': domain,
                    'chunk_type': chunk.chunk_type,
                    'semantic_score': chunk.semantic_score,
                    'model_name': self.model_name,
                    'device': self.device,
                    'batch_index': i + j
                }
                
                # Add chunk metadata
                metadata.update(chunk.metadata)
                
                embedding_obj = Embedding(
                    chunk_id=chunk.chunk_id,
                    embedding=embedding_array,
                    model_name=self.model_name,
                    metadata=metadata,
                    generation_time=time.time() - start_time,
                    embedding_dim=len(embedding_array)
                )
                
                embeddings.append(embedding_obj)
        
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        return embeddings
    
    def calculate_similarity(self, embedding1: Embedding, embedding2: Embedding) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            # Ensure embeddings are normalized
            emb1 = embedding1.embedding / np.linalg.norm(embedding1.embedding)
            emb2 = embedding2.embedding / np.linalg.norm(embedding2.embedding)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def find_similar_chunks(self, 
                          query_embedding: Embedding, 
                          chunk_embeddings: List[Embedding], 
                          top_k: int = 5) -> List[Tuple[Embedding, float]]:
        """
        Find most similar chunks to a query embedding.
        
        Args:
            query_embedding: Query embedding
            chunk_embeddings: List of chunk embeddings to search
            top_k: Number of top results to return
            
        Returns:
            List of (embedding, similarity_score) tuples
        """
        similarities = []
        
        for chunk_emb in chunk_embeddings:
            similarity = self.calculate_similarity(query_embedding, chunk_emb)
            similarities.append((chunk_emb, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_embeddings(self, embeddings: List[Embedding], filepath: str):
        """
        Save embeddings to file.
        
        Args:
            embeddings: List of embeddings to save
            filepath: Path to save file
        """
        try:
            # Convert embeddings to serializable format
            serializable_embeddings = []
            for emb in embeddings:
                serializable_emb = {
                    'chunk_id': emb.chunk_id,
                    'embedding': emb.embedding.tolist(),
                    'model_name': emb.model_name,
                    'metadata': emb.metadata,
                    'generation_time': emb.generation_time,
                    'embedding_dim': emb.embedding_dim
                }
                serializable_embeddings.append(serializable_emb)
            
            # Save as JSON
            with open(filepath, 'w') as f:
                json.dump(serializable_embeddings, f, indent=2)
            
            logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
            raise
    
    def load_embeddings(self, filepath: str) -> List[Embedding]:
        """
        Load embeddings from file.
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            List of embedding objects
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            embeddings = []
            for item in data:
                embedding = Embedding(
                    chunk_id=item['chunk_id'],
                    embedding=np.array(item['embedding']),
                    model_name=item['model_name'],
                    metadata=item['metadata'],
                    generation_time=item['generation_time'],
                    embedding_dim=item['embedding_dim']
                )
                embeddings.append(embedding)
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            raise


class MultiModelEmbeddingGenerator:
    """
    Multi-model embedding generator for ensemble approaches.
    """
    
    def __init__(self, model_names: List[str] = None):
        """
        Initialize multi-model generator.
        
        Args:
            model_names: List of model names to use
        """
        if model_names is None:
            model_names = [
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "multi-qa-MiniLM-L6-cos-v1"
            ]
        
        self.models = {}
        self._load_models(model_names)
        
        logger.info(f"MultiModelEmbeddingGenerator initialized with {len(self.models)} models")
    
    def _load_models(self, model_names: List[str]):
        """Load multiple models."""
        for model_name in model_names:
            try:
                generator = EmbeddingGenerator(model_name=model_name)
                self.models[model_name] = generator
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
    
    def generate_ensemble_embeddings(self, 
                                   chunks: List[Chunk], 
                                   domains: List[str] = None) -> Dict[str, List[Embedding]]:
        """
        Generate embeddings using all available models.
        
        Args:
            chunks: List of chunks
            domains: List of corresponding domains
            
        Returns:
            Dictionary mapping model names to embedding lists
        """
        results = {}
        
        for model_name, generator in self.models.items():
            try:
                embeddings = generator.generate_embeddings_batch(chunks, domains)
                results[model_name] = embeddings
                logger.info(f"Generated embeddings with {model_name}: {len(embeddings)}")
            except Exception as e:
                logger.error(f"Error generating embeddings with {model_name}: {e}")
        
        return results
    
    def get_best_model(self, domain: str = "generic") -> str:
        """
        Get the best model for a specific domain.
        
        Args:
            domain: Domain type
            
        Returns:
            Best model name
        """
        # Domain-specific model recommendations
        domain_models = {
            'business': 'all-mpnet-base-v2',
            'news': 'multi-qa-MiniLM-L6-cos-v1',
            'wikipedia': 'all-MiniLM-L6-v2'
        }
        
        return domain_models.get(domain, 'all-MiniLM-L6-v2')


# Add missing import
import re


if __name__ == "__main__":
    # Example usage
    from chunking import HybridChunker
    
    # Sample text
    sample_text = """
    This is a sample business website content. Our company provides excellent solutions.
    We have been in business for over 10 years and serve clients worldwide.
    Our team consists of experienced professionals dedicated to quality.
    """
    
    # Create chunks
    chunker = HybridChunker(max_chunk_size=200, min_chunk_size=50)
    chunks = chunker.chunk_text(sample_text, domain='business')
    
    # Generate embeddings
    embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    embeddings = embedding_gen.generate_embeddings_batch(chunks, domains=['business'] * len(chunks))
    
    print(f"Generated {len(embeddings)} embeddings")
    for i, emb in enumerate(embeddings):
        print(f"Embedding {i+1}: {emb.embedding_dim} dimensions, {emb.generation_time:.3f}s")