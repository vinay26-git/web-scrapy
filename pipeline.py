"""
Main Web Scraping Pipeline
Orchestrates scraping, chunking, and embedding generation
Author: Full-Stack Developer (Google, Amazon, Microsoft experience)
"""

import os
import json
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
from datetime import datetime
from tqdm import tqdm
from loguru import logger

from web_scraper import WebScraper, ScrapedContent
from chunking import HybridChunker, Chunk
from embeddings import EmbeddingGenerator, Embedding, MultiModelEmbeddingGenerator


@dataclass
class PipelineResult:
    """Data class representing the complete pipeline result."""
    url: str
    scraped_content: ScrapedContent
    chunks: List[Chunk]
    embeddings: List[Embedding]
    pipeline_metadata: Dict
    processing_time: float


class WebScrapingPipeline:
    """
    Complete web scraping pipeline that handles:
    1. Web scraping with Selenium
    2. Content cleaning and preprocessing
    3. Hybrid chunking
    4. Embedding generation
    """
    
    def __init__(self, 
                 output_dir: str = "output",
                 chunk_config: Dict = None,
                 embedding_config: Dict = None,
                 save_intermediate: bool = True):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to save results
            chunk_config: Configuration for chunking
            embedding_config: Configuration for embeddings
            save_intermediate: Whether to save intermediate results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "scraped").mkdir(exist_ok=True)
        (self.output_dir / "chunks").mkdir(exist_ok=True)
        (self.output_dir / "embeddings").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Default configurations
        self.chunk_config = chunk_config or {
            'max_chunk_size': 1000,
            'min_chunk_size': 100,
            'overlap_size': 100,
            'semantic_threshold': 0.7
        }
        
        self.embedding_config = embedding_config or {
            'model_name': 'all-MiniLM-L6-v2',
            'batch_size': 32,
            'max_length': 512
        }
        
        self.save_intermediate = save_intermediate
        
        # Initialize components
        self.scraper = None
        self.chunker = None
        self.embedding_generator = None
        
        self._initialize_components()
        
        logger.info("WebScrapingPipeline initialized successfully")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Initialize scraper
            self.scraper = WebScraper(headless=True, timeout=30)
            logger.info("Web scraper initialized")
            
            # Initialize chunker
            self.chunker = HybridChunker(**self.chunk_config)
            logger.info("Hybrid chunker initialized")
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(**self.embedding_config)
            logger.info("Embedding generator initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def process_single_url(self, url: str) -> Optional[PipelineResult]:
        """
        Process a single URL through the complete pipeline.
        
        Args:
            url: URL to process
            
        Returns:
            PipelineResult object or None if failed
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting pipeline for URL: {url}")
            
            # Step 1: Scrape content
            scraped_content = self.scraper.scrape_url(url)
            if not scraped_content:
                logger.error(f"Failed to scrape content from {url}")
                return None
            
            logger.info(f"Successfully scraped content: {len(scraped_content.content)} characters")
            
            # Save scraped content
            if self.save_intermediate:
                self._save_scraped_content(scraped_content)
            
            # Step 2: Chunk content
            chunks = self.chunker.chunk_text(
                scraped_content.content, 
                domain=scraped_content.website_type
            )
            
            if not chunks:
                logger.warning(f"No chunks created for {url}")
                return None
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Save chunks
            if self.save_intermediate:
                self._save_chunks(chunks, url)
            
            # Step 3: Generate embeddings
            domains = [scraped_content.website_type] * len(chunks)
            embeddings = self.embedding_generator.generate_embeddings_batch(chunks, domains)
            
            if not embeddings:
                logger.warning(f"No embeddings generated for {url}")
                return None
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Save embeddings
            if self.save_intermediate:
                self._save_embeddings(embeddings, url)
            
            # Create pipeline metadata
            pipeline_metadata = {
                'url': url,
                'website_type': scraped_content.website_type,
                'content_length': len(scraped_content.content),
                'chunk_count': len(chunks),
                'embedding_count': len(embeddings),
                'chunk_config': self.chunk_config,
                'embedding_config': self.embedding_config,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Create result
            result = PipelineResult(
                url=url,
                scraped_content=scraped_content,
                chunks=chunks,
                embeddings=embeddings,
                pipeline_metadata=pipeline_metadata,
                processing_time=time.time() - start_time
            )
            
            # Save complete result
            self._save_pipeline_result(result)
            
            logger.info(f"Pipeline completed for {url} in {result.processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing {url}: {e}")
            return None
    
    def process_multiple_urls(self, urls: List[str]) -> List[PipelineResult]:
        """
        Process multiple URLs through the pipeline.
        
        Args:
            urls: List of URLs to process
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        logger.info(f"Starting pipeline for {len(urls)} URLs")
        
        for i, url in enumerate(tqdm(urls, desc="Processing URLs")):
            try:
                result = self.process_single_url(url)
                if result:
                    results.append(result)
                
                # Add delay between requests
                if i < len(urls) - 1:
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                continue
        
        logger.info(f"Pipeline completed for {len(results)}/{len(urls)} URLs")
        return results
    
    def _save_scraped_content(self, scraped_content: ScrapedContent):
        """Save scraped content to file."""
        try:
            filename = f"scraped_{scraped_content.url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "scraped" / filename
            
            # Convert to serializable format
            data = {
                'url': scraped_content.url,
                'title': scraped_content.title,
                'content': scraped_content.content,
                'website_type': scraped_content.website_type,
                'metadata': scraped_content.metadata,
                'timestamp': scraped_content.timestamp
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving scraped content: {e}")
    
    def _save_chunks(self, chunks: List[Chunk], url: str):
        """Save chunks to file."""
        try:
            filename = f"chunks_{url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "chunks" / filename
            
            # Convert chunks to serializable format
            serializable_chunks = []
            for chunk in chunks:
                chunk_data = {
                    'text': chunk.text,
                    'chunk_id': chunk.chunk_id,
                    'start_index': chunk.start_index,
                    'end_index': chunk.end_index,
                    'chunk_type': chunk.chunk_type,
                    'metadata': chunk.metadata,
                    'semantic_score': chunk.semantic_score
                }
                serializable_chunks.append(chunk_data)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_chunks, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
    
    def _save_embeddings(self, embeddings: List[Embedding], url: str):
        """Save embeddings to file."""
        try:
            filename = f"embeddings_{url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "embeddings" / filename
            
            self.embedding_generator.save_embeddings(embeddings, str(filepath))
                
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def _save_pipeline_result(self, result: PipelineResult):
        """Save complete pipeline result."""
        try:
            filename = f"result_{result.url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "results" / filename
            
            # Convert result to serializable format
            result_data = {
                'url': result.url,
                'pipeline_metadata': result.pipeline_metadata,
                'processing_time': result.processing_time,
                'scraped_content_summary': {
                    'title': result.scraped_content.title,
                    'content_length': len(result.scraped_content.content),
                    'website_type': result.scraped_content.website_type
                },
                'chunks_summary': {
                    'count': len(result.chunks),
                    'chunk_ids': [chunk.chunk_id for chunk in result.chunks],
                    'total_text_length': sum(len(chunk.text) for chunk in result.chunks)
                },
                'embeddings_summary': {
                    'count': len(result.embeddings),
                    'embedding_dim': result.embeddings[0].embedding_dim if result.embeddings else 0,
                    'model_name': result.embeddings[0].model_name if result.embeddings else None
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving pipeline result: {e}")
    
    def load_pipeline_result(self, url: str) -> Optional[PipelineResult]:
        """
        Load a previously saved pipeline result.
        
        Args:
            url: URL of the result to load
            
        Returns:
            PipelineResult object or None if not found
        """
        try:
            filename = f"result_{url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "results" / filename
            
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                result_data = json.load(f)
            
            # Load individual components
            scraped_content = self._load_scraped_content(url)
            chunks = self._load_chunks(url)
            embeddings = self._load_embeddings(url)
            
            if not all([scraped_content, chunks, embeddings]):
                return None
            
            result = PipelineResult(
                url=url,
                scraped_content=scraped_content,
                chunks=chunks,
                embeddings=embeddings,
                pipeline_metadata=result_data['pipeline_metadata'],
                processing_time=result_data['processing_time']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading pipeline result: {e}")
            return None
    
    def _load_scraped_content(self, url: str) -> Optional[ScrapedContent]:
        """Load scraped content from file."""
        try:
            filename = f"scraped_{url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "scraped" / filename
            
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            return ScrapedContent(
                url=data['url'],
                title=data['title'],
                content=data['content'],
                website_type=data['website_type'],
                metadata=data['metadata'],
                raw_html="",  # Not saved for space
                timestamp=data['timestamp']
            )
            
        except Exception as e:
            logger.error(f"Error loading scraped content: {e}")
            return None
    
    def _load_chunks(self, url: str) -> Optional[List[Chunk]]:
        """Load chunks from file."""
        try:
            filename = f"chunks_{url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "chunks" / filename
            
            if not filepath.exists():
                return None
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            chunks = []
            for chunk_data in data:
                chunk = Chunk(
                    text=chunk_data['text'],
                    chunk_id=chunk_data['chunk_id'],
                    start_index=chunk_data['start_index'],
                    end_index=chunk_data['end_index'],
                    chunk_type=chunk_data['chunk_type'],
                    metadata=chunk_data['metadata'],
                    semantic_score=chunk_data['semantic_score']
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return None
    
    def _load_embeddings(self, url: str) -> Optional[List[Embedding]]:
        """Load embeddings from file."""
        try:
            filename = f"embeddings_{url.replace('://', '_').replace('/', '_')}.json"
            filepath = self.output_dir / "embeddings" / filename
            
            if not filepath.exists():
                return None
            
            return self.embedding_generator.load_embeddings(str(filepath))
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None
    
    def get_pipeline_statistics(self) -> Dict:
        """
        Get statistics about the pipeline processing.
        
        Returns:
            Dictionary with pipeline statistics
        """
        try:
            stats = {
                'total_results': 0,
                'total_chunks': 0,
                'total_embeddings': 0,
                'website_types': {},
                'processing_times': [],
                'success_rate': 0.0
            }
            
            results_dir = self.output_dir / "results"
            if not results_dir.exists():
                return stats
            
            result_files = list(results_dir.glob("*.json"))
            stats['total_results'] = len(result_files)
            
            for result_file in result_files:
                try:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                    
                    stats['total_chunks'] += data['chunks_summary']['count']
                    stats['total_embeddings'] += data['embeddings_summary']['count']
                    stats['processing_times'].append(data['processing_time'])
                    
                    website_type = data['scraped_content_summary']['website_type']
                    stats['website_types'][website_type] = stats['website_types'].get(website_type, 0) + 1
                    
                except Exception as e:
                    logger.warning(f"Error reading result file {result_file}: {e}")
            
            if stats['processing_times']:
                stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
                stats['min_processing_time'] = min(stats['processing_times'])
                stats['max_processing_time'] = max(stats['processing_times'])
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pipeline statistics: {e}")
            return {}
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.scraper:
                self.scraper.close()
            logger.info("Pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


if __name__ == "__main__":
    # Example usage
    urls = [
        "https://www.example.com",
        "https://www.bbc.com/news",
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    ]
    
    # Initialize pipeline
    pipeline = WebScrapingPipeline(
        output_dir="pipeline_output",
        chunk_config={
            'max_chunk_size': 800,
            'min_chunk_size': 100,
            'overlap_size': 50,
            'semantic_threshold': 0.6
        },
        embedding_config={
            'model_name': 'all-MiniLM-L6-v2',
            'batch_size': 16,
            'max_length': 512
        }
    )
    
    try:
        # Process URLs
        results = pipeline.process_multiple_urls(urls)
        
        # Print results
        for result in results:
            print(f"\nURL: {result.url}")
            print(f"Website Type: {result.scraped_content.website_type}")
            print(f"Content Length: {len(result.scraped_content.content)}")
            print(f"Chunks: {len(result.chunks)}")
            print(f"Embeddings: {len(result.embeddings)}")
            print(f"Processing Time: {result.processing_time:.2f}s")
        
        # Get statistics
        stats = pipeline.get_pipeline_statistics()
        print(f"\nPipeline Statistics:")
        print(f"Total Results: {stats['total_results']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Website Types: {stats['website_types']}")
        
    finally:
        pipeline.cleanup()