"""
Main pipeline orchestrator for the Web Scraper project.
Coordinates all components from scraping to embedding generation.
"""
import os
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json

from loguru import logger
from tqdm import tqdm

from .config import Config, WebsiteType
from .scraper import WebScraper, ScrapedContent
from .preprocessor import TextPreprocessor, ProcessedContent
from .chunker import HybridChunker, TextChunk
from .embeddings import EmbeddingGenerator, ChunkEmbedding


@dataclass
class PipelineResult:
    """Container for complete pipeline results."""
    scraped_contents: List[ScrapedContent]
    processed_contents: List[ProcessedContent]
    chunks: List[TextChunk]
    embeddings: List[ChunkEmbedding]
    statistics: Dict
    execution_time: float
    
    def save_summary(self, filepath: str):
        """Save pipeline summary to JSON file."""
        summary = {
            'execution_time_seconds': self.execution_time,
            'statistics': self.statistics,
            'scraped_count': len(self.scraped_contents),
            'processed_count': len(self.processed_contents),
            'chunks_count': len(self.chunks),
            'embeddings_count': len(self.embeddings),
            'urls_processed': [content.url for content in self.scraped_contents],
            'chunk_methods_used': list(set(chunk.chunk_method for chunk in self.chunks)),
            'embedding_model': self.embeddings[0].model_name if self.embeddings else None,
            'avg_quality_score': sum(
                content.quality_metrics.get('overall_quality_score', 0) 
                for content in self.processed_contents
            ) / max(len(self.processed_contents), 1)
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved pipeline summary to {filepath}")


class WebScrapingPipeline:
    """
    Enterprise-grade web scraping pipeline.
    
    Features:
    - Complete end-to-end processing
    - Error handling and recovery
    - Progress tracking and logging
    - Quality assessment
    - Batch processing optimization
    - Result persistence
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.scraper = WebScraper(config)
        self.preprocessor = TextPreprocessor(config)
        self.chunker = HybridChunker(config)
        self.embedding_generator = EmbeddingGenerator(config)
        
        # Setup logging
        self._setup_logging()
        
        logger.info("Initialized Web Scraping Pipeline")
    
    def _setup_logging(self):
        """Configure comprehensive logging."""
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Add file logger for pipeline execution
        logger.add(
            "logs/pipeline.log",
            rotation="50 MB",
            retention="30 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
            backtrace=True,
            diagnose=True
        )
    
    def scrape_urls(self, urls: List[str], 
                   website_types: Optional[List[WebsiteType]] = None) -> List[ScrapedContent]:
        """
        Scrape multiple URLs with progress tracking.
        
        Args:
            urls: List of URLs to scrape
            website_types: Optional list of website types (auto-detected if None)
            
        Returns:
            List of successfully scraped content
        """
        logger.info(f"Starting scraping phase for {len(urls)} URLs")
        
        scraped_contents = []
        
        with tqdm(total=len(urls), desc="Scraping URLs") as pbar:
            for i, url in enumerate(urls):
                website_type = website_types[i] if website_types else None
                
                try:
                    scraped_content = self.scraper.scrape_url(url, website_type)
                    if scraped_content:
                        scraped_contents.append(scraped_content)
                        logger.info(f"✓ Scraped: {url}")
                    else:
                        logger.warning(f"✗ Failed to scrape: {url}")
                        
                except Exception as e:
                    logger.error(f"✗ Error scraping {url}: {e}")
                
                pbar.update(1)
                pbar.set_postfix({
                    'Success': len(scraped_contents),
                    'Current': url[:50] + '...' if len(url) > 50 else url
                })
        
        success_rate = len(scraped_contents) / len(urls) * 100
        logger.info(f"Scraping completed: {len(scraped_contents)}/{len(urls)} URLs ({success_rate:.1f}% success rate)")
        
        return scraped_contents
    
    def process_contents(self, scraped_contents: List[ScrapedContent], 
                        min_quality_score: float = 50.0) -> List[ProcessedContent]:
        """
        Process scraped contents with quality filtering.
        
        Args:
            scraped_contents: List of scraped content to process
            min_quality_score: Minimum quality score threshold
            
        Returns:
            List of processed and filtered content
        """
        logger.info(f"Starting preprocessing phase for {len(scraped_contents)} contents")
        
        # Process all contents
        processed_contents = self.preprocessor.process_batch(scraped_contents)
        
        # Filter by quality
        high_quality_contents = self.preprocessor.filter_by_quality(
            processed_contents, min_quality_score
        )
        
        # Log quality statistics
        if processed_contents:
            quality_scores = [
                content.quality_metrics.get('overall_quality_score', 0)
                for content in processed_contents
            ]
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)
            
            logger.info(f"Quality stats - Avg: {avg_quality:.1f}, Min: {min_quality:.1f}, Max: {max_quality:.1f}")
        
        filter_rate = len(high_quality_contents) / max(len(processed_contents), 1) * 100
        logger.info(f"Preprocessing completed: {len(high_quality_contents)}/{len(processed_contents)} passed quality filter ({filter_rate:.1f}%)")
        
        return high_quality_contents
    
    def chunk_contents(self, processed_contents: List[ProcessedContent]) -> List[TextChunk]:
        """
        Chunk processed contents using hybrid approach.
        
        Args:
            processed_contents: List of processed content to chunk
            
        Returns:
            List of text chunks
        """
        logger.info(f"Starting chunking phase for {len(processed_contents)} contents")
        
        # Chunk all contents
        chunks = self.chunker.chunk_batch(processed_contents)
        
        # Log chunking statistics
        if chunks:
            token_counts = [chunk.token_count for chunk in chunks]
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            
            chunk_methods = {}
            for chunk in chunks:
                method = chunk.chunk_method
                chunk_methods[method] = chunk_methods.get(method, 0) + 1
            
            logger.info(f"Chunk stats - Count: {len(chunks)}, Avg tokens: {avg_tokens:.1f}")
            logger.info(f"Token range: {min_tokens} - {max_tokens}")
            logger.info(f"Chunk methods: {chunk_methods}")
        
        logger.info(f"Chunking completed: {len(chunks)} chunks created")
        return chunks
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[ChunkEmbedding]:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks to embed
            
        Returns:
            List of chunk embeddings
        """
        logger.info(f"Starting embedding generation for {len(chunks)} chunks")
        
        # Generate embeddings
        embeddings = self.embedding_generator.embed_chunks(chunks)
        
        # Log embedding statistics
        if embeddings:
            stats = self.embedding_generator.get_embedding_stats(embeddings)
            logger.info(f"Embedding stats - Dimension: {stats['dimension']}")
            logger.info(f"Model: {stats['model_name']}")
            logger.info(f"Avg magnitude: {stats['mean_magnitude']:.3f}")
            logger.info(f"Avg token count: {stats['avg_token_count']:.1f}")
        
        success_rate = len(embeddings) / max(len(chunks), 1) * 100
        logger.info(f"Embedding generation completed: {len(embeddings)}/{len(chunks)} ({success_rate:.1f}% success rate)")
        
        return embeddings
    
    def run_pipeline(self, urls: List[str], 
                    website_types: Optional[List[WebsiteType]] = None,
                    min_quality_score: float = 50.0,
                    save_results: bool = True,
                    output_dir: str = "output") -> PipelineResult:
        """
        Run the complete pipeline from URLs to embeddings.
        
        Args:
            urls: List of URLs to process
            website_types: Optional website type hints
            min_quality_score: Minimum content quality threshold
            save_results: Whether to save intermediate and final results
            output_dir: Directory to save results
            
        Returns:
            PipelineResult with all outputs and statistics
        """
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info("STARTING WEB SCRAPING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"URLs to process: {len(urls)}")
        logger.info(f"Min quality score: {min_quality_score}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directory
        if save_results:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Phase 1: Scraping
            logger.info("Phase 1/4: Web Scraping")
            scraped_contents = self.scrape_urls(urls, website_types)
            
            if not scraped_contents:
                logger.error("No content was successfully scraped. Pipeline terminated.")
                raise RuntimeError("Scraping phase failed completely")
            
            # Phase 2: Preprocessing
            logger.info("Phase 2/4: Text Preprocessing")
            processed_contents = self.process_contents(scraped_contents, min_quality_score)
            
            if not processed_contents:
                logger.error("No content passed quality filters. Pipeline terminated.")
                raise RuntimeError("All content filtered out during preprocessing")
            
            # Phase 3: Chunking
            logger.info("Phase 3/4: Hybrid Chunking")
            chunks = self.chunk_contents(processed_contents)
            
            if not chunks:
                logger.error("No chunks were created. Pipeline terminated.")
                raise RuntimeError("Chunking phase failed")
            
            # Phase 4: Embedding Generation
            logger.info("Phase 4/4: Embedding Generation")
            embeddings = self.generate_embeddings(chunks)
            
            if not embeddings:
                logger.error("No embeddings were generated. Pipeline terminated.")
                raise RuntimeError("Embedding generation failed")
            
            # Calculate final statistics
            execution_time = time.time() - start_time
            statistics = self._calculate_pipeline_statistics(
                scraped_contents, processed_contents, chunks, embeddings, execution_time
            )
            
            # Create result object
            result = PipelineResult(
                scraped_contents=scraped_contents,
                processed_contents=processed_contents,
                chunks=chunks,
                embeddings=embeddings,
                statistics=statistics,
                execution_time=execution_time
            )
            
            # Save results if requested
            if save_results:
                self._save_pipeline_results(result, output_dir)
            
            # Log final summary
            self._log_pipeline_summary(result)
            
            logger.info("=" * 60)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("=" * 60)
            logger.error("PIPELINE FAILED")
            logger.error("=" * 60)
            logger.error(f"Error: {e}")
            logger.error(f"Execution time: {execution_time:.2f} seconds")
            raise
    
    def _calculate_pipeline_statistics(self, scraped_contents: List[ScrapedContent],
                                     processed_contents: List[ProcessedContent],
                                     chunks: List[TextChunk],
                                     embeddings: List[ChunkEmbedding],
                                     execution_time: float) -> Dict:
        """Calculate comprehensive pipeline statistics."""
        stats = {
            'execution_time_seconds': execution_time,
            'execution_time_formatted': f"{execution_time:.2f}s",
            
            # Input/Output counts
            'input_urls': len(scraped_contents) if scraped_contents else 0,
            'scraped_contents': len(scraped_contents),
            'processed_contents': len(processed_contents),
            'total_chunks': len(chunks),
            'total_embeddings': len(embeddings),
            
            # Success rates
            'processing_success_rate': len(processed_contents) / max(len(scraped_contents), 1),
            'embedding_success_rate': len(embeddings) / max(len(chunks), 1),
            'overall_success_rate': len(embeddings) / max(len(scraped_contents), 1),
            
            # Content statistics
            'website_types': {},
            'languages': {},
            'chunk_methods': {},
            'quality_scores': [],
            
            # Size statistics
            'total_characters_processed': 0,
            'total_tokens_processed': 0,
            'avg_chunk_size_tokens': 0,
            'avg_chunk_size_characters': 0,
        }
        
        # Analyze website types
        for content in scraped_contents:
            website_type = content.website_type.value
            stats['website_types'][website_type] = stats['website_types'].get(website_type, 0) + 1
        
        # Analyze languages and quality
        for content in processed_contents:
            language = content.language
            stats['languages'][language] = stats['languages'].get(language, 0) + 1
            stats['quality_scores'].append(content.quality_metrics.get('overall_quality_score', 0))
            stats['total_characters_processed'] += len(content.cleaned_text)
        
        # Analyze chunks
        for chunk in chunks:
            method = chunk.chunk_method
            stats['chunk_methods'][method] = stats['chunk_methods'].get(method, 0) + 1
            stats['total_tokens_processed'] += chunk.token_count
        
        # Calculate averages
        if chunks:
            stats['avg_chunk_size_tokens'] = stats['total_tokens_processed'] / len(chunks)
            stats['avg_chunk_size_characters'] = sum(len(chunk.text) for chunk in chunks) / len(chunks)
        
        if stats['quality_scores']:
            stats['avg_quality_score'] = sum(stats['quality_scores']) / len(stats['quality_scores'])
            stats['min_quality_score'] = min(stats['quality_scores'])
            stats['max_quality_score'] = max(stats['quality_scores'])
        
        return stats
    
    def _save_pipeline_results(self, result: PipelineResult, output_dir: str):
        """Save all pipeline results to files."""
        logger.info(f"Saving pipeline results to {output_dir}")
        
        try:
            # Save embeddings (main output)
            embeddings_path = os.path.join(output_dir, "embeddings.pkl")
            self.embedding_generator.save_embeddings(result.embeddings, embeddings_path)
            
            # Save pipeline summary
            summary_path = os.path.join(output_dir, "pipeline_summary.json")
            result.save_summary(summary_path)
            
            # Save detailed statistics
            stats_path = os.path.join(output_dir, "detailed_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(result.statistics, f, indent=2, default=str)
            
            # Save scraped content metadata
            scraped_metadata = []
            for content in result.scraped_contents:
                scraped_metadata.append({
                    'url': content.url,
                    'title': content.title,
                    'website_type': content.website_type.value,
                    'scraped_at': content.scraped_at,
                    'content_length': len(content.content),
                    'word_count': len(content.content.split())
                })
            
            metadata_path = os.path.join(output_dir, "scraped_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(scraped_metadata, f, indent=2)
            
            logger.success(f"All results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _log_pipeline_summary(self, result: PipelineResult):
        """Log comprehensive pipeline summary."""
        stats = result.statistics
        
        logger.info("PIPELINE SUMMARY:")
        logger.info(f"• Execution time: {stats['execution_time_formatted']}")
        logger.info(f"• Input URLs: {stats['input_urls']}")
        logger.info(f"• Scraped content: {stats['scraped_contents']}")
        logger.info(f"• Processed content: {stats['processed_contents']}")
        logger.info(f"• Generated chunks: {stats['total_chunks']}")
        logger.info(f"• Generated embeddings: {stats['total_embeddings']}")
        logger.info(f"• Overall success rate: {stats['overall_success_rate']:.1%}")
        
        if stats.get('avg_quality_score'):
            logger.info(f"• Average quality score: {stats['avg_quality_score']:.1f}/100")
        
        if stats.get('avg_chunk_size_tokens'):
            logger.info(f"• Average chunk size: {stats['avg_chunk_size_tokens']:.0f} tokens")
        
        logger.info(f"• Website types: {dict(stats['website_types'])}")
        logger.info(f"• Languages detected: {dict(stats['languages'])}")
        logger.info(f"• Chunk methods: {dict(stats['chunk_methods'])}")


# Convenience function for quick pipeline execution
def run_web_scraping_pipeline(urls: List[str], 
                             config: Optional[Config] = None,
                             **kwargs) -> PipelineResult:
    """
    Convenience function to run the complete pipeline with minimal setup.
    
    Args:
        urls: List of URLs to process
        config: Optional configuration (uses defaults if None)
        **kwargs: Additional arguments for pipeline.run_pipeline()
        
    Returns:
        PipelineResult with all outputs
    """
    if config is None:
        config = Config()
    
    pipeline = WebScrapingPipeline(config)
    return pipeline.run_pipeline(urls, **kwargs)