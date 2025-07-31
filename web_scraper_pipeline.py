"""
Web Scraper Pipeline: Main Orchestrator
Coordinates the entire process from web scraping to embedding generation.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass, asdict
import pandas as pd

from loguru import logger

from config import config
from scraper import WebScraper
from text_processor import TextProcessor
from chunker import HybridChunker, TextChunk
from embedder import EmbeddingGenerator, EmbeddingResult, EmbeddingAnalyzer


@dataclass
class PipelineConfig:
    """Configuration for the pipeline execution."""
    urls: List[str]
    output_dir: str = "output"
    save_intermediate: bool = True
    use_cache: bool = True
    max_workers: int = 1  # For future parallel processing
    
    def __post_init__(self):
        """Ensure output directory exists."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


@dataclass
class PipelineResults:
    """Results from the complete pipeline execution."""
    processed_urls: int
    successful_scrapes: int
    total_chunks: int
    total_embeddings: int
    execution_time: float
    output_files: Dict[str, str]
    performance_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class WebScraperPipeline:
    """
    Main pipeline class that orchestrates the entire web scraping process.
    
    This class coordinates:
    1. Web scraping with Selenium
    2. Text preprocessing and cleaning
    3. Hybrid chunking (Recursive + Content-Aware)
    4. Embedding generation
    """
    
    def __init__(self, pipeline_config: PipelineConfig = None):
        self.config = pipeline_config or PipelineConfig([])
        self.results = None
        
        # Initialize components
        self.scraper = None
        self.text_processor = TextProcessor()
        self.embedder = EmbeddingGenerator()
        
        # Pipeline state
        self.scraped_data = []
        self.processed_data = []
        self.chunked_data = []
        self.embedding_data = []
        
        # Performance tracking
        self.start_time = None
        self.stage_times = {}
        
    def _log_stage_start(self, stage_name: str) -> None:
        """Log the start of a pipeline stage."""
        logger.info(f"=== Starting {stage_name} ===")
        self.stage_times[stage_name] = time.time()
    
    def _log_stage_end(self, stage_name: str) -> None:
        """Log the end of a pipeline stage."""
        duration = time.time() - self.stage_times[stage_name]
        logger.success(f"=== Completed {stage_name} in {duration:.2f}s ===")
        self.stage_times[f"{stage_name}_duration"] = duration
    
    def scrape_websites(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape websites and extract content.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of scraped data dictionaries
        """
        self._log_stage_start("Web Scraping")
        
        scraped_results = []
        
        try:
            with WebScraper() as scraper:
                self.scraper = scraper
                
                for i, url in enumerate(urls):
                    logger.info(f"Scraping URL {i+1}/{len(urls)}: {url}")
                    
                    try:
                        result = scraper.scrape_url(url)
                        if result:
                            scraped_results.append(result)
                            logger.success(f"Successfully scraped: {url}")
                        else:
                            logger.warning(f"Failed to scrape: {url}")
                    
                    except Exception as e:
                        logger.error(f"Error scraping {url}: {str(e)}")
                        continue
                    
                    # Add delay between requests
                    if i < len(urls) - 1:
                        time.sleep(2)
        
        except Exception as e:
            logger.error(f"Critical error in scraping stage: {str(e)}")
            raise
        
        finally:
            self.scraper = None
        
        self.scraped_data = scraped_results
        
        # Save intermediate results if requested
        if self.config.save_intermediate:
            output_file = Path(self.config.output_dir) / "scraped_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(scraped_results, f, indent=2, default=str)
            logger.info(f"Saved scraped data to {output_file}")
        
        self._log_stage_end("Web Scraping")
        return scraped_results
    
    def process_content(self, scraped_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process and clean the scraped content.
        
        Args:
            scraped_data: List of scraped data dictionaries
            
        Returns:
            List of processed data dictionaries
        """
        self._log_stage_start("Content Processing")
        
        processed_results = []
        
        for i, data in enumerate(scraped_data):
            logger.info(f"Processing content {i+1}/{len(scraped_data)}")
            
            try:
                # Extract necessary information
                content = data.get('content', '')
                site_type = data.get('site_type', 'general')
                metadata = data.get('metadata', {})
                
                # Process the content
                processed_result = self.text_processor.process_content(
                    content=content,
                    site_type=site_type,
                    metadata=metadata
                )
                
                # Add original data
                processed_result.update({
                    'url': data.get('url', ''),
                    'site_type': site_type,
                    'scraped_metadata': metadata,
                    'scraped_at': data.get('scraped_at', time.time())
                })
                
                processed_results.append(processed_result)
                
            except Exception as e:
                logger.error(f"Error processing content from {data.get('url', 'unknown')}: {str(e)}")
                continue
        
        self.processed_data = processed_results
        
        # Save intermediate results if requested
        if self.config.save_intermediate:
            output_file = Path(self.config.output_dir) / "processed_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_results, f, indent=2, default=str)
            logger.info(f"Saved processed data to {output_file}")
        
        self._log_stage_end("Content Processing")
        return processed_results
    
    def chunk_content(self, processed_data: List[Dict[str, Any]]) -> List[TextChunk]:
        """
        Chunk the processed content using hybrid chunking.
        
        Args:
            processed_data: List of processed data dictionaries
            
        Returns:
            List of TextChunk objects
        """
        self._log_stage_start("Content Chunking")
        
        all_chunks = []
        chunk_id_counter = 0
        
        for i, data in enumerate(processed_data):
            logger.info(f"Chunking content {i+1}/{len(processed_data)}")
            
            try:
                # Extract necessary information
                content = data.get('processed_content', '')
                site_type = data.get('site_type', 'general')
                
                if not content or len(content.strip()) < config.chunking.min_chunk_size:
                    logger.warning(f"Skipping empty or too short content from {data.get('url', 'unknown')}")
                    continue
                
                # Prepare metadata for chunks
                chunk_metadata = {
                    'source_url': data.get('url', ''),
                    'site_type': site_type,
                    'original_title': data.get('scraped_metadata', {}).get('title', ''),
                    'processing_stats': data.get('statistics', {}),
                    'key_phrases': data.get('key_phrases', []),
                    'source_index': i
                }
                
                # Create chunker for this site type
                chunker = HybridChunker(site_type=site_type)
                
                # Chunk the content
                chunks = chunker.chunk(content, metadata=chunk_metadata)
                
                # Update chunk IDs to be globally unique
                for chunk in chunks:
                    chunk.chunk_id = chunk_id_counter
                    chunk_id_counter += 1
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error chunking content from {data.get('url', 'unknown')}: {str(e)}")
                continue
        
        self.chunked_data = all_chunks
        
        # Save intermediate results if requested
        if self.config.save_intermediate:
            # Convert chunks to serializable format
            serializable_chunks = []
            for chunk in all_chunks:
                chunk_data = {
                    'chunk_id': chunk.chunk_id,
                    'text': chunk.text,
                    'start_position': chunk.start_position,
                    'end_position': chunk.end_position,
                    'metadata': chunk.metadata
                }
                serializable_chunks.append(chunk_data)
            
            output_file = Path(self.config.output_dir) / "chunked_data.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_chunks, f, indent=2, default=str)
            logger.info(f"Saved chunked data to {output_file}")
        
        self._log_stage_end("Content Chunking")
        return all_chunks
    
    def generate_embeddings(self, chunks: List[TextChunk]) -> List[EmbeddingResult]:
        """
        Generate embeddings for all chunks.
        
        Args:
            chunks: List of TextChunk objects
            
        Returns:
            List of EmbeddingResult objects
        """
        self._log_stage_start("Embedding Generation")
        
        if not chunks:
            logger.warning("No chunks provided for embedding generation")
            return []
        
        try:
            # Generate embeddings
            embeddings = self.embedder.generate_embeddings(
                chunks=chunks,
                use_cache=self.config.use_cache
            )
            
            self.embedding_data = embeddings
            
            # Save embeddings if requested
            if self.config.save_intermediate:
                output_file = Path(self.config.output_dir) / "embeddings.pkl"
                self.embedder.save_embeddings(embeddings, str(output_file))
                
                # Also save embeddings in JSON format (without the actual vectors for readability)
                json_data = []
                for emb in embeddings:
                    emb_data = {
                        'chunk_id': emb.chunk_id,
                        'text': emb.text[:200] + '...' if len(emb.text) > 200 else emb.text,
                        'embedding_dimension': len(emb.embedding),
                        'metadata': emb.metadata,
                        'model_name': emb.model_name,
                        'embedding_time': emb.embedding_time
                    }
                    json_data.append(emb_data)
                
                json_file = Path(self.config.output_dir) / "embeddings_info.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2, default=str)
                logger.info(f"Saved embedding info to {json_file}")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
        
        self._log_stage_end("Embedding Generation")
        return embeddings
    
    def run_pipeline(self, urls: List[str] = None) -> PipelineResults:
        """
        Run the complete pipeline from URLs to embeddings.
        
        Args:
            urls: List of URLs to process (optional, uses config if not provided)
            
        Returns:
            PipelineResults object with execution summary
        """
        self.start_time = time.time()
        
        # Use provided URLs or fall back to config
        if urls:
            self.config.urls = urls
        
        if not self.config.urls:
            raise ValueError("No URLs provided for processing")
        
        logger.info(f"Starting Web Scraper Pipeline with {len(self.config.urls)} URLs")
        logger.info(f"Output directory: {self.config.output_dir}")
        logger.info(f"Configuration: {config.embedding.model_name} on {config.embedding.device}")
        
        try:
            # Stage 1: Web Scraping
            scraped_data = self.scrape_websites(self.config.urls)
            
            if not scraped_data:
                logger.error("No data was successfully scraped")
                raise RuntimeError("Pipeline failed: No data scraped")
            
            # Stage 2: Content Processing
            processed_data = self.process_content(scraped_data)
            
            if not processed_data:
                logger.error("No data was successfully processed")
                raise RuntimeError("Pipeline failed: No data processed")
            
            # Stage 3: Content Chunking
            chunks = self.chunk_content(processed_data)
            
            if not chunks:
                logger.error("No chunks were generated")
                raise RuntimeError("Pipeline failed: No chunks generated")
            
            # Stage 4: Embedding Generation
            embeddings = self.generate_embeddings(chunks)
            
            if not embeddings:
                logger.error("No embeddings were generated")
                raise RuntimeError("Pipeline failed: No embeddings generated")
            
            # Calculate final results
            total_time = time.time() - self.start_time
            
            # Get performance statistics
            embedding_stats = self.embedder.get_performance_stats()
            
            # Analyze embeddings
            analyzer = EmbeddingAnalyzer()
            embedding_analysis = analyzer.analyze_embeddings(embeddings)
            
            # Create comprehensive results
            self.results = PipelineResults(
                processed_urls=len(self.config.urls),
                successful_scrapes=len(scraped_data),
                total_chunks=len(chunks),
                total_embeddings=len(embeddings),
                execution_time=total_time,
                output_files={
                    'scraped_data': str(Path(self.config.output_dir) / "scraped_data.json"),
                    'processed_data': str(Path(self.config.output_dir) / "processed_data.json"),
                    'chunked_data': str(Path(self.config.output_dir) / "chunked_data.json"),
                    'embeddings': str(Path(self.config.output_dir) / "embeddings.pkl"),
                    'embeddings_info': str(Path(self.config.output_dir) / "embeddings_info.json"),
                    'pipeline_results': str(Path(self.config.output_dir) / "pipeline_results.json")
                },
                performance_stats={
                    'stage_times': self.stage_times,
                    'embedding_stats': embedding_stats,
                    'embedding_analysis': embedding_analysis
                }
            )
            
            # Save complete results
            results_file = Path(self.config.output_dir) / "pipeline_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results.to_dict(), f, indent=2, default=str)
            
            # Create summary report
            self._create_summary_report()
            
            logger.success(f"Pipeline completed successfully in {total_time:.2f}s")
            logger.info(f"Processed {self.results.processed_urls} URLs")
            logger.info(f"Generated {self.results.total_embeddings} embeddings from {self.results.total_chunks} chunks")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def _create_summary_report(self) -> None:
        """Create a human-readable summary report."""
        if not self.results:
            return
        
        report_lines = [
            "="*60,
            "WEB SCRAPER PIPELINE - EXECUTION SUMMARY",
            "="*60,
            "",
            f"Execution Time: {self.results.execution_time:.2f} seconds",
            f"URLs Processed: {self.results.processed_urls}",
            f"Successful Scrapes: {self.results.successful_scrapes}",
            f"Total Chunks Generated: {self.results.total_chunks}",
            f"Total Embeddings Generated: {self.results.total_embeddings}",
            "",
            "STAGE BREAKDOWN:",
            "-" * 30
        ]
        
        # Add stage times
        for stage, duration in self.stage_times.items():
            if stage.endswith('_duration'):
                stage_name = stage.replace('_duration', '')
                report_lines.append(f"{stage_name:.<25} {duration:.2f}s")
        
        report_lines.extend([
            "",
            "PERFORMANCE METRICS:",
            "-" * 30
        ])
        
        # Add embedding performance
        emb_stats = self.results.performance_stats.get('embedding_stats', {})
        if emb_stats:
            report_lines.extend([
                f"Embedding Model: {config.embedding.model_name}",
                f"Device Used: {emb_stats.get('device', 'unknown')}",
                f"Average Time per Embedding: {emb_stats.get('avg_time_per_embedding', 0):.4f}s",
                f"Cache Hit Rate: {emb_stats.get('cache_hit_rate', 0):.2%}",
            ])
        
        # Add content analysis
        analysis = self.results.performance_stats.get('embedding_analysis', {})
        content_analysis = analysis.get('content_analysis', {})
        if content_analysis:
            report_lines.extend([
                "",
                "CONTENT ANALYSIS:",
                "-" * 30
            ])
            
            site_dist = content_analysis.get('site_type_distribution', {})
            for site_type, count in site_dist.items():
                report_lines.append(f"{site_type.capitalize():.<20} {count} chunks")
            
            text_stats = content_analysis.get('text_length_stats', {})
            if text_stats:
                report_lines.extend([
                    "",
                    f"Average Text Length: {text_stats.get('mean', 0):.0f} characters",
                    f"Text Length Range: {text_stats.get('min', 0)} - {text_stats.get('max', 0)}"
                ])
        
        report_lines.extend([
            "",
            "OUTPUT FILES:",
            "-" * 30
        ])
        
        for file_type, filepath in self.results.output_files.items():
            if Path(filepath).exists():
                report_lines.append(f"{file_type:.<25} {filepath}")
        
        report_lines.extend([
            "",
            "="*60,
            ""
        ])
        
        # Write report to file
        report_file = Path(self.config.output_dir) / "execution_summary.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Also log the summary
        logger.info(f"Execution summary saved to {report_file}")
        for line in report_lines:
            logger.info(line)


# Example usage and demonstration
if __name__ == "__main__":
    # Example URLs for testing
    test_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://www.microsoft.com/en-us/ai",
        "https://www.bbc.com/news/technology"
    ]
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        urls=test_urls,
        output_dir="example_output",
        save_intermediate=True,
        use_cache=True
    )
    
    # Initialize and run pipeline
    pipeline = WebScraperPipeline(pipeline_config)
    
    try:
        results = pipeline.run_pipeline()
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Processed {results.successful_scrapes} websites")
        print(f"📝 Generated {results.total_chunks} chunks")
        print(f"🔢 Created {results.total_embeddings} embeddings")
        print(f"⏱️  Total time: {results.execution_time:.2f} seconds")
        print(f"📁 Results saved to: {pipeline_config.output_dir}/")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}")