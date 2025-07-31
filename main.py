#!/usr/bin/env python3
"""
Main script for the Web Scraper pipeline.
Demonstrates how to use the complete pipeline for scraping, chunking, and embedding generation.
"""

import sys
import argparse
from pathlib import Path
from typing import List
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import WebScrapingPipeline, create_pipeline_from_config
from src.config import PipelineConfig, ScrapingConfig, ChunkingConfig, EmbeddingConfig


def main():
    """Main function to run the web scraping pipeline."""
    parser = argparse.ArgumentParser(description="Web Scraper Pipeline")
    parser.add_argument("--urls", nargs="+", help="URLs to scrape")
    parser.add_argument("--url-file", type=str, help="File containing URLs (one per line)")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--headless", action="store_true", default=True, help="Run browser in headless mode")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for text splitting")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", 
                       help="Embedding model to use")
    parser.add_argument("--example", action="store_true", help="Run with example URLs")
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("pipeline.log", rotation="10 MB", retention="7 days")
    
    # Determine URLs to process
    urls = []
    
    if args.example:
        urls = [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://www.bbc.com/news/technology",
            "https://www.microsoft.com/en-us/about"
        ]
        logger.info("Running with example URLs")
    elif args.urls:
        urls = args.urls
    elif args.url_file:
        with open(args.url_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        logger.error("No URLs provided. Use --urls, --url-file, or --example")
        return 1
    
    if not urls:
        logger.error("No valid URLs found")
        return 1
    
    logger.info(f"Processing {len(urls)} URLs")
    
    # Create pipeline
    if args.config:
        pipeline = create_pipeline_from_config(args.config)
    else:
        # Create custom configuration
        config = PipelineConfig(
            output_dir=args.output_dir,
            scraping=ScrapingConfig(headless=args.headless),
            chunking=ChunkingConfig(chunk_size=args.chunk_size),
            embedding=EmbeddingConfig(model_name=args.model)
        )
        pipeline = WebScrapingPipeline(config)
    
    try:
        # Process URLs
        results = pipeline.process_urls(urls)
        
        if not results:
            logger.error("No results generated")
            return 1
        
        # Save results
        pipeline.save_results(results, args.output_dir)
        
        # Print summary
        total_chunks = sum(len(r['chunks']) for r in results)
        print(f"\n{'='*60}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"URLs processed: {len(results)}/{len(urls)}")
        print(f"Total chunks created: {total_chunks}")
        print(f"Output directory: {args.output_dir}")
        print(f"Embedding model: {pipeline.config.embedding.model_name}")
        print(f"Embedding dimension: {pipeline.embedding_generator.get_embedding_dimension()}")
        print(f"{'='*60}")
        
        # Show detailed statistics
        for i, result in enumerate(results):
            stats = result['statistics']
            print(f"\nURL {i+1}: {result['url']}")
            print(f"  - Text length: {stats['text_statistics']['char_count']:,} characters")
            print(f"  - Chunks: {stats['chunk_statistics']['total_chunks']}")
            print(f"  - Avg chunk size: {stats['chunk_statistics']['avg_chunk_size']:.0f} characters")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return 1
    finally:
        pipeline.cleanup()


def run_quick_example():
    """Run a quick example with minimal setup."""
    print("Running quick example...")
    
    # Simple configuration
    config = PipelineConfig(
        scraping=ScrapingConfig(headless=True),
        chunking=ChunkingConfig(chunk_size=500),
        embedding=EmbeddingConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )
    
    pipeline = WebScrapingPipeline(config)
    
    # Single URL example
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    
    try:
        result = pipeline.process_url(url)
        
        if result:
            print(f"\nSuccessfully processed: {url}")
            print(f"Chunks created: {len(result['chunks'])}")
            print(f"Embedding dimension: {len(result['chunks'][0]['embedding'])}")
            
            # Show first chunk
            first_chunk = result['chunks'][0]
            print(f"\nFirst chunk preview:")
            print(f"Text: {first_chunk['text'][:200]}...")
            print(f"Embedding: {first_chunk['embedding'][:5]}...")
        else:
            print("Failed to process URL")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.cleanup()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run quick example
        run_quick_example()
    else:
        # Run with command line arguments
        sys.exit(main())