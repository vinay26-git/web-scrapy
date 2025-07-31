#!/usr/bin/env python3
"""
Comprehensive examples showing different ways to use the Web Scraper pipeline.
"""
import sys
import os

# Add src to path
sys.path.append('src')

from src.config import Config, WebsiteType
from src.pipeline import WebScrapingPipeline, run_web_scraping_pipeline
from src.scraper import WebScraper
from src.preprocessor import TextPreprocessor
from src.chunker import HybridChunker
from src.embeddings import EmbeddingGenerator
from loguru import logger


def example_1_simple_usage():
    """Example 1: Simplest usage with default settings."""
    print("\n" + "="*50)
    print("EXAMPLE 1: Simple Usage")
    print("="*50)
    
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://techcrunch.com/category/artificial-intelligence/"
    ]
    
    # One-liner pipeline execution
    result = run_web_scraping_pipeline(urls, output_dir="output/example1")
    
    print(f"✅ Processed {len(result.embeddings)} embeddings in {result.execution_time:.2f}s")
    return result


def example_2_custom_configuration():
    """Example 2: Custom configuration for specific needs."""
    print("\n" + "="*50)
    print("EXAMPLE 2: Custom Configuration")
    print("="*50)
    
    # Create custom configuration
    config = Config()
    
    # Customize scraping settings
    config.scraping.headless = True
    config.scraping.request_delay = 2.0  # Be more respectful
    config.scraping.max_retries = 5
    
    # Customize chunking for smaller chunks
    config.chunking.chunk_size = 500
    config.chunking.chunk_overlap = 100
    config.chunking.sentence_similarity_threshold = 0.8
    
    # Customize embeddings for faster processing
    config.embedding.batch_size = 8
    config.embedding.model_name = "sentence-transformers/all-MiniLM-L6-v2"
    
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.bbc.com/news/technology"
    ]
    
    result = run_web_scraping_pipeline(
        urls, 
        config=config,
        min_quality_score=60.0,  # Higher quality threshold
        output_dir="output/example2"
    )
    
    print(f"✅ Custom config: {len(result.embeddings)} embeddings generated")
    return result


def example_3_step_by_step():
    """Example 3: Step-by-step processing with intermediate inspection."""
    print("\n" + "="*50)
    print("EXAMPLE 3: Step-by-Step Processing")
    print("="*50)
    
    config = Config()
    pipeline = WebScrapingPipeline(config)
    
    urls = ["https://en.wikipedia.org/wiki/Natural_language_processing"]
    
    # Step 1: Scraping
    print("Step 1: Scraping...")
    scraped_contents = pipeline.scrape_urls(urls)
    print(f"Scraped {len(scraped_contents)} pages")
    
    # Inspect scraped content
    for content in scraped_contents:
        print(f"  • {content.title} ({len(content.content)} chars)")
    
    # Step 2: Preprocessing
    print("\nStep 2: Preprocessing...")
    processed_contents = pipeline.process_contents(scraped_contents, min_quality_score=50.0)
    print(f"Processed {len(processed_contents)} contents")
    
    # Inspect quality metrics
    for content in processed_contents:
        quality = content.quality_metrics['overall_quality_score']
        print(f"  • Quality: {quality:.1f}/100, Language: {content.language}")
    
    # Step 3: Chunking
    print("\nStep 3: Chunking...")
    chunks = pipeline.chunk_contents(processed_contents)
    print(f"Created {len(chunks)} chunks")
    
    # Inspect chunks
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"  • Chunk {i+1}: {len(chunk.text)} chars, {chunk.token_count} tokens")
        print(f"    Method: {chunk.chunk_method}")
    
    # Step 4: Embeddings
    print("\nStep 4: Generating embeddings...")
    embeddings = pipeline.generate_embeddings(chunks)
    print(f"Generated {len(embeddings)} embeddings")
    
    # Inspect embeddings
    if embeddings:
        stats = pipeline.embedding_generator.get_embedding_stats(embeddings)
        print(f"  • Dimension: {stats['dimension']}")
        print(f"  • Mean magnitude: {stats['mean_magnitude']:.3f}")
    
    return embeddings


def example_4_website_specific():
    """Example 4: Website type-specific processing."""
    print("\n" + "="*50)
    print("EXAMPLE 4: Website-Specific Processing")
    print("="*50)
    
    # Define URLs with explicit website types
    urls_and_types = [
        ("https://en.wikipedia.org/wiki/Web_scraping", WebsiteType.WIKIPEDIA),
        ("https://techcrunch.com/category/startups/", WebsiteType.NEWS),
        ("https://www.shopify.com/blog/ecommerce-business-blueprint", WebsiteType.BUSINESS)
    ]
    
    urls = [url for url, _ in urls_and_types]
    website_types = [wtype for _, wtype in urls_and_types]
    
    result = run_web_scraping_pipeline(
        urls,
        website_types=website_types,
        output_dir="output/example4"
    )
    
    # Analyze by website type
    type_counts = {}
    for content in result.scraped_contents:
        wtype = content.website_type.value
        type_counts[wtype] = type_counts.get(wtype, 0) + 1
    
    print(f"✅ Website type distribution: {type_counts}")
    return result


def example_5_error_handling():
    """Example 5: Robust error handling and recovery."""
    print("\n" + "="*50)
    print("EXAMPLE 5: Error Handling")
    print("="*50)
    
    # Mix of valid and invalid URLs
    urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",  # Valid
        "https://invalid-url-that-does-not-exist.com/page",  # Invalid
        "https://httpstat.us/404",  # Valid URL but 404
        "https://en.wikipedia.org/wiki/Data_science",  # Valid
    ]
    
    try:
        result = run_web_scraping_pipeline(
            urls,
            min_quality_score=30.0,  # Lower threshold for robustness
            output_dir="output/example5"
        )
        
        success_rate = len(result.scraped_contents) / len(urls)
        print(f"✅ Successfully processed {success_rate:.1%} of URLs")
        print(f"   Final embeddings: {len(result.embeddings)}")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None


def example_6_batch_processing():
    """Example 6: Large batch processing with progress tracking."""
    print("\n" + "="*50)
    print("EXAMPLE 6: Batch Processing")
    print("="*50)
    
    # Larger set of URLs for batch processing
    wikipedia_topics = [
        "Machine_learning", "Deep_learning", "Neural_network", 
        "Natural_language_processing", "Computer_vision"
    ]
    
    urls = [f"https://en.wikipedia.org/wiki/{topic}" for topic in wikipedia_topics]
    
    # Configure for batch processing
    config = Config()
    config.embedding.batch_size = 32  # Larger batches
    config.scraping.request_delay = 1.5  # Respectful scraping
    
    result = run_web_scraping_pipeline(
        urls,
        config=config,
        output_dir="output/example6_batch"
    )
    
    print(f"✅ Batch processed {len(urls)} URLs → {len(result.embeddings)} embeddings")
    
    # Show processing efficiency
    stats = result.statistics
    print(f"   Processing time: {stats['execution_time_formatted']}")
    print(f"   Avg time per URL: {stats['execution_time_seconds']/len(urls):.2f}s")
    
    return result


def example_7_embedding_analysis():
    """Example 7: Advanced embedding analysis and similarity."""
    print("\n" + "="*50)
    print("EXAMPLE 7: Embedding Analysis")
    print("="*50)
    
    urls = [
        "https://en.wikipedia.org/wiki/Python_(programming_language)",
        "https://en.wikipedia.org/wiki/JavaScript",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    result = run_web_scraping_pipeline(urls, output_dir="output/example7")
    
    if result.embeddings:
        # Calculate embedding statistics
        embedder = EmbeddingGenerator(Config())
        stats = embedder.get_embedding_stats(result.embeddings)
        
        print("📊 Embedding Analysis:")
        print(f"   Model: {stats['model_name']}")
        print(f"   Dimensions: {stats['dimension']}")
        print(f"   Total embeddings: {stats['count']}")
        print(f"   Avg magnitude: {stats['mean_magnitude']:.3f}")
        print(f"   Source URLs: {len(stats['source_urls'])}")
        
        # Simple similarity analysis between first two embeddings
        if len(result.embeddings) >= 2:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            emb1 = result.embeddings[0].embedding.reshape(1, -1)
            emb2 = result.embeddings[1].embedding.reshape(1, -1)
            similarity = cosine_similarity(emb1, emb2)[0][0]
            
            print(f"   Similarity (first two chunks): {similarity:.3f}")
    
    return result


def main():
    """Run all examples."""
    logger.remove()
    logger.add(sys.stdout, level="WARNING")  # Reduce logging for examples
    
    print("🚀 Web Scraper Examples")
    print("This will demonstrate various usage patterns of the Web Scraper pipeline.")
    
    # Create output directories
    os.makedirs("output", exist_ok=True)
    
    examples = [
        example_1_simple_usage,
        example_2_custom_configuration,
        example_3_step_by_step,
        example_4_website_specific,
        example_5_error_handling,
        example_6_batch_processing,
        example_7_embedding_analysis
    ]
    
    results = []
    
    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n🔄 Running Example {i}...")
            result = example_func()
            results.append(result)
            print(f"✅ Example {i} completed successfully")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
            results.append(None)
    
    # Summary
    successful = sum(1 for r in results if r is not None)
    print(f"\n📋 Summary: {successful}/{len(examples)} examples completed successfully")
    
    if successful > 0:
        total_embeddings = sum(len(r.embeddings) for r in results if r is not None)
        print(f"🎯 Total embeddings generated across all examples: {total_embeddings}")


if __name__ == "__main__":
    main()