"""
Comprehensive Example: Web Scraping Pipeline
Demonstrates the complete pipeline for business, news, and Wikipedia websites
Author: Full-Stack Developer (Google, Amazon, Microsoft experience)
"""

import time
from pathlib import Path
from loguru import logger

from pipeline import WebScrapingPipeline


def main():
    """Main example function demonstrating the complete pipeline."""
    
    # Configure logging
    logger.add("pipeline.log", rotation="10 MB", level="INFO")
    
    # Example URLs for different website types
    urls = {
        'business': [
            "https://www.apple.com",
            "https://www.microsoft.com",
            "https://www.google.com"
        ],
        'news': [
            "https://www.bbc.com/news",
            "https://www.reuters.com",
            "https://www.theguardian.com"
        ],
        'wikipedia': [
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Deep_learning"
        ]
    }
    
    # Configuration for different use cases
    configurations = {
        'high_quality': {
            'chunk_config': {
                'max_chunk_size': 1000,
                'min_chunk_size': 150,
                'overlap_size': 100,
                'semantic_threshold': 0.8
            },
            'embedding_config': {
                'model_name': 'all-mpnet-base-v2',
                'batch_size': 16,
                'max_length': 512
            }
        },
        'fast_processing': {
            'chunk_config': {
                'max_chunk_size': 800,
                'min_chunk_size': 100,
                'overlap_size': 50,
                'semantic_threshold': 0.6
            },
            'embedding_config': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 32,
                'max_length': 384
            }
        },
        'memory_efficient': {
            'chunk_config': {
                'max_chunk_size': 600,
                'min_chunk_size': 80,
                'overlap_size': 30,
                'semantic_threshold': 0.5
            },
            'embedding_config': {
                'model_name': 'all-MiniLM-L6-v2',
                'batch_size': 8,
                'max_length': 256
            }
        }
    }
    
    # Choose configuration based on your needs
    config_name = 'fast_processing'  # Change this as needed
    config = configurations[config_name]
    
    print(f"🚀 Starting Web Scraping Pipeline with {config_name} configuration")
    print(f"📊 Configuration: {config}")
    
    # Initialize pipeline
    pipeline = WebScrapingPipeline(
        output_dir=f"output_{config_name}",
        chunk_config=config['chunk_config'],
        embedding_config=config['embedding_config'],
        save_intermediate=True
    )
    
    try:
        # Process URLs by category
        all_results = []
        
        for category, category_urls in urls.items():
            print(f"\n📋 Processing {category} websites...")
            print(f"URLs: {category_urls}")
            
            # Process URLs in this category
            results = pipeline.process_multiple_urls(category_urls)
            all_results.extend(results)
            
            print(f"✅ Completed {category}: {len(results)}/{len(category_urls)} successful")
            
            # Print detailed results for this category
            for result in results:
                print(f"  📄 {result.url}")
                print(f"     Title: {result.scraped_content.title[:50]}...")
                print(f"     Content: {len(result.scraped_content.content)} chars")
                print(f"     Chunks: {len(result.chunks)}")
                print(f"     Embeddings: {len(result.embeddings)}")
                print(f"     Time: {result.processing_time:.2f}s")
        
        # Get overall statistics
        print(f"\n📈 Overall Pipeline Statistics:")
        stats = pipeline.get_pipeline_statistics()
        
        print(f"Total Results: {stats['total_results']}")
        print(f"Total Chunks: {stats['total_chunks']}")
        print(f"Total Embeddings: {stats['total_embeddings']}")
        print(f"Website Types: {stats['website_types']}")
        
        if 'avg_processing_time' in stats:
            print(f"Average Processing Time: {stats['avg_processing_time']:.2f}s")
            print(f"Min Processing Time: {stats['min_processing_time']:.2f}s")
            print(f"Max Processing Time: {stats['max_processing_time']:.2f}s")
        
        # Demonstrate loading a saved result
        if all_results:
            print(f"\n🔄 Demonstrating result loading...")
            first_result = all_results[0]
            loaded_result = pipeline.load_pipeline_result(first_result.url)
            
            if loaded_result:
                print(f"✅ Successfully loaded result for {loaded_result.url}")
                print(f"   Chunks: {len(loaded_result.chunks)}")
                print(f"   Embeddings: {len(loaded_result.embeddings)}")
            else:
                print(f"❌ Failed to load result for {first_result.url}")
        
        # Demonstrate embedding similarity search
        if all_results and all_results[0].embeddings:
            print(f"\n🔍 Demonstrating similarity search...")
            
            # Get first embedding as query
            query_embedding = all_results[0].embeddings[0]
            
            # Collect all embeddings from all results
            all_embeddings = []
            for result in all_results:
                all_embeddings.extend(result.embeddings)
            
            # Find similar chunks
            similar_chunks = pipeline.embedding_generator.find_similar_chunks(
                query_embedding, all_embeddings, top_k=3
            )
            
            print(f"Top 3 similar chunks to '{query_embedding.chunk_id}':")
            for i, (emb, similarity) in enumerate(similar_chunks, 1):
                print(f"  {i}. {emb.chunk_id} (similarity: {similarity:.3f})")
        
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📁 Results saved in: {pipeline.output_dir}")
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {e}")
        print(f"❌ Pipeline failed: {e}")
        
    finally:
        # Cleanup
        pipeline.cleanup()
        print(f"🧹 Cleanup completed")


def demonstrate_individual_components():
    """Demonstrate individual pipeline components."""
    
    print("\n🔧 Demonstrating Individual Components")
    
    # 1. Web Scraping
    print("\n1️⃣ Web Scraping Component:")
    from web_scraper import WebScraper
    
    with WebScraper(headless=True) as scraper:
        test_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        scraped_content = scraper.scrape_url(test_url)
        
        if scraped_content:
            print(f"   ✅ Successfully scraped: {scraped_content.url}")
            print(f"   📄 Title: {scraped_content.title}")
            print(f"   📊 Content length: {len(scraped_content.content)}")
            print(f"   🏷️  Website type: {scraped_content.website_type}")
        else:
            print(f"   ❌ Failed to scrape: {test_url}")
    
    # 2. Chunking
    print("\n2️⃣ Chunking Component:")
    from chunking import HybridChunker
    
    sample_text = """
    Python is a high-level, interpreted programming language. It was created by Guido van Rossum and first released in 1991.
    
    Python's design philosophy emphasizes code readability with its notable use of significant whitespace. Its language constructs and object-oriented approach aim to help programmers write clear, logical code for small and large-scale projects.
    
    Python features a dynamic type system and automatic memory management. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.
    """
    
    chunker = HybridChunker(max_chunk_size=200, min_chunk_size=50)
    chunks = chunker.chunk_text(sample_text, domain='wikipedia')
    
    print(f"   ✅ Created {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        print(f"   📄 Chunk {i}: {len(chunk.text)} chars, score: {chunk.semantic_score:.3f}")
    
    # 3. Embedding Generation
    print("\n3️⃣ Embedding Generation Component:")
    from embeddings import EmbeddingGenerator
    
    embedding_gen = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    embeddings = embedding_gen.generate_embeddings_batch(chunks, domains=['wikipedia'] * len(chunks))
    
    print(f"   ✅ Generated {len(embeddings)} embeddings")
    for i, emb in enumerate(embeddings, 1):
        print(f"   🔢 Embedding {i}: {emb.embedding_dim} dimensions, {emb.generation_time:.3f}s")


if __name__ == "__main__":
    print("🌐 Web Scraping Pipeline - Complete Example")
    print("=" * 50)
    
    # Demonstrate individual components first
    demonstrate_individual_components()
    
    # Run the complete pipeline
    main()
    
    print("\n✨ Example completed successfully!")