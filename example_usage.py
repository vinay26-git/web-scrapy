"""
Example Usage of the Web Scraper Pipeline

This script demonstrates how to use the web scraper pipeline
to scrape different types of websites and generate embeddings.
"""

from web_scraper_pipeline import WebScraperPipeline, PipelineConfig
from loguru import logger
import sys

def example_basic_usage():
    """Basic usage example with a few URLs."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Define URLs to scrape
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://www.google.com/ai/",
        "https://techcrunch.com/category/artificial-intelligence/"
    ]
    
    # Create pipeline configuration
    config = PipelineConfig(
        urls=urls,
        output_dir="basic_example_output",
        save_intermediate=True,
        use_cache=True
    )
    
    # Initialize and run pipeline
    pipeline = WebScraperPipeline(config)
    
    try:
        results = pipeline.run_pipeline()
        
        print(f"\n✅ SUCCESS!")
        print(f"📊 Processed {results.successful_scrapes}/{results.processed_urls} websites")
        print(f"📝 Generated {results.total_chunks} chunks")
        print(f"🔢 Created {results.total_embeddings} embeddings")
        print(f"⏱️  Total time: {results.execution_time:.2f} seconds")
        print(f"📁 Results saved to: {config.output_dir}/")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {str(e)}")
        return None

def example_wikipedia_focused():
    """Example focused on Wikipedia articles."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Wikipedia-Focused Scraping")
    print("="*60)
    
    # Wikipedia URLs on AI topics
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Natural_language_processing",
        "https://en.wikipedia.org/wiki/Computer_vision",
        "https://en.wikipedia.org/wiki/Neural_network"
    ]
    
    config = PipelineConfig(
        urls=urls,
        output_dir="wikipedia_example_output",
        save_intermediate=True,
        use_cache=True
    )
    
    pipeline = WebScraperPipeline(config)
    
    try:
        results = pipeline.run_pipeline()
        
        print(f"\n✅ Wikipedia scraping completed!")
        print(f"📚 Processed {results.successful_scrapes} Wikipedia articles")
        print(f"📝 Generated {results.total_chunks} chunks")
        print(f"🔢 Created {results.total_embeddings} embeddings")
        
        # Analyze the results
        if results.performance_stats:
            analysis = results.performance_stats.get('embedding_analysis', {})
            content_analysis = analysis.get('content_analysis', {})
            
            if content_analysis:
                print(f"\n📊 Content Analysis:")
                site_dist = content_analysis.get('site_type_distribution', {})
                for site_type, count in site_dist.items():
                    print(f"   {site_type}: {count} chunks")
                
                text_stats = content_analysis.get('text_length_stats', {})
                if text_stats:
                    print(f"   Average chunk length: {text_stats.get('mean', 0):.0f} characters")
        
        return results
        
    except Exception as e:
        print(f"❌ Wikipedia scraping failed: {str(e)}")
        return None

def example_news_focused():
    """Example focused on news websites."""
    print("\n" + "="*60)
    print("EXAMPLE 3: News-Focused Scraping")
    print("="*60)
    
    # News URLs (note: these might be dynamic, so results may vary)
    urls = [
        "https://www.bbc.com/news/technology",
        "https://techcrunch.com/",
        "https://www.wired.com/category/artificial-intelligence/",
    ]
    
    config = PipelineConfig(
        urls=urls,
        output_dir="news_example_output",
        save_intermediate=True,
        use_cache=True
    )
    
    pipeline = WebScraperPipeline(config)
    
    try:
        results = pipeline.run_pipeline()
        
        print(f"\n✅ News scraping completed!")
        print(f"📰 Processed {results.successful_scrapes} news sites")
        print(f"📝 Generated {results.total_chunks} chunks")
        print(f"🔢 Created {results.total_embeddings} embeddings")
        
        return results
        
    except Exception as e:
        print(f"❌ News scraping failed: {str(e)}")
        return None

def example_business_focused():
    """Example focused on business websites."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Business-Focused Scraping")
    print("="*60)
    
    # Business/company URLs
    urls = [
        "https://www.microsoft.com/en-us/ai",
        "https://www.ibm.com/artificial-intelligence",
        "https://cloud.google.com/ai-platform",
        "https://aws.amazon.com/machine-learning/",
    ]
    
    config = PipelineConfig(
        urls=urls,
        output_dir="business_example_output",
        save_intermediate=True,
        use_cache=True
    )
    
    pipeline = WebScraperPipeline(config)
    
    try:
        results = pipeline.run_pipeline()
        
        print(f"\n✅ Business scraping completed!")
        print(f"🏢 Processed {results.successful_scrapes} business sites")
        print(f"📝 Generated {results.total_chunks} chunks")
        print(f"🔢 Created {results.total_embeddings} embeddings")
        
        return results
        
    except Exception as e:
        print(f"❌ Business scraping failed: {str(e)}")
        return None

def example_mixed_content():
    """Example with mixed content types."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Mixed Content Types")
    print("="*60)
    
    # Mix of different site types
    urls = [
        # Wikipedia
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        # Business
        "https://www.microsoft.com/en-us/ai",
        # News
        "https://www.bbc.com/news/technology",
        # Another Wikipedia
        "https://en.wikipedia.org/wiki/Machine_learning",
        # Another Business
        "https://cloud.google.com/ai-platform"
    ]
    
    config = PipelineConfig(
        urls=urls,
        output_dir="mixed_example_output",
        save_intermediate=True,
        use_cache=True
    )
    
    pipeline = WebScraperPipeline(config)
    
    try:
        results = pipeline.run_pipeline()
        
        print(f"\n✅ Mixed content scraping completed!")
        print(f"🌐 Processed {results.successful_scrapes} websites of different types")
        print(f"📝 Generated {results.total_chunks} chunks")
        print(f"🔢 Created {results.total_embeddings} embeddings")
        
        # Show distribution by site type
        if results.performance_stats:
            analysis = results.performance_stats.get('embedding_analysis', {})
            content_analysis = analysis.get('content_analysis', {})
            
            if content_analysis:
                print(f"\n📊 Site Type Distribution:")
                site_dist = content_analysis.get('site_type_distribution', {})
                for site_type, count in site_dist.items():
                    print(f"   {site_type.capitalize()}: {count} chunks")
                
                chunking_dist = content_analysis.get('chunking_method_distribution', {})
                print(f"\n🔧 Chunking Method Distribution:")
                for method, count in chunking_dist.items():
                    print(f"   {method}: {count} chunks")
        
        return results
        
    except Exception as e:
        print(f"❌ Mixed content scraping failed: {str(e)}")
        return None

def demonstrate_performance_analysis(results):
    """Demonstrate how to analyze pipeline performance."""
    if not results:
        print("No results to analyze.")
        return
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Basic metrics
    efficiency = results.total_embeddings / results.execution_time
    print(f"⚡ Processing Efficiency: {efficiency:.1f} embeddings/second")
    
    success_rate = results.successful_scrapes / results.processed_urls
    print(f"✅ Scraping Success Rate: {success_rate:.1%}")
    
    chunks_per_site = results.total_chunks / results.successful_scrapes if results.successful_scrapes > 0 else 0
    print(f"📄 Average Chunks per Site: {chunks_per_site:.1f}")
    
    # Stage breakdown
    stage_times = results.performance_stats.get('stage_times', {})
    print(f"\n⏱️  Stage Performance:")
    for stage, duration in stage_times.items():
        if stage.endswith('_duration'):
            stage_name = stage.replace('_duration', '').replace('_', ' ').title()
            percentage = (duration / results.execution_time) * 100
            print(f"   {stage_name:.<25} {duration:.2f}s ({percentage:.1f}%)")
    
    # Embedding stats
    emb_stats = results.performance_stats.get('embedding_stats', {})
    if emb_stats:
        print(f"\n🔢 Embedding Performance:")
        print(f"   Cache Hit Rate: {emb_stats.get('cache_hit_rate', 0):.1%}")
        print(f"   Average Time per Embedding: {emb_stats.get('avg_time_per_embedding', 0):.4f}s")
        print(f"   Device Used: {emb_stats.get('device', 'unknown')}")

def main():
    """Run all examples."""
    print("🚀 Web Scraper Pipeline - Example Usage")
    print("=" * 80)
    
    # Configure logging
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Run examples
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Wikipedia Focused", example_wikipedia_focused),
        ("News Focused", example_news_focused),
        ("Business Focused", example_business_focused),
        ("Mixed Content", example_mixed_content),
    ]
    
    all_results = []
    
    for name, example_func in examples:
        print(f"\n🎯 Running: {name}")
        try:
            result = example_func()
            if result:
                all_results.append((name, result))
        except KeyboardInterrupt:
            print(f"\n⚠️  Interrupted during {name}")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            continue
    
    # Performance analysis on the last successful result
    if all_results:
        print(f"\n📊 Analyzing performance from: {all_results[-1][0]}")
        demonstrate_performance_analysis(all_results[-1][1])
    
    print(f"\n🎉 Completed {len(all_results)} examples successfully!")
    print("Check the output directories for generated files.")

if __name__ == "__main__":
    main()