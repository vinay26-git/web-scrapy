#!/usr/bin/env python3
"""
Simple example demonstrating the web scraping pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import WebScrapingPipeline
from src.config import PipelineConfig, ScrapingConfig, ChunkingConfig, EmbeddingConfig


def main():
    """Run a simple example of the web scraping pipeline."""
    print("🚀 Starting Web Scraper Pipeline Example")
    print("=" * 50)
    
    # Create a simple configuration
    config = PipelineConfig(
        scraping=ScrapingConfig(
            headless=True,  # Run in headless mode
            page_load_timeout=20,
            max_retries=2
        ),
        chunking=ChunkingConfig(
            chunk_size=800,  # Smaller chunks for demo
            chunk_overlap=100,
            preserve_paragraphs=True
        ),
        embedding=EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=16,  # Smaller batch for demo
            device="auto"
        ),
        output_dir="./example_output"
    )
    
    # Create pipeline
    print("📦 Initializing pipeline...")
    pipeline = WebScrapingPipeline(config)
    
    # Example URLs for different website types
    urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",  # Wikipedia
        "https://www.bbc.com/news/technology",  # News
        "https://www.microsoft.com/en-us/about"  # Business
    ]
    
    print(f"🌐 Processing {len(urls)} URLs...")
    print("URLs:", urls)
    print("-" * 50)
    
    try:
        # Process each URL individually to show progress
        results = []
        for i, url in enumerate(urls, 1):
            print(f"\n📄 Processing URL {i}/{len(urls)}: {url}")
            
            result = pipeline.process_url(url)
            
            if result:
                results.append(result)
                stats = result['statistics']
                print(f"✅ Success! Created {len(result['chunks'])} chunks")
                print(f"   Text length: {stats['text_statistics']['char_count']:,} characters")
                print(f"   Avg chunk size: {stats['chunk_statistics']['avg_chunk_size']:.0f} characters")
                print(f"   Embedding dimension: {stats['embedding_statistics']['dimension']}")
            else:
                print(f"❌ Failed to process {url}")
        
        if results:
            # Save results
            print(f"\n💾 Saving results...")
            pipeline.save_results(results)
            
            # Show summary
            print("\n" + "=" * 50)
            print("📊 PIPELINE SUMMARY")
            print("=" * 50)
            print(f"✅ Successfully processed: {len(results)}/{len(urls)} URLs")
            
            total_chunks = sum(len(r['chunks']) for r in results)
            print(f"📝 Total chunks created: {total_chunks}")
            
            # Show details for each result
            for i, result in enumerate(results, 1):
                print(f"\n📄 Result {i}: {result['url']}")
                print(f"   Website type: {result['scraped_content']['website_type']}")
                print(f"   Title: {result['scraped_content']['title'][:60]}...")
                print(f"   Chunks: {len(result['chunks'])}")
                
                # Show first chunk preview
                if result['chunks']:
                    first_chunk = result['chunks'][0]
                    preview = first_chunk['text'][:100].replace('\n', ' ')
                    print(f"   First chunk: {preview}...")
            
            print(f"\n📁 Results saved to: {config.output_dir}")
            print("🎉 Pipeline completed successfully!")
            
        else:
            print("❌ No results generated")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        pipeline.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())