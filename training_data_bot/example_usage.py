"""
Example usage of the UnifiedLoader.

This script demonstrates various ways to use the unified loader
for loading different types of documents.
"""


import asyncio
import sys
from pathlib import Path

# ADD this instead:
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training_data_bot.sources import UnifiedLoader
from training_data_bot.core import setup_logging


async def example_1_single_file():
    """Example 1: Load a single file."""
    print("\n" + "="*60)
    print("Example 1: Loading a Single File")
    print("="*60)
    
    loader = UnifiedLoader()
    
    # Load a single text file (type is automatically detected)
    doc = await loader.load_single("test_documents/sample.txt")
    
    print(f"Document Title: {doc.title}")
    print(f"Document Type: {doc.doc_type.value}")
    print(f"Word Count: {doc.word_count}")
    print(f"Content Preview: {doc.content[:100]}...")


async def example_2_multiple_files():
    """Example 2: Load multiple files at once."""
    print("\n" + "="*60)
    print("Example 2: Loading Multiple Files")
    print("="*60)
    
    loader = UnifiedLoader()
    
    # Load multiple files of different types
    sources = [
        "test_documents/sample.txt",
        "test_documents/sample.json",
        "test_documents/sample.csv",
    ]
    
    documents = await loader.load_multiple(sources, max_workers=4)
    
    print(f"Loaded {len(documents)} documents:")
    for doc in documents:
        print(f"  - {doc.title} ({doc.doc_type.value}): {doc.word_count} words")


async def example_3_directory():
    """Example 3: Load all files from a directory."""
    print("\n" + "="*60)
    print("Example 3: Loading Entire Directory")
    print("="*60)
    
    loader = UnifiedLoader()
    
    # Load all supported files from a directory
    documents = await loader.load_directory("test_documents")
    
    print(f"Loaded {len(documents)} documents from directory")
    
    # Group by type
    by_type = {}
    for doc in documents:
        doc_type = doc.doc_type.value
        by_type[doc_type] = by_type.get(doc_type, 0) + 1
    
    print("\nDocuments by type:")
    for doc_type, count in sorted(by_type.items()):
        print(f"  - {doc_type}: {count}")


async def example_4_web_content():
    """Example 4: Load content from a web URL."""
    print("\n" + "="*60)
    print("Example 4: Loading Web Content")
    print("="*60)
    
    loader = UnifiedLoader()
    
    try:
        # Load content from a URL (automatically detected as web content)
        doc = await loader.load_single("https://example.com")
        
        print(f"Web Page Title: {doc.title}")
        print(f"Word Count: {doc.word_count}")
        print(f"Content Preview: {doc.content[:150]}...")
    except Exception as e:
        print(f"Could not load web content: {e}")


async def example_5_mixed_sources():
    """Example 5: Load from mixed sources (files + URLs)."""
    print("\n" + "="*60)
    print("Example 5: Loading Mixed Sources")
    print("="*60)
    
    loader = UnifiedLoader()
    
    # Mix of local files and URLs
    sources = [
        "test_documents/sample.txt",
        "test_documents/sample.json",
        "https://example.com",
    ]
    
    documents = await loader.load_multiple(sources, max_workers=2, skip_errors=True)
    
    print(f"Loaded {len(documents)} documents from mixed sources:")
    for doc in documents:
        source_type = "Web" if doc.doc_type.value == "url" else "Local"
        print(f"  - {doc.title} ({source_type}, {doc.doc_type.value})")


async def example_6_grouped_by_type():
    """Example 6: Load and group documents by type."""
    print("\n" + "="*60)
    print("Example 6: Loading with Grouping by Type")
    print("="*60)
    
    loader = UnifiedLoader()
    
    sources = [
        "test_documents/sample.txt",
        "test_documents/sample.md",
        "test_documents/sample.json",
        "test_documents/sample.csv",
    ]
    
    # Load and automatically group by document type
    grouped = await loader.load_mixed_batch(
        sources,
        max_workers=4,
        group_by_type=True
    )
    
    print("Documents grouped by type:")
    for doc_type, docs in grouped.items():
        print(f"\n{doc_type.value.upper()} ({len(docs)} documents):")
        for doc in docs:
            print(f"  - {doc.title}: {doc.word_count} words")


async def example_7_error_handling():
    """Example 7: Error handling with skip_errors."""
    print("\n" + "="*60)
    print("Example 7: Error Handling")
    print("="*60)
    
    loader = UnifiedLoader()
    
    # Some sources that might fail
    sources = [
        "test_documents/sample.txt",  # Valid
        "nonexistent_file.txt",        # Will fail
        "test_documents/sample.json",  # Valid
        "another_missing.pdf",         # Will fail
    ]
    
    # With skip_errors=True, failed loads won't crash the whole batch
    documents = await loader.load_multiple(sources, skip_errors=True)
    
    print(f"Attempted to load {len(sources)} sources")
    print(f"Successfully loaded {len(documents)} documents")
    print(f"Failed: {len(sources) - len(documents)}")


async def example_8_getting_info():
    """Example 8: Getting loader information."""
    print("\n" + "="*60)
    print("Example 8: Loader Information")
    print("="*60)
    
    loader = UnifiedLoader()
    
    # Get information about the loader
    print(f"Loader: {loader}")
    print(f"\nSupported formats: {', '.join(loader.get_supported_formats())}")
    
    print("\nRegistered specialized loaders:")
    loader_info = loader.get_loader_info()
    for loader_name, formats in loader_info.items():
        print(f"  - {loader_name}: {', '.join(formats)}")


async def run_all_examples():
    """Run all examples."""
    # Setup logging
    setup_logging(level="WARNING", structured=False)  # Reduce log noise
    
    print("\n" + "="*80)
    print(" "*25 + "UNIFIED LOADER EXAMPLES")
    print("="*80)
    
    try:
        await example_1_single_file()
        await example_2_multiple_files()
        await example_3_directory()
        await example_4_web_content()
        await example_5_mixed_sources()
        await example_6_grouped_by_type()
        await example_7_error_handling()
        await example_8_getting_info()
        
        print("\n" + "="*80)
        print(" "*28 + "EXAMPLES COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_examples())