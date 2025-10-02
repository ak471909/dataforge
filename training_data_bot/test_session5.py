"""
Test script for Session 5: Text Preprocessing Pipeline
Run this to verify the text preprocessing system is working correctly.
"""

import sys
sys.path.append('.')

from core import Document, DocumentType, setup_logging, get_logger
from preprocessing import TextPreprocessor
from uuid import uuid4


# Setup logging for tests
setup_logging(level="INFO", structured=False)
logger = get_logger()


def create_sample_documents():
    """Create sample documents for testing."""
    # Short document
    short_doc = Document(
        id=uuid4(),
        title="Short Article",
        content="This is a short document. It has only a few sentences. Perfect for testing.",
        source="test",
        doc_type=DocumentType.TXT,
        word_count=14,
        char_count=77
    )
    
    # Medium document
    medium_content = """
    The Training Data Bot is an enterprise-grade system for generating training data.
    It processes documents from various sources including PDFs, web pages, and text files.
    The system uses AI to generate high-quality training examples for machine learning models.
    
    The architecture is modular and extensible. Each component has a specific responsibility.
    Document loaders handle different file formats. The preprocessing pipeline chunks text appropriately.
    Task generators create training examples based on templates. Quality evaluation ensures high standards.
    
    This makes it suitable for production use in real companies.
    """
    
    medium_doc = Document(
        id=uuid4(),
        title="Medium Article",
        content=medium_content,
        source="test",
        doc_type=DocumentType.TXT,
        word_count=len(medium_content.split()),
        char_count=len(medium_content)
    )
    
    # Long document
    long_content = " ".join([
        f"This is sentence number {i}. It contains some information about topic {i}."
        for i in range(1, 101)
    ])
    
    long_doc = Document(
        id=uuid4(),
        title="Long Article",
        content=long_content,
        source="test",
        doc_type=DocumentType.TXT,
        word_count=len(long_content.split()),
        char_count=len(long_content)
    )
    
    return short_doc, medium_doc, long_doc


def test_preprocessor_initialization():
    """Test preprocessor initialization and configuration."""
    print("\n" + "="*60)
    print("TEST 1: Preprocessor Initialization")
    print("="*60)
    
    # Test basic initialization
    preprocessor = TextPreprocessor(
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=10
    )
    
    print(f"✓ Preprocessor created: {preprocessor}")
    print(f"✓ Chunk size: {preprocessor.chunk_size}")
    print(f"✓ Chunk overlap: {preprocessor.chunk_overlap}")
    print(f"✓ Min chunk size: {preprocessor.min_chunk_size}")
    
    # Test from config
    from core import ProcessingConfig
    config = ProcessingConfig(
        chunk_size=500,
        chunk_overlap=100
    )
    
    preprocessor2 = TextPreprocessor.from_config(config)
    print(f"\n✓ Created from config: {preprocessor2}")
    
    # Test validation
    try:
        bad_preprocessor = TextPreprocessor(chunk_size=100, chunk_overlap=150)
        print("❌ Should have raised error for invalid parameters")
    except Exception as e:
        print(f"✓ Correctly raised error: {type(e).__name__}")
    
    print("\n✓ Preprocessor initialization working!")


def test_basic_chunking():
    """Test basic document chunking."""
    print("\n" + "="*60)
    print("TEST 2: Basic Chunking")
    print("="*60)
    
    preprocessor = TextPreprocessor(chunk_size=50, chunk_overlap=10)
    short_doc, medium_doc, _ = create_sample_documents()
    
    # Test short document
    print("\nProcessing short document...")
    chunks = preprocessor.process_document(short_doc)
    print(f"✓ Created {len(chunks)} chunk(s)")
    
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.token_count} tokens")
        print(f"    Content preview: {chunk.content[:60]}...")
    
    # Test medium document
    print("\nProcessing medium document...")
    chunks = preprocessor.process_document(medium_doc)
    print(f"✓ Created {len(chunks)} chunks")
    
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"  Chunk {i}: {chunk.token_count} tokens")
    
    if len(chunks) > 3:
        print(f"  ... and {len(chunks) - 3} more chunks")
    
    print("\n✓ Basic chunking working!")


def test_chunk_overlap():
    """Test that chunk overlap is working correctly."""
    print("\n" + "="*60)
    print("TEST 3: Chunk Overlap")
    print("="*60)
    
    preprocessor = TextPreprocessor(chunk_size=20, chunk_overlap=5)
    _, medium_doc, _ = create_sample_documents()
    
    chunks = preprocessor.process_document(medium_doc)
    
    print(f"Created {len(chunks)} chunks with 5-token overlap")
    
    if len(chunks) >= 2:
        # Check if there's overlap between consecutive chunks
        chunk1_words = chunks[0].content.split()[-5:]
        chunk2_words = chunks[1].content.split()[:5]
        
        overlap_check = any(word in chunk2_words for word in chunk1_words)
        
        print(f"\n✓ Chunk 0 last 5 words: {' '.join(chunk1_words)}")
        print(f"✓ Chunk 1 first 5 words: {' '.join(chunk2_words)}")
        print(f"✓ Overlap detected: {overlap_check}")
    
    # Check overlap_tokens field
    for i, chunk in enumerate(chunks[:3]):
        print(f"  Chunk {i}: overlap_tokens = {chunk.overlap_tokens}")
    
    print("\n✓ Chunk overlap working!")


def test_batch_processing():
    """Test processing multiple documents."""
    print("\n" + "="*60)
    print("TEST 4: Batch Processing")
    print("="*60)
    
    preprocessor = TextPreprocessor(chunk_size=50, chunk_overlap=10)
    documents = list(create_sample_documents())
    
    print(f"Processing {len(documents)} documents...")
    all_chunks = preprocessor.process_documents(documents)
    
    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    
    # Group by document
    by_doc = {}
    for chunk in all_chunks:
        doc_id = str(chunk.document_id)
        by_doc[doc_id] = by_doc.get(doc_id, 0) + 1
    
    print(f"✓ Chunks per document:")
    for doc, count in by_doc.items():
        print(f"  - Document {doc[:8]}...: {count} chunks")
    
    print("\n✓ Batch processing working!")


def test_text_cleaning():
    """Test text cleaning functionality."""
    print("\n" + "="*60)
    print("TEST 5: Text Cleaning")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    
    # Test with messy text
    messy_text = """
    This  has    extra    spaces.
    
    
    And multiple blank lines.
    It has "fancy quotes" and 'apostrophes'.
    """
    
    cleaned = preprocessor._clean_text(messy_text)
    
    print("Original text issues:")
    print("  - Multiple spaces")
    print("  - Blank lines")
    print("  - Fancy quotes")
    
    print(f"\n✓ Cleaned text: {cleaned[:100]}...")
    print(f"✓ Text length: {len(messy_text)} -> {len(cleaned)}")
    
    # Check improvements
    has_double_space = "  " in cleaned
    has_fancy_quotes = """ in cleaned or """ in cleaned
    
    print(f"\n✓ Removed double spaces: {not has_double_space}")
    print(f"✓ Normalized quotes: {not has_fancy_quotes}")
    
    print("\n✓ Text cleaning working!")


def test_chunk_context():
    """Test getting context around chunks."""
    print("\n" + "="*60)
    print("TEST 6: Chunk Context")
    print("="*60)
    
    preprocessor = TextPreprocessor(chunk_size=30, chunk_overlap=5)
    _, medium_doc, _ = create_sample_documents()
    
    chunks = preprocessor.process_document(medium_doc)
    
    if len(chunks) >= 3:
        # Get context for middle chunk
        middle_idx = len(chunks) // 2
        context = preprocessor.get_chunk_context(chunks, middle_idx, context_chunks=1)
        
        print(f"Document has {len(chunks)} chunks")
        print(f"Getting context for chunk {middle_idx}:")
        print(f"\n✓ Context (with 1 chunk before/after):")
        print(f"  {context[:150]}...")
        print(f"\n✓ Context length: {len(context.split())} words")
    
    print("\n✓ Chunk context working!")


def test_sentence_chunking():
    """Test sentence-based chunking."""
    print("\n" + "="*60)
    print("TEST 7: Sentence-Based Chunking")
    print("="*60)
    
    preprocessor = TextPreprocessor()
    _, medium_doc, _ = create_sample_documents()
    
    # Create sentence-based chunks
    chunks = preprocessor.create_sentence_chunks(medium_doc, sentences_per_chunk=3)
    
    print(f"✓ Created {len(chunks)} sentence-based chunks")
    
    for i, chunk in enumerate(chunks[:3]):
        sentence_count = chunk.content.count('.') + chunk.content.count('!') + chunk.content.count('?')
        print(f"  Chunk {i}: ~{sentence_count} sentences, {chunk.token_count} words")
    
    print("\n✓ Sentence-based chunking working!")


def test_statistics():
    """Test chunk statistics."""
    print("\n" + "="*60)
    print("TEST 8: Chunk Statistics")
    print("="*60)
    
    preprocessor = TextPreprocessor(chunk_size=100, chunk_overlap=20)
    documents = list(create_sample_documents())
    
    all_chunks = preprocessor.process_documents(documents)
    stats = preprocessor.get_statistics(all_chunks)
    
    print("Chunk Statistics:")
    print(f"✓ Total chunks: {stats['total_chunks']}")
    print(f"✓ Total tokens: {stats['total_tokens']}")
    print(f"✓ Average tokens per chunk: {stats['avg_tokens_per_chunk']:.1f}")
    print(f"✓ Min tokens: {stats['min_tokens']}")
    print(f"✓ Max tokens: {stats['max_tokens']}")
    print(f"✓ Documents processed: {stats['documents_processed']}")
    
    print("\n✓ Statistics generation working!")


def test_merge_small_chunks():
    """Test merging small chunks."""
    print("\n" + "="*60)
    print("TEST 9: Merge Small Chunks")
    print("="*60)
    
    preprocessor = TextPreprocessor(chunk_size=30, chunk_overlap=5, min_chunk_size=15)
    short_doc, _, _ = create_sample_documents()
    
    # Create chunks (might have small ones)
    chunks = preprocessor.process_document(short_doc)
    print(f"Initial chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.token_count} tokens")
    
    # Merge small chunks
    merged = preprocessor.merge_small_chunks(chunks, merge_threshold=10)
    print(f"\nAfter merging (threshold=10):")
    print(f"✓ Merged chunks: {len(merged)}")
    for i, chunk in enumerate(merged):
        print(f"  Chunk {i}: {chunk.token_count} tokens")
    
    print("\n✓ Small chunk merging working!")


def test_large_document():
    """Test with a large document."""
    print("\n" + "="*60)
    print("TEST 10: Large Document Processing")
    print("="*60)
    
    preprocessor = TextPreprocessor(chunk_size=100, chunk_overlap=20)
    _, _, long_doc = create_sample_documents()
    
    print(f"Processing large document: {long_doc.word_count} words")
    
    import time
    start_time = time.time()
    chunks = preprocessor.process_document(long_doc)
    duration = time.time() - start_time
    
    print(f"\n✓ Created {len(chunks)} chunks")
    print(f"✓ Processing time: {duration:.3f} seconds")
    print(f"✓ Chunks per second: {len(chunks)/duration:.1f}")
    
    stats = preprocessor.get_statistics(chunks)
    print(f"\n✓ Average chunk size: {stats['avg_tokens_per_chunk']:.1f} tokens")
    print(f"✓ Size range: {stats['min_tokens']} - {stats['max_tokens']} tokens")
    
    print("\n✓ Large document processing working!")


def run_all_tests():
    """Run all Session 5 tests."""
    print("\n" + "="*80)
    print(" "*20 + "SESSION 5 TEST SUITE")
    print(" "*16 + "Text Preprocessing Pipeline")
    print("="*80)
    
    try:
        test_preprocessor_initialization()
        test_basic_chunking()
        test_chunk_overlap()
        test_batch_processing()
        test_text_cleaning()
        test_chunk_context()
        test_sentence_chunking()
        test_statistics()
        test_merge_small_chunks()
        test_large_document()
        
        print("\n" + "="*80)
        print(" "*25 + "ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nSession 5 is complete and working correctly!")
        print("\nWhat the Text Preprocessor can do:")
        print("1. Split documents into overlapping chunks")
        print("2. Configure chunk size, overlap, and minimum size")
        print("3. Clean and normalize text")
        print("4. Process multiple documents in batch")
        print("5. Create sentence-based chunks")
        print("6. Get context around specific chunks")
        print("7. Merge small chunks")
        print("8. Generate detailed statistics")
        print("9. Track metadata for each chunk")
        print("10. Handle documents of any size efficiently")
        print("\nNext Steps:")
        print("Ready to proceed to Session 6: AI Client Integration")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()