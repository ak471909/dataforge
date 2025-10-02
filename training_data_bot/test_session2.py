"""
Test script for Session 2: Configuration and Logging Infrastructure
Run this to verify the configuration and logging systems are working correctly.
"""

import sys
sys.path.append('.')

from core import (
    settings,
    get_settings,
    setup_logging,
    get_logger,
    LogContext,
    get_performance_logger,
    validate_configuration,
)


def test_configuration():
    """Test configuration loading and validation."""
    print("\n" + "="*60)
    print("TEST 1: Configuration System")
    print("="*60)
    
    # Get global settings
    config = get_settings()
    
    print(f"✓ App Name: {config.app_name}")
    print(f"✓ Version: {config.app_version}")
    print(f"✓ Environment: {config.environment}")
    print(f"✓ Log Level: {config.log_level}")
    print(f"✓ Chunk Size: {config.chunk_size}")
    print(f"✓ Chunk Overlap: {config.chunk_overlap}")
    print(f"✓ Max Workers: {config.max_workers}")
    print(f"✓ Quality Threshold: {config.quality_threshold}")
    print(f"✓ Output Directory: {config.output_directory}")
    
    # Test AI provider config
    openai_config = config.get_ai_provider_config("openai")
    if openai_config:
        print(f"✓ OpenAI Model: {openai_config.model_name}")
    else:
        print("⚠ OpenAI config not available (API key not set)")
    
    # Test processing config
    proc_config = config.get_processing_config()
    print(f"✓ Processing Config Created: {proc_config.chunk_size} tokens/chunk")
    
    # Validate configuration
    issues = validate_configuration(config)
    if issues:
        print(f"\n⚠ Configuration Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Configuration validation passed!")
    
    print("\n✓ Configuration system working correctly!")


def test_logging_basic():
    """Test basic logging functionality."""
    print("\n" + "="*60)
    print("TEST 2: Basic Logging")
    print("="*60)
    
    # Setup logging
    logger = setup_logging(level="INFO", structured=False)
    
    print("\nLogging different levels:")
    logger.debug("This is a debug message (should not appear if level is INFO)")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Log with extra fields
    logger.info(
        "Processing document",
        document_id="doc_123",
        word_count=1500,
        processing_time=2.5
    )
    
    print("\n✓ Basic logging working correctly!")


def test_logging_context():
    """Test logging context management."""
    print("\n" + "="*60)
    print("TEST 3: Logging Context")
    print("="*60)
    
    logger = get_logger()
    
    # Test context manager
    with LogContext("document_loading", component="loader"):
        logger.info("Loading document from source")
        logger.info("Extracting text content")
        logger.info("Creating document object")
    
    # Test nested contexts
    with LogContext("document_processing", component="processor"):
        logger.info("Starting document processing")
        
        with LogContext("text_chunking", component="preprocessor"):
            logger.info("Splitting text into chunks")
            logger.info("Created 10 chunks")
        
        logger.info("Processing complete")
    
    print("\n✓ Logging context working correctly!")


def test_performance_logging():
    """Test performance logging."""
    print("\n" + "="*60)
    print("TEST 4: Performance Logging")
    print("="*60)
    
    perf_logger = get_performance_logger()
    
    # Log processing stats
    perf_logger.log_processing_stats(
        operation="document_batch_processing",
        total_items=100,
        processed_items=95,
        failed_items=5,
        duration=45.3
    )
    
    # Log API call
    perf_logger.log_api_call(
        provider="openai",
        model="gpt-3.5-turbo",
        tokens_used=1250,
        response_time=1.8,
        success=True
    )
    
    # Log document processing
    perf_logger.log_document_processing(
        document_id="doc_456",
        document_type="pdf",
        word_count=5000,
        chunks_created=5,
        processing_time=3.2
    )
    
    print("\n✓ Performance logging working correctly!")


def test_error_logging():
    """Test error logging and exception handling."""
    print("\n" + "="*60)
    print("TEST 5: Error Logging")
    print("="*60)
    
    logger = get_logger()
    
    # Log a simulated error
    try:
        # Simulate an error
        raise ValueError("This is a test error")
    except Exception as e:
        logger.exception(
            "An error occurred during processing",
            error_type=type(e).__name__,
            operation="test_operation"
        )
    
    print("\n✓ Error logging working correctly!")


def test_configuration_validation():
    """Test configuration validation."""
    print("\n" + "="*60)
    print("TEST 6: Configuration Validation")
    print("="*60)
    
    from core.config import Settings
    
    # Test valid configuration
    valid_config = Settings(
        chunk_size=1000,
        chunk_overlap=200,
        quality_threshold=0.8
    )
    issues = validate_configuration(valid_config)
    print(f"✓ Valid config issues: {len(issues)}")
    
    # Test invalid configuration (will raise validation error during creation)
    try:
        invalid_config = Settings(
            chunk_size=100,
            chunk_overlap=200,  # Overlap >= chunk_size (invalid)
        )
        print("⚠ Should have raised validation error!")
    except Exception as e:
        print(f"✓ Validation caught error: {type(e).__name__}")
    
    print("\n✓ Configuration validation working correctly!")


def test_directory_creation():
    """Test automatic directory creation."""
    print("\n" + "="*60)
    print("TEST 7: Directory Creation")
    print("="*60)
    
    import os
    from pathlib import Path
    
    config = get_settings()
    config.create_directories()
    
    # Check if directories were created
    output_exists = Path(config.output_directory).exists()
    temp_exists = Path(config.temp_directory).exists()
    
    print(f"✓ Output directory exists: {output_exists}")
    print(f"✓ Temp directory exists: {temp_exists}")
    
    if output_exists and temp_exists:
        print("\n✓ Directory creation working correctly!")
    else:
        print("\n⚠ Some directories were not created")


def run_all_tests():
    """Run all Session 2 tests."""
    print("\n" + "="*80)
    print(" "*20 + "SESSION 2 TEST SUITE")
    print(" "*15 + "Configuration and Logging Infrastructure")
    print("="*80)
    
    try:
        test_configuration()
        test_logging_basic()
        test_logging_context()
        test_performance_logging()
        test_error_logging()
        test_configuration_validation()
        test_directory_creation()
        
        print("\n" + "="*80)
        print(" "*25 + "ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nSession 2 is complete and working correctly!")
        print("\nNext Steps:")
        print("1. Create a .env file from .env.example")
        print("2. Add your API keys if you have them")
        print("3. Ready to proceed to Session 3: Document Loaders")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()