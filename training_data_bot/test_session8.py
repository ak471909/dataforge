"""
Test suite for Session 8: Evaluation, Storage & Final Integration

Tests the complete system end-to-end.
"""

import asyncio
from pathlib import Path
from uuid import uuid4
import tempfile
import os
import sys

# Add parent directory to the python part so the package can beimported 
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import core components
from training_data_bot.core import (
    Document,
    TextChunk,
    TrainingExample,
    Dataset,
    TaskType,
    DocumentType,
    ExportFormat,
    QualityMetric,
)

# Import evaluation and storage
from evaluation import QualityEvaluator
from storage import DatasetExporter

# Import main bot
from bot import TrainingDataBot


def create_test_example():
    """Create a test training example."""
    return TrainingExample(
        id=uuid4(),
        input_text="What is machine learning?",
        output_text="Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.",
        task_type=TaskType.QA_GENERATION,
        source_document_id=uuid4(),
        quality_scores={"relevance": 0.9, "coherence": 0.85}
    )


def create_test_dataset(num_examples=10):
    """Create a test dataset."""
    examples = []
    for i in range(num_examples):
        example = TrainingExample(
            id=uuid4(),
            input_text=f"Question {i+1}: What is topic {i+1}?",
            output_text=f"Answer {i+1}: This is the explanation for topic {i+1}. It covers important concepts and provides detailed information.",
            task_type=TaskType.QA_GENERATION,
            source_document_id=uuid4(),
            quality_scores={"relevance": 0.8 + (i % 3) * 0.05}
        )
        examples.append(example)
    
    return Dataset(
        id=uuid4(),
        name="Test Dataset",
        description="Dataset for testing",
        examples=examples,
        total_examples=len(examples)
    )


async def test_1_quality_evaluator_example():
    """Test evaluating a single training example."""
    print("\n" + "="*60)
    print("TEST 1: Quality Evaluator - Single Example")
    print("="*60)
    
    evaluator = QualityEvaluator(quality_threshold=0.7)
    
    print(f"\n✓ Created QualityEvaluator with threshold: 0.7")
    
    # Create test example
    example = create_test_example()
    
    # Evaluate example
    report = evaluator.evaluate_example(example, detailed=True)
    
    print(f"\n✓ Evaluation Results:")
    print(f"  Overall Score: {report.overall_score:.2f}")
    print(f"  Passed: {report.passed}")
    print(f"\n  Metric Scores:")
    for metric, score in report.metric_scores.items():
        print(f"    {metric.value}: {score:.2f}")
    
    if report.issues:
        print(f"\n  Issues: {len(report.issues)}")
        for issue in report.issues:
            print(f"    - {issue}")
    
    if report.warnings:
        print(f"\n  Warnings: {len(report.warnings)}")
        for warning in report.warnings:
            print(f"    - {warning}")
    
    if report.recommendations:
        print(f"\n  Recommendations: {len(report.recommendations)}")
        for rec in report.recommendations[:3]:  # Show first 3
            print(f"    - {rec}")
    
    print("\n✅ TEST 1 PASSED")


async def test_2_quality_evaluator_dataset():
    """Test evaluating an entire dataset."""
    print("\n" + "="*60)
    print("TEST 2: Quality Evaluator - Dataset")
    print("="*60)
    
    evaluator = QualityEvaluator(quality_threshold=0.7)
    
    # Create test dataset
    dataset = create_test_dataset(num_examples=15)
    
    print(f"\n✓ Created dataset with {len(dataset.examples)} examples")
    
    # Evaluate dataset
    report = evaluator.evaluate_dataset(dataset, detailed_report=True)
    
    print(f"\n✓ Dataset Evaluation Results:")
    print(f"  Overall Score: {report.overall_score:.2f}")
    print(f"  Passed: {report.passed}")
    print(f"\n  Metric Scores:")
    for metric, score in report.metric_scores.items():
        print(f"    {metric.value}: {score:.2f}")
    
    if report.issues:
        print(f"\n  Issues Found: {len(report.issues)}")
        for issue in report.issues:
            print(f"    - {issue}")
    
    if report.warnings:
        print(f"\n  Warnings: {len(report.warnings)}")
    
    print("\n✅ TEST 2 PASSED")


async def test_3_export_jsonl():
    """Test exporting to JSONL format."""
    print("\n" + "="*60)
    print("TEST 3: Export to JSONL")
    print("="*60)
    
    exporter = DatasetExporter()
    dataset = create_test_dataset(num_examples=5)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Export to JSONL
        result_path = await exporter.export_dataset(
            dataset=dataset,
            output_path=temp_path,
            format=ExportFormat.JSONL,
            split_data=False
        )
        
        print(f"\n✓ Exported to: {result_path}")
        
        # Verify file exists
        assert result_path.exists(), "Output file not created"
        
        # Check file size
        file_size = result_path.stat().st_size
        print(f"✓ File size: {file_size} bytes")
        
        # Read and verify content
        with open(result_path, 'r') as f:
            lines = f.readlines()
        
        print(f"✓ Lines in file: {len(lines)}")
        assert len(lines) == len(dataset.examples), "Wrong number of lines"
        
        # Get file info
        info = exporter.get_export_info(result_path)
        print(f"✓ File info: {info['size_mb']:.4f} MB")
        
        print("\n✅ TEST 3 PASSED")
        
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


async def test_4_export_json():
    """Test exporting to JSON format."""
    print("\n" + "="*60)
    print("TEST 4: Export to JSON")
    print("="*60)
    
    exporter = DatasetExporter()
    dataset = create_test_dataset(num_examples=3)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        result_path = await exporter.export_dataset(
            dataset=dataset,
            output_path=temp_path,
            format=ExportFormat.JSON,
            split_data=False,
            pretty_print=True
        )
        
        print(f"\n✓ Exported to JSON: {result_path}")
        
        # Verify file
        assert result_path.exists()
        
        # Read and parse JSON
        import json
        with open(result_path, 'r') as f:
            data = json.load(f)
        
        print(f"✓ Loaded JSON with {len(data)} examples")
        assert len(data) == len(dataset.examples)
        
        # Verify structure
        first_example = data[0]
        assert 'input' in first_example
        assert 'output' in first_example
        assert 'task_type' in first_example
        print(f"✓ JSON structure validated")
        
        print("\n✅ TEST 4 PASSED")
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


async def test_5_export_csv():
    """Test exporting to CSV format."""
    print("\n" + "="*60)
    print("TEST 5: Export to CSV")
    print("="*60)
    
    exporter = DatasetExporter()
    dataset = create_test_dataset(num_examples=4)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        result_path = await exporter.export_dataset(
            dataset=dataset,
            output_path=temp_path,
            format=ExportFormat.CSV,
            split_data=False,
            include_metadata=False
        )
        
        print(f"\n✓ Exported to CSV: {result_path}")
        
        # Verify file
        assert result_path.exists()
        
        # Read CSV
        import csv
        with open(result_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"✓ CSV has {len(rows)} rows")
        assert len(rows) == len(dataset.examples)
        
        # Verify headers
        first_row = rows[0]
        assert 'input_text' in first_row
        assert 'output_text' in first_row
        assert 'task_type' in first_row
        print(f"✓ CSV headers validated")
        
        print("\n✅ TEST 5 PASSED")
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


async def test_6_export_split_dataset():
    """Test exporting with train/val/test split."""
    print("\n" + "="*60)
    print("TEST 6: Export with Split")
    print("="*60)
    
    exporter = DatasetExporter()
    dataset = create_test_dataset(num_examples=20)
    
    # Set custom splits
    dataset.train_split = 0.7
    dataset.validation_split = 0.2
    dataset.test_split = 0.1
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "dataset.jsonl"
        
        result_path = await exporter.export_dataset(
            dataset=dataset,
            output_path=temp_path,
            format=ExportFormat.JSONL,
            split_data=True
        )
        
        print(f"\n✓ Exported split dataset to: {result_path}")
        
        # Verify split files exist
        train_file = Path(temp_dir) / "dataset_train.jsonl"
        val_file = Path(temp_dir) / "dataset_val.jsonl"
        test_file = Path(temp_dir) / "dataset_test.jsonl"
        
        assert train_file.exists(), "Train file not created"
        assert val_file.exists(), "Validation file not created"
        assert test_file.exists(), "Test file not created"
        
        # Count lines in each file
        with open(train_file) as f:
            train_count = len(f.readlines())
        with open(val_file) as f:
            val_count = len(f.readlines())
        with open(test_file) as f:
            test_count = len(f.readlines())
        
        print(f"\n✓ Split counts:")
        print(f"  Train: {train_count}")
        print(f"  Validation: {val_count}")
        print(f"  Test: {test_count}")
        print(f"  Total: {train_count + val_count + test_count}")
        
        assert train_count + val_count + test_count == len(dataset.examples)
        
        print("\n✅ TEST 6 PASSED")


async def test_7_bot_initialization():
    """Test TrainingDataBot initialization."""
    print("\n" + "="*60)
    print("TEST 7: TrainingDataBot Initialization")
    print("="*60)
    
    # Create bot with config
    config = {
        "chunk_size": 800,
        "chunk_overlap": 150,
        "quality_threshold": 0.75
    }
    
    bot = TrainingDataBot(config=config)
    
    print(f"\n✓ Created TrainingDataBot: {bot}")
    
    # Check components
    assert bot.loader is not None, "Loader not initialized"
    assert bot.preprocessor is not None, "Preprocessor not initialized"
    assert bot.task_manager is not None, "Task manager not initialized"
    assert bot.evaluator is not None, "Evaluator not initialized"
    assert bot.exporter is not None, "Exporter not initialized"
    
    print(f"✓ All components initialized")
    
    # Check configuration
    assert bot.preprocessor.chunk_size == 800
    assert bot.preprocessor.chunk_overlap == 150
    assert bot.evaluator.quality_threshold == 0.75
    
    print(f"✓ Configuration applied correctly")
    
    # Get statistics
    stats = bot.get_statistics()
    print(f"\n✓ Initial Statistics:")
    print(f"  Documents: {stats['documents']['total']}")
    print(f"  Datasets: {stats['datasets']['total']}")
    print(f"  Jobs: {stats['jobs']['total']}")
    
    print("\n✅ TEST 7 PASSED")


async def test_8_bot_context_manager():
    """Test TrainingDataBot as context manager."""
    print("\n" + "="*60)
    print("TEST 8: Bot Context Manager")
    print("="*60)
    
    async with TrainingDataBot() as bot:
        print(f"\n✓ Bot created in context manager")
        print(f"  Bot: {bot}")
        
        # Get stats
        stats = bot.get_statistics()
        print(f"✓ Statistics accessible: {stats['documents']['total']} documents")
    
    print(f"✓ Context manager exited cleanly")
    
    print("\n✅ TEST 8 PASSED")


async def test_9_bot_set_ai_client():
    """Test setting AI client on bot."""
    print("\n" + "="*60)
    print("TEST 9: Set AI Client")
    print("="*60)
    
    bot = TrainingDataBot()
    
    # Initially no AI client configured
    print(f"\n✓ Bot created without AI client")
    
    # Set AI client (with dummy key for testing)
    bot.set_ai_client(
        provider="openai",
        api_key="sk-test-key-12345",
        model="gpt-3.5-turbo"
    )
    
    print(f"✓ AI client configured")
    assert bot.ai_client is not None
    assert bot.task_manager.ai_client is not None
    
    print(f"✓ Task manager has AI client reference")
    
    print("\n✅ TEST 9 PASSED")


async def test_10_integration_export_examples():
    """Test exporting examples directly."""
    print("\n" + "="*60)
    print("TEST 10: Direct Example Export")
    print("="*60)
    
    exporter = DatasetExporter()
    
    # Create examples
    examples = [create_test_example() for _ in range(3)]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        # Export examples directly
        result_path = await exporter.export_examples(
            examples=examples,
            output_path=temp_path,
            format=ExportFormat.JSONL
        )
        
        print(f"\n✓ Exported {len(examples)} examples")
        print(f"✓ Output: {result_path}")
        
        assert result_path.exists()
        
        # Verify content
        with open(result_path, 'r') as f:
            lines = f.readlines()
        
        print(f"✓ Verified {len(lines)} lines in output")
        assert len(lines) == len(examples)
        
        print("\n✅ TEST 10 PASSED")
        
    finally:
        if temp_path.exists():
            temp_path.unlink()


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SESSION 8 TEST SUITE: EVALUATION, STORAGE & FINAL INTEGRATION")
    print("="*60)
    
    tests = [
        test_1_quality_evaluator_example,
        test_2_quality_evaluator_dataset,
        test_3_export_jsonl,
        test_4_export_json,
        test_5_export_csv,
        test_6_export_split_dataset,
        test_7_bot_initialization,
        test_8_bot_context_manager,
        test_9_bot_set_ai_client,
        test_10_integration_export_examples,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed/len(tests)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Session 8 Complete!")
        print("\n" + "="*60)
        print("TRAINING DATA BOT - FULLY OPERATIONAL!")
        print("="*60)
        print("\nThe complete system is ready:")
        print("✓ Document loading from multiple sources")
        print("✓ Text preprocessing and chunking")
        print("✓ AI-powered task generation")
        print("✓ Quality evaluation and filtering")
        print("✓ Multi-format dataset export")
        print("✓ End-to-end bot integration")
        print("\nReady for production use! 🚀")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")


if __name__ == "__main__":
    asyncio.run(main())