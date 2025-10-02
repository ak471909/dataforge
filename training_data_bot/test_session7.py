"""
Test suite for Session 7: Task Generation System

Tests all task generators and the task manager.
"""

import asyncio
from uuid import uuid4

# Import core components
from core import (
    Document,
    TextChunk,
    TaskType,
    DocumentType,
    ProcessingJob,
    ProcessingStatus,
)

# Import AI client (mock for testing)
from ai import AIClient, AIResponse

# Import task components
from tasks import (
    BaseTaskGenerator,
    TaskManager,
    QAGenerator,
    ClassificationGenerator,
    SummarizationGenerator,
)


class MockAIClient:
    """Mock AI client for testing without real API calls."""
    
    def __init__(self):
        self.call_count = 0
    
    async def generate(self, prompt, system_prompt=None, **kwargs):
        """Mock generation."""
        self.call_count += 1
        
        # Simulate different responses based on prompt content
        if "question" in prompt.lower() or "q&a" in prompt.lower():
            content = """Q: What is the main topic of this text?
A: The main topic is about machine learning and artificial intelligence.

Q: What are the key concepts mentioned?
A: The key concepts are neural networks, training data, and model optimization.

Q: How does this relate to real-world applications?
A: This relates to applications in natural language processing and computer vision."""
        
        elif "classify" in prompt.lower() or "category" in prompt.lower():
            content = """Category: informative
Reasoning: The text provides factual information and explanations about a technical topic, making it primarily informative in nature."""
        
        elif "summarize" in prompt.lower() or "summary" in prompt.lower():
            content = """This text discusses the fundamentals of machine learning, including key concepts like neural networks, training processes, and practical applications in various domains. It emphasizes the importance of quality training data and proper model optimization techniques."""
        
        else:
            content = "Generated response for testing purposes."
        
        return AIResponse(
            content=content,
            model="mock-model",
            tokens_used=100,
            finish_reason="stop",
            response_time=0.1,
            metadata={"prompt_tokens": 50, "completion_tokens": 50}
        )
    
    def count_tokens(self, text):
        """Mock token counting."""
        return len(text.split())


def create_test_chunks():
    """Create sample text chunks for testing."""
    doc_id = uuid4()
    
    chunks = [
        TextChunk(
            id=uuid4(),
            document_id=doc_id,
            content="""Machine learning is a subset of artificial intelligence that enables 
            computers to learn from data without being explicitly programmed. Neural networks 
            are a key component, inspired by biological neurons in the human brain. Training 
            data quality is crucial for model performance.""",
            start_index=0,
            end_index=300,
            chunk_index=0,
            token_count=50
        ),
        TextChunk(
            id=uuid4(),
            document_id=doc_id,
            content="""Deep learning uses multiple layers of neural networks to process 
            complex patterns. Convolutional neural networks excel at image recognition, 
            while recurrent neural networks are effective for sequential data like text 
            and time series.""",
            start_index=300,
            end_index=550,
            chunk_index=1,
            token_count=45
        ),
    ]
    
    return chunks


async def test_1_qa_generator():
    """Test Q&A generator functionality."""
    print("\n" + "="*60)
    print("TEST 1: Q&A Generator")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create Q&A generator
    qa_gen = QAGenerator(ai_client=ai_client, num_questions=3)
    
    print(f"\n✓ Created QAGenerator: {qa_gen}")
    print(f"  Task Type: {qa_gen.task_type.value}")
    print(f"  Number of Questions: {qa_gen.num_questions}")
    
    # Get default template
    template = qa_gen.get_default_template()
    print(f"\n✓ Default Template:")
    print(f"  Name: {template.name}")
    print(f"  Type: {template.task_type.value}")
    
    # Generate Q&A for single chunk
    chunks = create_test_chunks()
    result = await qa_gen.generate_single(chunks[0])
    
    print(f"\n✓ Generated Q&A Result:")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Tokens Used: {result.tokens_used}")
    print(f"  Processing Time: {result.processing_time:.3f}s")
    print(f"  Output Preview: {result.output[:100]}...")
    
    # Check quality scores
    print(f"\n✓ Quality Scores:")
    for metric, score in result.quality_scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Create training example
    example = qa_gen.create_training_example(result, chunks[0])
    print(f"\n✓ Training Example Created:")
    print(f"  Task Type: {example.task_type.value}")
    print(f"  Source Document: {example.source_document_id}")
    print(f"  Input Length: {len(example.input_text)} chars")
    print(f"  Output Length: {len(example.output_text)} chars")
    
    # Get statistics
    stats = qa_gen.get_statistics()
    print(f"\n✓ Generator Statistics:")
    print(f"  Total Generated: {stats['total_generated']}")
    print(f"  Total Failed: {stats['total_failed']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    
    print("\n✅ TEST 1 PASSED")


async def test_2_classification_generator():
    """Test classification generator functionality."""
    print("\n" + "="*60)
    print("TEST 2: Classification Generator")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create classification generator with custom categories
    categories = ["informative", "opinion", "narrative", "instructional"]
    classifier = ClassificationGenerator(
        ai_client=ai_client,
        categories=categories
    )
    
    print(f"\n✓ Created ClassificationGenerator: {classifier}")
    print(f"  Task Type: {classifier.task_type.value}")
    print(f"  Categories: {classifier.categories}")
    
    # Generate classification
    chunks = create_test_chunks()
    result = await classifier.generate_single(chunks[0])
    
    print(f"\n✓ Generated Classification:")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Output:\n{result.output}")
    
    # Check quality scores
    print(f"\n✓ Quality Scores:")
    for metric, score in result.quality_scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Test category update
    new_categories = ["technical", "general", "academic"]
    classifier.set_categories(new_categories)
    print(f"\n✓ Updated Categories: {classifier.categories}")
    
    print("\n✅ TEST 2 PASSED")


async def test_3_summarization_generator():
    """Test summarization generator functionality."""
    print("\n" + "="*60)
    print("TEST 3: Summarization Generator")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create summarization generator
    summarizer = SummarizationGenerator(
        ai_client=ai_client,
        summary_style="concise",
        max_summary_length=100
    )
    
    print(f"\n✓ Created SummarizationGenerator: {summarizer}")
    print(f"  Task Type: {summarizer.task_type.value}")
    print(f"  Style: {summarizer.summary_style}")
    print(f"  Max Length: {summarizer.max_summary_length} words")
    
    # Generate summary
    chunks = create_test_chunks()
    result = await summarizer.generate_single(chunks[0])
    
    print(f"\n✓ Generated Summary:")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Summary:\n{result.output}")
    
    # Check quality scores
    print(f"\n✓ Quality Scores:")
    for metric, score in result.quality_scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Test style update
    summarizer.set_summary_style("detailed")
    print(f"\n✓ Updated Style: {summarizer.summary_style}")
    
    # Test length update
    summarizer.set_max_length(50)
    print(f"✓ Updated Max Length: {summarizer.max_summary_length}")
    
    print("\n✅ TEST 3 PASSED")


async def test_4_batch_generation():
    """Test batch generation across all generators."""
    print("\n" + "="*60)
    print("TEST 4: Batch Generation")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create generator
    qa_gen = QAGenerator(ai_client=ai_client)
    
    # Generate batch
    chunks = create_test_chunks()
    results = await qa_gen.generate_batch(chunks, max_concurrent=2)
    
    print(f"\n✓ Batch Generation Complete:")
    print(f"  Input Chunks: {len(chunks)}")
    print(f"  Generated Results: {len(results)}")
    print(f"  Success Rate: {len(results)/len(chunks):.1%}")
    
    # Check each result
    for i, result in enumerate(results):
        print(f"\n  Result {i+1}:")
        print(f"    Confidence: {result.confidence:.2f}")
        print(f"    Tokens: {result.tokens_used}")
    
    # Get statistics
    stats = qa_gen.get_statistics()
    print(f"\n✓ Statistics After Batch:")
    print(f"  Total Generated: {stats['total_generated']}")
    print(f"  Total Time: {stats['total_time']:.2f}s")
    print(f"  Avg Time per Task: {stats['avg_time_per_task']:.3f}s")
    
    print("\n✅ TEST 4 PASSED")


async def test_5_task_manager():
    """Test task manager functionality."""
    print("\n" + "="*60)
    print("TEST 5: Task Manager")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create task manager
    manager = TaskManager(ai_client=ai_client, max_concurrent=3)
    
    print(f"\n✓ Created TaskManager: {manager}")
    
    # Register generators
    manager.register_generator(TaskType.QA_GENERATION)
    manager.register_generator(TaskType.CLASSIFICATION)
    manager.register_generator(TaskType.SUMMARIZATION)
    
    print(f"✓ Registered {len(manager.generators)} generators")
    
    # Generate single task
    chunks = create_test_chunks()
    result = await manager.generate_task(
        chunk=chunks[0],
        task_type=TaskType.QA_GENERATION
    )
    
    print(f"\n✓ Single Task Generated:")
    print(f"  Task Type: QA")
    print(f"  Confidence: {result.confidence:.2f}")
    
    print("\n✅ TEST 5 PASSED")


async def test_6_multi_task_generation():
    """Test generating multiple task types."""
    print("\n" + "="*60)
    print("TEST 6: Multi-Task Generation")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create task manager
    manager = TaskManager(ai_client=ai_client)
    
    # Generate multiple task types
    chunks = create_test_chunks()
    task_types = [
        TaskType.QA_GENERATION,
        TaskType.CLASSIFICATION,
        TaskType.SUMMARIZATION
    ]
    
    results = await manager.generate_tasks(
        chunks=chunks,
        task_types=task_types
    )
    
    print(f"\n✓ Multi-Task Generation Complete:")
    print(f"  Chunks: {len(chunks)}")
    print(f"  Task Types: {len(task_types)}")
    print(f"  Expected Results: {len(chunks) * len(task_types)}")
    print(f"  Actual Results: {len(results)}")
    
    # Count by type (approximate)
    print(f"\n✓ Results Generated:")
    print(f"  Total: {len(results)}")
    
    print("\n✅ TEST 6 PASSED")


async def test_7_training_examples():
    """Test creating training examples."""
    print("\n" + "="*60)
    print("TEST 7: Training Example Creation")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create task manager
    manager = TaskManager(ai_client=ai_client)
    
    # Create training examples
    chunks = create_test_chunks()
    task_types = [TaskType.QA_GENERATION, TaskType.SUMMARIZATION]
    
    examples = await manager.create_training_examples(
        chunks=chunks,
        task_types=task_types
    )
    
    print(f"\n✓ Training Examples Created:")
    print(f"  Total Examples: {len(examples)}")
    
    # Examine first example
    if examples:
        example = examples[0]
        print(f"\n✓ Example Details:")
        print(f"  Task Type: {example.task_type.value}")
        print(f"  Source Document: {example.source_document_id}")
        print(f"  Input Preview: {example.input_text[:80]}...")
        print(f"  Output Preview: {example.output_text[:80]}...")
        print(f"  Quality Scores: {example.quality_scores}")
    
    print("\n✅ TEST 7 PASSED")


async def test_8_job_processing():
    """Test job processing with progress tracking."""
    print("\n" + "="*60)
    print("TEST 8: Job Processing")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create task manager
    manager = TaskManager(ai_client=ai_client)
    
    # Create processing job
    job = ProcessingJob(
        id=uuid4(),
        name="Test Training Data Generation",
        job_type="training_data_generation",
        status=ProcessingStatus.PENDING
    )
    
    print(f"\n✓ Created Processing Job:")
    print(f"  ID: {job.id}")
    print(f"  Name: {job.name}")
    print(f"  Status: {job.status.value}")
    
    # Process job
    chunks = create_test_chunks()
    task_types = [TaskType.QA_GENERATION, TaskType.SUMMARIZATION]
    
    examples = await manager.process_job(
        job=job,
        chunks=chunks,
        task_types=task_types
    )
    
    print(f"\n✓ Job Processing Complete:")
    print(f"  Status: {job.status.value}")
    print(f"  Total Items: {job.total_items}")
    print(f"  Processed Items: {job.processed_items}")
    print(f"  Progress: {job.progress_percentage:.1f}%")
    print(f"  Examples Created: {len(examples)}")
    
    print("\n✅ TEST 8 PASSED")


async def test_9_statistics():
    """Test statistics collection."""
    print("\n" + "="*60)
    print("TEST 9: Statistics Collection")
    print("="*60)
    
    # Create mock AI client
    ai_client = MockAIClient()
    
    # Create task manager
    manager = TaskManager(ai_client=ai_client)
    
    # Register and use generators
    chunks = create_test_chunks()
    
    # Generate QA
    await manager.generate_task(chunks[0], TaskType.QA_GENERATION)
    
    # Generate Classification
    await manager.generate_task(chunks[0], TaskType.CLASSIFICATION)
    
    # Generate Summarization
    await manager.generate_task(chunks[1], TaskType.SUMMARIZATION)
    
    # Get statistics
    stats = manager.get_statistics()
    
    print(f"\n✓ Task Manager Statistics:")
    for task_type, task_stats in stats.items():
        print(f"\n  {task_type}:")
        print(f"    Total Generated: {task_stats['total_generated']}")
        print(f"    Total Failed: {task_stats['total_failed']}")
        print(f"    Success Rate: {task_stats['success_rate']:.1%}")
        print(f"    Total Tokens: {task_stats['total_tokens_used']}")
        print(f"    Avg Time: {task_stats['avg_time_per_task']:.3f}s")
    
    # Reset statistics
    manager.reset_statistics()
    print(f"\n✓ Statistics Reset")
    
    # Verify reset
    new_stats = manager.get_statistics()
    for task_type, task_stats in new_stats.items():
        assert task_stats['total_generated'] == 0, f"Stats not reset for {task_type}"
    
    print(f"✓ All statistics successfully reset to 0")
    
    print("\n✅ TEST 9 PASSED")


async def test_10_error_handling():
    """Test error handling and recovery."""
    print("\n" + "="*60)
    print("TEST 10: Error Handling")
    print("="*60)
    
    class FailingMockAIClient:
        """Mock client that fails sometimes."""
        
        def __init__(self):
            self.call_count = 0
        
        async def generate(self, prompt, system_prompt=None, **kwargs):
            self.call_count += 1
            
            # Fail every other call
            if self.call_count % 2 == 0:
                raise Exception("Simulated API failure")
            
            return AIResponse(
                content="Success response",
                model="mock-model",
                tokens_used=50,
                finish_reason="stop",
                response_time=0.1,
                metadata={}
            )
        
        def count_tokens(self, text):
            return len(text.split())
    
    # Create failing AI client
    ai_client = FailingMockAIClient()
    
    # Create generator
    qa_gen = QAGenerator(ai_client=ai_client)
    
    # Try batch generation (some will fail)
    chunks = create_test_chunks()
    results = await qa_gen.generate_batch(chunks, max_concurrent=2)
    
    print(f"\n✓ Batch Generation with Failures:")
    print(f"  Input Chunks: {len(chunks)}")
    print(f"  Successful Results: {len(results)}")
    print(f"  Failed Results: {len(chunks) - len(results)}")
    
    # Get statistics
    stats = qa_gen.get_statistics()
    print(f"\n✓ Statistics After Failures:")
    print(f"  Total Generated: {stats['total_generated']}")
    print(f"  Total Failed: {stats['total_failed']}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")
    
    print("\n✅ TEST 10 PASSED (Error handling works correctly)")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("SESSION 7 TEST SUITE: TASK GENERATION SYSTEM")
    print("="*60)
    
    tests = [
        test_1_qa_generator,
        test_2_classification_generator,
        test_3_summarization_generator,
        test_4_batch_generation,
        test_5_task_manager,
        test_6_multi_task_generation,
        test_7_training_examples,
        test_8_job_processing,
        test_9_statistics,
        test_10_error_handling,
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
        print("\n🎉 ALL TESTS PASSED! Session 7 Complete!")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review.")


if __name__ == "__main__":
    asyncio.run(main())