"""
Example usage of Session 7: Task Generation System

Demonstrates how to use the task generators to create training data.
"""

import asyncio
import sys
from pathlib import Path
from uuid import uuid4

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core imports
from training_data_bot.core import Document, TextChunk, TaskType, DocumentType

# AI imports
from training_data_bot.ai import AIClient

# Task imports
from training_data_bot.tasks import (
    TaskManager,
    QAGenerator,
    ClassificationGenerator,
    SummarizationGenerator,
)


async def example_1_basic_qa_generation():
    """Example 1: Basic Q&A generation."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Q&A Generation")
    print("="*60)
    
    # Setup (in production, use real API keys)
    # ai_client = AIClient(provider="openai", api_key="your-key")
    
    # For demo, we'll show the structure
    print("\n# Step 1: Initialize AI Client")
    print("ai_client = AIClient(provider='openai', api_key='your-key')")
    
    print("\n# Step 2: Create Q&A Generator")
    print("qa_gen = QAGenerator(ai_client=ai_client, num_questions=5)")
    
    print("\n# Step 3: Prepare Text Chunk")
    print("""chunk = TextChunk(
    id=uuid4(),
    document_id=uuid4(),
    content="Machine learning is transforming...",
    start_index=0,
    end_index=200,
    chunk_index=0,
    token_count=50
)""")
    
    print("\n# Step 4: Generate Q&A")
    print("result = await qa_gen.generate_single(chunk)")
    
    print("\n# Step 5: Create Training Example")
    print("example = qa_gen.create_training_example(result, chunk)")
    
    print("\n✓ Result: Training example ready for model fine-tuning!")


async def example_2_multiple_task_types():
    """Example 2: Generate multiple task types."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Multiple Task Types")
    print("="*60)
    
    print("\n# Step 1: Initialize Task Manager")
    print("ai_client = AIClient(provider='anthropic', api_key='your-key')")
    print("manager = TaskManager(ai_client=ai_client)")
    
    print("\n# Step 2: Prepare Multiple Chunks")
    print("""chunks = [
    TextChunk(...),  # Chapter 1
    TextChunk(...),  # Chapter 2
    TextChunk(...),  # Chapter 3
]""")
    
    print("\n# Step 3: Define Task Types")
    print("""task_types = [
    TaskType.QA_GENERATION,
    TaskType.CLASSIFICATION,
    TaskType.SUMMARIZATION
]""")
    
    print("\n# Step 4: Generate All Tasks")
    print("results = await manager.generate_tasks(chunks, task_types)")
    
    print("\n# Step 5: Create Training Examples")
    print("examples = await manager.create_training_examples(chunks, task_types)")
    
    print(f"\n✓ Result: {len([1,2,3]) * 3} training examples created!")
    print("  - 3 Q&A pairs")
    print("  - 3 Classifications")
    print("  - 3 Summaries")


async def example_3_custom_categories():
    """Example 3: Custom classification categories."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Classification Categories")
    print("="*60)
    
    print("\n# Step 1: Define Custom Categories")
    print("""categories = [
    "technical_documentation",
    "marketing_content",
    "customer_support",
    "product_description",
    "blog_post"
]""")
    
    print("\n# Step 2: Create Classifier")
    print("classifier = ClassificationGenerator(")
    print("    ai_client=ai_client,")
    print("    categories=categories,")
    print("    include_reasoning=True")
    print(")")
    
    print("\n# Step 3: Classify Text")
    print("result = await classifier.generate_single(chunk)")
    
    print("\n# Example Output:")
    print("""Category: technical_documentation
Reasoning: The text contains technical terminology, code examples, 
and follows a structured documentation format with clear sections 
and API references.""")
    
    print("\n✓ Perfect for training domain-specific classifiers!")


async def example_4_batch_processing():
    """Example 4: Batch processing with progress."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Processing")
    print("="*60)
    
    print("\n# Step 1: Load Large Document")
    print("""from training_data_bot.sources import UnifiedLoader
from training_data_bot.preprocessing import TextPreprocessor

loader = UnifiedLoader()
preprocessor = TextPreprocessor(chunk_size=1000, chunk_overlap=200)

# Load and chunk document
document = await loader.load_single('large_textbook.pdf')
chunks = preprocessor.process_document(document)

print(f"Created {len(chunks)} chunks from document")""")
    
    print("\n# Step 2: Setup Batch Processing")
    print("""manager = TaskManager(ai_client=ai_client, max_concurrent=10)

# Create processing job
job = ProcessingJob(
    id=uuid4(),
    name="Textbook Q&A Generation",
    job_type="qa_generation"
)""")
    
    print("\n# Step 3: Process with Progress Tracking")
    print("""examples = await manager.process_job(
    job=job,
    chunks=chunks,
    task_types=[TaskType.QA_GENERATION]
)

print(f"Progress: {job.progress_percentage}%")
print(f"Created {len(examples)} examples")""")
    
    print("\n✓ Efficient batch processing with progress monitoring!")


async def example_5_custom_templates():
    """Example 5: Using custom task templates."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Task Templates")
    print("="*60)
    
    print("\n# Step 1: Create Custom Template")
    print("""from core import TaskTemplate

custom_template = TaskTemplate(
    id=uuid4(),
    name="Advanced Q&A with Difficulty Levels",
    task_type=TaskType.QA_GENERATION,
    description="Generate Q&A pairs with difficulty ratings",
    prompt_template=\"\"\"
Read this text and generate 5 questions with answers.
Rate each question's difficulty (Easy/Medium/Hard).

Text: {text}

Format:
Q: [question]
Difficulty: [Easy/Medium/Hard]
A: [answer]
\"\"\",
    parameters={
        "num_questions": 5,
        "temperature": 0.8,
        "max_tokens": 1500
    }
)""")
    
    print("\n# Step 2: Use Custom Template")
    print("""result = await qa_gen.generate_single(
    chunk=chunk,
    template=custom_template
)""")
    
    print("\n✓ Full control over generation format and style!")


async def example_6_quality_filtering():
    """Example 6: Quality filtering and assessment."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Quality Filtering")
    print("="*60)
    
    print("\n# Step 1: Generate with Quality Scores")
    print("""results = await manager.generate_tasks(chunks, task_types)

# Each result has quality scores
for result in results:
    print(f"Confidence: {result.confidence}")
    print(f"Quality Scores: {result.quality_scores}")""")
    
    print("\n# Step 2: Filter by Quality")
    print("""quality_threshold = 0.7

high_quality_results = [
    r for r in results 
    if r.confidence >= quality_threshold
]

print(f"High quality: {len(high_quality_results)}/{len(results)}")""")
    
    print("\n# Step 3: Create Examples from Best Results")
    print("""examples = [
    generator.create_training_example(result, chunk)
    for result, chunk in zip(high_quality_results, chunks)
]""")
    
    print("\n✓ Ensure only high-quality data for training!")


async def example_7_statistics_monitoring():
    """Example 7: Monitor generation statistics."""
    print("\n" + "="*60)
    print("EXAMPLE 7: Statistics Monitoring")
    print("="*60)
    
    print("\n# Step 1: Generate Training Data")
    print("""manager = TaskManager(ai_client=ai_client)

# Generate lots of examples
examples = await manager.create_training_examples(
    chunks=all_chunks,
    task_types=[
        TaskType.QA_GENERATION,
        TaskType.CLASSIFICATION,
        TaskType.SUMMARIZATION
    ]
)""")
    
    print("\n# Step 2: Check Statistics")
    print("""stats = manager.get_statistics()

for task_type, metrics in stats.items():
    print(f"{task_type}:")
    print(f"  Generated: {metrics['total_generated']}")
    print(f"  Failed: {metrics['total_failed']}")
    print(f"  Success Rate: {metrics['success_rate']:.1%}")
    print(f"  Tokens Used: {metrics['total_tokens_used']}")
    print(f"  Avg Time: {metrics['avg_time_per_task']:.2f}s")
    print(f"  Total Cost: ${metrics['estimated_cost']:.2f}")""")
    
    print("\n✓ Track performance and costs!")


async def example_8_real_world_pipeline():
    """Example 8: Complete real-world pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 8: Complete Training Data Pipeline")
    print("="*60)
    
    print("\n# Complete Pipeline:")
    print("""
from training_data_bot.sources import UnifiedLoader
from training_data_bot.preprocessing import TextPreprocessor
from training_data_bot.tasks import TaskManager
from training_data_bot.storage import DatasetExporter

async def create_training_dataset(source_files):
    # Step 1: Load documents
    loader = UnifiedLoader()
    documents = await loader.load_multiple(source_files)
    print(f"Loaded {len(documents)} documents")
    
    # Step 2: Preprocess into chunks
    preprocessor = TextPreprocessor(chunk_size=800, chunk_overlap=150)
    all_chunks = []
    for doc in documents:
        chunks = preprocessor.process_document(doc)
        all_chunks.extend(chunks)
    print(f"Created {len(all_chunks)} chunks")
    
    # Step 3: Generate training data
    manager = TaskManager(ai_client=ai_client)
    examples = await manager.create_training_examples(
        chunks=all_chunks,
        task_types=[
            TaskType.QA_GENERATION,
            TaskType.SUMMARIZATION
        ]
    )
    print(f"Generated {len(examples)} training examples")
    
    # Step 4: Filter by quality
    quality_threshold = 0.75
    high_quality = [
        ex for ex in examples
        if all(score >= quality_threshold 
               for score in ex.quality_scores.values())
    ]
    print(f"High quality examples: {len(high_quality)}")
    
    # Step 5: Export dataset
    exporter = DatasetExporter()
    dataset_path = await exporter.export(
        examples=high_quality,
        output_path="training_data.jsonl",
        format="jsonl"
    )
    print(f"Dataset exported to: {dataset_path}")
    
    return dataset_path

# Run the pipeline
dataset = await create_training_dataset([
    "textbook_chapter1.pdf",
    "textbook_chapter2.pdf",
    "documentation.md"
])
""")
    
    print("\n✓ Complete automated training data generation!")


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("SESSION 7 EXAMPLES: TASK GENERATION SYSTEM")
    print("="*60)
    print("\nThese examples demonstrate the task generation capabilities.")
    print("In production, replace mock clients with real AI API keys.\n")
    
    examples = [
        example_1_basic_qa_generation,
        example_2_multiple_task_types,
        example_3_custom_categories,
        example_4_batch_processing,
        example_5_custom_templates,
        example_6_quality_filtering,
        example_7_statistics_monitoring,
        example_8_real_world_pipeline,
    ]
    
    for example in examples:
        await example()
        await asyncio.sleep(0.1)  # Small delay for readability
    
    print("\n" + "="*60)
    print("🎉 ALL EXAMPLES COMPLETE!")
    print("="*60)
    print("\nNext Steps:")
    print("1. Set up AI API keys (OpenAI or Anthropic)")
    print("2. Load your documents")
    print("3. Generate training data")
    print("4. Export to JSONL for model fine-tuning")
    print("\nReady for Session 8: Evaluation & Storage!")


if __name__ == "__main__":
    asyncio.run(main())