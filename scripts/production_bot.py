"""
Production-ready script to run the Training Data Bot.

Usage:
    python scripts/production_bot.py --config config/production.yaml
"""

import asyncio
import argparse
from pathlib import Path
import sys
import yaml
from dotenv import load_dotenv
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training_data_bot import TrainingDataBot
from training_data_bot.core import TaskType, ExportFormat


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


async def main():
    """Main production workflow."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Training Data Bot - Production Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/production.yaml"),
        help="Path to configuration file"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory or file path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/training_data.jsonl"),
        help="Output file path"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["qa_generation", "summarization"],
        help="Task types to generate"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Verify API key
    api_key = os.getenv("TDB_OPENAI_API_KEY")
    if not api_key:
        print("ERROR: TDB_OPENAI_API_KEY not found in environment")
        sys.exit(1)
    
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Tasks: {', '.join(args.tasks)}")
    
    try:
        # Initialize bot
        print("\nInitializing Training Data Bot...")
        bot = TrainingDataBot(config=config)
        
        # Configure AI client
        bot.set_ai_client(
            provider=config['ai']['default_provider'],
            api_key=api_key,
            model=config['ai']['openai']['model']
        )
        
        # Load documents
        print(f"\nLoading documents from: {args.input}")
        if args.input.is_dir():
            documents = await bot.load_documents(str(args.input))
        else:
            documents = await bot.load_documents([str(args.input)])
        
        print(f"Loaded {len(documents)} document(s)")
        
        # Process documents
        print("\nProcessing documents (this may take a while)...")
        task_types = [TaskType(task) for task in args.tasks]
        
        dataset = await bot.process_documents(
            documents=documents,
            task_types=task_types,
            quality_filter=True
        )
        
        print(f"Generated {len(dataset.examples)} training examples")
        
        # Evaluate quality
        print("\nEvaluating dataset quality...")
        report = await bot.evaluate_dataset(dataset)
        print(f"Quality Score: {report.overall_score:.2f}")
        print(f"Passed Quality Check: {report.passed}")
        
        # Export dataset
        print(f"\nExporting to: {args.output}")
        await bot.export_dataset(
            dataset=dataset,
            output_path=args.output,
            format=ExportFormat.JSONL,
            split_data=True
        )
        
        # Statistics
        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60)
        stats = bot.get_statistics()
        print(f"Documents Processed: {stats['documents']['total']}")
        print(f"Training Examples: {stats['datasets']['total_examples']}")
        print(f"Output: {args.output}")
        
        # Cleanup
        await bot.cleanup()
        
        print("\nSuccess!")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())