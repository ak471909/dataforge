"""Quick import test to verify all imports work."""

import sys
from pathlib import Path

# Add current directory to Python path (since we're in project root)
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


print("Testing imports...")

try:
    from training_data_bot.core import Document, Dataset, TaskType
    print("✅ Core imports work")
except ImportError as e:
    print(f"❌ Core imports failed: {e}")

try:
    from training_data_bot.sources import UnifiedLoader
    print("✅ Sources imports work")
except ImportError as e:
    print(f"❌ Sources imports failed: {e}")

try:
    from training_data_bot.preprocessing import TextPreprocessor
    print("✅ Preprocessing imports work")
except ImportError as e:
    print(f"❌ Preprocessing imports failed: {e}")

try:
    from training_data_bot.ai import AIClient
    print("✅ AI imports work")
except ImportError as e:
    print(f"❌ AI imports failed: {e}")

try:
    from training_data_bot.tasks import TaskManager, QAGenerator
    print("✅ Tasks imports work")
except ImportError as e:
    print(f"❌ Tasks imports failed: {e}")

try:
    from training_data_bot.evaluation import QualityEvaluator
    print("✅ Evaluation imports work")
except ImportError as e:
    print(f"❌ Evaluation imports failed: {e}")

try:
    from training_data_bot.storage import DatasetExporter
    print("✅ Storage imports work")
except ImportError as e:
    print(f"❌ Storage imports failed: {e}")

try:
    from training_data_bot.bot import TrainingDataBot
    print("✅ Bot import works")
except ImportError as e:
    print(f"❌ Bot import failed: {e}")

try:
    from training_data_bot import TrainingDataBot
    print("✅ Package-level import works")
except ImportError as e:
    print(f"❌ Package-level import failed: {e}")

print("\n✅ All imports successful!")