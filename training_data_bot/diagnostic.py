# diagnostic.py
import sys
from pathlib import Path

current = Path(__file__).resolve().parent
print(f"Current directory: {current}")
print(f"Parent directory: {current.parent}")

# Check if __init__.py exists
init_file = current / "__init__.py"
print(f"__init__.py exists: {init_file.exists()}")

# Add parent to path
sys.path.insert(0, str(current.parent))

# Try import
try:
    import training_data_bot
    print("✓ Import successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")