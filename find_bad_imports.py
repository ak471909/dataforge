"""Find files with incorrect imports."""

from pathlib import Path
import re

def check_file(filepath):
    """Check a file for bad import patterns."""
    try:
        content = filepath.read_text(encoding='utf-8')
        lines = content.split('\n')
        bad_imports = []
        
        for i, line in enumerate(lines, 1):
            # Check for "from ai import" (should be "from training_data_bot.ai import")
            if re.match(r'^\s*from\s+ai\s+import', line):
                bad_imports.append((i, line.strip(), "Should be: from training_data_bot.ai import"))
            
            # Check for "from core import" (should be "from training_data_bot.core import")
            if re.match(r'^\s*from\s+core\s+import', line):
                bad_imports.append((i, line.strip(), "Should be: from training_data_bot.core import"))
            
            # Check for other standalone module imports
            for module in ['tasks', 'sources', 'preprocessing', 'evaluation', 'storage']:
                pattern = rf'^\s*from\s+{module}\s+import'
                if re.match(pattern, line):
                    bad_imports.append((i, line.strip(), f"Should be: from training_data_bot.{module} import"))
        
        return bad_imports
    except Exception as e:
        return []

def main():
    """Check all Python files."""
    # Get the current script's directory
    current_dir = Path(__file__).parent
    package_dir = current_dir / "training_data_bot"
    
    if not package_dir.exists():
        print("❌ training_data_bot directory not found!")
        return
    
    print("🔍 Scanning for bad imports...\n")
    
    issues_found = False
    
    for py_file in sorted(package_dir.rglob("*.py")):
        # Skip test files
        if "test_" in py_file.name:
            continue
        
        bad_imports = check_file(py_file)
        
        if bad_imports:
            issues_found = True
            # Use package_dir as the base for relative path
            rel_path = py_file.relative_to(package_dir)
            print(f"❌ {rel_path}")
            for line_num, line, suggestion in bad_imports:
                print(f"   Line {line_num}: {line}")
                print(f"            {suggestion}")
            print()

            
if __name__ == "__main__":
    main()