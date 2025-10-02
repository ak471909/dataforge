"""
Code Aggregator Script

Collects all Python code from the training_data_bot project and writes it
to a single text file with clear section headers.

Usage: python collect_code.py
"""

from pathlib import Path
from typing import List, Tuple


def get_all_python_files(base_path: Path) -> List[Path]:
    """
    Recursively find all Python files in the project.
    
    Args:
        base_path: Root directory to search
        
    Returns:
        Sorted list of Python file paths
    """
    # Patterns to exclude
    exclude_patterns = [
        '__pycache__',
        '.git',
        'venv',
        'env',
        '.venv',
        'logs',
        'output',
        'temp',
        'test_documents',
        '.pytest_cache',
    ]
    
    python_files = []
    
    for py_file in base_path.rglob('*.py'):
        # Skip if in excluded directory
        if any(pattern in str(py_file) for pattern in exclude_patterns):
            continue
        python_files.append(py_file)
    
    return sorted(python_files)


def read_file_content(file_path: Path) -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File content as string
    """
    try:
        return file_path.read_text(encoding='utf-8')
    except Exception as e:
        return f"# Error reading file: {e}\n"


def write_aggregated_code(output_file: Path, files_and_content: List[Tuple[str, str]]):
    """
    Write all code to a single output file.
    
    Args:
        output_file: Path to output file
        files_and_content: List of (relative_path, content) tuples
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("TRAINING DATA BOT - COMPLETE PROJECT CODE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total files: {len(files_and_content)}\n")
        f.write(f"Generated: {Path.cwd()}\n")
        f.write("=" * 80 + "\n\n\n")
        
        # Write each file
        for relative_path, content in files_and_content:
            # Section header
            f.write("=" * 80 + "\n")
            f.write(f"FILE: {relative_path}\n")
            f.write("=" * 80 + "\n\n")
            
            # File content
            f.write(content)
            
            # Add spacing between files
            f.write("\n\n\n")
        
        # Write footer
        f.write("=" * 80 + "\n")
        f.write("END OF PROJECT CODE\n")
        f.write("=" * 80 + "\n")


def main():
    """Main execution function."""
    # Get project root (assumes script is in project root)
    project_root = Path(__file__).parent
    
    # Output file path
    output_file = project_root / "project_complete_code.txt"
    
    print("Collecting Python files...")
    
    # Get all Python files
    python_files = get_all_python_files(project_root)
    
    print(f"Found {len(python_files)} Python files")
    
    # Read all files
    files_and_content = []
    for py_file in python_files:
        # Get relative path for cleaner output
        try:
            relative_path = py_file.relative_to(project_root)
        except ValueError:
            relative_path = py_file
        
        # Read content
        content = read_file_content(py_file)
        
        # Store as tuple
        files_and_content.append((str(relative_path), content))
        
        print(f"  ✓ {relative_path}")
    
    # Write aggregated file
    print(f"\nWriting to {output_file}...")
    write_aggregated_code(output_file, files_and_content)
    
    # Print summary
    total_lines = sum(content.count('\n') for _, content in files_and_content)
    total_chars = sum(len(content) for _, content in files_and_content)
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Output file: {output_file}")
    print(f"Total files: {len(files_and_content)}")
    print(f"Total lines: {total_lines:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()