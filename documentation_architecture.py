"""
Project Architecture Documentation Generator

Creates a complete visual documentation of the project structure with:
- Directory tree
- File descriptions
- Code statistics
- Module relationships

Usage: python document_architecture.py
"""

from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def count_lines(file_path: Path) -> Tuple[int, int, int]:
    """
    Count lines in a file.
    
    Returns:
        (total_lines, code_lines, comment_lines)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        total = len(lines)
        
        code = 0
        comments = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Check for multiline comments
            if '"""' in stripped or "'''" in stripped:
                in_multiline_comment = not in_multiline_comment
                comments += 1
                continue
            
            if in_multiline_comment:
                comments += 1
            elif stripped.startswith('#'):
                comments += 1
            else:
                code += 1
        
        return total, code, comments
    except Exception:
        return 0, 0, 0


def get_file_description(file_path: Path) -> str:
    """Extract description from file docstring."""
    try:
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        in_docstring = False
        docstring_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            if '"""' in stripped:
                if in_docstring:
                    break
                in_docstring = True
                # Get content after opening quotes
                after_quotes = stripped.split('"""', 1)[1]
                if after_quotes and after_quotes != '"""':
                    docstring_lines.append(after_quotes)
                continue
            
            if in_docstring and stripped:
                docstring_lines.append(stripped)
        
        return ' '.join(docstring_lines[:2]) if docstring_lines else "No description"
    except Exception:
        return "Error reading file"


def build_tree(directory: Path, prefix: str = "", exclude_patterns: List[str] = None) -> List[str]:
    """Build directory tree structure."""
    if exclude_patterns is None:
        exclude_patterns = [
            '__pycache__', '.git', 'venv', 'env', '.venv', 
            'logs', 'temp', '.pytest_cache', 'output',
            '.egg-info', 'build', 'dist'
        ]
    
    tree_lines = []
    
    try:
        items = sorted(directory.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        items = [item for item in items if not any(pattern in str(item) for pattern in exclude_patterns)]
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            current_prefix = "└── " if is_last else "├── "
            tree_lines.append(prefix + current_prefix + item.name)
            
            if item.is_dir():
                extension = "    " if is_last else "│   "
                tree_lines.extend(build_tree(item, prefix + extension, exclude_patterns))
    except PermissionError:
        pass
    
    return tree_lines


def generate_architecture_doc(output_file: Path):
    """Generate complete architecture documentation."""
    project_root = Path(__file__).parent
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write("TRAINING DATA BOT - PROJECT ARCHITECTURE DOCUMENTATION\n")
        f.write("=" * 80 + "\n\n")
        
        # Table of Contents
        f.write("TABLE OF CONTENTS\n")
        f.write("-" * 80 + "\n")
        f.write("1. Project Overview\n")
        f.write("2. Directory Structure\n")
        f.write("3. Module Breakdown\n")
        f.write("4. File Descriptions\n")
        f.write("5. Code Statistics\n")
        f.write("6. Dependencies\n")
        f.write("\n\n")
        
        # 1. Project Overview
        f.write("=" * 80 + "\n")
        f.write("1. PROJECT OVERVIEW\n")
        f.write("=" * 80 + "\n\n")
        f.write("Project Name: Training Data Bot\n")
        f.write("Version: 0.1.0\n")
        f.write("Purpose: Enterprise-grade training data curation bot for LLM fine-tuning\n")
        f.write("\nArchitecture Pattern: Separate Project & Package (Industry Standard)\n")
        f.write("\nKey Features:\n")
        f.write("- Multi-format document loading (PDF, DOCX, TXT, MD, HTML, CSV, JSON, URLs)\n")
        f.write("- Automatic text preprocessing and chunking\n")
        f.write("- AI-powered task generation (Q&A, Classification, Summarization)\n")
        f.write("- Quality evaluation and filtering\n")
        f.write("- Multi-format export (JSONL, JSON, CSV, Parquet)\n")
        f.write("\n\n")
        
        # 2. Directory Structure
        f.write("=" * 80 + "\n")
        f.write("2. DIRECTORY STRUCTURE\n")
        f.write("=" * 80 + "\n\n")
        
        tree_lines = build_tree(project_root)
        f.write("my_training_data_bot/\n")
        for line in tree_lines:
            f.write(line + "\n")
        f.write("\n\n")
        
        # 3. Module Breakdown
        f.write("=" * 80 + "\n")
        f.write("3. MODULE BREAKDOWN\n")
        f.write("=" * 80 + "\n\n")
        
        modules = {
            "training_data_bot/": "Main package - contains all core functionality",
            "training_data_bot/core/": "Foundation - data models, config, logging, exceptions",
            "training_data_bot/sources/": "Document loading - PDF, DOCX, web, unified loader",
            "training_data_bot/preprocessing/": "Text processing - chunking and preparation",
            "training_data_bot/ai/": "AI integration - OpenAI, Anthropic providers",
            "training_data_bot/tasks/": "Task generation - Q&A, classification, summarization",
            "training_data_bot/evaluation/": "Quality assessment - multi-metric evaluation",
            "training_data_bot/storage/": "Data export - JSONL, JSON, CSV, Parquet formats",
            "config/": "Configuration files - production, development, staging",
            "scripts/": "Operational scripts - production runner, batch processing",
            "output/": "Generated training data outputs",
            "documents/": "Input documents for processing",
        }
        
        for module, description in modules.items():
            f.write(f"{module}\n")
            f.write(f"  → {description}\n\n")
        
        f.write("\n")
        
        # 4. File Descriptions
        f.write("=" * 80 + "\n")
        f.write("4. FILE DESCRIPTIONS\n")
        f.write("=" * 80 + "\n\n")
        
        package_dir = project_root / "training_data_bot"
        python_files = sorted(package_dir.rglob("*.py"))
        
        current_dir = None
        for py_file in python_files:
            if '__pycache__' in str(py_file):
                continue
            
            rel_path = py_file.relative_to(project_root)
            file_dir = str(rel_path.parent)
            
            if file_dir != current_dir:
                current_dir = file_dir
                f.write(f"\n{file_dir}/\n")
                f.write("-" * 80 + "\n")
            
            description = get_file_description(py_file)
            f.write(f"\n{py_file.name}\n")
            f.write(f"  {description}\n")
        
        f.write("\n\n")
        
        # 5. Code Statistics
        f.write("=" * 80 + "\n")
        f.write("5. CODE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        stats_by_module = defaultdict(lambda: {"files": 0, "total_lines": 0, "code_lines": 0, "comment_lines": 0})
        
        for py_file in python_files:
            if '__pycache__' in str(py_file):
                continue
            
            rel_path = py_file.relative_to(project_root)
            module = str(rel_path.parent)
            
            total, code, comments = count_lines(py_file)
            
            stats_by_module[module]["files"] += 1
            stats_by_module[module]["total_lines"] += total
            stats_by_module[module]["code_lines"] += code
            stats_by_module[module]["comment_lines"] += comments
        
        # Print statistics by module
        f.write(f"{'Module':<50} {'Files':<8} {'Total':<8} {'Code':<8} {'Comments':<8}\n")
        f.write("-" * 80 + "\n")
        
        grand_total = {"files": 0, "total_lines": 0, "code_lines": 0, "comment_lines": 0}
        
        for module in sorted(stats_by_module.keys()):
            stats = stats_by_module[module]
            f.write(f"{module:<50} {stats['files']:<8} {stats['total_lines']:<8} "
                   f"{stats['code_lines']:<8} {stats['comment_lines']:<8}\n")
            
            for key in grand_total:
                grand_total[key] += stats[key]
        
        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<50} {grand_total['files']:<8} {grand_total['total_lines']:<8} "
               f"{grand_total['code_lines']:<8} {grand_total['comment_lines']:<8}\n")
        
        f.write("\n\n")
        
        # 6. Dependencies
        f.write("=" * 80 + "\n")
        f.write("6. DEPENDENCIES\n")
        f.write("=" * 80 + "\n\n")
        
        requirements_file = project_root / "requirements.txt"
        if requirements_file.exists():
            f.write("Core Dependencies:\n")
            f.write("-" * 80 + "\n")
            requirements = requirements_file.read_text(encoding='utf-8')
            for line in requirements.split('\n'):
                if line.strip() and not line.startswith('#'):
                    f.write(f"  - {line.strip()}\n")
        
        f.write("\n\n")
        
        # Footer
        f.write("=" * 80 + "\n")
        f.write("END OF ARCHITECTURE DOCUMENTATION\n")
        f.write("=" * 80 + "\n")


def main():
    """Main execution."""
    project_root = Path(__file__).parent
    output_file = project_root / "project_architecture.txt"
    
    print("Generating project architecture documentation...")
    print(f"Output: {output_file}")
    
    generate_architecture_doc(output_file)
    
    file_size = output_file.stat().st_size / 1024
    
    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print(f"Architecture documentation created: {output_file}")
    print(f"File size: {file_size:.2f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()