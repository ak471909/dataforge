"""
Setup script for Training Data Bot package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="training-data-bot",
    version="0.1.0",
    description="Enterprise-grade training data curation bot for LLM fine-tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Training Data Bot Team",
    author_email="team@company.com",
    url="https://github.com/yourcompany/training-data-bot",
    
    # Package discovery
    packages=find_packages(exclude=["tests*", "examples*", "scripts*"]),
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Core dependencies
    install_requires=[
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "beautifulsoup4>=4.12.0",
        "PyMuPDF>=1.22.0",
        "python-docx>=0.8.11",
        "openai>=1.0.0",
        "anthropic>=0.25.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "api": [
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)