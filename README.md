# Training Data Bot

A lightweight, Docker-deployable system for automatic training data curation. It ingests multi-format documents, generates QA/classification/summaries using LLMs, applies quality filters, and exports ML-ready datasets in seconds. Inspired by real-world data bottlenecks in LLM fine-tuning

## Features

- Load documents from multiple sources (PDF, DOCX, TXT, MD, HTML, CSV, JSON, URLs)
- Automatic text preprocessing and chunking
- AI-powered task generation (Q&A, classification, summarization)
- Quality evaluation and filtering
- Multi-format export (JSONL, JSON, CSV, Parquet)

## Installation
```bash
# Clone the repository
git clone https://github.com/yourcompany/training-data-bot.git
cd training-data-bot

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
