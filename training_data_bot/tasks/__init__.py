"""
Tasks module for training data generation.

This module provides task generators for creating different types
of training data from text chunks.
"""

from training_data_bot.tasks.base import BaseTaskGenerator
from training_data_bot.tasks.manager import TaskManager
from training_data_bot.tasks.generators import (
    QAGenerator,
    ClassificationGenerator,
    SummarizationGenerator,
)

from ..core import TaskTemplate

__all__ = [
    # Base
    "BaseTaskGenerator",
    
    # Manager
    "TaskManager",
    
    # Generators
    "QAGenerator",
    "ClassificationGenerator",
    "SummarizationGenerator",

    # Template
    "TaskTemplate",
]