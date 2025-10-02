"""
Classification task generator.

Generates text classification training examples from text chunks.
"""

import time
from typing import Optional, List
from uuid import uuid4

from training_data_bot.core import (
    TextChunk,
    TaskTemplate,
    TaskResult,
    TaskType,
    LogContext,
)
from training_data_bot.tasks.base import BaseTaskGenerator


class ClassificationGenerator(BaseTaskGenerator):
    """
    Text classification task generator.
    
    Generates classification examples with labels and reasoning
    suitable for training classification models.
    """
    
    def __init__(self, ai_client, **kwargs):
        """
        Initialize the classification generator.
        
        Args:
            ai_client: AI client for generation
            **kwargs: Additional parameters including:
                - categories: List of classification categories
                - include_reasoning: Whether to include reasoning
        """
        super().__init__(
            ai_client=ai_client,
            task_type=TaskType.CLASSIFICATION,
            **kwargs
        )
        
        self.categories = kwargs.get("categories", [
            "informative", "opinion", "narrative", "instructional", "analytical"
        ])
        self.include_reasoning = kwargs.get("include_reasoning", True)
    
    def get_default_template(self) -> TaskTemplate:
        """Get the default classification template."""
        return TaskTemplate(
            id=uuid4(),
            name="Default Text Classification",
            task_type=TaskType.CLASSIFICATION,
            description="Classify text into predefined categories",
            prompt_template="""Classify the following text into one of these categories: {categories}

Text:
{text}

Instructions:
- Choose the most appropriate category
- Provide a brief reasoning for your classification
- Be objective and analytical

Format your response as:
Category: [chosen category]
Reasoning: [brief explanation]

Classification:""",
            parameters={
                "categories": ", ".join(self.categories),
                "temperature": 0.3,
                "max_tokens": 300,
            },
            version="1.0"
        )
    
    async def generate_single(
        self,
        chunk: TextChunk,
        template: Optional[TaskTemplate] = None,
        **kwargs
    ) -> TaskResult:
        """
        Generate classification for a single text chunk.
        
        Args:
            chunk: Text chunk to classify
            template: Optional custom template
            **kwargs: Additional generation parameters
            
        Returns:
            TaskResult with classification and reasoning
        """
        with LogContext("classification_generate_single", chunk_id=str(chunk.id)):
            # Use provided template or default
            template = template or self.get_default_template()
            
            # Get categories from kwargs or use defaults
            categories = kwargs.get("categories", self.categories)
            
            # Build the prompt
            prompt = self._build_prompt(
                chunk=chunk,
                template=template,
                categories=", ".join(categories)
            )
            
            # Track start time
            start_time = time.time()
            
            try:
                # Generate with AI
                response = await self.ai_client.generate(
                    prompt=prompt,
                    system_prompt="You are an expert at text classification and analysis.",
                    temperature=template.parameters.get("temperature", 0.3),
                    max_tokens=template.parameters.get("max_tokens", 300),
                )
                
                # Calculate metrics
                processing_time = time.time() - start_time
                confidence = self._calculate_confidence(response, chunk)
                quality_scores = self._assess_quality(response.content, chunk, categories)
                
                # Update statistics
                self.stats["total_generated"] += 1
                self.stats["total_tokens_used"] += response.tokens_used
                self.stats["total_time"] += processing_time
                
                # Create task result
                result = TaskResult(
                    id=uuid4(),
                    task_id=template.id,
                    input_chunk_id=chunk.id,
                    output=response.content,
                    confidence=confidence,
                    quality_scores=quality_scores,
                    processing_time=processing_time,
                    model_used=response.model,
                    tokens_used=response.tokens_used
                )
                
                self.logger.debug(
                    f"Generated classification",
                    chunk_id=str(chunk.id),
                    confidence=confidence,
                    tokens_used=response.tokens_used
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Classification generation failed: {e}")
                self.stats["total_failed"] += 1
                raise
    
    def _assess_quality(
        self,
        output: str,
        chunk: TextChunk,
        categories: List[str]
    ) -> dict:
        """
        Assess quality of generated classification.
        
        Args:
            output: Generated classification text
            chunk: Source text chunk
            categories: Valid categories
            
        Returns:
            Dictionary of quality scores
        """
        scores = super()._assess_quality(output, chunk)
        
        # Parse classification result
        parsed = self._parse_classification(output)
        
        # Check if category is valid
        if parsed["category"]:
            # Normalize and check
            category_lower = parsed["category"].lower()
            valid_categories = [c.lower() for c in categories]
            scores["valid_category"] = 1.0 if category_lower in valid_categories else 0.0
        else:
            scores["valid_category"] = 0.0
        
        # Check if reasoning is provided
        if parsed["reasoning"]:
            reasoning_length = len(parsed["reasoning"].split())
            # Good reasoning is 10-50 words
            scores["reasoning_quality"] = 1.0 if 10 <= reasoning_length <= 50 else 0.7
        else:
            scores["reasoning_quality"] = 0.0 if self.include_reasoning else 1.0
        
        # Check structure
        scores["structure_score"] = 1.0 if parsed["category"] else 0.0
        
        return scores
    
    def _parse_classification(self, output: str) -> dict:
        """
        Parse classification result from output.
        
        Args:
            output: Generated text with classification
            
        Returns:
            Dictionary with category and reasoning
        """
        result = {
            "category": None,
            "reasoning": None
        }
        
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Category:'):
                result["category"] = line.split(':', 1)[1].strip()
            elif line.startswith('Reasoning:'):
                result["reasoning"] = line.split(':', 1)[1].strip()
            elif result["reasoning"] and line and not line.startswith('Category:'):
                # Continue reasoning
                result["reasoning"] += ' ' + line
        
        return result
    
    def _format_input(self, chunk: TextChunk) -> str:
        """
        Format chunk for classification input.
        
        Args:
            chunk: Text chunk
            
        Returns:
            Formatted input text with instruction
        """
        return f"Classify this text into one of the predefined categories:\n{chunk.content}"
    
    def set_categories(self, categories: List[str]):
        """
        Update the classification categories.
        
        Args:
            categories: New list of categories
        """
        self.categories = categories
        self.logger.info(f"Updated categories to: {categories}")