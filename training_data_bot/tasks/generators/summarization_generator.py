"""
Summarization task generator.

Generates text summaries from text chunks for training data.
"""

import time
from typing import Optional
from uuid import uuid4

from training_data_bot.core import (
    TextChunk,
    TaskTemplate,
    TaskResult,
    TaskType,
    LogContext,
)
from training_data_bot.tasks.base import BaseTaskGenerator


class SummarizationGenerator(BaseTaskGenerator):
    """
    Text summarization generator.
    
    Generates concise summaries from text chunks suitable for
    training summarization models.
    """
    
    def __init__(self, ai_client, **kwargs):
        """
        Initialize the summarization generator.
        
        Args:
            ai_client: AI client for generation
            **kwargs: Additional parameters including:
                - summary_style: Style of summary (concise, detailed, bullet)
                - max_summary_length: Maximum words in summary
        """
        super().__init__(
            ai_client=ai_client,
            task_type=TaskType.SUMMARIZATION,
            **kwargs
        )
        
        self.summary_style = kwargs.get("summary_style", "concise")
        self.max_summary_length = kwargs.get("max_summary_length", 100)
    
    def get_default_template(self) -> TaskTemplate:
        """Get the default summarization template."""
        style_instructions = {
            "concise": "Create a brief, one-paragraph summary",
            "detailed": "Create a comprehensive summary with key details",
            "bullet": "Create a bullet-point summary of main points"
        }
        
        instruction = style_instructions.get(
            self.summary_style,
            "Create a concise summary"
        )
        
        return TaskTemplate(
            id=uuid4(),
            name="Default Summarization",
            task_type=TaskType.SUMMARIZATION,
            description="Generate summaries from text",
            prompt_template="""Read the following text and create a summary.

Text:
{text}

Instructions:
- {instruction}
- Keep it under {max_length} words
- Capture the main ideas and key points
- Use clear, concise language
- Maintain factual accuracy

Summary:""",
            parameters={
                "instruction": instruction,
                "max_length": self.max_summary_length,
                "temperature": 0.5,
                "max_tokens": 500,
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
        Generate summary for a single text chunk.
        
        Args:
            chunk: Text chunk to summarize
            template: Optional custom template
            **kwargs: Additional generation parameters
            
        Returns:
            TaskResult with generated summary
        """
        with LogContext("summarization_generate_single", chunk_id=str(chunk.id)):
            # Use provided template or default
            template = template or self.get_default_template()
            
            # Get style instructions
            style_instructions = {
                "concise": "Create a brief, one-paragraph summary",
                "detailed": "Create a comprehensive summary with key details",
                "bullet": "Create a bullet-point summary of main points"
            }
            
            style = kwargs.get("summary_style", self.summary_style)
            instruction = style_instructions.get(style, "Create a concise summary")
            
            # Build the prompt
            prompt = self._build_prompt(
                chunk=chunk,
                template=template,
                instruction=instruction,
                max_length=kwargs.get("max_summary_length", self.max_summary_length)
            )
            
            # Track start time
            start_time = time.time()
            
            try:
                # Generate with AI
                response = await self.ai_client.generate(
                    prompt=prompt,
                    system_prompt="You are an expert at creating clear, concise summaries that capture key information.",
                    temperature=template.parameters.get("temperature", 0.5),
                    max_tokens=template.parameters.get("max_tokens", 500),
                )
                
                # Calculate metrics
                processing_time = time.time() - start_time
                confidence = self._calculate_confidence(response, chunk)
                quality_scores = self._assess_quality(response.content, chunk)
                
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
                    f"Generated summary",
                    chunk_id=str(chunk.id),
                    confidence=confidence,
                    tokens_used=response.tokens_used
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Summarization generation failed: {e}")
                self.stats["total_failed"] += 1
                raise
    
    def _assess_quality(self, output: str, chunk: TextChunk) -> dict:
        """
        Assess quality of generated summary.
        
        Args:
            output: Generated summary text
            chunk: Source text chunk
            
        Returns:
            Dictionary of quality scores
        """
        scores = super()._assess_quality(output, chunk)
        
        # Summary-specific quality checks
        summary_length = len(output.split())
        source_length = len(chunk.content.split())
        
        # Check if summary is shorter than source (compression ratio)
        if source_length > 0:
            compression_ratio = summary_length / source_length
            # Good summaries are 10-50% of original length
            if 0.1 <= compression_ratio <= 0.5:
                scores["compression_score"] = 1.0
            elif compression_ratio < 0.1:
                scores["compression_score"] = 0.7  # Too short
            else:
                scores["compression_score"] = 0.5  # Not compressed enough
        else:
            scores["compression_score"] = 0.0
        
        # Check length against target
        target_length = self.max_summary_length
        if summary_length <= target_length:
            scores["length_compliance"] = 1.0
        else:
            # Penalize for exceeding target
            overage = (summary_length - target_length) / target_length
            scores["length_compliance"] = max(1.0 - overage, 0.0)
        
        # Check for completeness (summary should be substantial)
        if summary_length >= 20:
            scores["completeness"] = 1.0
        elif summary_length >= 10:
            scores["completeness"] = 0.7
        else:
            scores["completeness"] = 0.3
        
        return scores
    
    def _format_input(self, chunk: TextChunk) -> str:
        """
        Format chunk for summarization input.
        
        Args:
            chunk: Text chunk
            
        Returns:
            Formatted input text with instruction
        """
        return f"Summarize the following text:\n{chunk.content}"
    
    def set_summary_style(self, style: str):
        """
        Update the summary style.
        
        Args:
            style: New style (concise, detailed, or bullet)
        """
        valid_styles = ["concise", "detailed", "bullet"]
        if style in valid_styles:
            self.summary_style = style
            self.logger.info(f"Updated summary style to: {style}")
        else:
            raise ValueError(f"Invalid style. Choose from: {valid_styles}")
    
    def set_max_length(self, max_length: int):
        """
        Update the maximum summary length.
        
        Args:
            max_length: Maximum words in summary
        """
        if max_length > 0:
            self.max_summary_length = max_length
            self.logger.info(f"Updated max summary length to: {max_length}")
        else:
            raise ValueError("max_length must be positive")