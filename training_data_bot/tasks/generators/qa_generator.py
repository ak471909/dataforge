"""
Q&A task generator.

Generates question-answer pairs from text chunks for training data.
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
from training_data_bot.ai import AIResponse
from training_data_bot.tasks.base import BaseTaskGenerator


class QAGenerator(BaseTaskGenerator):
    """
    Question-Answer pair generator.
    
    Generates high-quality Q&A pairs from text chunks suitable for
    training question-answering models.
    """
    
    def __init__(self, ai_client, **kwargs):
        """
        Initialize the Q&A generator.
        
        Args:
            ai_client: AI client for generation
            **kwargs: Additional parameters
        """
        super().__init__(
            ai_client=ai_client,
            task_type=TaskType.QA_GENERATION,
            **kwargs
        )
        
        self.num_questions = kwargs.get("num_questions", 3)
        self.question_types = kwargs.get("question_types", [
            "factual", "analytical", "application"
        ])
    
    def get_default_template(self) -> TaskTemplate:
        """Get the default Q&A generation template."""
        return TaskTemplate(
            id=uuid4(),
            name="Default Q&A Generation",
            task_type=TaskType.QA_GENERATION,
            description="Generate question-answer pairs from text",
            prompt_template="""Read the following text carefully and generate {num_questions} high-quality question-answer pairs.

Text:
{text}

Requirements:
- Questions should be clear and specific
- Answers should be accurate and found in the text
- Cover different aspects of the text
- Use varied question types: factual, analytical, and application-based

Format each Q&A pair as:
Q: [question]
A: [answer]

Generate {num_questions} Q&A pairs:""",
            parameters={
                "num_questions": self.num_questions,
                "temperature": 0.7,
                "max_tokens": 1000,
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
        Generate Q&A pairs from a single text chunk.
        
        Args:
            chunk: Text chunk to process
            template: Optional custom template
            **kwargs: Additional generation parameters
            
        Returns:
            TaskResult with generated Q&A pairs
        """
        with LogContext("qa_generate_single", chunk_id=str(chunk.id)):
            # Use provided template or default
            template = template or self.get_default_template()
            
            # Build the prompt
            prompt = self._build_prompt(
                chunk=chunk,
                template=template,
                num_questions=kwargs.get("num_questions", self.num_questions)
            )
            
            # Track start time
            start_time = time.time()
            
            try:
                # Generate with AI
                response = await self.ai_client.generate(
                    prompt=prompt,
                    system_prompt="You are an expert at creating educational question-answer pairs.",
                    temperature=template.parameters.get("temperature", 0.7),
                    max_tokens=template.parameters.get("max_tokens", 1000),
                )

                # Check if output contains Q&A format
                if not self._is_valid_qa_format(response.content):
                    self.logger.warning(
                        f"Generated output doesn't contain Q&A format, skipping chunk {chunk.id}"
                    )
                    raise ValueError("Output is not in Q&A format")

                
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
                    f"Generated Q&A pairs",
                    chunk_id=str(chunk.id),
                    confidence=confidence,
                    tokens_used=response.tokens_used
                )
                
                return result
                
            except Exception as e:
                self.logger.error(f"Q&A generation failed: {e}")
                self.stats["total_failed"] += 1
                raise
    
    def _assess_quality(self, output: str, chunk: TextChunk) -> dict:
        """
        Assess quality of generated Q&A pairs.
        
        Args:
            output: Generated Q&A text
            chunk: Source text chunk
            
        Returns:
            Dictionary of quality scores
        """
        scores = super()._assess_quality(output, chunk)
        
        # Q&A specific quality checks
        qa_pairs = self._parse_qa_pairs(output)
        
        # Check if we got the expected number of pairs
        expected_count = self.num_questions
        actual_count = len(qa_pairs)
        scores["count_accuracy"] = min(actual_count / expected_count, 1.0) if expected_count > 0 else 0.0
        
        # Check Q&A structure
        scores["structure_score"] = 1.0 if actual_count > 0 else 0.0
        
        # Check average lengths
        if qa_pairs:
            avg_q_length = sum(len(q.split()) for q, _ in qa_pairs) / len(qa_pairs)
            avg_a_length = sum(len(a.split()) for _, a in qa_pairs) / len(qa_pairs)
            
            # Good questions are 5-20 words
            scores["question_length"] = 1.0 if 5 <= avg_q_length <= 20 else 0.5
            
            # Good answers are 10-100 words
            scores["answer_length"] = 1.0 if 10 <= avg_a_length <= 100 else 0.7
        
        return scores
    
    def _parse_qa_pairs(self, output: str) -> list:
        """
        Parse Q&A pairs from output text.
        
        Args:
            output: Generated text with Q&A pairs
            
        Returns:
            List of (question, answer) tuples
        """
        pairs = []
        lines = output.split('\n')
        
        current_question = None
        current_answer = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('Q:') or line.startswith('Question:'):
                # Save previous pair if exists
                if current_question and current_answer:
                    pairs.append((current_question, current_answer))
                
                # Start new question
                current_question = line.split(':', 1)[1].strip() if ':' in line else line
                current_answer = None
                
            elif line.startswith('A:') or line.startswith('Answer:'):
                # Start answer
                current_answer = line.split(':', 1)[1].strip() if ':' in line else line
            
            elif current_answer is not None and line:
                # Continue answer
                current_answer += ' ' + line
        
        # Add last pair
        if current_question and current_answer:
            pairs.append((current_question, current_answer))
        
        return pairs
    
    def _format_input(self, chunk: TextChunk) -> str:
        """
        Format chunk for Q&A input.
        
        Args:
            chunk: Text chunk
            
        Returns:
            Formatted input text
        """
        return f"Text for Q&A generation:\n{chunk.content}"
    
    def _is_valid_qa_format(self, text: str) -> bool:
        """
        Validate that text contains Q&A format.
        
        Args:
            text: Generated text to validate
            
        Returns:
            True if text contains Q&A pairs, False otherwise
        """
        # Check for Q: and A: markers
        has_question = any(marker in text for marker in ["Q:", "Question:"])
        has_answer = any(marker in text for marker in ["A:", "Answer:"])
        
        # Must have both questions and answers
        return has_question and has_answer