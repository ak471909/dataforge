"""
OpenAI AI provider implementation.

This module provides integration with OpenAI's API for text generation.
"""

import asyncio
import time
from typing import List, Optional

from training_data_bot.core import (
    AIProviderError,
    RateLimitError,
    AuthenticationError,
    get_logger,
)
from training_data_bot.ai.providers.base import BaseAIProvider, AIResponse


class OpenAIProvider(BaseAIProvider):
    """OpenAI API provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        timeout: float = 30.0,
        **kwargs
    ):
        """Initialize OpenAI provider."""
        super().__init__(api_key, model, max_tokens, temperature, timeout, **kwargs)
        self.logger = get_logger("ai.OpenAIProvider")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import AsyncOpenAI
            
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout
            )
            
            self.logger.info(
                f"Initialized OpenAI provider with model: {self.model}"
            )
            
        except ImportError:
            raise AIProviderError(
                "OpenAI package not installed. Install with: pip install openai",
                provider="openai"
            )
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize OpenAI client: {e}",
                provider="openai",
                cause=e
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """Generate a response using OpenAI."""
        start_time = time.time()
        
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Get parameters
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k not in ['max_tokens', 'temperature']}
            )
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            
            # Get token usage
            tokens_used = response.usage.total_tokens if response.usage else 0
            
            self.logger.debug(
                f"Generated response: {tokens_used} tokens in {response_time:.2f}s"
            )
            
            return AIResponse(
                content=content,
                model=self.model,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                response_time=response_time,
                metadata={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Handle specific error types
            error_message = str(e).lower()
            
            if "rate_limit" in error_message or "429" in error_message:
                raise RateLimitError(
                    f"OpenAI rate limit exceeded: {e}",
                    provider="openai",
                    model=self.model,
                    cause=e
                )
            elif "authentication" in error_message or "401" in error_message:
                raise AuthenticationError(
                    f"OpenAI authentication failed: {e}",
                    provider="openai",
                    cause=e
                )
            else:
                raise AIProviderError(
                    f"OpenAI generation failed: {e}",
                    provider="openai",
                    model=self.model,
                    cause=e
                )
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[AIResponse]:
        """Generate responses for multiple prompts."""
        tasks = [
            self.generate(prompt, system_prompt, **kwargs)
            for prompt in prompts
        ]
        
        # Execute all tasks concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                self.logger.error(f"Batch generation failed for prompt {i}: {response}")
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        try:
            import tiktoken
            
            # Get encoding for the model
            try:
                encoding = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # Fallback to cl100k_base for newer models
                encoding = tiktoken.get_encoding("cl100k_base")
            
            tokens = encoding.encode(text)
            return len(tokens)
            
        except ImportError:
            # Fallback to rough approximation if tiktoken not available
            self.logger.warning(
                "tiktoken not installed, using rough token approximation"
            )
            return len(text.split()) * 4 // 3  # Rough approximation
    
    async def close(self):
        """Close the OpenAI client."""
        if self.client:
            await self.client.close()
            self.logger.info("Closed OpenAI client")