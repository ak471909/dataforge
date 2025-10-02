"""
Anthropic AI provider implementation.

This module provides integration with Anthropic's Claude API for text generation.
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


class AnthropicProvider(BaseAIProvider):
    """Anthropic Claude API provider implementation."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-sonnet-20240229",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        timeout: float = 30.0,
        **kwargs
    ):
        """Initialize Anthropic provider."""
        super().__init__(api_key, model, max_tokens, temperature, timeout, **kwargs)
        self.logger = get_logger("ai.AnthropicProvider")
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Anthropic client."""
        try:
            from anthropic import AsyncAnthropic
            
            self.client = AsyncAnthropic(
                api_key=self.api_key,
                timeout=self.timeout
            )
            
            self.logger.info(
                f"Initialized Anthropic provider with model: {self.model}"
            )
            
        except ImportError:
            raise AIProviderError(
                "Anthropic package not installed. Install with: pip install anthropic",
                provider="anthropic"
            )
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize Anthropic client: {e}",
                provider="anthropic",
                cause=e
            )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """Generate a response using Anthropic Claude."""
        start_time = time.time()
        
        try:
            # Get parameters
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            temperature = kwargs.get('temperature', self.temperature)
            
            # Build request parameters
            request_params = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            # Add any extra parameters
            request_params.update({
                k: v for k, v in kwargs.items() 
                if k not in ['max_tokens', 'temperature']
            })
            
            # Make API call
            response = await self.client.messages.create(**request_params)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Extract response data
            content = response.content[0].text
            finish_reason = response.stop_reason
            
            # Get token usage
            tokens_used = (
                response.usage.input_tokens + response.usage.output_tokens
                if response.usage else 0
            )
            
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
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Handle specific error types
            error_message = str(e).lower()
            
            if "rate_limit" in error_message or "429" in error_message:
                raise RateLimitError(
                    f"Anthropic rate limit exceeded: {e}",
                    provider="anthropic",
                    model=self.model,
                    cause=e
                )
            elif "authentication" in error_message or "401" in error_message:
                raise AuthenticationError(
                    f"Anthropic authentication failed: {e}",
                    provider="anthropic",
                    cause=e
                )
            else:
                raise AIProviderError(
                    f"Anthropic generation failed: {e}",
                    provider="anthropic",
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
        """Count tokens using Anthropic's tokenizer."""
        try:
            # Anthropic uses a similar tokenization to GPT
            # Rough approximation: ~4 chars per token
            return len(text) // 4
            
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}, using rough approximation")
            return len(text.split()) * 4 // 3  # Rough approximation
    
    async def close(self):
        """Close the Anthropic client."""
        if self.client:
            await self.client.close()
            self.logger.info("Closed Anthropic client")