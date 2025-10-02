"""
Main AI client for managing AI providers.

This module provides a unified interface for interacting with different
AI providers (OpenAI, Anthropic, etc.).
"""

from typing import Dict, List, Optional, Union
import asyncio

from training_data_bot.core import (
    AIProviderConfig,
    AIProviderError,
    ConfigurationError,
    get_logger,
    get_performance_logger,
    LogContext,
)
from training_data_bot.ai.providers import (
    BaseAIProvider,
    AIResponse,
    OpenAIProvider,
    AnthropicProvider,
)


class AIClient:
    """
    Main AI client for managing providers and generating responses.
    
    Provides a unified interface for interacting with different AI providers.
    """
    
    # Registry of available providers
    PROVIDERS = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
    }
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the AI client.
        
        Args:
            provider: Provider name ("openai" or "anthropic")
            api_key: API key for the provider
            model: Model name to use
            **kwargs: Additional provider-specific parameters
        """
        self.logger = get_logger("ai.AIClient")
        self.perf_logger = get_performance_logger()
        
        self.provider_name = provider.lower()
        self.provider_instance: Optional[BaseAIProvider] = None
        
        # Initialize provider
        self._initialize_provider(api_key, model, **kwargs)
    
    def _initialize_provider(
        self,
        api_key: Optional[str],
        model: Optional[str],
        **kwargs
    ):
        """Initialize the AI provider."""
        if self.provider_name not in self.PROVIDERS:
            raise ConfigurationError(
                f"Unknown AI provider: {self.provider_name}. "
                f"Available providers: {', '.join(self.PROVIDERS.keys())}"
            )
        
        provider_class = self.PROVIDERS[self.provider_name]
        
        # Validate API key
        if not api_key:
            raise ConfigurationError(
                f"API key required for {self.provider_name} provider"
            )
        
        # Create provider instance
        try:
            self.provider_instance = provider_class(
                api_key=api_key,
                model=model or self._get_default_model(),
                **kwargs
            )
            
            self.logger.info(
                f"Initialized AI client with {self.provider_name} provider",
                provider=self.provider_name,
                model=self.provider_instance.model
            )
            
        except Exception as e:
            raise AIProviderError(
                f"Failed to initialize {self.provider_name} provider: {e}",
                provider=self.provider_name,
                cause=e
            )
    
    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        defaults = {
            "openai": "gpt-3.5-turbo",
            "anthropic": "claude-3-sonnet-20240229",
        }
        return defaults.get(self.provider_name, "gpt-3.5-turbo")
    
    @classmethod
    def from_config(cls, config: AIProviderConfig) -> "AIClient":
        """
        Create AI client from configuration.
        
        Args:
            config: AI provider configuration
            
        Returns:
            Configured AIClient instance
        """
        return cls(
            provider=config.provider_name,
            api_key=config.api_key,
            model=config.model_name,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout,
        )
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> AIResponse:
        """
        Generate a response from the AI model.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            AIResponse object with generated content
        """
        with LogContext("ai_generate", component="AIClient"):
            self.logger.info("Generating AI response")
            
            try:
                response = await self.provider_instance.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    **kwargs
                )
                
                # Log performance metrics
                self.perf_logger.log_api_call(
                    provider=self.provider_name,
                    model=response.model,
                    tokens_used=response.tokens_used,
                    response_time=response.response_time,
                    success=True
                )
                
                return response
                
            except Exception as e:
                self.logger.error(f"AI generation failed: {e}")
                raise
    
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        max_concurrent: int = 5,
        **kwargs
    ) -> List[AIResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional generation parameters
            
        Returns:
            List of AIResponse objects
        """
        with LogContext("ai_generate_batch", component="AIClient"):
            self.logger.info(
                f"Generating {len(prompts)} AI responses",
                batch_size=len(prompts),
                max_concurrent=max_concurrent
            )
            
            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def generate_with_semaphore(prompt):
                async with semaphore:
                    try:
                        return await self.generate(prompt, system_prompt, **kwargs)
                    except Exception as e:
                        self.logger.error(f"Batch generation failed for prompt: {e}")
                        return None
            
            # Execute all tasks
            tasks = [generate_with_semaphore(p) for p in prompts]
            responses = await asyncio.gather(*tasks)
            
            # Filter out None responses
            valid_responses = [r for r in responses if r is not None]
            
            self.logger.info(
                f"Batch generation complete",
                total_prompts=len(prompts),
                successful=len(valid_responses),
                failed=len(prompts) - len(valid_responses)
            )
            
            return valid_responses
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return self.provider_instance.count_tokens(text)
    
    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Estimate cost for token usage (rough approximation).
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Estimated cost in USD
        """
        # Rough cost estimates (as of 2024)
        costs = {
            "openai": {
                "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
                "gpt-4": {"input": 0.03, "output": 0.06},
            },
            "anthropic": {
                "claude-3-sonnet": {"input": 0.003, "output": 0.015},
                "claude-3-opus": {"input": 0.015, "output": 0.075},
            }
        }
        
        # Find matching cost structure
        provider_costs = costs.get(self.provider_name, {})
        model_name = self.provider_instance.model.lower()
        
        # Find matching model
        model_costs = None
        for model_key, cost_data in provider_costs.items():
            if model_key in model_name:
                model_costs = cost_data
                break
        
        if not model_costs:
            # Default rough estimate
            return (input_tokens + output_tokens) * 0.002 / 1000
        
        input_cost = (input_tokens / 1000) * model_costs["input"]
        output_cost = (output_tokens / 1000) * model_costs["output"]
        
        return input_cost + output_cost
    
    def get_provider_info(self) -> Dict[str, any]:
        """
        Get information about the current provider.
        
        Returns:
            Dictionary with provider information
        """
        return {
            "provider": self.provider_name,
            **self.provider_instance.get_model_info()
        }
    
    async def close(self):
        """Close the AI client and cleanup resources."""
        if self.provider_instance:
            await self.provider_instance.close()
            self.logger.info("Closed AI client")
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return f"AIClient(provider={self.provider_name}, model={self.provider_instance.model})"