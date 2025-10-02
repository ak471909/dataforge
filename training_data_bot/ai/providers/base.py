"""
Base AI provider interface.

This module defines the abstract base class that all AI providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class AIResponse:
    """Response from an AI provider."""
    content: str
    model: str
    tokens_used: int
    finish_reason: str
    response_time: float
    metadata: Dict[str, Any]


class BaseAIProvider(ABC):
    """
    Abstract base class for AI providers.
    
    All AI providers (OpenAI, Anthropic, etc.) must implement this interface.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        timeout: float = 30.0,
        **kwargs
    ):
        """
        Initialize the AI provider.
        
        Args:
            api_key: API key for the provider
            model: Model name to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0 to 2.0)
            timeout: Request timeout in seconds
            **kwargs: Additional provider-specific parameters
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.extra_params = kwargs
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    async def generate_batch(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[AIResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of user prompts
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            List of AIResponse objects
        """
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Close any open connections or cleanup resources."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.__class__.__name__,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "timeout": self.timeout
        }
    
    def __repr__(self) -> str:
        """String representation of the provider."""
        return f"{self.__class__.__name__}(model={self.model})"