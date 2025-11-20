"""
LLM client module for OpenAI.
"""
import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        pass


class OpenAIClient(LLMClient):
    """Client for OpenAI's API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var
            model: Model name to use
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install openai: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using GPT.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated text response
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=2048
        )

        return response.choices[0].message.content


def get_llm_client(
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMClient:
    """
    Factory function to get an OpenAI LLM client.

    Args:
        api_key: API key for OpenAI
        model: Optional model name override

    Returns:
        LLM client instance
    """
    kwargs = {"api_key": api_key}
    if model:
        kwargs["model"] = model
    return OpenAIClient(**kwargs)
