import os
from typing import Optional
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
from src.agent_tiers.infrastructure.config import settings


import openai


def get_openai_model(model_choice: Optional[str] = None) -> OpenAIChatModel:
    """
    Get OpenAI model configuration based on environment variables.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI-compatible model
    """
    llm_model = model_choice or settings.LLM_MODEL
    api_key = settings.OPENAI_API_KEY

    provider = OpenAIProvider(api_key=api_key)
    return OpenAIChatModel(llm_model, provider=provider)


def get_openai_embedding_client() -> openai.AsyncOpenAI:
    """
    Get OpenAI embedding client configuration based on environment variables.

    Returns:
        Configured OpenAI-compatible client for embeddings
    """
    api_key = settings.OPENAI_API_KEY

    return openai.AsyncOpenAI(api_key=api_key)


def get_openai_embedding_model() -> str:
    """
    Get OpenAI embedding model name from environment.

    Returns:
        Embedding model name
    """
    return settings.EMBEDDING_MODEL


def get_ollama_model(model_choice: Optional[str] = None) -> OpenAIChatModel:
    """
    Get Ollama model configuration from local server.

    Args:
        model_choice: Optional override for model choice 

    Returns:
        Configured Ollama model via OpenAI-compatible endpoint
    """
    base_url = settings.OLLAMA_BASE_URL
    
    llm_model = model_choice or settings.OLLAMA_LLM_MODEL
    
    api_key = "ollama" 

    provider = OpenAIProvider(api_key=api_key, base_url=base_url)
    
    return OpenAIChatModel(llm_model, provider=provider)


def get_ollama_embedding_client() -> openai.AsyncOpenAI:
    """
    Get Ollama embedding client configuration from local server.

    Returns:
        Configured Ollama client for embeddings via OpenAI-compatible endpoint
    """
    base_url = settings.OLLAMA_BASE_URL 
    
    api_key = "ollama"

    return openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

