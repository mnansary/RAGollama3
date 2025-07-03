# LLMservice.py

from typing import Generator
from langchain_community.llms import Ollama
from .config import MODEL_NAME # Assuming MODEL_NAME is in your config file

class LLMService:
    def __init__(self, llm_model: str = MODEL_NAME, temperature: float = 0.0,num_ctx=70000):
        """
        Initializes a lean Language Model Service.
        This service is a thin wrapper around the Ollama client, responsible
        only for sending prompts and receiving responses.

        Args:
            llm_model (str): The name of the Ollama model to use (e.g., 'llama3').
            temperature (float): The temperature for LLM generation (0.0 for deterministic output).
        """
        self.llm = Ollama(model=llm_model, temperature=temperature, num_ctx=num_ctx)
        self.llm_model_name = llm_model
        print(f"✅ LLMService initialized with model: {self.llm_model_name}")

        # A warm-up call to load the model into memory for faster subsequent responses.
        print("Warming up the LLM... (this may take a moment)")
        self.invoke("Respond only with OK.")
        print("LLM is ready.")

    def invoke(self, prompt: str) -> str:
        """
        Invokes the LLM with a given prompt and returns the complete response as a single string.

        Args:
            prompt (str): The complete, formatted prompt string to send to the LLM.

        Returns:
            str: The LLM's response, stripped of leading/trailing whitespace.
        """
        return self.llm.invoke(prompt).strip()

    def stream(self, prompt: str) -> Generator[str, None, None]:
        """
        Invokes the LLM with a given prompt and streams the response chunk by chunk.

        Args:
            prompt (str): The complete, formatted prompt string to send to the LLM.

        Yields:
            Generator[str, None, None]: A generator that yields response chunks as they are generated.
        """
        for chunk in self.llm.stream(prompt):
            yield chunk