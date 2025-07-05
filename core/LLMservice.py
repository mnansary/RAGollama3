from typing import Generator, Any
from langchain_community.llms import Ollama
from .config import MODEL_NAME # Assuming MODEL_NAME is in your config file

class LLMService:
    def __init__(self, llm_model: str = MODEL_NAME, **default_params: Any):
        """
        Initializes a flexible Language Model Service.
        This service wraps the Ollama client and allows for both default and
        per-call parameter configurations.

        Args:
            llm_model (str): The name of the Ollama model to use (e.g., 'llama3').
            **default_params: Default parameters for the Ollama model
                              (e.g., temperature=0.1, num_ctx=8192). These can
                              be overridden in each `invoke` or `stream` call.
        """
        # Set default temperature if not provided
        if 'temperature' not in default_params:
            default_params['temperature'] = 0.1

        self.llm = Ollama(model=llm_model, **default_params)
        self.llm_model_name = llm_model
        print(f"✅ LLMService initialized with model: {self.llm_model_name} and defaults: {default_params}")

        # A warm-up call to load the model into memory for faster subsequent responses.
        print("Warming up the LLM... (this may take a moment)")
        # Use invoke directly on the Ollama instance to avoid recursion
        self.llm.invoke("Respond only with OK.")
        print("LLM is ready.")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """
        Invokes the LLM with a given prompt and returns the complete response.
        Allows for overriding default LLM parameters on a per-call basis.

        Args:
            prompt (str): The complete, formatted prompt string to send to the LLM.
            **kwargs: Variable keyword arguments to pass to the Ollama client
                      (e.g., temperature=0.7, max_tokens=512, top_p=0.9).
                      These will override the defaults set during initialization.

        Returns:
            str: The LLM's response, stripped of leading/trailing whitespace.
        """
        return self.llm.invoke(prompt, **kwargs).strip()

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Invokes the LLM with a given prompt and streams the response chunk by chunk.
        Allows for overriding default LLM parameters on a per-call basis.

        Args:
            prompt (str): The complete, formatted prompt string to send to the LLM.
            **kwargs: Variable keyword arguments to pass to the Ollama client
                      (e.g., temperature=0.7, max_tokens=4000, top_p=0.9).
                      These will override the defaults set during initialization.

        Yields:
            Generator[str, None, None]: A generator that yields response chunks as they are generated.
        """
        for chunk in self.llm.stream(prompt, **kwargs):
            yield chunk