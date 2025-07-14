
import requests
import json
import base64
import os
from dotenv import load_dotenv
from typing import Generator, Dict, Any

# Load environment variables from a .env file
load_dotenv()

class LLMService:
    """
    A synchronous client for an OpenAI-compatible LLM API.
    (The class definition remains unchanged from the previous version)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str
    ):
        """
        Initializes the client.

        Args:
            api_key (str): The API key for authentication.
            base_url (str): The base URL of the nginx-forwarded service.
            model (str): The name of the model to use for requests.
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.chat_url = f"{self.base_url}/v1/chat/completions"

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        print(f"✅ LLMService initialized for model '{self.model}' at endpoint: {self.chat_url}")

    def _process_stream(self, response: requests.Response) -> Generator[str, None, None]:
        """A helper generator to process and yield streaming content."""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data: '):
                    content = decoded_line[len('data: '):]
                    if content.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(content)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        text_chunk = delta.get("content")
                        if text_chunk:
                            yield text_chunk
                    except json.JSONDecodeError:
                        print(f"\n[Warning] Could not decode JSON chunk: {content}")

    def invoke(self, prompt: str, **kwargs: Any) -> str:
        """Sends a request for a single, complete response (non-streaming)."""
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "stream": False, **kwargs}
        
        try:
            response = requests.post(self.chat_url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"\n[Error] An error occurred during invoke: {e}")
            raise

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Connects to the streaming endpoint and yields text chunks as they arrive."""
        messages = [{"role": "user", "content": prompt}]
        payload = {"model": self.model, "messages": messages, "stream": True, **kwargs}
        
        try:
            response = requests.post(self.chat_url, headers=self.headers, json=payload, timeout=120, stream=True)
            response.raise_for_status()
            yield from self._process_stream(response)
        except requests.exceptions.RequestException as e:
            print(f"\n[Error] An error occurred during stream: {e}")
            raise

    def image_stream(self, prompt: str, image_path: str, **kwargs: Any) -> Generator[str, None, None]:
        """Sends an image and a text prompt for a streaming response."""
        try:
            with open(image_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"\n[Error] Image file not found at: {image_path}")
            raise

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        }]
        payload = {"model": self.model, "messages": messages, "stream": True, **kwargs}

        try:
            response = requests.post(self.chat_url, headers=self.headers, json=payload, timeout=180, stream=True)
            response.raise_for_status()
            yield from self._process_stream(response)
        except requests.exceptions.RequestException as e:
            print(f"\n[Error] An error occurred during image stream: {e}")
            raise

# -------------------------------------------------------------------------------------
# Example Usage - MODIFIED TO USE .ENV
# -------------------------------------------------------------------------------------

def main():
    """
    Demonstrates the usage of the OpenAI-compatible LLMService,
    loading configuration from the .env file.
    """
    # --- CHANGE: Load settings from environment variables ---
    base_url = os.getenv("LLM_MODEL_BASE_URL")
    api_key = os.getenv("LLM_MODEL_API_KEY")
    model_name=os.getenv("LLM_MODEL_NAME")

    # --- CHANGE: Add validation to ensure variables were loaded ---
    if not base_url or not api_key or not model_name:
        print("🔴 ERROR: Missing LLM_MODEL_BASE_URL or LLM_MODEL_API_KEY  or LLM_MODEL_NAME in the .env file.")
        print("Please create a .env file with the required variables.")
        return

    # Initialize the service client with the loaded settings
    service = LLMService(api_key=api_key, base_url=base_url,model=model_name)

    # --- The rest of the main function remains the same ---
    
    print("\n--- 1. Testing invoke() method ---")
    try:
        response_text = service.invoke(
            prompt="What is the capital of France and what is its most famous landmark?",
            max_tokens=100
        )
        print("\n[Full Response from invoke()]:\n", response_text)
    except Exception as e:
        print(f"\n[Invoke Failed]: {e}")
        
    print("\n" + "="*50 + "\n")

    print("--- 2. Testing stream() method ---")
    try:
        stream_prompt = "Tell me a short story about a robot who discovers music."
        print(f"[Streaming Response for: '{stream_prompt}']:\n")
        for chunk in service.stream(prompt=stream_prompt, max_tokens=256):
            print(chunk, end="", flush=True)
        print("\n\n--- Text stream finished ---")
    except Exception as e:
        print(f"\n[Stream Failed]: {e}")

    print("\n" + "="*50 + "\n")

    print("--- 3. Testing image_stream() method ---")
    try:
        image_prompt = "Describe this image in a few sentences. What is the main subject?"
        print(f"[Streaming Response for image with prompt: '{image_prompt}']:\n")
        for chunk in service.image_stream(prompt=image_prompt, image_path="./archive/Screenshot From 2025-07-14 17-07-27.png", max_tokens=200):
            print(chunk, end="", flush=True)
        print("\n\n--- Image stream finished ---")
    except Exception as e:
        print(f"\n[Image Stream Failed]: {e}")


if __name__ == "__main__":
    main()