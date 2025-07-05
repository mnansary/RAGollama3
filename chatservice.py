import json
import sys
from typing import Dict, Any, Deque, Tuple, Generator
import re
# Use standard library deque for efficient history management
from collections import deque

# Import all our custom components
from core.RetriverService import RetrieverService
from core.LLMservice import LLMService
from core.prompts import ANALYST_PROMPT, STRATEGIST_PROMPTS
from core.config import VECTOR_DB_PATH, MODEL_NAME,EMBEDDING_MODEL

class ProactiveChatService:
    def __init__(self, history_length: int = 100):
        # NOTE: num_passages_to_retrieve is removed from here.
        print("Initializing ProactiveChatService...")
        self.retriever = RetrieverService(vector_db_path=VECTOR_DB_PATH,embedding_model=EMBEDDING_MODEL) # Simplified init
        self.llm_service = LLMService(llm_model=MODEL_NAME)
        self.history: Deque[Tuple[str, str]] = deque(maxlen=history_length)
        print(f"✅ ProactiveChatService initialized successfully. History window: {history_length} turns.")

    def _format_history(self) -> str:
        """Formats the conversation history into a readable string for prompts."""
        if not self.history:
            return "No conversation history yet."
        return "\n".join([f"User: {user_q}\nAI: {ai_a}" for user_q, ai_a in self.history])

    def _run_analyst_stage(self, user_input: str, history_str: str) -> Dict[str, Any] | None:
        """
        Executes the Analyst stage: gets a structured JSON plan from the LLM.
        This version is robust against conversational text surrounding the JSON object.
        """
        print("\n----- 🕵️ Analyst Stage -----")
        prompt = ANALYST_PROMPT.format(history=history_str, question=user_input)
        try:
            response_text = self.llm_service.invoke(
                prompt,
                temperature=0.0,
                max_tokens=51200
            )
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                json_str = match.group(0)
                plan = json.loads(json_str)
                print("✅ Analyst plan generated and extracted successfully:")
                #print(json.dumps(plan, indent=2))
                return plan
            else:
                print(f"❌ CRITICAL: Analyst stage failed. No valid JSON block found in the response.")
                print(f"LLM Response was: {response_text}")
                return None
        except json.JSONDecodeError as e:
            print(f"❌ CRITICAL: Analyst stage failed with JSONDecodeError: {e}")
            print(f"Problematic JSON string was: {json_str}")
            return None
        except Exception as e:
            print(f"❌ CRITICAL: An unexpected error occurred in the Analyst stage: {e}")
            return None
                
    def _run_retriever_stage(self, plan: Dict[str, Any]) -> Tuple[str, list]:
        """Executes the Retriever stage based on the Analyst's detailed plan."""
        print("\n----- 📚 Retriever Stage -----")
        query = plan.get("query_for_retriever", "")
        k = plan.get("k_for_retriever", 3)
        filters = plan.get("metadata_filter", None)

        if k == 0 or not query:
            print("Skipping retrieval as per Analyst's plan.")
            return "No retrieval was performed.", []

        print(f"🔍 Querying retriever with: '{query}', k={k}, filters={filters}")
        retrieval_results = self.retriever.retrieve(query, k=k, filters=filters)
        retrieved_passages = retrieval_results.get("retrieved_passages", [])

        if not retrieved_passages:
            print("⚠️ Retriever found no documents.")
            return "No information found matching the criteria.", []
        
        combined_context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_passages])
        print(f"✅ Retriever found {len(retrieved_passages)} documents.")
        return combined_context, retrieved_passages

    def _run_strategist_stage(self, plan: Dict[str, Any], context: str, user_input: str, history_str: str) -> Generator[str, None, None]:
        """Executes the Strategist stage: returns a generator that streams the final response."""
        print("\n----- 🎭 Strategist Stage -----")
        strategy = plan.get("response_strategy", "RESPOND_WARMLY")
        print(f"✍️ Executing strategy: '{strategy}'")

        prompt_template = STRATEGIST_PROMPTS.get(strategy)
        if not prompt_template:
            print(f"❌ WARNING: Invalid strategy '{strategy}'. Defaulting.")
            prompt_template = STRATEGIST_PROMPTS["RESPOND_WARMLY"]

        prompt = prompt_template.format(
            context=context,
            question=user_input,
            history=history_str
        )
        return self.llm_service.stream(
            prompt,
            temperature=0.0,
            max_tokens=120000,
            top_p=0.95,
            repetition_penalty=1.2
        )

    def chat(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """Main entry point, orchestrating the new, more robust pipeline."""
        print(f"\n==================== NEW CHAT TURN: User said '{user_input}' ====================")
        history_str = self._format_history()

        plan = self._run_analyst_stage(user_input, history_str)
        if not plan:
            yield {"type": "error", "content": "I'm sorry, I'm having a little trouble. Could you rephrase?"}
            return

        combined_context, retrieved_passages = self._run_retriever_stage(plan)
        
        answer_generator = self._run_strategist_stage(plan, combined_context, user_input, history_str)
        
        full_answer_list = []
        for chunk in answer_generator:
            full_answer_list.append(chunk)
            yield {
                "type": "answer_chunk",
                "content": chunk
            }
        
        final_answer = "".join(full_answer_list).strip()
        self.history.append((user_input, final_answer))

        # --- START OF THE FIX ---
        sources = []
        if retrieved_passages:
            # Use a set to automatically handle duplicates, then convert to a list
            unique_urls = set()
            for doc in retrieved_passages:
                # Safely get the metadata and then the URL
                if doc.get("metadata") and doc["metadata"].get("url"):
                    unique_urls.add(doc["metadata"]["url"])
            sources = list(unique_urls)
        print(sources)
        # --- END OF THE FIX ---

        yield {
            "type": "final_data",
            # Use the Bengali key "তথ্যসূত্র" as intended
            "content": {"sources": sources}
        }
        print("\n-------------------- STREAM COMPLETE --------------------")

if __name__ == "__main__":
    # 1. Initialize the service
    chat_service = ProactiveChatService(history_length=5)

    # 2. Define test cases
    test_conversation = [
        "জন্ম নিবন্ধন করার প্রক্রিয়া কি?",
        "বিআরটিএ অফিসের ফোন নাম্বার কি?",
        "আপনি কি আমার হয়ে একটি আবেদনপত্র পূরণ করে দিতে পারবেন?",
        "আচ্ছা, বাংলাদেশের বর্তমান প্রধানমন্ত্রীর নাম কি?",
    ]

    # 3. Loop through the conversation and process each turn
    for turn in test_conversation:
        print(f"\n\n\n>>>>>>>>>>>>>>>>>> User Input: {turn} <<<<<<<<<<<<<<<<<<")
        print("\n<<<<<<<<<<<<<<<<<< Bot Response >>>>>>>>>>>>>>>>>>")

        final_sources = []
        try:
            # The client code iterates through the generator yielded by chat()
            for event in chat_service.chat(turn):
                if event["type"] == "answer_chunk":
                    # Print each chunk as it arrives to simulate a streaming UI
                    print(event["content"], end="", flush=True)
                elif event["type"] == "final_data":
                    # Capture the sources from the final event
                    final_sources = event["content"].get("তথ্যসূত্র", [])
                elif event["type"] == "error":
                    print(event["content"], end="", flush=True)
            
            # After the stream is complete, print the sources if any were found
            if final_sources:
                print(f"\n\n[তথ্যসূত্র: {', '.join(final_sources)}]")

        except Exception as e:
            print(f"\n\nAn unexpected error occurred: {e}")

        print("\n<<<<<<<<<<<<<<<<<< End of Response >>>>>>>>>>>>>>>>>>")