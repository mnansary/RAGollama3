# In chatservice.py

import json
from typing import Generator, Deque, Tuple, Dict, Any, List
from collections import deque

# Import your existing services and prompts
from core.RetriverService import RetrieverService
from core.LLMservice import LLMService
from core.prompts import (ANSWERING_PROMPT, 
                          FOLLOW_UP_CHECKER_PROMPT, 
                          REWRITER_SPECIALIST_PROMPT, 
                          DISAMBIGUATION_PROMPT, 
                          SUMMARIZER_PROMPT, 
                          SUGGESTER_PROMPT)

# Import configuration
from core.config import VECTOR_DB_PATH, EMBEDDING_MODEL, MODEL_NAME

class ChatService:
    def __init__(
        self,
        num_passages_to_retrieve: int = 1,
        context_window_size: int = 10,
        summary_size: int = 5
    ):
        """Initializes the main chat service with a hybrid RAG strategy."""
        self.retriever = RetrieverService(
            vector_db_path=VECTOR_DB_PATH,
            embedding_model_name=EMBEDDING_MODEL,
            num_passages_to_retrieve=num_passages_to_retrieve
        )
        self.llm_service = LLMService(llm_model=MODEL_NAME)
        self.context_window: Deque[Tuple[str, str]] = deque(maxlen=context_window_size)
        self.summary_size = summary_size

    def _get_history_summary(self) -> str:
        # ... (This function remains unchanged) ...
        if not self.context_window:
            return "No conversation history."
        history_list = list(self.context_window)
        return "\n".join([f"Q: {q}\nA: {a}" for q, a in history_list])
            
    def _generate_suggestions(self, context: str, question: str) -> List[str]:
        # ... (This function remains unchanged but will be skipped) ...
        try:
            prompt = SUGGESTER_PROMPT.format(context=context, question=question)
            suggestions_str = self.llm_service.invoke(prompt)
            return [line.split('. ', 1)[1] for line in suggestions_str.strip().split('\n') if '. ' in line]
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return []

    def chat(self, user_input: str) -> Generator[Dict[str, Any], None, None]:
        """
        Main chat entry point using a hybrid strategy based on conversation history.
        """
        history_summary = self._get_history_summary()
        full_answer = ""
        retrieved_passages = []
        question_for_generation = user_input

        # --- MODE 1: NEW CONVERSATION - "Retrieve First, Validate Second" ---
        if "No conversation history" in history_summary:
            yield {"type": "debug", "content": "Pipeline Mode: New Conversation. Strategy: Retrieve First."}
            
            # Step 1: Attempt direct retrieval
            yield {"type": "debug", "content": f"Step 1: Retrieving documents for initial query: '{user_input}'..."}
            retrieval_results = self.retriever.retrieve(user_input)
            retrieved_passages = retrieval_results.get("retrieved_passages", [])

            if not retrieved_passages:
                yield {"type": "debug", "content": "❌ Retrieval failed. No relevant documents found."}
                yield {"type": "error", "content": "দুঃখিত, আপনার প্রশ্নের সাথে সম্পর্কিত কোনো তথ্য খুঁজে পাওয়া যায়নি।"}
                return
            
            yield {"type": "debug", "content": f"✅ Retrieval successful. Found {len(retrieved_passages)} passages. Now validating context..."}
            combined_context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_passages])

            # Step 2: Attempt to generate a definitive answer
            answer_prompt = ANSWERING_PROMPT.format(history=history_summary, context=combined_context, question=user_input)
            full_answer = self.llm_service.invoke(answer_prompt)

            # Step 3: Assess the answer. If not definitive, ask for clarification.
            if "NOT_SURE_ANSWER" in full_answer or not full_answer.strip():
                yield {"type": "debug", "content": "⚠️ LLM couldn't form a definitive answer. The query is likely ambiguous. Asking for clarification."}
                disambiguation_prompt = DISAMBIGUATION_PROMPT.format(history=history_summary, question=user_input)
                clarification_question = self.llm_service.invoke(disambiguation_prompt)
                
                if clarification_question != "NOT_AMBIGUOUS":
                    yield {"type": "clarification", "content": clarification_question}
                else:
                    yield {"type": "error", "content": "আমি আপনার প্রশ্নটি বুঝতে পেরেছি, কিন্তু আমার কাছে এই মুহূর্তে কোনো নির্দিষ্ট উত্তর নেই।"}
                return
            
            # If the answer is definitive, stream it
            for chunk in full_answer:
                yield {"type": "answer_chunk", "content": chunk}

        # --- MODE 2: ONGOING CONVERSATION - Now with 2-Step Logic ---
        else:
            yield {"type": "debug", "content": "Pipeline Mode: Ongoing Conversation. Strategy: Decide, then Rewrite."}
            
            # Create the history string once
            raw_history = "\n".join([f"Q: {q}\nA: {a}" for q, a in self.context_window])
            
            # --- Step 2a: The Gatekeeper ---
            yield {"type": "debug", "content": "Step 1: Checking if question is a follow-up..."}
            checker_prompt = FOLLOW_UP_CHECKER_PROMPT.format(history=raw_history, new_question=user_input)
            decision = self.llm_service.invoke(checker_prompt)

            # --- Step 2b: The Decision ---
            if decision == "FOLLOW_UP":
                yield {"type": "debug", "content": "✅ Follow-up detected. Calling specialist rewriter..."}
                
                # Call the specialist rewriter prompt
                rewriter_prompt = REWRITER_SPECIALIST_PROMPT.format(history=raw_history, new_question=user_input)
                rewritten_question = self.llm_service.invoke(rewriter_prompt)
                
                question_for_retrieval = rewritten_question
                yield {"type": "debug", "content": f"✅ Rewritten question: '{question_for_retrieval}'"}
            else: # Decision is NEW_TOPIC
                question_for_retrieval = user_input
                yield {"type": "debug", "content": "✅ New topic detected. Clearing conversation history."}
                self.context_window.clear()
            
            question_for_generation = question_for_retrieval

            # --- The rest of the pipeline continues as before ---
            yield {"type": "debug", "content": "Step 2: Retrieving documents..."}
            retrieval_results = self.retriever.retrieve(question_for_retrieval)
            retrieved_passages = retrieval_results.get("retrieved_passages", [])

            if not retrieved_passages:
                yield {"type": "debug", "content": "❌ Retrieval failed."}
                yield {"type": "error", "content": "দুঃখিত, এই সম্পর্কিত কোনো তথ্য খুঁজে পাওয়া যায়নি।"}
                return
            
            yield {"type": "debug", "content": f"✅ Retrieval successful. Found {len(retrieved_passages)} passages."}
            combined_context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_passages])

            yield {"type": "debug", "content": "Step 3: Generating answer..."}
            answer_prompt = ANSWERING_PROMPT.format(history=history_summary, context=combined_context, question=question_for_retrieval)
            
            for chunk in self.llm_service.stream(answer_prompt):
                if "NOT_SURE_ANSWER" in full_answer + chunk: break
                full_answer += chunk
                yield {"type": "answer_chunk", "content": chunk}

            if "NOT_SURE_ANSWER" in full_answer or not full_answer.strip():
                 yield {"type": "debug", "content": "❌ LLM could not answer from context."}
                 yield {"type": "error", "content": "দুঃখিত, এই বিষয়টি সম্পর্কে আমার কাছে কোনো নির্দিষ্ট তথ্য নেই।"}
                 return
        
        # --- COMMON FINAL STEPS ---
        yield {"type": "debug", "content": "✅ Answer generation complete."}
        
        # Post-Generation: Citations and Suggestions
        sources = list(set([doc["url"] for doc in retrieved_passages if doc.get("url")]))
        
        # *** NEW LOGIC: Suggestions are now skipped ***
        yield {"type": "debug", "content": "Skipping follow-up suggestions as per configuration."}
        suggestions = []

        # Update State & Yield Final Payload
        # Use full_answer which was accumulated during the streaming loop
        self.context_window.append((user_input, full_answer))
        final_payload = {"sources": sources, "suggested_questions": suggestions}
        yield {"type": "final_data", "content": final_payload}