from typing import Dict, Generator, List, Tuple, Deque
from collections import deque
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from embedding import CustomEmbeddings

class BanglaRAGService:
    def __init__(
        self,
        vector_db_path: str = "data/20240901/database",
        embedding_model: str = "l3cube-pune/bengali-sentence-similarity-sbert",
        llm_model: str = "llama3.3",
        temperature: float = 0.2,
        context_window_size: int = 5
    ):
        """Initialize the Bangla RAG service."""
        self.embedding_model = CustomEmbeddings(embedding_model)
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 docs

        self.llm = Ollama(model=llm_model, temperature=temperature)

        self.context_window: Deque[str] = deque(maxlen=context_window_size)

        self._prepare_prompts()

    def _prepare_prompts(self) -> None:
        """Initialize prompts for relevance checking and answer generation."""
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine if the passage contains enough information to help answer the question.
            The passage does not need to contain the exact answer word-for-word, but it should provide useful details.
            If the passage contains **some relevant information** that helps answer the question, answer YES.
            If the passage is completely unrelated, answer NO.
            
            Respond ONLY with "YES" or "NO".
            """),
            ("user", "Question: {question}\nPassage: {passage}")
        ])

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable chatbot, designed to assist users with their inquiries in a detailed and informative manner. 
            Your responses should answer the user's questions , provide additional context, relevant examples, and insights related to the question at hand. 
            Ensure your tone is professional, yet approachable, and remember to communicate in Bengali (বাংলা).
            Please provide a response that is not long. 
            """),
            ("user", "Context: {context}\nQuestion: {question}\nHistory: {history}")
        ])

        self.no_answer_prompt=ChatPromptTemplate.from_messages([
            ("system", """You do not not the answer to the given question within current context. Its not answerable. 
            Ask the user if you could help with a new topic. But your response can not be generic. Response according to the context and question.
            Ensure your tone is professional, yet approachable, and remember to communicate in Bengali (বাংলা).
            Please provide a response that is not long. 
            """),
            ("user", "Context: {context}\nQuestion: {question}")
        ])

    def _invoke_llm_stream(self, prompt: str) -> Generator[str, None, None]:
        """Helper function to invoke the LLM with streaming."""
        try:
            for chunk in self.llm.stream(prompt):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"

    def _invoke_llm(self, prompt: str) -> str:
        """Helper function for non-streaming LLM calls (used for relevance checking)."""
        try:
            return self.llm.invoke(prompt).strip()
        except Exception as e:
            return f"Error: {str(e)}"

    def _check_context_relevance(self, question: str) -> Tuple[bool, str, str]:
        """Determine if the question relates to the previous context and can be answered."""
        history = "\n".join(self.context_window)
        prompt = f"""Determine if this new question relates to the previous conversation AND can be answered using the history as context.
        The history does not need to contain the exact answer word-for-word, but it should provide useful details.
        If the history contains **some relevant information** that helps answer the question, answer YES.
        If the history is completely unrelated, answer NO.
        Respond ONLY with "YES" or "NO".
        
        Previous Context:
        {history}
        
        New Question: {question}"""

        relevance = self._invoke_llm(prompt)
        return relevance.strip().upper() == "YES", relevance, history

    def _update_state(self, question: str, answer: str, relevant: bool) -> None:
        """Update conversation history based on relevance."""
        if relevant and answer:
            entry = f"Q: {question}\nA: {answer}"
            self.context_window.append(entry)
        else:
            self.context_window.clear()

    def process_query(self, question: str) -> Generator[str, None, None]:
        """Main processing pipeline for handling user queries with streaming."""
        self.current_source_passage = None

        # No history scenario
        if not self.context_window:
            docs = self.retriever.invoke(question)
            if not docs:
                yield "এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
                return

            relevant_passage = None
            for doc in docs:
                relevance_response = self._invoke_llm(self.relevance_prompt.format(question=question, passage=doc.page_content))
                if relevance_response == "YES":
                    relevant_passage = doc.page_content
                    break

            if not relevant_passage:
                self._update_state(question, None, False)
                yield "এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
                return

            self.current_source_passage = relevant_passage
            answer_prompt = self.answer_prompt.format(
                question=question,
                context=relevant_passage,
                history=""
            )

            full_answer = ""
            for chunk in self._invoke_llm_stream(answer_prompt):
                full_answer += chunk
                yield chunk  # Stream the response

            self._update_state(question, full_answer, True)
            return

        # History exists, check relevance
        context_relevant, context_response, history = self._check_context_relevance(question)
        if context_relevant:
            answer_prompt = self.answer_prompt.format(
                question=question,
                context=self.current_source_passage,
                history=history
            )

            full_answer = ""
            for chunk in self._invoke_llm_stream(answer_prompt):
                full_answer += chunk
                yield chunk  # Stream response

            self._update_state(question, full_answer, True)
            return

        # If context is not relevant, use the "no answer" prompt
        no_answer_prompt = self.no_answer_prompt.format(
            question=question,
            context=self.current_source_passage
        )

        full_response = ""
        for chunk in self._invoke_llm_stream(no_answer_prompt):
            full_response += chunk
            yield chunk  # Stream response

        self._update_state(question, None, False)
