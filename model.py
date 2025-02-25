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
        temperature: float = 0.1,
        context_window_size: int = 5
    ):
        """Initialize the Bangla RAG service."""
        self.embedding_model = CustomEmbeddings(embedding_model)
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.llm = Ollama(model=llm_model, temperature=temperature)
        self.llm_model_name = llm_model

        self.context_window: Deque[str] = deque(maxlen=context_window_size)
        self.current_source_passage = None
        self.state = "WAITING_FOR_QUESTION"  # State machine for multi-turn handling
        self.pending_question = None  # Store question during confirmation

        self._prepare_prompts()

    def _prepare_prompts(self) -> None:
        """Initialize prompts, including one for analyzing context switch responses."""
        self.relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine if the passage contains enough information to help answer the question.
            Answer YES or NO."""),
            ("user", "Question: {question}\nPassage: {passage}")
        ])

        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable chatbot. Answer the user's question in Bengali (বাংলা) professionally and approachably, using only certain information from the context."""),
            ("user", "Context: {context}\nQuestion: {question}\nHistory: {history}")
        ])

        self.no_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You don’t know the answer within the current context. Ask the user if they’d like help with a new topic in Bengali (বাংলা), tailoring the response to the question."""),
            ("user", "Context: {context}\nQuestion: {question}")
        ])

        self.context_switch_prompt = ChatPromptTemplate.from_messages([
            ("system", """The new question isn’t related to the previous context. Inform the user in Bengali (বাংলা) and ask if they want to switch, summarizing the current context."""),
            ("user", "Current Context: {current_context}\nNew Question: {question}")
        ])

        self.check_context_relevance_prompt = ChatPromptTemplate.from_messages([
            ("system", """Determine if the new question continues the previous conversation. Answer YES or NO."""),
            ("user", "Previous Context: {history}\nNew Question: {question}")
        ])

        # New prompt to analyze user response for context switch
        self.confirm_switch_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the user’s response to determine if they want to switch context.
            The user was asked if they want to switch context after a new, unrelated question.
            Respond with 'YES' if they agree to switch, 'NO' if they disagree, or 'UNCLEAR' if the response is ambiguous."""),
            ("user", "User Response: {response}")
        ])

    def _invoke_llm_stream(self, prompt: str) -> Generator[str, None, None]:
        """Stream LLM response."""
        for chunk in self.llm.stream(prompt):
            yield chunk

    def _invoke_llm(self, prompt: str) -> str:
        """Non-streaming LLM call."""
        return self.llm.invoke(prompt).strip()

    def _check_context_relevance(self, question: str) -> Tuple[bool, str, str]:
        """Check if the question relates to previous context."""
        history = "\n".join(self.context_window)
        prompt = self.check_context_relevance_prompt.format(history=history, question=question)
        relevance = self._invoke_llm(prompt)
        return relevance.strip().upper() == "YES", relevance, history

    def _update_state(self, question: str, answer: str, relevant: bool, passage_context=None) -> None:
        """Update conversation history and passage context."""
        if relevant and answer:
            entry = f"Question: {question}\nAnswer: {answer}"
            self.context_window.append(entry)
            if passage_context:
                self.current_source_passage = passage_context
        elif not relevant:
            self.context_window.clear()
            self.current_source_passage = None

    def process_query(self, input_str: str) -> Generator[str, None, None]:
        """Main processing pipeline with state management."""

        if self.state == "WAITING_FOR_QUESTION":
            question = input_str
            if not self.context_window:
                docs = self.retriever.invoke(question)
                if not docs:
                    yield "দুঃখিত, এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
                    return

                relevant_passage = None
                for doc in docs:
                    relevance_response = self._invoke_llm(self.relevance_prompt.format(question=question, passage=doc.page_content))
                    if relevance_response == "YES":
                        relevant_passage = doc.page_content
                        break

                if not relevant_passage:
                    self._update_state(question, None, False)
                    yield "দুঃখিত, এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
                    return

                answer_prompt = self.answer_prompt.format(
                    question=question,
                    context=relevant_passage,
                    history=""
                )

                full_answer = ""
                for chunk in self._invoke_llm_stream(answer_prompt):
                    full_answer += chunk
                    yield chunk

                self._update_state(question, full_answer, True, relevant_passage)
                return

            # Check relevance with existing history
            context_relevant, _, history = self._check_context_relevance(question)
            if context_relevant:
                answer_prompt = self.answer_prompt.format(
                    question=question,
                    context=self.current_source_passage,
                    history=history
                )

                full_answer = ""
                for chunk in self._invoke_llm_stream(answer_prompt):
                    full_answer += chunk
                    yield chunk

                self._update_state(question, full_answer, True, self.current_source_passage)
            else:
                # Ask for context switch
                current_context_summary = history if history else "কোনো পূর্ববর্তী প্রসঙ্গ নেই"
                switch_prompt = self.context_switch_prompt.format(
                    current_context=current_context_summary,
                    question=question
                )
                for chunk in self._invoke_llm_stream(switch_prompt):
                    yield chunk
                self.state = "WAITING_FOR_SWITCH_CONFIRMATION"
                self.pending_question = question
            return

        elif self.state == "WAITING_FOR_SWITCH_CONFIRMATION":
            # Analyze response with LLM
            confirm_prompt = self.confirm_switch_prompt.format(response=input_str)
            decision = self._invoke_llm(confirm_prompt).strip().upper()

            if decision == "YES":
                self.state = "WAITING_FOR_QUESTION"
                docs = self.retriever.invoke(self.pending_question)
                if not docs:
                    yield "দুঃখিত, এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
                    return

                relevant_passage = None
                for doc in docs:
                    relevance_response = self._invoke_llm(self.relevance_prompt.format(question=self.pending_question, passage=doc.page_content))
                    if relevance_response == "YES":
                        relevant_passage = doc.page_content
                        break

                if not relevant_passage:
                    self._update_state(self.pending_question, None, False)
                    yield "দুঃখিত, এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
                    return

                answer_prompt = self.answer_prompt.format(
                    question=self.pending_question,
                    context=relevant_passage,
                    history=""
                )

                full_answer = ""
                for chunk in self._invoke_llm_stream(answer_prompt):
                    full_answer += chunk
                    yield chunk

                self._update_state(self.pending_question, full_answer, True, relevant_passage)
                self.pending_question = None
            elif decision == "NO":
                self.state = "WAITING_FOR_QUESTION"
                yield "ঠিক আছে, আমি বর্তমান প্রসঙ্গে থাকব। আপনি কীভাবে এগিয়ে যেতে চান?"
                self.pending_question = None
            else:  # UNCLEAR
                yield "দুঃখিত, আপনার উত্তরটি পরিষ্কার নয়। দয়া করে আরেকবার বলুন, আপনি কি প্রসঙ্গ পরিবর্তন করতে চান?"
            return