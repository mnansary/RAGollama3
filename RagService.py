from typing import Generator, Deque, Tuple
from collections import deque
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import torch
from config import VECTOR_DB_PATH, EMBEDDING_MODEL, MODEL_NAME


# === Custom Embedding ===
class CustomEmbeddings:
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)

    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], convert_to_numpy=True).tolist()[0]


class BanglaRAGService:
    def __init__(
        self,
        vector_db_path: str = VECTOR_DB_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        llm_model: str = MODEL_NAME,
        temperature: float = 0.0,
        context_window_size: int = 10,
        summary_size:int=5
    ):
        self.embedding_model = CustomEmbeddings(embedding_model)
        self.vectorstore = Chroma(
            persist_directory=vector_db_path,
            embedding_function=self.embedding_model
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        self.llm = Ollama(model=llm_model, temperature=temperature, num_ctx=32000)
        self.llm_model_name = llm_model

        self.context_window: Deque[Tuple[str, str, str]] = deque(maxlen=context_window_size)
        self.current_source_passage = None
        self.summary_size = summary_size  # Number of past interactions to summarize
        self.debug_info = ""  # For displaying debug details in Gradio UI

        self._prepare_prompts()
        self._invoke_llm("This is just to get you going and loading the model to make new questions faster. No need to reply anything other than OK.")

    def _prepare_prompts(self) -> None:
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """
        You are a knowledgeable chatbot. Answer the user's question in Bengali (বাংলা) using only the provided context.

        Guidelines:
        - Your tone should indicate that you are answering from your knowledge base. Do not speak as if the user already possesses the information (e.g., avoid phrases like "আপনার কাছে আছে"). Instead, present facts directly and neutrally.
        - Answer sufficiently and concisely.
        - Use the context and past conversation to answer the question precisely.
        - Do not rephrase or repeat the question in your answer unnecessarily. If it is necessary to refer to the question, do so briefly.
        - Use the context provided to answer the question.
        - If unsure or context is insufficient, respond exactly with 'NOT_SURE_ANSWER'.
             """),
            ("user", "Conversation History Summary:\n{history}\n\nContext:\n{context}\n\nRewritten Question: {question}\nOriginal Question: {original}")
        ])

        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a helpful assistant. 
            - Based on the summary of a past conversation and a new question, decide whether the new question relates to the past conversation.
            - If it does, rewrite the new question to be standalone and unambiguous. 
                - Replace vague terms like "এখানে", "ওখানে", "তিনি", etc., with explicit references from the summary.
                - Make sure the rewritten question is specific enough for retrieving relevant documents from a large database.
                - Keep it concise, precise, and free of redundancy.
            - If the new question is unrelated, return exactly: NEW_QUES
            - ONLY RETURN THE REWRITTEN QUESTION. DO NOT EXPLAIN OR ADD ANY OTHER TEXT.

            Examples:

            Summary:
            Q1: রাজশাহী কৃষি সম্প্রসারণ অধিদপ্তরের ঠিকানা কী?
            A1: রাজশাহী কৃষি সম্প্রসারণ অধিদপ্তরের ঠিকানা হলো লালন শাহ সড়ক, রাজশাহী।
            New Question: ওখানে সপ্তাহে কয়দিন খোলা থাকে?
            Rewritten Question: রাজশাহী কৃষি সম্প্রসারণ অধিদপ্তর সপ্তাহে কয়দিন খোলা থাকে?

            Summary:
            Q1: পদ্মা সেতুর দৈর্ঘ্য কত?
            A1: পদ্মা সেতুর দৈর্ঘ্য ৬.১৫ কিমি।
            Q2: এটি কবে উদ্বোধন হয়?
            A2: ২৫ জুন ২০২২ সালে।
            New Question: ময়মনসিংহ মেডিকেল কলেজ কোথায়?
            Rewritten Question: NEW_QUES
             
            Reasoning:
            - The first example the new question is directly related to the previous conversation about the Rajshahi Agricultural Extension Department, so it is rewritten to be specific.
            - The second example is unrelated to the previous questions about the Padma Bridge, so it returns NEW_QUES.

            IMPORTANT:FOR UNRELATED QUESTIONS RETRUN EXACTY: NEW_QUES 
              
            """),
            ("user", "Summary:\n{summary}\nNew Question: {new_q}")
        ])

        self.summarize_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a summarizer. Given a chat history with alternating questions and answers, summarize the conversation concisely."),
            ("user", "Conversation:\n{history}")
        ])

    def _invoke_llm_stream(self, prompt: str) -> Generator[str, None, None]:
        for chunk in self.llm.stream(prompt):
            yield chunk

    def _invoke_llm(self, prompt: str) -> str:
        return self.llm.invoke(prompt).strip()

    def _summarize_history(self) -> str:
        if not self.context_window:
            return ""
        history_text = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in self.context_window])
        prompt = self.summarize_prompt.format(history=history_text)
        return self._invoke_llm(prompt)

    def _rewrite_if_related(self, new_question: str) -> str:
        if not self.context_window:
            return new_question

        if len(self.context_window) <= self.summary_size:
            summary = "\n".join([f"Q: {q}\nA: {a}" for q, a, _ in self.context_window])
        else:
            summary = self._summarize_history()

        prompt = self.rewrite_prompt.format(summary=summary, new_q=new_question)
        return self._invoke_llm(prompt)

    def _update_state(self, question: str, answer: str, relevant: bool, passage_context: str = None) -> None:
        if relevant and answer:
            self.context_window.append((question, answer, passage_context or ""))
            self.current_source_passage = passage_context
        elif not relevant:
            self.context_window.clear()
            self.current_source_passage = None

    def process_query(self, input_str: str) -> Generator[str, None, None]:
        original_question = input_str
        rewritten_question = self._rewrite_if_related(original_question)

        if rewritten_question == "NEW_QUES":
            self.context_window.clear()
            self.current_source_passage = None
            question = original_question
            relevant = False
        else:
            question = rewritten_question
            relevant = True

        docs = self.retriever.invoke(question)
        if not docs:
            self.debug_info = f"User Question: {original_question}\nRewritten Question: {rewritten_question}\nRetrieved Context: NONE\nGiven Answer: NOT_FOUND"
            yield "দুঃখিত, এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
            return

        combined_passages = "\n".join([doc.page_content for doc in docs])
        history = self._summarize_history() if self.context_window else ""

        prompt = self.answer_prompt.format(
            question=question,
            context=combined_passages,
            original=original_question,
            history=history
        )

        full_answer = ""
        for chunk in self._invoke_llm_stream(prompt):
            full_answer += chunk
            if "NOT" not in full_answer:
                yield chunk

        if "NOT" in full_answer:
            self.debug_info = f"User Question: {original_question}\nRewritten Question: {rewritten_question}\nRetrieved Context: {combined_passages}\nGiven Answer: NOT_SURE_ANSWER"
            yield "দুঃখিত, এই প্রশ্নের উত্তর আমার ডেটাবেজে নেই।"
            return

        self._update_state(original_question, full_answer, relevant, combined_passages)
        self.debug_info = f"User Question: {original_question}\nRewritten Question: {rewritten_question}\nRetrieved Context: {combined_passages}\nGiven Answer: {full_answer}"
        yield ""
