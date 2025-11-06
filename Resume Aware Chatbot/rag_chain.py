import os
from typing import Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Optional local embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


def _get_embeddings():
    use_local = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    if use_local:
        # Free, CPU-friendly embedding model (zero token cost)
        # Good trade-off for a small resume corpus
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        model = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)


def _get_llm() -> ChatOpenAI:
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")  # cost-effective & capable
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    return ChatOpenAI(model=model, temperature=temperature)


def build_or_load_vectorstore(docs, embeddings, index_path: Optional[str]) -> FAISS:
    """
    Build FAISS from docs or load from disk if present. Saves reprocessing of document everytime
    """
    if index_path and os.path.isdir(index_path):
        # Dangerous deserialization is necessary for FAISS load
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    vs = FAISS.from_documents(docs, embeddings)
    if index_path:
        os.makedirs(index_path, exist_ok=True)
        vs.save_local(index_path)
    return vs


def get_retriever(vs: FAISS):
    """
    Use MMR to diversify retrieved chunks.
    """
    k = int(os.getenv("RETRIEVER_K", "4"))
    fetch_k = int(os.getenv("RETRIEVER_FETCH_K", "12"))
    lambda_mult = float(os.getenv("MMR_LAMBDA", "0.7"))
    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    )


def build_conv_rag_chain(retriever) -> ConversationalRetrievalChain:
    """
    Build a Conversational RAG chain with a strict QA prompt.
    Returns answers and source documents.
    """
    SYSTEM_INSTRUCTIONS = """
You are an interview assistant that answers ONLY using the candidate's resume content.
If the answer is not present in the resume context, say "I don't know."
Keep answers concise and helpful for interviews. Use bullet points when listing items.
Cite pages if relevant.
"""
    QA_TEMPLATE = """{system}

Context from resume:
{context}

Chat history:
{chat_history}

Question: {question}

Guidelines:
- Answer using ONLY the context from the resume.
- If not found in the context, say "I don't know."
- Be concise and specific. Include page numbers if helpful.

Answer:
"""

    qa_prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["system", "context", "chat_history", "question"],
        partial_variables={"system": SYSTEM_INSTRUCTIONS.strip()},
    )

    llm = _get_llm()

    # NOTE: ConversationalRetrievalChain condenses the question using chat history,
    # retrieves docs, and then answers with the qa prompt.
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        qa_prompt=qa_prompt,
        return_source_documents=True,
        verbose=False,
    )
    return chain


def bootstrap_rag(docs) -> Tuple[ConversationalRetrievalChain, FAISS]:
    """
    Compose embeddings, vector store, retriever, and chain.
    """
    embeddings = _get_embeddings()
    index_path = os.getenv("VECTORSTORE_PATH", ".faiss_index") or None
    vs = build_or_load_vectorstore(docs, embeddings, index_path)
    retriever = get_retriever(vs)
    chain = build_conv_rag_chain(retriever)
    return chain, vs
