from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument
from app.services.rag_service import rag_service
from app.services.memory_service import chat_memory
from app.services.response_service import response_service
from app.core.config import settings

router = APIRouter()

@router.post(
    "/chat", 
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process chat message",
    description="Send a question about the resume and get an AI-powered response with source citations"
)
async def chat_endpoint(request: ChatRequest):
    """Process a chat message and return response with sources"""
    
    if not rag_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "RAG service not initialized",
                "message": "Please wait and try again",
                "retry_after": 5
            }
        )
    
    try:
        # Get chat history in modern format
        chat_history = chat_memory.get_langchain_format(request.session_id)
        
        # Query RAG chain with latest async pattern
        result = await rag_service.query(request.question, chat_history)
        
    except Exception as e:
        print(f"❌ RAG chain error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Processing failed",
                "message": "Failed to process question. Please try again.",
                "type": str(type(e).__name__)
            }
        )
    
    answer = result.get("answer", "").strip()
    source_docs = result.get("source_documents", []) or []
    
    # Format sources with latest schema
    raw_sources = rag_service.format_sources(source_docs)
    sources = [
        SourceDocument(source=src["source"], page=src["page"]) 
        for src in raw_sources
    ]
    
    # Check for unknown answers
    if response_service.is_unknown_answer(answer):
        answer = await response_service.handle_unknown_answer(
            request.question, raw_sources, source_docs, request.session_id
        )
    
    # Update memory
    chat_memory.add_exchange(request.session_id, request.question, answer)
    
    return ChatResponse(response=answer, sources=sources)

@router.get(
    "/chat/models",
    summary="Get available models",
    description="List available LLM and embedding models"
)
async def get_available_models():
    """Get information about available models"""
    return {
        "llm_models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "embedding_models": ["text-embedding-3-small", "text-embedding-3-large"],
        "current_config": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.embeddings_model,
            "use_local_embeddings": settings.use_local_embeddings
        }
    }