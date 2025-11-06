import os
from typing import List, Dict, Any
from email_sender import send_email_to_user

class ResponseService:
    @staticmethod
    def is_unknown_answer(answer: str) -> bool:
        """Detect if the answer indicates unknown information"""
        lower = answer.lower()
        unknown_phrases = ["i don't know", "idk", "not found in", "cannot find"]
        return (
            any(phrase in lower for phrase in unknown_phrases) 
            or len(answer.strip()) < 10
        )
    
    @staticmethod
    async def handle_unknown_answer(
        question: str, 
        sources: List[Dict], 
        source_docs: List[Any], 
        session_id: str
    ) -> str:
        """Handle unknown answers by sending email notification"""
        try:
            snippets = "\n\n".join([
                f"- Page {s.get('page', '?')}: {doc.page_content[:200]}..." 
                for s, doc in zip(sources, source_docs) 
                if hasattr(doc, 'page_content')
            ])
            
            email_body = f"""Unanswered interview question detected.

Question: {question}

Retrieved context:
{snippets if snippets else '(no relevant context found)'}

Session: {session_id}
Timestamp: {os.environ.get('TZ', 'UTC')}
"""
            send_email_to_user("Interview Question - Needs Review", email_body)
        except Exception as email_error:
            print(f"Failed to send email notification: {email_error}")
        
        return "I couldn't find that information in the resume. The question has been forwarded for review."

response_service = ResponseService()