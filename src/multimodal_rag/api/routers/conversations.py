"""GET /conversations/{conversation_id}: fetch a conversation's full
turn history, each with its own citations -- for reloading/resuming a
conversation (see the frontend chat UI) rather than relying on the
client to have kept it.
"""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ..db import Database
from ..dependencies import get_db
from ..schemas import ConversationResponse

router = APIRouter()


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str, db: Database = Depends(get_db)
) -> ConversationResponse:
    exists = await run_in_threadpool(db.conversation_exists, conversation_id)
    if not exists:
        raise HTTPException(
            status_code=404, detail=f"No conversation with id={conversation_id!r}"
        )
    messages = await run_in_threadpool(db.get_conversation_messages, conversation_id)
    return ConversationResponse(conversation_id=conversation_id, messages=messages)
