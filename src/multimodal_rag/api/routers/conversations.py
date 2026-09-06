"""GET /conversations: list recent conversations (preview + activity), for
a "previous conversations" picker. GET /conversations/{conversation_id}:
fetch one conversation's full turn history, each with its own citations
-- for reloading/resuming a conversation (see the frontend chat UI)
rather than relying on the client to have kept it. DELETE
/conversations/{conversation_id}: permanently remove one.
"""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ..db import Database
from ..dependencies import get_db
from ..schemas import ConversationDeleteResponse, ConversationListResponse, ConversationResponse

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(db: Database = Depends(get_db)) -> ConversationListResponse:
    conversations = await run_in_threadpool(db.list_conversations)
    return ConversationListResponse(conversations=conversations)


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


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(
    conversation_id: str, db: Database = Depends(get_db)
) -> ConversationDeleteResponse:
    exists = await run_in_threadpool(db.conversation_exists, conversation_id)
    if not exists:
        raise HTTPException(
            status_code=404, detail=f"No conversation with id={conversation_id!r}"
        )
    await run_in_threadpool(db.delete_conversation, conversation_id)
    return ConversationDeleteResponse(status="deleted", conversation_id=conversation_id)
