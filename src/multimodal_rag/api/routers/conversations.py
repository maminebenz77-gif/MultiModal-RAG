"""GET /conversations: list recent conversations (preview + activity), for
a "previous conversations" picker. GET /conversations/{conversation_id}:
fetch one conversation's full turn history, each with its own citations
-- for reloading/resuming a conversation (see the frontend chat UI)
rather than relying on the client to have kept it. DELETE
/conversations/{conversation_id}: permanently remove one.

A conversation belongs to whoever started it. Someone else's is a 404,
never a 403 -- the same answer as one that doesn't exist.
"""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ...identity import Principal
from ..db import Database
from ..dependencies import get_db
from ..identity import get_principal
from ..schemas import ConversationDeleteResponse, ConversationListResponse, ConversationResponse

router = APIRouter()


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    db: Database = Depends(get_db), principal: Principal = Depends(get_principal)
) -> ConversationListResponse:
    conversations = await run_in_threadpool(db.list_conversations, principal)
    return ConversationListResponse(conversations=conversations)


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> ConversationResponse:
    exists = await run_in_threadpool(db.conversation_exists, principal, conversation_id)
    if not exists:
        raise HTTPException(
            status_code=404, detail=f"No conversation with id={conversation_id!r}"
        )
    messages = await run_in_threadpool(
        db.get_conversation_messages, principal, conversation_id
    )
    return ConversationResponse(conversation_id=conversation_id, messages=messages)


@router.delete("/conversations/{conversation_id}", response_model=ConversationDeleteResponse)
async def delete_conversation(
    conversation_id: str,
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> ConversationDeleteResponse:
    exists = await run_in_threadpool(db.conversation_exists, principal, conversation_id)
    if not exists:
        raise HTTPException(
            status_code=404, detail=f"No conversation with id={conversation_id!r}"
        )
    await run_in_threadpool(db.delete_conversation, principal, conversation_id)
    return ConversationDeleteResponse(status="deleted", conversation_id=conversation_id)
