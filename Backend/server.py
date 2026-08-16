import os
import json
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import uvicorn

# pyrefly: ignore [missing-import]
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agent import build_agent
from contextlib import asynccontextmanager
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup AsyncSqliteSaver as an async context manager during app lifecycle
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app.state.agent = build_agent(checkpointer)
        yield

app = FastAPI(title="LangChain Tutor Agent Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest, fastapi_request: Request):
    agent = fastapi_request.app.state.agent
    async def event_generator() -> AsyncGenerator[dict, None]:
        from tools import db_get_student_profile
        from datetime import datetime
        from langgraph.types import Command
        
        thread_id = request.thread_id
        config = {"configurable": {"thread_id": thread_id}}
        
        # Check if the thread is currently interrupted/paused
        try:
            state = agent.get_state(config)
            is_interrupted = len(state.next) > 0
        except Exception:
            is_interrupted = False
            
        if is_interrupted:
            # Resume the graph with the student's answer
            inputs = Command(resume=request.message)
        else:
            # New day context check (Context Engineering)
            profile = db_get_student_profile(thread_id)
            should_inject_profile = False
            profile_summary = ""
            
            if profile:
                today_str = datetime.now().date().isoformat()
                has_old_session = False
                for topic, data in profile.items():
                    if data.get("last_seen") and data.get("last_seen") != today_str:
                        has_old_session = True
                        break
                
                if has_old_session:
                    try:
                        # Clear old checkpoint history to avoid transcript replaying
                        agent.checkpointer.delete_thread(thread_id)
                    except Exception:
                        pass
                    
                    # Format summary of previous student performance
                    profile_summary = "Student Profile Summary:\n"
                    for topic, data in profile.items():
                        weak_spots_str = ", ".join(data.get("weak_spots", []))
                        profile_summary += (
                            f"- Topic '{topic}': {data.get('correct')} correct, {data.get('incorrect')} incorrect. "
                            f"Weak spots: [{weak_spots_str}]. Last seen: {data.get('last_seen')}.\n"
                        )
                    should_inject_profile = True
            
            if should_inject_profile:
                context_prompt = (
                    f"{profile_summary}\n"
                    "IMPORTANT: This is a new session/day. Review the profile. "
                    "If the student struggled with a topic last time (e.g. has incorrect answers or weak spots), "
                    "you MUST greet them and refer back to it encouragingly (e.g., 'Last time you struggled with X, let's revisit that today')."
                )
                inputs = {
                    "messages": [
                        SystemMessage(content=context_prompt),
                        HumanMessage(content=request.message)
                    ]
                }
            else:
                inputs = {"messages": [HumanMessage(content=request.message)]}
        
        try:
            async for event in agent.astream_events(inputs, config=config, version="v2"):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    # Ignore inner model streams originating from tool calls (e.g. explain_concept or quiz generation inside tool)
                    tags = event.get("tags", [])
                    metadata = event.get("metadata", {})
                    if "seq:step:1" in tags or metadata.get("langgraph_node") == "agent":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            content = chunk.content
                            if isinstance(content, str):
                                yield {
                                    "event": "token",
                                    "data": json.dumps({"text": content})
                                }
                            elif isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        yield {
                                            "event": "token",
                                            "data": json.dumps({"text": item.get("text", "")})
                                        }
                                    elif isinstance(item, str):
                                        yield {
                                            "event": "token",
                                            "data": json.dumps({"text": item})
                                        }

                elif kind == "on_tool_start":
                    name = event.get("name")
                    yield {
                        "event": "status",
                        "data": json.dumps({"tool": name, "phase": "start"})
                    }

                elif kind == "on_tool_end":
                    name = event.get("name")
                    output = event.get("data", {}).get("output")
                    
                    # Convert tool output object to dict/primitive if needed
                    if hasattr(output, "content"):
                        raw_output = output.content
                    else:
                        raw_output = output
                    
                    if isinstance(raw_output, str):
                        try:
                            raw_output = json.loads(raw_output)
                        except Exception:
                            pass

                    yield {
                        "event": "tool_result",
                        "data": json.dumps({"tool": name, "output": raw_output})
                    }

            yield {
                "event": "done",
                "data": json.dumps({})
            }

        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)})
            }

    return EventSourceResponse(event_generator())

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
