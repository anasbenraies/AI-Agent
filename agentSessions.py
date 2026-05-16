# api.py  — Step 2: per-session memory (full replacement)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as APIModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel as LCBaseModel
from tools import search_tool, wikipedia_tool, write_to_file_tool, translate_tool
from datetime import datetime, timedelta
import uuid

load_dotenv()

# ── LangChain setup ────────────────────────────────────────────────────────────

class ResearchResponse(LCBaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are a research assistant that will help generate a research paper.
        Answer the user query and use necessary tools.
        Wrap the output in this format and provide no other text\n{format_instructions}
     """),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),
]).partial(format_instructions=parser.get_format_instructions())

tools = [search_tool, wikipedia_tool, write_to_file_tool, translate_tool]


# ── session store ──────────────────────────────────────────────────────────────

SESSION_TTL_MINUTES = 60    # evict sessions idle for longer than this

class Session:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True         # one Memory object per session
        )
        self.executor = AgentExecutor(
            agent=create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt),
            tools=tools,
            verbose=True,
            memory=self.memory           # executor is bound to this session's memory
        )
        self.last_active = datetime.utcnow()

    def touch(self):
        self.last_active = datetime.utcnow()

    def is_expired(self) -> bool:
        return datetime.utcnow() - self.last_active > timedelta(minutes=SESSION_TTL_MINUTES)


# dict that maps session_id (str) → Session
session_store: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    """Return existing session or raise 404."""
    session = session_store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found. POST /sessions to create one.")
    if session.is_expired():
        del session_store[session_id]
        raise HTTPException(status_code=410, detail="Session expired. POST /sessions to start a new one.")
    session.touch()
    return session


# ── FastAPI ────────────────────────────────────────────────────────────────────

app = FastAPI()


# -- request / response schemas ------------------------------------------------

class SessionResponse(APIModel):
    session_id: str

class QueryRequest(APIModel):
    query: str

class QueryResponse(APIModel):
    session_id: str
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    raw: str | None = None


# -- endpoints -----------------------------------------------------------------

@app.post("/sessions", response_model=SessionResponse, status_code=201)
async def create_session():
    """
    Client calls this once to get a session_id.
    They include that ID in every subsequent /research request.
    """
    session_id = str(uuid.uuid4())
    session_store[session_id] = Session()
    return SessionResponse(session_id=session_id)


@app.post("/research/{session_id}", response_model=QueryResponse)
async def research(session_id: str, request: QueryRequest):
    """
    Run a research query inside a specific session.
    The session's memory accumulates across calls with the same session_id.
    """
    session = get_session(session_id)

    raw_response = session.executor.invoke({"query": request.query})

    try:
        structured = parser.parse(raw_response["output"])
        return QueryResponse(session_id=session_id, **structured.dict())
    except Exception:
        return QueryResponse(
            session_id=session_id,
            topic="unknown", summary="", sources=[], tools_used=[],
            raw=raw_response["output"]
        )


@app.delete("/sessions/{session_id}", status_code=204)
async def close_session(session_id: str):
    """Explicit cleanup — client calls this when done."""
    session_store.pop(session_id, None)


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """Inspect the raw memory of a session — useful for debugging."""
    session = get_session(session_id)
    messages = session.memory.chat_memory.messages
    return {
        "session_id": session_id,
        "turns": [{"type": m.type, "content": m.content} for m in messages]
    }