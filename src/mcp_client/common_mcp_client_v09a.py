#python -m uvicorn common_mcp_client_v09a:app --reload --host 127.0.0.1 --port 8001
import asyncio
import nest_asyncio
import os

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal, List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig


nest_asyncio.apply()

# 환경 변수 로드
load_dotenv("D:/projects/myFren/src/.env")
MODEL_NAME  = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
default_model = ChatOpenAI(model=MODEL_NAME , temperature=0)

# ---------------------------------------------------------------------------
# Agent 초기화
# ---------------------------------------------------------------------------
async def setup_agent():    
    model = ChatOpenAI(model=MODEL_NAME, temperature=0)
    
    client = MultiServerMCPClient({
        "common_mcp_server": {
            "url": "http://localhost:8000/sse",
            "transport": "sse",
        }
    })

    tools = await client.get_tools()

    if not tools:
        raise RuntimeError("No tools found. Check MCP server configuration.")

    return create_react_agent(model, tools)

AGENT = asyncio.run(setup_agent())

# ---------------------------------------------------------------------------
# LangGraph 상태 정의
# ---------------------------------------------------------------------------
class State(MessagesState):
    messages: Annotated[list, add_messages]
    summary: str
    qclass: str

# ---------------------------------------------------------------------------
# query classification
# ---------------------------------------------------------------------------
def query_classification(state: State) -> Literal["memorizer", "agent"]:
    query = state["messages"][-1].content
    prompt = ChatPromptTemplate.from_template("{query}에 대한 내용을 기억해야할 쿼리는 'REMEMBER', 수정해야할 쿼리는 'UPDATE', " \
                                                "검색해야할 쿼리는 'SEARCH', 별의미없는 잡담은 'CHAT', 기타는 'ETC' 중에 하나로 분류해줘." \
                                                "답변은 반드시 위에 있는 분류명으로만 간결하게 대답해줘.")
    chain = prompt | default_model | StrOutputParser() # LCEL의 기본 파이프라인    
    qclass = chain.invoke({"query": query})

    print("===========qclass: ", qclass)

    if qclass in ["REMEMBER", "UPDATE", "ETC"]:
        return "memorizer"
    else:
        return "agent"

def memorizer(state: State):
    message = state["messages"][-1].content
    print("===========memorizer:", message)
    #save vectorstore.
    

# ---------------------------------------------------------------------------
# Agent 노드 정의
# ---------------------------------------------------------------------------
def agent_node(state: State, _agent):
    summary = state.get("summary", "")

    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]

    response = asyncio.run(_agent.ainvoke({"messages": messages}))
    print(response)
    return {"messages": [response["messages"][-1]]}

# -----------------------------------------------------
# Summarizing
# -----------------------------------------------------
# 대화 종료 또는 요약 결정 로직
def should_summary(state: State) -> Literal["summarizer", "end"]:
    # 메시지 목록 확인
    messages = state["messages"]
    print("===========MESSAGE.LENGTH: ", len(messages))
    # 메시지 수가 6개 초과라면 요약 노드로 이동
    if len(messages) > 6:
        return "summarizer"
    return "end"

# 대화 내용 요약 및 메시지 정리 로직
def summarizer(state: State):
    # 이전 요약 정보 확인
    summary = state.get("summary", "")
    # 이전 요약 정보가 있다면 요약 메시지 생성
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above in Korean:"
        )
    else:
        # 요약 메시지 생성
        summary_message = "Create a summary of the conversation above in Korean:"

    # 요약 메시지와 이전 메시지 결합
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    # 모델 호출
    response = default_model.invoke(messages)
    # 오래된 메시지 삭제
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    # 요약 정보 반환
    return {"summary": response.content, "messages": delete_messages}

# 업데이트 정보 출력 함수
def print_update(event):
    for value in event.items():
        if 'agent' in value:
            print("===========print_update.MESSAGE: ")
            print(value[-1]['messages'][-1].content)
        elif 'summarizer' in value:
            print("===========print_update.SUMMARY: ")
            print(value[-1]['summary'])

# -----------------------------------------------------
# Graph
# -----------------------------------------------------
def build_graph(_agent):
    graph = StateGraph(State)
    graph.add_node("agent", lambda s: agent_node(s, _agent)) # type: ignore
    graph.add_node("memorizer", memorizer)
    graph.add_node("summarizer", summarizer)

    graph.add_conditional_edges(
        START,
        query_classification,
    )
    graph.add_edge("memorizer", "agent")
    graph.add_conditional_edges(
        "agent",
        should_summary,
        {
            "summarizer": "summarizer",
            "end": END
        },
    )
    graph.add_edge("summarizer", END)

    # 메모리 저장소 생성
    inmemory = MemorySaver()
    return graph.compile(checkpointer=inmemory)

APP_GRAPH = build_graph(AGENT)

# ---------------------------------------------------------------------------
# Chat 함수
# ---------------------------------------------------------------------------
def chat_with_bot(message: str, thread_id: str):

    lc_messages = [HumanMessage(content=message)]

    config = RunnableConfig(
        recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
        configurable={"thread_id": thread_id},  # 스레드 ID 설정
    )

    final_response = ""

    for event in APP_GRAPH.stream(
        {"messages": lc_messages}, # type: ignore
        config=config, 
        stream_mode="updates"
    ): 
        print_update(event)
        for value in event.values():
            if "messages" in value:
                final_response =  value['messages'][-1].content
        
    return final_response

# -----------------------------------------------------
# FastAPI Schema
# -----------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ChatResponse(BaseModel):
    response: str


# -----------------------------------------------------
# FastAPI
# -----------------------------------------------------
app = FastAPI(title="Voice Chatbot with MCP + LangGraph")

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    result = await asyncio.to_thread(
        chat_with_bot,
        req.message,
        req.thread_id
    )

    return ChatResponse(response=result)

@app.post("/video", response_model=ChatResponse)
async def video(req: ChatRequest):

    print(f"###### VIDEO message arrived: {req.message}")

    return ChatResponse(response="chatbot received video message.")
