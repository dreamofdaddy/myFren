import asyncio
import nest_asyncio
import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, time as dt_time
import time

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.graph import END
from langgraph.graph.message import add_messages
from typing import Annotated, Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

import speech_recognition as sr
import edge_tts


# 환경 변수 로드
load_dotenv("/home/dody/work/kosa-ict-genai-2025-2nd/src/exercise/dody/.env")
default_model_name = os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
default_model = ChatOpenAI(model_name=default_model_name, temperature=0)

# ---------------------------------------------------------------------------
# Agent 초기화
# ---------------------------------------------------------------------------
@st.cache_resource
def init_agent_app():
    async def setup():
        model = ChatOpenAI(model=default_model_name, temperature=0)
        client = MultiServerMCPClient({
            "common_mcp_server": {
                "url": "http://localhost:8000/sse",
                "transport": "sse",
            }
        })
        tools = await client.get_tools()
        if not tools:
            raise RuntimeError("No tools found. Check MCP server configuration.")
        _agent = create_react_agent(model, tools)
        return _agent

    return asyncio.run(setup())

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
    output_parser = StrOutputParser()
    chain = prompt | default_model | output_parser # LCEL의 기본 파이프라인
    
    qclass = chain.invoke({"query": query})
    print("===========qclass: ", qclass)
    if qclass == "REMEMBER":
        return "memorizer"
    elif qclass == "UPDATE":
        return "memorizer"
    elif qclass == "ETC":
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

@st.cache_resource
def init_graph(_agent):
    workflow = StateGraph(State)
    workflow.add_node("agent", lambda s: agent_node(s, _agent))
    workflow.add_node("memorizer", memorizer)
    workflow.add_node("summarizer", summarizer)

    workflow.add_conditional_edges(
        START,
        query_classification,
    )
    workflow.add_edge("memorizer", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_summary,
        {
            "summarizer": "summarizer",
            "end": END
        },
    )
    workflow.add_edge("summarizer", END)

    # 메모리 저장소 생성
    inmemory = MemorySaver()
    return workflow.compile(checkpointer=inmemory)

# STT
def get_audio_input():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        audio = r.listen(source)
    # 구글 웹 음성 API로 인식하기 
    try:
        print("Google Speech Recognition thinks you said : " + r.recognize_google(audio, language='ko'))
        return r.recognize_google(audio, language='ko')
    except sr.UnknownValueError as e:
        print("Google Speech Recognition could not understand audio".format(e))
        return None
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return None

# TTS 
async def text_to_speech(text):
    voice = "ko-KR-SunHiNeural"   # 여성/남성=ko-KR-InJoonNeural"
    output_file = "korean_conv_tts.mp3"

    tts = edge_tts.Communicate(text, voice, rate="+15%")
    await tts.save(output_file)
    os.system(f"mpg123 {output_file}")


# ---------------------------------------------------------------------------
# Chat 함수
# ---------------------------------------------------------------------------
def chat_with_bot(messages, agent_app, thread_id):
    # user/assistant dict → LangChain Message 객체 변환
    lc_messages: list[BaseMessage] = []
    lc_messages.append(HumanMessage(content=messages[-1]["content"]))

    config = RunnableConfig(
        recursion_limit=10,  # 최대 10개의 노드까지 방문. 그 이상은 RecursionError 발생
        configurable={"thread_id": thread_id},  # 스레드 ID 설정
    )
    for event in agent_app.stream({"messages": lc_messages}, config=config, stream_mode="updates"):
        print_update(event)
        for value in event.items():
            if 'agent' in value:
                response =  value[-1]['messages'][-1].content
        
    return response

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
async def main():
    st.title("Voice Chatbot with MCP + LangGraph")

    _agent = init_agent_app()
    agent_app = init_graph(_agent)
    #agent_app.get_graph().draw_mermaid_png(output_file_path="./graph_v04.png")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "medicine_alarms" not in st.session_state:
        # 기본 약 먹는 시간 설정
        st.session_state.medicine_alarms = {
            "아침": {"time": dt_time(8, 0), "enabled": False, "triggered": False},
            "점심": {"time": dt_time(12, 30), "enabled": False, "triggered": False},
            "저녁": {"time": dt_time(19, 0), "enabled": False, "triggered": False}
        }
    if "medicine_history" not in st.session_state:
        # 복약 이력
        st.session_state.medicine_history = []
    if "pending_medicine_check" not in st.session_state:
        # 복약 확인 대기 중인 알람
        st.session_state.pending_medicine_check = None
    if "medicine_check_time" not in st.session_state:
        # 복약 확인 시작 시간
        st.session_state.medicine_check_time = None
    if "medicine_success_message" not in st.session_state:
        # 복약 성공 메시지
        st.session_state.medicine_success_message = None
    if "medicine_info_message" not in st.session_state:
        # 복약 정보 메시지
        st.session_state.medicine_info_message = None
    if "medicine_just_processed" not in st.session_state:
        # 복약 알림 처리 직후 플래그
        st.session_state.medicine_just_processed = False
    
    # 사이드바 - 시계 및 약 알람
    with st.sidebar:
        # 현재 시간 크게 표시
        current_time = datetime.now()
        
        # 오전/오후 구분
        hour_12 = current_time.hour % 12
        if hour_12 == 0:
            hour_12 = 12
        am_pm = "오전" if current_time.hour < 12 else "오후"
        time_str = f"{am_pm} {hour_12:02d}:{current_time.minute:02d}:{current_time.second:02d}"
        
        # 요일 한글 변환
        weekday_kr = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        weekday_str = weekday_kr[current_time.weekday()]
        
        st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin-bottom: 20px;'>
                <h1 style='font-size: 3em; margin: 0; color: #1f77b4;'>{time_str}</h1>
                <p style='font-size: 1.2em; margin: 5px 0; color: #666;'>{current_time.strftime('%Y년 %m월 %d일')}</p>
                <p style='font-size: 1em; color: #888;'>{weekday_str}</p>
            </div>
        """, unsafe_allow_html=True)

        # 마이크 버튼을 약복용 알림 위쪽에 배치
        if st.button("🎤 마이크 켜기", use_container_width=True, help="마이크 켜기", key="mic_button_sidebar"):
            with st.spinner("음성을 인식하는 중..."):
                voice_input = get_audio_input()
                if voice_input:
                    st.session_state.messages.append({"role": "user", "content": voice_input})
                    st.session_state.waiting_for_response = True
                    st.rerun()
                else:
                    st.error("음성 인식에 실패했습니다.")
        st.divider()

        st.header("💊 약 복용 알림")
        
        # 복약 이력 통계
        if st.session_state.medicine_history:
            total_count = len(st.session_state.medicine_history)
            taken_count = sum(1 for h in st.session_state.medicine_history if h["taken"])
            completion_rate = (taken_count / total_count * 100) if total_count > 0 else 0
            
            st.metric("복약 순응도", f"{completion_rate:.1f}%", f"{taken_count}/{total_count}회")
            st.divider()
        
        # 각 시간대별 약 알람 설정
        for period, alarm_data in st.session_state.medicine_alarms.items():
            st.subheader(f"{period} 약")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                hour = st.number_input(
                    "시",
                    min_value=0,
                    max_value=23,
                    value=alarm_data["time"].hour,
                    key=f"hour_{period}",
                    step=1
                )
            
            with col2:
                minute = st.number_input(
                    "분",
                    min_value=0,
                    max_value=59,
                    value=alarm_data["time"].minute,
                    key=f"minute_{period}",
                    step=1
                )
                
            with col3:
                enabled = st.checkbox(
                    "활성",
                    value=alarm_data["enabled"],
                    key=f"enable_{period}"
                )
            
            # 시간 업데이트
            st.session_state.medicine_alarms[period]["time"] = dt_time(hour, minute)
            st.session_state.medicine_alarms[period]["enabled"] = enabled
            
            # 알람 체크
            if alarm_data["enabled"] and not alarm_data["triggered"]:
                alarm_time = alarm_data["time"]
                if current_time.hour == alarm_time.hour and current_time.minute == alarm_time.minute:
                    st.session_state.medicine_alarms[period]["triggered"] = True
                    st.session_state.pending_medicine_check = period
                    st.session_state.medicine_check_time = datetime.now()
                    st.balloons()
                    
                    # 알람음
                    st.markdown("""
                        <audio autoplay>
                            <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
                        </audio>
                    """, unsafe_allow_html=True)
                    await text_to_speech("약 드실 시간입니다! 1분이내 약복용이 확인되지 않으면 보호자에게 알림이 전송됩니다.")
            
            # 다음 분이 되면 triggered 리셋
            if alarm_data["triggered"]:
                alarm_time = alarm_data["time"]
                if current_time.hour != alarm_time.hour or current_time.minute != alarm_time.minute:
                    st.session_state.medicine_alarms[period]["triggered"] = False
        
        # 전체 알람 초기화 버튼
        # if st.button("🔄 모든 알람 리셋", use_container_width=True):
        #     for period in st.session_state.medicine_alarms:
        #         st.session_state.medicine_alarms[period]["triggered"] = False
        #         st.session_state.medicine_alarms[period]["enabled"] = False
        #     st.success("모든 알람이 리셋되었습니다!")
        #     st.rerun()
        
        st.divider()
        st.caption("💡 체크박스를 활성화하여 약 복용 시간에 알림을 받으세요")
    
    # 복약 확인 대화 상자 (알람이 울렸을 때)
    if st.session_state.pending_medicine_check:
        period = st.session_state.pending_medicine_check
        elapsed_time = (datetime.now() - st.session_state.medicine_check_time).total_seconds()

        # 30초 경과 체크
        if elapsed_time < 30:
            # 대화 상자 표시
            st.warning(f"### 💊 {period} 약 드실 시간입니다!")
            st.info(f"약을 드셨나요? ({int(30 - elapsed_time)}초 후 보호자에게 알림 전송)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("✅ 예, 먹었어요", use_container_width=True, key="medicine_yes"):
                    st.session_state.messages.append({"role": "user", "content": f"[약 복용 알림] {period} 약 복용 시간({st.session_state.medicine_check_time.strftime('%H:%M')})에 복약이 확인되었습니다. 보호자에게 카톡메세지를 발송합니다."})
                    response = chat_with_bot(st.session_state.messages, agent_app, "user_id")
                    await text_to_speech("복약이 기록되었습니다!")
                    # 복약 완료 기록
                    st.session_state.medicine_history.append({
                        "period": period,
                        "time": st.session_state.medicine_check_time,
                        "taken": True,
                        "response_time": datetime.now()
                    })
                    st.session_state.pending_medicine_check = None
                    st.session_state.medicine_check_time = None
                    st.session_state.medicine_success_message = "복약이 기록되었습니다! 👍"
                    st.session_state.medicine_just_processed = True
                    st.rerun()
            
            with col2:
                if st.button("❌ 아니오, 안 먹었어요", use_container_width=True, key="medicine_no"):
                    await text_to_speech("복약이 미완료 기록되었습니다! 약 먹는 것을 잊지 마세요!")
                    # 복약 미완료 기록
                    st.session_state.medicine_history.append({
                        "period": period,
                        "time": st.session_state.medicine_check_time,
                        "taken": False,
                        "response_time": datetime.now()
                    })
                    st.session_state.pending_medicine_check = None
                    st.session_state.medicine_check_time = None
                    st.session_state.medicine_info_message = "약 먹는 것을 잊지 마세요!"
                    st.session_state.medicine_just_processed = True
                    st.rerun()
            
            with col3:
                if st.button("⏰ 나중에", use_container_width=True, key="medicine_later"):
                    await text_to_speech("복약이 연기되었습니다! 잠시 후 다시 알려드릴게요")
                    st.session_state.pending_medicine_check = None
                    st.session_state.medicine_check_time = None
                    st.session_state.medicine_info_message = "잠시 후 다시 알려드릴게요"
                    st.session_state.medicine_just_processed = True
                    st.rerun()
        else:
            # 30초 경과 - 보호자에게 SMS 발송
            st.error("⚠️ 약 복용 확인 시간이 초과되었습니다!")
            st.warning("보호자에게 알림을 발송합니다...")
            
            # SMS 발송 (MCP 서버 통해)
            emergency_phone = os.getenv("EMERGENCY_PHONE_NUMBER", "").split(",")[0]
            if emergency_phone:
                # MCP 서버의 send_sms_message 호출
                message = f"[약 복용 알림] {period} 약 복용 시간({st.session_state.medicine_check_time.strftime('%H:%M')})에 복약 확인이 되지 않았습니다. 확인 부탁드립니다."
                
                # 여기서 실제로 SMS를 보내려면 agent를 통해 Tool 호출해야 하지만,
                # 간단히 시뮬레이션으로 표시
                st.session_state.messages.append({"role": "user", "content": "[약 복용 알림]약 복용 시간을 초과했습니다. 보호자에게 카톡 메세지를 발송합니다."})
                response = chat_with_bot(st.session_state.messages, agent_app, "user_id")
                await text_to_speech(response)     #TTS
                
                st.info(f"📱 SMS 발송: {emergency_phone}")
                st.caption(f"내용: {message}")
            
            # 미복약 기록
            st.session_state.medicine_history.append({
                "period": period,
                "time": st.session_state.medicine_check_time,
                "taken": False,
                "response_time": None,
                "alert_sent": True
            })
            
            st.session_state.pending_medicine_check = None
            st.session_state.medicine_check_time = None
            
            if st.button("확인", use_container_width=True):
                st.rerun()
    
    # 복약 알림 메시지 표시 (복약 알림이 없을 때만)
    if not st.session_state.pending_medicine_check:
        if st.session_state.medicine_success_message:
            st.success(st.session_state.medicine_success_message)
            st.session_state.medicine_success_message = None
        elif st.session_state.medicine_info_message:
            st.info(st.session_state.medicine_info_message)
            st.session_state.medicine_info_message = None
    
    
    # 채팅 히스토리 표시 (복약 알림이 없고 복약 처리 직후가 아닐 때만)
    if not st.session_state.pending_medicine_check and not st.session_state.get("medicine_just_processed", False):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    elif not st.session_state.pending_medicine_check and st.session_state.get("medicine_just_processed", False):
        # 복약 처리 직후 플래그 리셋 (채팅 히스토리 표시 건너뛴 후)
        st.session_state.medicine_just_processed = False
        st.rerun()  # 플래그 리셋 후 다시 렌더링

    # 텍스트 입력 및 응답 생성 (복약 알림이 없고 복약 처리 직후가 아닐 때만)
    if not st.session_state.pending_medicine_check and not st.session_state.get("medicine_just_processed", False):
        # 텍스트 입력 (하단 고정)
        user_input = st.chat_input("메시지를 입력하세요...")

        if user_input:
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.waiting_for_response = True
            st.rerun()  # 사용자 메시지 먼저 표시
            
        # 응답 생성 (사용자 메시지가 표시된 후)
        if st.session_state.get("waiting_for_response", False):
            with st.spinner("답변 생성 중..."):
                response = chat_with_bot(st.session_state.messages, agent_app, "user_id")
                await text_to_speech(response)     #TTS

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.waiting_for_response = False  # 플래그 해제
            st.rerun()  # 응답 표시
    
    # 사이드바 시간 업데이트를 위한 주기적 rerun (알람이 활성화되고 사용자 입력이 없는 경우에만)
    if any(alarm["enabled"] for alarm in st.session_state.medicine_alarms.values()):
        time.sleep(1)
        st.rerun()


if __name__ == "__main__":
    nest_asyncio.apply()  # 이미 실행 중인 이벤트 루프 중첩 허용
    asyncio.get_event_loop().run_until_complete(main())
