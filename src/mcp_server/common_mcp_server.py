#%pip install google-search-results
#python common_mcp_server.py (server run command)

import os
import urllib
import openai
import pytz
import requests
import json
from pykrx import stock
from dotenv import load_dotenv
from urllib.parse import quote
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import LLMMathChain
from mcp.server.fastmcp import FastMCP
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from naver_sens import SensClient


mcp = FastMCP("common_mcp_server", port=8000)

load_dotenv("D:/projects/myFren/src/.env")

openai.api_key = os.getenv("OPENAI_API_KEY")
default_model = ChatOpenAI(model=os.getenv("OPENAI_DEFAULT_MODEL"), temperature=0, streaming=True, verbose=True) # type: ignore

FILE_NAME = "./persona.txt"
loader = TextLoader(FILE_NAME, encoding="utf-8")
embeddings_model = OpenAIEmbeddings()

index = VectorstoreIndexCreator(
    vectorstore_cls=FAISS,
    embedding=embeddings_model,
).from_loaders([loader])
# save to vectorstore.
index.vectorstore.save_local("persona") # type: ignore

access_key = os.getenv("SMS_ACCESS_KEY")
secret_key = os.getenv("SMS_SECRET_KEY")
service_key = os.getenv("SMS_SERVICE_KEY")
sens_client = SensClient(
    service_id=service_key, # type: ignore
    access_key=access_key, # type: ignore
    secret_key=secret_key # type: ignore
)

@mcp.tool("private_info")
async def private_info(query):
    """나의 페르소나 정보. 나와 나의가족, 우리집 등에 대한 정보가 필요할 때 참조하여 답변한다."""
    vectorstore = FAISS.load_local("persona", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    print("========= private_info query: "+query)
    response = vectorstore.similarity_search(query, k=1)
    return response

@mcp.tool("datecalc")
async def datecalc(query):
    """날짜계산, 시간계산에 유용하다."""
    print("========= datecalc: " + query)
    return query

@mcp.tool("currtime")
async def currtime():
    """오늘(현재, 지금, 올해, 이번달, 이번년도)의 년도, 날짜, 시간 정보를 반환합니다."""
    kst = pytz.timezone('Asia/Seoul')
    curr_time = datetime.now(kst)
    print("========= get_time_info ")
    return curr_time.strftime('%Y-%m-%d %H:%M:%S')

@mcp.tool("calculator")
async def calculator(query : str) -> str:
    """숫자를 계산해야하는 질문에 유용하다. 시간이나 날짜의 계산에는 사용하면 않된다."""
    llm_math_chain = LLMMathChain.from_llm(OpenAI())
    print("========= calculator: " + query)
    return llm_math_chain.invoke(query)

@mcp.tool("serpapi_search")
async def serpapi_search(query):
    """A search engine. Useful for when you need to answer questions about current events. Input should be a search query."""
    serpapi = SerpAPIWrapper()
    print("========= serpapi_search: "+query)
    return serpapi.run(query)

@mcp.tool("wikipedia_search")
async def wikipedia_search(query):
    """
    A wrapper around Wikipedia. Useful for when you need to answer general questions 
    about people, places, companies, facts, historical events, or other subjects. Input should be a search query.
    """
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()) # type: ignore
    print("========= wikipedia_search: "+query)
    return wikipedia.run(query)

@mcp.tool("foodhouse_search")
async def foodhouse_search(keyword):
    """카페, 레스토랑, 음식점 등 맛집 정보를 반환합니다. 사용자 쿼리 중 맛집 검색을 위한 키워드를 추출하여 파라미터로 전달합니다."""
    client_id = "SvLZxpvr8jnKZ1UzWKDB"
    client_secret = "2Cn80Ux0VV"
    encText = urllib.parse.quote(keyword) # type: ignore
    url = "https://openapi.naver.com/v1/search/blog?query=" + encText # JSON 결과
    request = urllib.request.Request(url) # type: ignore
    request.add_header("X-Naver-Client-Id",client_id)
    request.add_header("X-Naver-Client-Secret",client_secret)
    response = urllib.request.urlopen(request) # type: ignore
    rescode = response.getcode()
    if(rescode==200):
        response_body = response.read()
    else:
        print("Error Code:" + rescode)

    print("========= foodhouse_search: "+keyword)
    return response_body.decode('utf-8')

@mcp.tool("camp_search")
async def camp_search(keyword):
    """캠핑장 정보를 반환합니다. 사용자 쿼리 중 캠핑장 검색을 위한 키워드를 추출하여 파라미터로 전달합니다."""
    ServiceKey = os.getenv("GOCAMPING_SERVICE_KEY")
    keyword = quote(keyword.strip())
    print("========= camp_search: "+keyword)
    url = f"http://apis.data.go.kr/B551011/GoCamping/searchList?serviceKey={ServiceKey}&keyword={keyword}&numOfRows=10&pageNo=1&MobileOS=ETC&MobileApp=TestApp&_type=json"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Request Exception: {e}")
        return None

@mcp.tool("stock_search")
async def stock_search(start_date, end_date, ticker):
    """Return prices within given dates for ticker stock. start_date and end_date shoule be 'YYYYMMDD' format."""
    start_date = start_date.strip()
    end_date = end_date.strip()
    ticker = ticker.strip()
    stock_name = stock.get_market_ticker_name(ticker)
    df = stock.get_market_ohlcv(start_date, end_date, ticker)
    df['종목명'] = [stock_name] * len(df)

    print("========= stock_search " + ticker +" / "+ stock_name + start_date +" / "+ end_date)
    return json.dumps(df.to_dict(orient='records'), ensure_ascii=False)

@mcp.tool("kakao_send_message")
async def kakao_send_message(send_message):
    """카카오톡(카톡) 메세지를 보낸다. 수신자에게 보낼 메세지를 파라미터로 전달해야 한다."""
    print(f"========= kakao_send_message: {send_message}")

    #access_token must be renewed every 12 hours.
    with open("./token.json", "r") as kakao:
        tokens = json.load(kakao)
        access_token = tokens["access_token"]

    url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
        "Authorization": f"Bearer {access_token}"
    }
    data = {
        "object_type": "text",
        "text": send_message,
        "link": {
            "web_url": "https://developers.kakao.com",
            "mobile_web_url": "https://developers.kakao.com"
        },
        "button_title": "확인"
    }
    data = {"template_object": json.dumps(data)}
    response = requests.post(url, headers=headers, data=data)

    if response.status_code == 200:
        return f"카톡 메시지가 성공적으로 전송되었습니다.({response.status_code})"
    else:
        return f"카톡 메시지 전송 중 문제가 발생했습니다.: {response.status_code}, {response.text}"

@mcp.tool("send_sms_message")
async def send_sms_message(rcv_phone_number, subject, send_message):
    """SMS 메세지를 발송한다. 수신자 전화번호와 수신자에게 보낼 메세지를 파라미터로 전달해야 한다.
    발송 메세지는 최대 2000byte 까지만 가능하며, 전화번호는 반드시 핸드폰 전화번호 11자리 번호만 사용해야 한다."""
    print(f"========= send_sms_message: {rcv_phone_number} / {subject} / {send_message}")

    sndr_phone_number = os.getenv("SMS_SENDER_PHONE_NUMBER")
    subject = "[알림]"+subject

    response = sens_client.send_message(
        from_num=sndr_phone_number,     # 발신번호 # type: ignore
        to_num=rcv_phone_number,        # 수신번호
        content=send_message,           # 문자내용
        subject=subject
    )
    if response.status_code == 202:
        return "SMS 메시지가 성공적으로 발송되었습니다."
    else:
        return f"SMS 메시지 발송 중 문제가 발생했습니다.: {response.status_code}, {response.text}"

@mcp.tool("alarm_reservation")
async def alarm_reservation(rcv_phone_number, subject, send_message, reservation_time):
    """알람(알림)을 예약한다. 알람시간이 되면 알람시스템에서 SMS 메세지를 발송한다. 
    알람수신자 전화번호와 수신자에게 보낼 알람메세지, 예약시간(yyyy-MM-dd HH:mm)을 파라미터로 전달해야 한다.
    알람 메세지는 최대 2000byte 까지만 가능하며, 전화번호는 반드시 핸드폰 전화번호 11자리 번호만 사용해야 한다."""
    print(f"========= alarm_reservation: {rcv_phone_number} / {subject} / {send_message} / {reservation_time}")

    sndr_phone_number = os.getenv("SMS_SENDER_PHONE_NUMBER")
    subject = "[알림]"+subject

    response = sens_client.send_message(
        from_num=sndr_phone_number,     # 발신번호 # type: ignore
        to_num=rcv_phone_number,        # 수신번호
        content=send_message,           # 문자내용
        subject=subject,
        reserveTime=reservation_time,
        reserveTimeZone="Asia/Seoul"
    )
    if response.status_code == 202:
        return "일람 예약이 성공적으로 등록되었습니다."
    else:
        return f"알람 예약 등록 중 문제가 발생했습니다.: {response.status_code}"

@mcp.tool("send_emergency_call")
async def send_emergency_call(emergency_message):
    """사용자가 응급상황이거나 재난, 위험에 처해 있을때 응급구조요청 메세지를 발송한다. 
    메세지 수신대상자는 사전에 정의되어 있으며, 수신자에게 보낼 구조요청 메세지는 
    assistant가 사용자의 상황을 잘 파악하여 작성한 후 파라미터로 전달한다.
    응급구조요청 메세지는 최대 90byte 이내로 가능하면 간단명료하게 작성한다."""
    print(f"========= send_emergency_call: {emergency_message}")

    sndr_phone_number = os.getenv("SMS_SENDER_PHONE_NUMBER")
    subject = "[응급구조요청]도움이 필요합니다."

    emergency_phone_numbers = os.getenv("EMERGENCY_PHONE_NUMBER")
    rcv_phone_numbers = emergency_phone_numbers.split(",") # type: ignore
    for rcv_phone_number in rcv_phone_numbers:
        try:
            response = sens_client.send_message(
                from_num=sndr_phone_number,     # 발신번호 # type: ignore
                to_num=rcv_phone_number,        # 수신번호
                content=emergency_message,           # 문자내용
                subject=subject
            )
            if response.status_code == 202:
                print("응급구조요청 메시지가 성공적으로 발송되었습니다.")
            else:
                raise ValueError(f"응급구조요청 메시지 발송 중 문제가 발생했습니다.: {response.status_code}")
        except Exception as e:
            return e
    return "응급구조요청 메시지가 성공적으로 발송되었습니다."


if __name__ == "__main__":
    mcp.run(transport="sse")
