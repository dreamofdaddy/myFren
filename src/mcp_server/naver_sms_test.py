import os
from dotenv import load_dotenv
from naver_sens import SensClient


load_dotenv("/home/dody/work/kosa-ict-genai-2025-2nd/src/exercise/dody/.env")
access_key = os.getenv("SMS_ACCESS_KEY")
secret_key = os.getenv("SMS_SECRET_KEY")
service_key = os.getenv("SMS_SERVICE_KEY")
sndr_number = os.getenv("SMS_SENDER_PHONE_NUMBER")
rcv_number = "01027181102"

sens_client = SensClient(
    service_id=service_key,
    access_key=access_key,
    secret_key=secret_key
)
response = sens_client.send_message(
    from_num=sndr_number,  # 발신번호
    to_num=rcv_number,   # 수신번호
    content="[알림] SMS 발송 테스트입니다."  # 문자 내용
)
print("============메시지 전송 상태: ", response.status_code)
print(response.text)
