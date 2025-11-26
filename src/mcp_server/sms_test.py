import os
import hmac, hashlib, base64
import time, requests, json
from dotenv import load_dotenv

load_dotenv("/home/dody/work/kosa-ict-genai-2025-2nd/src/exercise/dody/.env")

def make_signature(secret_key, access_key, timestamp, uri):
    secret_key = bytes(secret_key, 'UTF-8')
    method = "POST"
    message = method + " "  + uri + "\\n" + timestamp + "\\n" + access_key
    message = bytes(message, 'UTF-8')
    signingKey = base64.b64encode(hmac.new(secret_key, message, digestmod=hashlib.sha256).digest())
    return signingKey

access_key = os.getenv("SMS_ACCESS_KEY")
secret_key = os.getenv("SMS_SECRET_KEY")
service_key = os.getenv("SMS_SERVICE_KEY")

# <https://api.ncloud-docs.com/docs/ko/ai-application-service-sens-smsv2>
url = "https://sens.apigw.ntruss.com"
uri = f"/sms/v2/services/{service_key}/messages"

timestamp = int(time.time() * 1000)
timestamp = str(timestamp)

# sender phone number
sndr_number = os.getenv("SMS_SENDER_PHONE_NUMBER")
print(f"================sndr_number: {sndr_number}")

# receiver phone number
rcv_number = "01027181102"

# sms subject
sms_subject = "[MYFREN]문자안내"
# sms contents
sms_contents = "SMS 발송 테스트 입니다.(from MYFREN.)"

header = {
    "Content-Type": "application/json; charset=utf-8",
    "x-ncp-apigw-timestamp": timestamp,
    "x-ncp-iam-access-key": access_key,
    "x-ncp-apigw-signature-v2": make_signature(secret_key, access_key, timestamp, uri)
}

# from : SMS 인증한 사용자만 가능
data = {
    "type":"SMS",
    "from":sndr_number,
    "content":sms_contents,
    "subject":sms_subject,
    "messages":[
        {
            "to":rcv_number,
        }
    ]
}

response = requests.post(url+uri,headers=header,data=json.dumps(data))

print("============메시지 전송 상태: ", response.status_code)
print(response.text)
