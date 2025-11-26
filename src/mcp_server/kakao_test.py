import requests
import json

with open("./token.json", "r") as kakao:
    tokens = json.load(kakao)
    access_token = tokens["access_token"]

url = "https://kapi.kakao.com/v2/api/talk/memo/default/send"
send_message = "카톡메세지 발송 테스트."
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
    "button_title": "바로 확인"
}
data = {"template_object": json.dumps(data)}
response = requests.post(url, headers=headers, data=data)

if response.status_code == 200:
    print( f"메시지를 성공적으로 전송했습니다.({response.status_code})")
else:
    print(f"메시지 전송중 문제가 발생했습니다.: {response.status_code}, {response.text}")
