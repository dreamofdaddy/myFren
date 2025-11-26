import requests
import json

### 카카오 로그인 코드받기(12시간용)
#https://kauth.kakao.com/oauth/authorize?client_id=5d7f8ef9a90e511fbf5a8f25e0ab5988&redirect_uri=https://example.com/oauth&response_type=code
#https://example.com/oauth?code=SkrAnse6i_0K5O5s7vfPFGkLwesolFRmEuLzkrJ5nRu4vSp2-E5vhgAAAAQKFxDvAAABmoVLJwwq17LwdM8QAg


url = 'https://kauth.kakao.com/oauth/token'
client_id = '5d7f8ef9a90e511fbf5a8f25e0ab5988'
redirect_uri = 'https://example.com/oauth'
code = 'SkrAnse6i_0K5O5s7vfPFGkLwesolFRmEuLzkrJ5nRu4vSp2-E5vhgAAAAQKFxDvAAABmoVLJwwq17LwdM8QAg'

data = {
    'grant_type':'authorization_code',
    'client_id':client_id,
    'redirect_uri':redirect_uri,
    'code': code,
}

response = requests.post(url, data=data)
tokens = response.json()

#발행된 토큰 저장
with open("./token.json","w") as kakao:
    json.dump(tokens, kakao)
