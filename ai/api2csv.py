import requests
import pprint
import json
import pandas as pd
from pandas.io.json import json_normalize


# url = "https://api.odcloud.kr/api/gov24/v3/serviceList?page=1&perPage=1000&serviceKey=llT3a%2Fs%2BeadzZv098xtjQLa2I8J5xLpfITc7Gem3WOGpwhYjbouTgVyXQ7gLs4HgMErxbIXYtwZEqhP76zLCLQ%3D%3D&type=json"
# # pageNo=1&numOfRows=10000&
# response = requests.get(url)
# contents = response.text
# pp = pprint.PrettyPrinter(indent=4)
# # print(pp.pprint(contents))

# json_ob = json.loads(contents)
# # print(json_ob)
# # print(type(json_ob))

# body = json_ob['data']
# # print(body)

# dataframe= json_normalize(body)
# print(dataframe)

# dataframe.to_csv('serviceList.csv',index=False,encoding='utf-8-sig')

import requests
import pandas as pd

# API 설정
api_key = 'llT3a%2Fs%2BeadzZv098xtjQLa2I8J5xLpfITc7Gem3WOGpwhYjbouTgVyXQ7gLs4HgMErxbIXYtwZEqhP76zLCLQ%3D%3D'
base_url = f"https://api.odcloud.kr/api/gov24/v3/supportConditions?serviceKey={api_key}&type=json"

# 초기 설정
per_page = 100
all_data = []
page = 1

while True:
    url = f"{base_url}&page={page}&perPage={per_page}"
    response = requests.get(url)
    data = response.json()

    # 데이터가 없는 경우 루프 종료
    if not data['data']:
        break

    # 데이터 추가
    all_data.extend(data['data'])
    page += 1

# 데이터프레임으로 변환
df = pd.DataFrame(all_data)

# 전체 행 개수 출력
print(f"Total number of rows: {len(df)}")

# CSV 파일로 저장
df.to_csv('supportConditions.csv', index=False, encoding='utf-8-sig')