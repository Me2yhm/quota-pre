import requests
from datetime import date

today = date.today().strftime("%Y-%m-%d")

# 股指期货预测信号-接口示例
url = "http://172.16.7.32:9007/spif_signal"
payload = {
    "date": f"{today}",
    "content": """
今日实盘\n
IC:小跌, 正确
[0.0014, 0.7061, 0.2564, 0.0356, 0.0005]
IH:小跌, 实际持平
[0.0011, 0.6952, 0.0255, 0.278, 0.0002]
IM:小跌, 实际持平
[0.0593, 0.6122, 0.0244, 0.2259, 0.0782]
IF:小涨, 实际持平
[0.0502, 0.4482, 0.0009, 0.5007, 0.0]
\n
今日预测\n
IC:小跌
[0.0003, 0.9733, 0.0005, 0.0259, 0.0]
IH:持平[0.0005, 0.0111, 0.981, 0.0073, 0.0]
IM:小跌
[0.158, 0.5604, 0.0633, 0.1681, 0.0502]
IF:大跌
[0.6312, 0.0149, 0.0, 0.3539, 0.0]
\n
<font color="comment">五个分类概率[大跌,小跌,持平,小涨,大涨]</font>
""",
}
response = requests.request("POST", url=url, json=payload)
assert response.status_code == 200, f"请求异常={response.status_code}"
print(response.text)

# response.text
# 成功：{"code":0,"message":"success","data":{}}
# 失败：{"code":1,"message":"error...","data":{}}
