import requests
from datetime import date

today = date.today().strftime("%Y-%m-%d")

# 股指期货预测信号-接口示例
url = "http://172.16.7.32:9007/spif_signal"
payload = {
    "date": f"{today}",
    "content": """
今日实盘\n
IC:持平, 小涨
[0.0038, 0.1903, 0.6669, 0.105, 0.0339]
IH:小涨, 小跌
[0.0005, 0.1984, 0.1913, 0.4324, 0.1774]
IM:持平, 小跌
[0.0038, 0.1513, 0.7186, 0.0942, 0.032]
IF:小跌, 正确
[0.025, 0.8871, 0.0372. 0.0507, 0.0001]
\n
明日预测\n
IC:持平
[0.0002, 0.0675, 0.7906, 0.0411, 0.1006]
IH:持平
[0.0, 0.0756, 0.6354, 0.0714, 0.2175]
IM:小跌
[0.0335, 0.5421, 0.0049, 0.3926, 0.027]
IF:大跌
[0.9801, 0.0166, 0.0001, 0.0031, 0.0001]
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
