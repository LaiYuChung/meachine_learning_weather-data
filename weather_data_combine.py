#匯入套件
#合併爬取後的天氣資料
import pandas as pd
import os

#合併短、中、長期資料
#範例為檔案放在histort_data資料夾路徑底下，可根據當下狀況更改路徑
#下述操作為根據cwb_weather_data_crawling設定之檔名，如檔名不同需修改此部分
file_name = os.listdir("./history_data/")
data1y = pd.DataFrame()
for i in range(1,2):
    data = pd.read_csv("./history_data/"+file_name[-i])
    data1y = pd.concat([data1y,data])
print(len(data1y))
    
data5y = pd.DataFrame()
for i in range(5):
    data = pd.read_csv("./history_data/"+file_name[5+i])
    data5y = pd.concat([data5y,data])
print(len(data5y))

data10y = pd.DataFrame()
for i in range(10):
    data = pd.read_csv("./history_data/"+file_name[i])
    data10y = pd.concat([data10y,data])
print(len(data10y))

#因原先資料欄位沒有對齊，因此合併欄位資料，並刪除多餘欄位，如無以上問題可不需做此動作
data5y["dew_point_temp"] = list(data5y["dew_point_temp"])[:1096] + list(data5y["dew point temp"])[1096:]
data5y["relative_humidity"] = list(data5y["relative_humidity"])[:1096] + list(data5y["relative humidity"])[1096:]
data5y.drop(["dew point temp","relative humidity"],axis=1,inplace=True)

data10y["dew_point_temp"] = list(data10y["dew_point_temp"])[:2922] + list(data10y["dew point temp"])[2922:]
data10y["relative_humidity"] = list(data10y["relative_humidity"])[:2922] + list(data10y["relative humidity"])[2922:]
data10y.drop(["dew point temp","relative humidity"],axis=1,inplace=True)

# 儲存成三種不同時間區間的資料
data1y.to_csv("history_data_1_year",index=False)
data5y.to_csv("history_data_5_year",index=False)
data10y.to_csv("history_data_10_year",index=False)

#因原先資料有缺失值，補上缺失值當下個月份之平均值，並回存檔案
#如無以上問題可不需做此動作
d1 = pd.read_csv("history_data_10_year")
d1["dew_point_temp"][2517:2519] = [17.0, 17.0]
d1["relative_humidity"][2517:2519] = [75.3, 75.3]
d1.to_csv("history_data_10_year")

d1 = pd.read_csv("history_data_5_year")
d1["dew_point_temp"][691:693] = [17.0, 17.0]
d1["relative_humidity"][691:693] = [75.3, 75.3]
d1.to_csv("history_data_5_year")
