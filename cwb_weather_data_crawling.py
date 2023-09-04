#匯入套件
#需要安裝chromedriver.exe，放到與此檔案同一個路徑底下
#根據當前使用電腦的Chrome版本下載
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
import time
import pandas as pd

#爬取中央氣象局開放觀測資料月報表，index可以篩選想要爬取的欄位
def get_daily_data(date,d):
    browser =webdriver.Chrome()
    browser.get("https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp")
    w1 = browser.window_handles[0]
    time.sleep(1)
    select = browser.find_element(By.ID,"station")
    option = select.find_element(By.CSS_SELECTOR,"option[value='466920']")
    option.click()
    select = browser.find_element(By.ID,"datatype")
    option = select.find_element(By.CSS_SELECTOR,"option[value='mn']")
    option.click()
   
    search_input = browser.find_element(By.ID,"datepicker")  
    search_input.send_keys(date) #YYYY-MM
    search_button = browser.find_element(By.ID,"doquery") 
    search_button.click()
    time.sleep(2)

    w2 = browser.window_handles[1]
    browser.switch_to.window(w2)
    
    #日期 測站氣壓 海平面氣壓 溫度 最高溫度 最低溫度 露點溫度 相對溼度 能見度 總雲量
    index = [5,2,3,8,9,11,13,14,31,35] 
    df = pd.DataFrame()
    all_data = []
    for i in range(4,4+d):
        data = []
        d = browser.find_element(By.XPATH,"/html/body/div[2]/table/tbody/tr[{}]/td[{}]".format(i,index[0]))   
        data.append(d.text[:10])
        for j in range(1,len(index)):
            d = browser.find_element(By.XPATH,"/html/body/div[2]/table/tbody/tr[{}]/td[{}]".format(i,index[j]))   
            if d.text.strip() != "X":
                data.append(float(d.text.strip())) 
            else:
                data.append(None)
        all_data.append(data)
    df = pd.DataFrame(all_data,columns=["date","station_pressure","sea_level_pressure","temp","max_temp","min_temp",
                                        "dew_point_temp","relative_humidity","visibility","cloud"])
    print(len(df),end=",")
    return df

#根據不同月份其中的天數爬取資料
def get_temp_data(year,days):
    month = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    months = [str(year)+"-"+i for i in month]
    df = pd.DataFrame()
    for m,d in zip(months,days):
        a = get_daily_data(m,d)
        df = pd.concat([df,a])
    print(len(df))
    df.to_csv("{}_weather_data.csv".format(year),index=False)
    print("data save in {}_weather_data.csv".format(year))

#開始抓取每年資料，自動儲存成csv檔案
#day = 非閏年每個月的天數
#days2 = 閏年每個月的天數
#d_year = 非閏年年份
#d2_year = 閏年年份
days = [31,28,31,30,31,30,31,31,30,31,30,31]
days2 = [31,29,31,30,31,30,31,31,30,31,30,31]
d_year = [2022,2021,2019,2018,2017,2015,2014,2013]
d2_year = [2020,2016]

#取非閏年資料
for i in d_year:
    get_temp_data(i,days)

#取閏年資料
for i in d2_year:
    get_temp_data(i,days2)
