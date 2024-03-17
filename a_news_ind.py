import os
import sys
import urllib.request
from urllib.parse import *
import requests
from bs4 import BeautifulSoup
import json
import re
from collections import Counter
import datetime
import numpy as np
import time


# 뉴스api: 49674d443d0d4e7995cf0ac5fef58827




import requests
import os
import sys
import urllib.request



class news_ind:
    def __init__(self,minute):

        self.url = "https://openapi.naver.com/v1/search/news.json"

        self.naver_headers= {"X-Naver-Client-Id": "p_rolCKyuutHJ0EzJbRr",
                                "X-Naver-Client-Secret": "WUAUiEvKcb"
                            }
        self.oversea_api= '49674d443d0d4e7995cf0ac5fef58827'

        self.minute= minute  #불러올 기간


    def naver_trend(self):
        # 트랜드
        client_id = self.naver_headers["X-Naver-Client-Id"]
        client_secret = self.naver_headers["X-Naver-Client-Secret"]
        self.url = "https://openapi.naver.com/v1/datalab/search";
        body = "{\"startDate\":\"2023-01-01\",\"endDate\":\"2023-04-30\",\"timeUnit\":\"month\",\"keywordGroups\":[{\"groupName\":\"한글\",\"keywords\":[\"한글\",\"korean\"]},{\"groupName\":\"영어\",\"keywords\":[\"영어\",\"english\"]}],\"device\":\"pc\",\"ages\":[\"1\",\"2\"],\"gender\":\"f\"}";

        request = urllib.request.Request(self.url)
        request.add_header("X-Naver-Client-Id",client_id)
        request.add_header("X-Naver-Client-Secret",client_secret)
        request.add_header("Content-Type","application/json")
        response = urllib.request.urlopen(request, data=body.encode("utf-8"))
        rescode = response.getcode()

        if(rescode==200):
            response_body = response.read()
            print(response_body.decode('utf-8'))
        else:
            print("Error Code:" + rescode)


    def naver_korea_news(self,keywards):
        # 검색할 키워드 리스트
        keywords = keywards
        # 결과를 저장할 dict
        results = {}
        # 최신순
        results_list=[] #최신순 종합할때 사용


        #기간별
        keyword_count={} # 기간동안 키워드 카운트
        filtered_articles = [] # 각 키워드 기간값 내 저장
        filtered_results = {} # 전체 키워드 저장

        #datetime.timedelta(minutes=self.minute)
        start_date = datetime.datetime.now()- datetime.timedelta(minutes=self.minute)   # 현재 시간에서 10시간 전
        end_date = datetime.datetime.now()  # 현재 시간


        # 각 키워드에 대한 기사 검색 및 저장

        for keyword in keywords:

            filtered_articles = []  # 각 키워드 기간값 내 저장

            params = {
                "query": keyword,
                "display": 100,  # 가져올 기사 수 설정 (최대 100)
                "sort": 'sim', # 정확도순sim   최신순 date
                   }
            response = requests.get(self.url, headers=self.naver_headers, params=params)
            data = response.json()


            # 기사 결과를 dict에 저장
            results[keyword] = data["items"]
            articles = data["items"]

            # 최신순
            results_list.extend(articles)
            results_list.sort(key=lambda x: x['pubDate'], reverse=True)
            latest_articles = results_list[:100] # 최신 100개 기사 추출

            #기간만큼 최대 (100개), articles는 키워드 하나의 뉴스
            for idx,article in enumerate(articles):
                published_date = datetime.datetime.strptime(article['pubDate'],
                                                            '%a, %d %b %Y %H:%M:%S %z')  # assuming date format is 'YYYY-MM-DD'
                published_date = published_date.replace(tzinfo=None)  # 뉴스데이터의 날짜값

                # Check if the article is in the desired date range

                if start_date <= published_date <= end_date:
                    filtered_articles.append(article)
                    print(article['link'], '키워드:', keyword, idx)

            # Save the filtered articles

            filtered_results[keyword] = filtered_articles
            keyword_count[keyword]= len(filtered_articles)


        # 종합 (키워드별로 n개씩 뽑음)
        sort_res={}
        for keyword, articles in results.items():
            for idx, article in enumerate(articles):
                sort_res[str(keyword)+'_'+str(idx+1)]=article['title']

        return sort_res,keyword_count





    def oversea_news(self,country):
        api_key= self.oversea_api
        #url = 'https://newsapi.org/v2/top-headlines' 헤드라인
        url = 'https://newsapi.org/v2/everything'
        params = {
            'apiKey': api_key,
            'q': 'finance',
            #'category': 'technology',  # 헤드라인에만 사용
            #'country': country , # 국가에 맞게 변경 가능 # 헤드라인에만 사용
            'pageSize': 100,
            'sortBy': 'relevancy',  # 관련성 순으로 정렬

             }

        '''''
        'business': 비즈니스 관련 뉴스
        'entertainment': 엔터테인먼트 관련 뉴스
        'health': 건강 관련 뉴스
        'science': 과학 관련 뉴스
        'sports': 스포츠 관련 뉴스
        'technology': 기술 관련 뉴스
        '''''

        response = requests.get(url, params=params)
        data = response.json()

        if data['status'] == 'ok':
            articles = data['articles']
            for article in articles:
                print(article['title'],article['url'])
        else:
            print('Error:', data['message'])




    def start_news_ind(self,pos_neg_kewards):
        while True:
            try:
                neg_value=0
                pos_value=0

                print('--------------------------------------------부정 키워드 -------------------------------------------')
                res,neg_keyword_count=self.naver_korea_news(pos_neg_kewards['neg'])
                print('--------------------------------------------긍정 키워드 -------------------------------------------')
                res,pos_keyword_count=self.naver_korea_news(pos_neg_kewards['pos'])

                print('--------------------------------------------결 과  -------------------------------------------')
                for idx,count_dict in enumerate([neg_keyword_count,pos_keyword_count]):

                    if idx == 0 :# 부정 키워드
                        neg_value= np.sum(list(count_dict.values()))
                        neg_v=list(count_dict.values())

                    else: #긍정 키워드
                        pos_value = np.sum(list(count_dict.values()))
                        pos_v=list(count_dict.values())

                res = []
                for neg, pos in zip(neg_v, pos_v):
                    if neg == 0:
                        res.append(1.1)
                    else:
                        res.append(pos / neg)

                return_value2 = np.mean(res)
                return_value = pos_value/neg_value

                print(neg_keyword_count,'======>',neg_value,'neg값')
                print(pos_keyword_count,'======>',pos_value,'pos값')
                print('korea news::::::' ,'갯수중시 sum value: ', return_value,'비율중시 mean value',return_value2)

            except:
                print('오류')

            time.sleep(60)



# 국내 해외 공통
pos_keyword = ['연준 금리 동결 인하 결정', '부채 한도협상 진전 ', 'FOMC 금리동결 인하 결정','CPI 예상 하회','CPI 낙관' ]
neg_keyword = ['연준 금리 인상 결정 ', '부채 한도협상 난항', 'FOMC 금리인상 결정', 'CPI 예상 상회','CPI 우려']
minute = 600 # 뉴스 불러올 기간

# 해외
country= 'us'

'''''
전체국가 : 공백
'us': 미국 (United States)
'gb': 영국 (United Kingdom)
'ca': 캐나다 (Canada)
'au': 오스트레일리아 (Australia)
'fr': 프랑스 (France)
'de': 독일 (Germany)
'jp': 일본 (Japan)
'kr': 대한민국 (South Korea)
'cn': 중국 (China)
'in': 인도 (India)
'''''


total_keyword={'pos':pos_keyword,'neg':neg_keyword}
news= news_ind(minute)
news.oversea_news(country)
news.start_news_ind(total_keyword) # 인풋은 dict 형태 ( 한국뉴스 데이터 시작)

