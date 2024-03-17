
import datetime
import torch
from torch.distributions import Categorical
import psycopg2
import e_train as params
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np

import a_Env as env_
env=env_.Env()


class Rule_agent():
    def __init__(self,
                 cash,  # 초기 보유현금
                 cost,  # 수수료 %
                 input_,  # 인풋 데이터
                 ori_input,
                 price_data,  # 주가 데이터
                 date_data,  # 날짜 데이터
                 input_dim,  # feature 수 (숏이나 롱의 dim 하나만 들어온다 ( APPO py 에선 dict으로 들어옴)
                 deposit,  # 증거금
                 slippage,  # 슬리피지
                 if_short_or_long
                 ):
        # 클래스 상속

        self.short_or_long = if_short_or_long

        self.scale_input = input_
        self.long_ori_input=ori_input[0] #롱 ori인풋
        self.short_ori_input= ori_input[1]
        self.dim = input_dim
        self.price_data = price_data  # 종가 데이터
        self.date_data = date_data[:]  # 백테스팅때 사용
        self.close_df_data=0 # 종가데이터
        self.close_date_index= False # 종가 데이터 구하는 인덱스
        self.time_bool = 'None' #현 시각이 기준시각보다 많은지 적은지


        # 에이전트 변수
        self.init_cash = cash
        self.init_cost = cost # 수수료
        self.init_slip = slippage

        self.cash = cash  # 가진 현금
        self.cost = cost  # 수수료 비용
        self.deposit = deposit  # 증거금
        self.stock = 0  # 가진 주식수
        self.slip = slippage  # 슬리피지
        self.PV = self.cash  # 현 포트폴리오 벨류 저장
        self.PV_list = [self.cash]


        self.next_step = []
        self.action_data = []
        self.reward_data = []
        self.step_data = []

        self.action=torch.Tensor([1])
        self.unit=[0,0,0]

        # 롱포지션
        self.long_price = []
        self.long_aver_price = 0
        self.long_unit = 0

        self.B= False
        self.B1=False # B1 매수 진입시 True
        self.B2=False
        self.B3=False
        self.B4=False
        self.B4B=False
        self.B5=False
        self.B6=False
        self.B7=False

        self.B_buy_price=0
        self.B1_buy_price=0 # 매수했을때 가격
        self.B2_buy_price=0
        self.B3_buy_price=0
        self.B4_buy_price=0
        self.B4B_buy_price=0
        self.B5_buy_price=0
        self.B6_buy_price=0
        self.B7_buy_price=0

        self.EL=False
        self.EL1=False
        self.EL2=False
        self.EL3=False
        self.EL4=False
        self.EL4B=False
        self.EL5=False
        self.EL6=False
        self.EL7=False

        self.EL_exit_price=0
        self.EL1_exit_price = 0
        self.EL2_exit_price = 0
        self.EL3_exit_price = 0
        self.EL4_exit_price = 0
        self.EL4B_exit_price= 0
        self.EL5_exit_price = 0
        self.EL6_exit_price = 0
        self.EL7_exit_price = 0


        # 숏 포지션이 있을경우
        self.stock = 0
        self.short_unit = 0
        self.short_price = []  # 매수했던 가격*계약수
        self.short_aver_price = 0

        self.S =False
        self.S1 = False
        self.S2 = False
        self.S3 = False
        self.S4 = False
        self.S4B= False
        self.S5 = False
        self.S6 = False
        self.S7 = False

        self.S_sell_price=0
        self.S1_sell_price=0
        self.S2_sell_price=0
        self.S3_sell_price=0
        self.S4_sell_price=0
        self.S4B_sell_price=0
        self.S5_sell_price=0
        self.S6_sell_price=0
        self.S7_sell_price=0

        self.ES=False
        self.ES1=False
        self.ES2=False
        self.ES3=False
        self.ES4=False
        self.ES4B=False
        self.ES5=False
        self.ES6=False
        self.ES7=False

        self.ES_exit_price=0
        self.ES1_exit_price=0
        self.ES2_exit_price=0
        self.ES3_exit_price=0
        self.ES4_exit_price=0
        self.ES4B_exit_price=0
        self.ES5_exit_price=0
        self.ES6_exit_price=0
        self.ES7_exit_price=0

        self.past_PV = self.cash  # 한스탭이전 포트폴리오 벨류 (초기는 현금)
        self.gamma = 0.99
        self.Cumulative_reward = 0  # 누적 리워드
        self.old_prob = 0  # old prob 저장
        self.total_loss = 0  # actor_loss+critic_loss
        self.back_testing = False  # 백테스팅 or 학습일경우 False



        #SB시그널의 파라미터
        self.curr_exit_name = 'None'
        self.my_position='None'   #None 관망 long 롱 short 숏

        self.COND=0
        self.L_COND = 0
        self.NLRCCI_COND= 0

        self.condition1 = False
        self.condition2 = False
        self.condition3 = False
        self.condition4 = False

        self.EnPrice=0
        self.EnPrice1=0
        self.EnPrice2=0
        self.EnPrice3=0

        self.B1_count=0
        self.S1_count=0



    def reset(self):
        self.cash = self.init_cash  # 가진 현금
        self.cost = self.init_cost  # 수수료 퍼센트
        self.PV = self.init_cash  # 포트폴리오 벨류 저장
        self.slip = self.init_slip
        self.past_PV = self.cash  # 이전 포트폴리오 벨류 (초기는 현금과같음))
        self.long_price = []
        self.deposit = self.deposit  # 증거금
        self.long_aver_price = 0
        self.long_unit = 0
        self.stock = 0  # 가진 주식수
        self.back_testing = False
        self.step_data = []
        self.PV_list = [self.cash]

        self.next_step = []
        self.action_data = []
        self.reward_data = []
        self.step_data = []

        # 숏 포지션이 있을경우
        self.stock = 0
        self.short_unit = 0
        self.short_price = []  # 매수했던 가격*계약수
        self.short_aver_price = 0


    def close_search(self): # 마지막가격 서치 ( 1분봉을 뽑음)
        # 전날 종가 구해야함.
        # 모든 날짜 종가 가져오기
          # 1.모든 날짜에서 마지막꺼 추출

        self.name = params.stock_name
        connection = psycopg2.connect(dbname='postgres', user='postgres', password='snowai**', host='172.30.1.96',
                                      port='5432', sslmode='require')
        db = connection.cursor()


        # DB 데이터 호출(기존 datacount보다 하루치를 더뽑음 초기의 이전 종가를 구하기위해)
        if params.part_time == False:  # 부분 시간 학습 (ex 정규장이면 정규장만 불러오기)
            db.execute(
                f"SELECT open,close,high,low,volume,datetime FROM (SELECT open,close,high,low,volume,datetime FROM snowball.price_ai WHERE symbol={self.name} ORDER BY datetime DESC limit {params.data_count+1440}) as foo order by datetime asc;")
        data_set = db.fetchall()


        open = pd.Series([float(t[0]) for t in data_set])
        close = pd.Series([float(t[1]) for t in data_set])
        high = pd.Series([float(t[2]) for t in data_set])
        low = pd.Series([float(t[3]) for t in data_set])
        vol = pd.Series([float(t[4]) for t in data_set])
        date = pd.Series([t[5] for t in data_set])

        self.close_ = close
        self.open = open
        self.low = low
        self.high = high
        self.vol_ = vol
        self.date_ = date

        scaler = MinMaxScaler()  # 0-1사이로 정규화  평균0.5 분산1
        close_1 = scaler.fit_transform(self.close_.values.reshape(-1, 1))
        vol_1 = scaler.fit_transform(self.vol_.values.reshape(-1, 1))
        high_1 = scaler.fit_transform(self.high.values.reshape(-1, 1))
        open_1 = scaler.fit_transform(self.open.values.reshape(-1, 1))
        low_1 = scaler.fit_transform(self.low.values.reshape(-1, 1))

        close_ = self.close_  # 스케일링 이전 데이터
        vol_ = self.vol_
        open_ = self.open
        low_ = self.low
        high_ = self.high

        close_s = close_1.reshape(-1)  # 스케일링 데이터
        vol_s = vol_1.reshape(-1)
        low_s = low_1.reshape(-1)
        high_s = high_1.reshape(-1)
        open_s = open_1.reshape(-1)
        date = self.date_

        str_date=[str(date_) for date_ in date] # 1분 모든 날짜 데이터
        date= pd.to_datetime(str_date)
        date=pd.DataFrame({'date':date,'close':close_.values},index=date)
        close_df_data = date.between_time('16:59:00', '16:59:00') #종가 데이터

        date_pre= [str(date0) for date0 in close_df_data['date']] #나온 날짜를 str형식
        date_pre= pd.DataFrame({'date_pre':pd.Series(date_pre)})
        close_df_data = close_df_data.reset_index()

        close_df_data= pd.concat([close_df_data,date_pre],axis=1)

        return close_df_data



    def last_close(self,step):

        if step==2: # 처음에만 실행(스탭 1부터 룰베이스 적용되므로)
            self.close_df_data= self.close_search()  # 전날 종가 가져온다 (가격, 날짜, 현스탭)
            self.close_date=self.close_df_data['date']
            self.yesterday_close=self.close_df_data['close']

        #이전날 종가 구하기 ##################### 날짜, 시간 추가
        ind_date = datetime.datetime.strptime(self.date_data[step], '%Y-%m-%d %H:%M:%S')  # 날짜형으로 변환(지표의 현재날짜 시간)
        ind_date2= datetime.datetime.strptime(self.date_data[step][:10], '%Y-%m-%d') #지표의 현재 날짜만
        당일날00시 = datetime.datetime.strptime(self.date_data[step][:10]+' 00:00:00','%Y-%m-%d %H:%M:%S')
        다음날00시 = datetime.datetime.strptime(self.date_data[step][:10]+' 00:00:00','%Y-%m-%d %H:%M:%S') + datetime.timedelta(days=1)
        당일날17시 = datetime.datetime.strptime(self.date_data[step][:10]+' 17:00:00','%Y-%m-%d %H:%M:%S')

        last_date = ind_date2 + datetime.timedelta(days=-1)  # 이전 날짜
        last_date= str(last_date)[:10]

        last_close=False # 전일종가 ( 있으면 넣고 없으면 False)

        #전일 종가의 시간 구하기
        if 당일날00시<=ind_date and ind_date<=당일날17시: # 현 지표 시각이 00시 ~ 17:00 사이면 이전 날짜 종가
            last_close_time=last_date+' 16:59:00' #이전날짜 종가 시각
            try:  # 전일 종가가 존재하는경우
                self.close_date_index = self.close_df_data['date_pre'].tolist().index(last_close_time)
                last_close = self.close_df_data['close'].iloc[self.close_date_index]  # 전일 종가 가져옴

            except:  # 전일 종가가 존재하지 않는경우(주말,공휴일이거나 처음이라 없을때)
                try: #이전 종가를 가져온다
                    self.close_date_index = self.close_date_index-1
                    if self.close_date_index<0: #인덱스가 0보다 작은경우는 처음시작부분에서 데이터가 없을때다
                        self.close_data_index=False
                    last_close = self.close_df_data['close'].iloc[self.close_date_index]
                except: #그래도 없는경우
                    last_close = False


        if 다음날00시>ind_date and ind_date>당일날17시 : #17시이후 , 다음날 00시이전이면 오늘 날짜 종가
            last_close_time= str(ind_date2)[:10]+' 16:59:00' # 종가

            try: #전일 종가가 존재하는경우
                self.close_date_index = self.close_df_data['date_pre'].tolist().index(last_close_time)
                last_close=self.close_df_data['close'].iloc[self.close_date_index] #전일 종가 가져옴

            except: #전일 종가가 존재하지 않는경우
                self.close_date_index = self.close_date_index - 1
                if self.close_date_index<0: #인덱스가 0보다 작은경우는 처음시작부분에서 데이터가 없을때다
                    self.close_data_index=False
                try:
                    last_close = self.close_df_data['close'].iloc[self.close_date_index]
                except:
                    last_close= False



                #except:  # 그래도 없는경우 (처음시작부분)
                #last_close = False


        return last_close
        #if #(10월 26일인데 24시 이전이면 26일 종가 사용)
        #10월 26일 16:59 종가 100
        #10월 26일 18시 (장시작)


        ####################################################################################################

    def date_time(self,step, 기준시간): #현재시각이 설정한 기준시간보다 높으면 over, 낮으면 under 반환

        time_=self.date_data[step][-8:] # 현재시각
        time_= datetime.datetime.strptime(time_, '%H:%M:%S')
        기준시간=datetime.datetime.strptime(기준시간, '%H:%M:%S')


        if time_>기준시간:
            self.time_bool ='over'

        if time_< 기준시간:
            self.time_bool='under'

        if time_==기준시간:
            self.time_bool='same'

        return self.time_bool

    def long_B(self,adj_price_data,policy,step): # 매수 ( 원하는 폴리시만큼 만큼)

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit = self.long_decide_action(policy, self.deposit)

        self.B = True  # 매수완
        self.B_buy_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.EL = False  # Exit long 했는지
        self.EL_exit_price = 0  # Exit long 가격


    def long_B1(self,adj_price_data,step): #최초 매수
        policy = torch.Tensor([0, 0, 1])
        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit = self.long_decide_action(policy, self.deposit)

        self.B1 = True  # 매수완
        self.B1_buy_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.EL1 = False  # Exit long 했는지
        self.EL1_exit_price = 0  # Exit long 가격


    def long_B2(self,adj_price_data,step): #추가매수
        policy = torch.Tensor([0, 0, 1])

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit = self.long_decide_action(policy, self.deposit)

        self.B2 = True  # B2매수완료
        self.B2_buy_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.EL2 = False
        self.EL2_exit_price = 0


    def long_B3(self,adj_price_data,step): # 추매
        policy =torch.Tensor([0,0,1])

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit = self.long_decide_action(policy, self.deposit)

        self.B3 = True # B3 매수 완료
        self.B3_buy_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.EL3 = False
        self.EL3_exit_price = 0


    def long_B4(self,adj_price_data ,step): # 추매
        policy =torch.Tensor([0,0,1])

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit  = self.long_decide_action(policy,self.deposit)

        self.B4 =True
        self.B4_buy_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.EL4 = False
        self.EL4_exit_price = 0

    def long_B4B(self, adj_price_data, step):  # 추매
        policy = torch.Tensor([0, 0, 1])

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit = self.long_decide_action(policy, self.deposit)

        self.B4B = True
        self.B4_buy_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.EL4B = False
        self.EL4B_exit_price = 0



    def long_B5(self,adj_price_data,step): # 추매
        policy = torch.Tensor([0,0,1])

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit = self.long_decide_action(policy, self.deposit)

        self.B5 = True
        self.B5_buy_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.EL5= False
        self.EL5_exit_price = 0

    def long_B6(self, adj_price_data,step): # 추매
        policy = torch.Tensor([0,0,1])

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            self.action, self.unit = self.SC_decide_action(policy)
        else:
            self.action, self.unit = self.long_decide_action(policy,self.deposit)

        self.B6 = True
        self.B6_buy_price = adj_price_data[step]

        self.EL6 =False
        self.EL6_exit_price = 0





    def long_EL1(self,adj_price_data,step):
        self.action = torch.Tensor([0])  # 청산
        self.unit = [1, 0, 0]  # 1계약
        self.EL1 = True
        self.EL1_exit_price = adj_price_data[step]  # 청산가 기입

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            policy = torch.Tensor([1, 0, 0])
            self.action, self.unit = self.SC_decide_action(policy)

        self.B1 = False  # 청산했으므로 제거
        self.B1_buy_price = 0


    def long_EL2(self,adj_price_data,step):
        self.action = torch.Tensor([0])
        self.unit = [1, 0, 0]  # 1계약
        self.EL2 = True  # EL1은 초기 매수(추매 아님)
        self.EL2_exit_price = adj_price_data[step]


        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            policy = torch.Tensor([1, 0, 0])
            self.action, self.unit = self.SC_decide_action(policy)

        self.B2 = False  # B2 에 해당하는 추가매매 계약 청산
        self.B2_buy_price = 0


    def long_EL3(self,adj_price_data,step):
        self.action = torch.Tensor([0])
        self.unit = [1, 0, 0]  # 1계약
        self.EL3 = True  # EL1은 초기 매수(추매 아님)
        self.EL3_exit_price = adj_price_data[step]

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            policy = torch.Tensor([1, 0, 0])
            self.action, self.unit = self.SC_decide_action(policy)

        self.B3 = False  # B2 에 해당하는 추가매매 계약 청산
        self.B3_buy_price = 0


    def long_EL4(self,adj_price_data,step):
        self.action = torch.Tensor([0])
        self.unit = [1, 0, 0]  # 1계약
        self.EL4 = True  # EL1은 초기 매수(추매 아님)
        self.EL4_exit_price = adj_price_data[step]

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            policy = torch.Tensor([1, 0, 0])
            self.action, self.unit = self.SC_decide_action(policy)

        self.B4 = False  # B2 에 해당하는 추가매매 계약 청산
        self.B4_buy_price = 0


    def long_EL4B(self,adj_price_data,step):
        self.action = torch.Tensor([0])
        self.unit = [1, 0, 0]  # 1계약
        self.EL4B = True  # EL1은 초기 매수(추매 아님)
        self.EL4B_exit_price = adj_price_data[step]

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            policy = torch.Tensor([1, 0, 0])
            self.action, self.unit = self.SC_decide_action(policy)

        self.B4B = False  # B2 에 해당하는 추가매매 계약 청산
        self.B4B_buy_price = 0

    def long_EL5(self,adj_price_data,step):
        self.action = torch.Tensor([0])
        self.unit = [1, 0, 0]  # 1계약
        self.EL5 = True  # EL1은 초기 매수(추매 아님)
        self.EL5_exit_price = adj_price_data[step]

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            policy = torch.Tensor([1, 0, 0])
            self.action, self.unit = self.SC_decide_action(policy)

        self.B5 = False  # B2 에 해당하는 추가매매 계약 청산
        self.B5_buy_price = 0


    def long_EL6(self,adj_price_data,step):
        self.action = torch.Tensor([0])
        self.unit = [1, 0, 0]  # 1계약
        self.EL6 = True  # EL1은 초기 매수(추매 아님)
        self.EL6_exit_price = adj_price_data[step]

        if params.train_stock_or_future == 'coin' or params.train_stock_or_future =='stock':
            policy = torch.Tensor([1, 0, 0])
            self.action, self.unit = self.SC_decide_action(policy)

        self.B6 = False  # B2 에 해당하는 추가매매 계약 청산
        self.B6_buy_price = 0


    def short_S(self,adj_price_data,policy,step): #원하는 만큼 매도
        self.action, self.unit = self.short_decide_action(policy, self.deposit)

        self.S = True  # 매수완
        self.S_sell_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.ES = False  # Exit short 했는지
        self.ES_exit_price = 0  # Exit short 가격

    def short_S1(self,adj_price_data,step): #최초 매도
        policy = torch.Tensor([0, 0, 1])
        self.action, self.unit = self.short_decide_action(policy, self.deposit)

        self.S1 = True  # 매수완
        self.S1_sell_price = adj_price_data[step]  # 매수 가격 포인트가치 합산안된 ori 가격

        self.ES1 = False  # Exit short 했는지
        self.ES1_exit_price = 0  # Exit short 가격



    def short_S2(self,adj_price_data,step): #추가 매도
        policy = torch.Tensor([0, 0, 1])
        self.action, self.unit = self.short_decide_action(policy, self.deposit)

        self.S2 = True  # 추매 완
        self.S2_sell_price = adj_price_data[step]

        self.ES2 = False
        self.ES2_exit_price = 0




    def short_S3(self,adj_price_data,step): #주가매도
        policy = torch.Tensor([0,0,1])
        self.action, self.unit = self.short_decide_action(policy, self.deposit)

        self.S3 = True
        self.S3_sell_price = adj_price_data[step]

        self.ES3 = False
        self.ES3_exit_price = 0



    def short_S4(self,adj_price_data,step): #주가매도
        policy = torch.Tensor([0,0,1])
        self.action, self.unit = self.short_decide_action(policy, self.deposit)

        self.S4 = True
        self.S4_sell_price = adj_price_data[step]

        self.ES4 = False
        self.ES4_exit_price = 0



    def short_S4B(self,adj_price_data,step): #주가매도
        policy = torch.Tensor([0,0,1])
        self.action, self.unit = self.short_decide_action(policy, self.deposit)

        self.S4B = True
        self.S4B_sell_price = adj_price_data[step]

        self.ES4B = False
        self.ES4B_exit_price = 0



    def short_S5(self,adj_price_data,step): #주가매도
        policy = torch.Tensor([0,0,1])
        self.action, self.unit = self.short_decide_action(policy, self.deposit)

        self.S5 = True
        self.S5_sell_price = adj_price_data[step]

        self.ES5 = False
        self.ES5_exit_price = 0


    def short_S6(self,adj_price_data, step):
        policy =torch.Tensor([0,0,1])
        self.action,self.unit = self.short_decide_action(policy, self.deposit)

        self.S6= True
        self.S6_sell_price = adj_price_data[step]

        self.ES6 = False
        self.ES6_exit_price=0


    def short_ES1(self,adj_price_data,step): #최초 청산
        self.action = torch.Tensor([0])  # 청산
        self.unit = [1, 0, 0]  # 1계약
        self.ES1 = True
        self.ES1_exit_price = adj_price_data[step]  # 청산가 기입

        self.S1 = False  # 청산했으므로 제거
        self.S1_sell_price = 0


    def short_ES2(self,adj_price_data,step): #최초 청산
        self.action = torch.Tensor([0])
        self.unit = [1, 0, 0]  # 1계약
        self.ES2 = True  # EL1은 초기 매수(추매 아님)
        self.ES2_exit_price = adj_price_data[step]

        self.S2 = False  # S2 에 해당하는 추가매매 계약 청산
        self.S2_sell_price = 0


    def short_ES3(self, adj_price_data, step):
        self.action= torch.Tensor([0])
        self.unit = [1,0,0]
        self.ES3 =True
        self.ES3_exit_price = adj_price_data[step]
        self.S3 = False
        self.S3_sell_price = 0


    def short_ES4(self, adj_price_data, step):
        self.action= torch.Tensor([0])
        self.unit = [1,0,0]
        self.ES4 =True
        self.ES4_exit_price = adj_price_data[step]

        self.S4 = False
        self.S4_sell_price = 0

    def short_ES4B(self, adj_price_data, step):
        self.action= torch.Tensor([0])
        self.unit = [1,0,0]
        self.ES4B =True
        self.ES4B_exit_price = adj_price_data[step]

        self.S4B = False
        self.S4B_sell_price = 0

    def short_ES5(self, adj_price_data, step):
        self.action= torch.Tensor([0])
        self.unit = [1,0,0]
        self.ES5 =True
        self.ES5_exit_price = adj_price_data[step]

        self.S5 = False
        self.S5_sell_price = 0

    def short_ES6(self, adj_price_data, step):
        self.action= torch.Tensor([0])
        self.unit = [1,0,0]
        self.ES6 =True
        self.ES6_exit_price = adj_price_data[step]

        self.S6 = False
        self.S6_sell_price = 0



    def Exit_all_long(self,adj_price_data,step): #전량청산
        # 전량 청산
        self.action = torch.Tensor([0])
        self.unit = [self.long_unit, 0, 0]  # 가지고있는 전량 청산

        self.B= False
        self.B1 = False
        self.B2 = False
        self.B3 = False
        self.B4 = False
        self.B5 = False
        self.B6 = False

        self.B_buy_price=0
        self.B1_buy_price = 0
        self.B2_buy_price = 0
        self.B3_buy_price = 0
        self.B4_buy_price = 0
        self.B5_buy_price = 0
        self.B6_buy_price = 0

        self.EL=True
        self.EL1 = True
        self.EL2 = True
        self.EL3 = True
        self.EL4 = True
        self.EL5 = True
        self.EL6= True

        self.EL_exit_price =adj_price_data[step]
        self.EL1_exit_price = adj_price_data[step]
        self.EL2_exit_price = adj_price_data[step]
        self.EL3_exit_price = adj_price_data[step]
        self.EL4_exit_price = adj_price_data[step]
        self.EL5_exit_price = adj_price_data[step]
        self.EL6_exit_price = adj_price_data[step]





    def Exit_all_short(self,adj_price_data,step):
        self.action = torch.Tensor([0])
        self.unit=[self.short_unit,0,0]

        self.S= False
        self.S1 = False
        self.S2 = False
        self.S3 = False
        self.S4 = False
        self.S5 = False
        self.S6= False

        self.S_sell_price =0
        self.S1_sell_price = 0
        self.S2_sell_price = 0
        self.S3_sell_price = 0
        self.S4_sell_price = 0
        self.S5_sell_price = 0
        self.S6_sell_price = 0

        self.ES= True
        self.ES1 = True
        self.ES2 = True
        self.ES3 = True
        self.ES4 = True
        self.ES5 = True
        self.ES6= True

        self.ES_exit_price = adj_price_data[step]
        self.ES1_exit_price = adj_price_data[step]
        self.ES2_exit_price = adj_price_data[step]
        self.ES3_exit_price = adj_price_data[step]
        self.ES4_exit_price = adj_price_data[step]
        self.ES5_exit_price = adj_price_data[step]
        self.ES6_exit_price = adj_price_data[step]


    def Rule_base(self, is_short_or_long, step):  # 룰베이스 로직

        한계약매매 = True  # 분할매매 여부 False 면 분할매매 True면 한계약 매매
        self.action = torch.Tensor([1])  # 초기상태 관망
        self.unit = [0, 0, 0]

        adj_price_data = self.price_data  # 원래 가격

        # buy 가 들어갈때 all short exit
        # sell이 들어갈때 S1= True
        # EL 이 들어갈때 if B1=True
        # ES 가 들어갈때 is S1=True

        ind_CCI = self.long_ori_input[params.long_ind_name.index('CCI_trend')]  #지표 불러오기
        NNCO_up_L = self.long_ori_input[params.long_ind_name.index('NNCO_up_L')]
        NNCO_down_L =self.long_ori_input[params.long_ind_name.index('NNCO_down_L')]

        '''''
        알고리즘 전략 부분
        '''''
        return self.action, self.unit



    def SC_decide_action(self, policy):  # stock이나 코인의 경우 decide_action

        policy1 = torch.clamp(policy, 0, 1)
        action_s = Categorical(policy1)
        action = action_s.sample()  # 매도시 최소 1주 매도
        policy1 = policy1.to('cpu')
        self.sell_policy=policy1[0]
        self.buy_poliy=policy1[2]

        max_trade_cash = self.init_cash  # 최대 트레이딩 가능 캐시 ###################

        limit_buy_cash = self.init_cash*0.2  # 스탭당 최대 매수가능 캐시 ################
        min_buy_cash = 0  # self.init_cash*0.2 #매수시 최소 매수 캐시

        limit_sell_unit = 10000000  # 스탭당  최대 판매 유닛
        min_sell_unit = 10000 #self.init_cash*0.2 #매도시 최소 판매 유닛 ######################



        if self.back_testing == True:  # 백테스팅일때 최대 한개
            action = torch.argmax(policy1)

            max_trade_cash = self.init_cash   # 최대 트레이딩 가능 캐시 #######################

            limit_buy_cash = self.init_cash*0.5   # 스탭당 최대 매수가능 캐시 ############
            min_buy_cash = 0  # self.init_cash*0.2 #매수시 최소 매수 캐시

            limit_sell_unit = 100000  # 스탭당  최대 판매 유닛
            min_sell_unit = 10000  # self.init_cash*0.2 #매도시 최소 판매 유닛 ##############


        if action == 0:  # 매도
            unit0 = policy1[0] * self.stock

            if params.train_stock_or_future == 'coin': # 코인인경우

                if unit0 >= limit_sell_unit:
                    unit0 = torch.Tensor([limit_sell_unit]).to(params.device)

                if unit0 <= min_sell_unit: # 매도 너무 적은경우 최소 매도만큼
                    if unit0==0:
                        unit0=torch.Tensor([0]).to(params.device)
                    else:
                        unit0 = torch.Tensor([min_sell_unit]).to(params.device)

                unit = [unit0.item(), 0, 0]

            else:
                unit0 = max((policy1[0] * self.stock), 1)
                unit0 = round(float(unit0))
                unit = [unit0, 0, 0]

                if unit0 >= limit_sell_unit:
                    unit0 = torch.Tensor([limit_sell_unit]).to(params.device)

                if torch.Tensor([unit0]).to('cpu')*torch.Tensor([(self.price-self.slip)]).to('cpu') <= min_sell_cash:
                    unit0,_ = divmod(min_sell_cash,float( torch.Tensor([unit0]).to('cpu')* torch.Tensor([(self.price-self.slip)]).to('cpu')))


        elif action == 2:  # 매수
            limit_cash = limit_buy_cash #설정한 최대 매수가능 캐시
            limit_cash =min(limit_cash, self.cash)
            unit2 = (policy1[2] * limit_cash) / (self.price+self.slip)

            limit_buy_unit = limit_buy_cash/(self.price+self.slip)  #설정한 최대 매수 가능 유닛
            min_buy_unit = min_buy_cash/(self.price+self.slip) # 설정한 최소 매수 수량
            max_trade_unit = max(max_trade_cash/(self.price+self.slip),0) #최대 보유가능 수량

            if params.train_stock_or_future == 'coin':
                if unit2 >= limit_buy_unit:
                    unit2 = torch.Tensor([limit_buy_unit]).to(params.device)

                if torch.Tensor([unit2]).to(params.device) <= torch.Tensor([min_buy_unit]).to(params.device): # 매수 너무 적은경우 최소 매수량 만큼
                    if unit2==0:
                        unit2=torch.Tensor([0]).to(params.device)
                    else:
                        unit2 = torch.Tensor([min_buy_unit]).to(params.device)

                if torch.Tensor([self.stock]).to(params.device) + torch.Tensor([unit2]).to(params.device) >= torch.Tensor([max_trade_unit]).to(params.device):  # 앞으로 살 갯수 + 이미 가지고있는게 max이하여야함
                    unit2 = max_trade_unit - self.stock
                    unit2 = torch.max(torch.Tensor([unit2]),torch.Tensor([0]))
                unit = [0, 0, unit2.item()]

            else:
                unit2 = max(((policy1[2] * self.cash) / self.price), 1)
                unit2 = round(float(unit2))

                if self.stock + unit2 >= max_trade_unit:  # 앞으로 살 갯수 + 이미 가지고있는게 max이하여야함
                    unit2 = max_trade_unit - self.stock
                    unit2 = torch.Tensor([unit2])

                if unit2 >= limit_buy_unit:
                    unit2 = torch.Tensor([limit_buy_unit]).to(params.device)

                if unit2 < 1:  # 소수점단위의 주식수는 없으므로
                    unit2 = torch.Tensor([0])
                unit = [0, 0, unit2]


        else:  # 관망
            unit = [0, 0, 0]

        return action, unit







    def long_decide_action(self, policy, deposit):  # 롱포지션 ,관망(청산) : 롱온리 액션
        policy1 = torch.clamp(policy, 0, 1)
        action_s = Categorical(policy1)
        action = action_s.sample()

        PV_reward= False # 맨처음 스탭에서 0.99폴리시로 리워드 한번먹고 계속 같은 행동하면 계약수가 모자라서 잃어도 리워드 감소가 없음 . 이를 방지하기 위해 이런경우 PV로 리워드 계산

        ori_unit=[0,0,0] # 리워드 계산시 사용되는 unit
        policy1=policy1.to('cpu')

        # 학습일때
        limit_long_unit = 3  # 스탭당  최대 롱 계약
        limit_long_eq= 3
        min_sell_cash = self.init_cash*0.2 #최소 청산 캐시
        max_trade_unit = 3

        if self.back_testing == True:  # 백테스팅중인 경우
            action = torch.argmax(policy1)
            limit_long_unit = 1
            min_sell_cash = self.init_cash * 0.2  # 최소 청산 캐시
            limit_long_eq = 1
            max_trade_unit = 1


        if action == 0:  # 롱 청산
            if params.coin_or_stock=='coin':
                unit0 = policy1[0] * self.long_unit
                unit0 = float(unit0)

                ori_unit = [unit0, 0, 0]
                if unit0 >= limit_long_eq:
                    unit0 = limit_long_eq

                if unit0 * self.price <= min_sell_cash:  # 매도액 너무 적은경우 최소 매도액만큼
                    if unit0 <= 0:  # inf값 방지
                        unit0 = 0
                    else:
                        unit0 = min_sell_cash / (unit0 * (self.price - self.slip))

                unit = [unit0, 0, 0]



            if params.coin_or_stock=='stock' or params.coin_or_stock=='future':
                unit0 = max((policy1[0] * self.long_unit),1)
                unit0 = round(float(unit0))

                ori_unit = [unit0, 0, 0]

                if unit0>=limit_long_eq:
                    unit0=limit_long_eq

                elif unit0<1:
                    unit0=1

                unit = [unit0, 0, 0]


        elif action == 2:  # 롱 포지션

            if params.coin_or_stock == 'coin':
                unit2= float(policy1[2] * self.cash)/ float(deposit)
                unit2=float(unit2)


            if params.coin_or_stock == 'stock' or params.coin_or_stock == 'future':
                quotient, remainder = divmod(float(policy1[2]*self.cash), float(deposit))
                remainder = remainder/float(policy1[2] * self.cash)
                unit2 = round(float(quotient))


            if unit2 >= limit_long_unit: #유닛수가 리미트보다 크면 리미트로 제한
                unit2 = limit_long_unit

            if self.long_unit+unit2>=max_trade_unit: # 앞으로 살 갯수 + 이미 가지고있는게 max이하여야함
                unit2= max_trade_unit-self.long_unit

            unit = [0, 0, unit2]


        else:  # 관망
            unit = [0, 0, 0]


        return action, unit






    def short_decide_action(self, policy, deposit):  # 롱숏 포지션
        # PV 가 낮은데 리워드가 높은경우 : 매수시 리워드 받고나서 계속 가지고 있다가 잃는경우

        policy1 = torch.clamp(policy, 0, 1)
        action_s = Categorical(policy1)
        action = action_s.sample()

        ori_unit=[0,0,0] # 리워드 계산시 사용되는 unit
        policy1=policy1.to('cpu')

        # 학습일때
        limit_short_unit = 3# 스탭당  최대 숏 계약
        limit_short_eq=3
        min_sell_cash = self.init_cash*0.2 #최소 청산 캐시
        max_trade_unit=3

        PV_reward = False


        if self.back_testing==True: # 백테스팅일때
            action = torch.argmax(policy1)
            limit_short_unit= limit_short_unit  #스탭당 최대 거래 계약수 제한
            limit_short_eq = limit_short_eq
            min_sell_cash = self.init_cash * 0.2  # 최소 청산 캐시(룰베이스에선 항상 전량청산)
            max_trade_unit = 1  #최대 보유 유닛 제한


        if action == 0:  # 숏 청산

            if params.coin_or_stock == 'coin':
                unit0 = policy1[0] * self.short_unit
                unit0 = float(unit0)



            if params.coin_or_stock == 'stock' or params.coin_or_stock == 'future':
                unit0 = max((policy1[0]*self.short_unit),1)
                unit0 = round(float(unit0))


            if unit0 >= limit_short_eq:
                unit0 = limit_short_eq

            if unit0 * self.price <= min_sell_cash:  # 매도액 너무 적은경우 최소 매도액만큼
                if unit0 <= 0:  # inf값 방지
                    unit0 = 0
                else:
                    unit0 = min_sell_cash / (unit0 * (self.price - self.slip))

            unit=[unit0,0,0]




        elif action == 2:  # 숏 매수
            if params.coin_or_stock == 'coin':
                #quotient, remainder = divmod(float(policy1[2] * self.cash), float(deposit))
                unit2= float(policy1[2] * self.cash)/float(deposit)
                #remainder=remainder/float(policy1[2] * self.cash)
                #unit2 = quotient+remainder

                unit2 = float(unit2)


            if params.coin_or_stock == 'stock' or params.coin_or_stock == 'future':
                quotient, remainder = divmod(float(policy1[2]*self.cash), float(deposit))
                remainder=remainder/float(policy1[2] * self.cash)
                unit2 = quotient
                unit2 = round(float(unit2))


            if unit2 >= limit_short_unit:
                unit2 = limit_short_unit

            if self.short_unit + unit2 >= max_trade_unit:
                unit2 = max_trade_unit - self.short_unit

            unit = [0, 0, unit2]


        else:  #관망
            unit =[0,0,0]
        return action, unit




