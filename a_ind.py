
import math
import scipy.signal
import torch

#정규화 및 전처리 계산
from sklearn.preprocessing import MinMaxScaler

#시각화및 저장,계산
import numpy as np
import pandas as pd
# 시각화및 저장,계산
import numpy as np
import pandas as pd
import scipy.signal
import torch
import matplotlib.pyplot as plt

# 정규화 및 전처리 계산
from sklearn.preprocessing import MinMaxScaler
import e_train as params
import psycopg2



#API및 데이터 불러오기
#기타
#크롤링



class ind_env:
    def __init__(self):
        self.MD=0
        pass


    def AccumN(self,data,period): # 시리즈 데이터의 기간만큼 누적합 한다
        data= pd.Series(data).rolling(period).sum()
        data= data.dropna().reset_index()[0]
        return data


    def period_equal(self,data): #데이터의 기간을 일치 시켜준다
        len_data=[] #길이저장
        res=[] #결과저장

        for data_dim in range(len(data)): # start_period 찾는다
            len_data.append(len(data[data_dim])) # 데이터 길이 저장
        start_period=min(len_data)

        for data_dim in range(len(data)): #기간 일치
            res_data=data[data_dim][-start_period:]
            res.append(res_data)

        return res #들어온 순서대로 출력



    def ind_data_create_second(self,  # 추가 지표들 불러올때 실행 (기준지표와 동일한 date를 맞춰서 불러온다) 전진분석때는?
                    minute,
                    data_count,
                    coin_or_stock,
                    point_value,
                    name, #기준 종목 ex NQ
                    second_name,# 날짜 맞출종목
                    third_name, # 기준종목의 price_ai2이름
                    fourth_name): # 맞출 종목의 price_ai2이름


        self.name = name
        self.name2= second_name

        connection =psycopg2.connect(dbname='postgres', user='postgres', password='snowai**', host='172.30.1.96',
                                      port='5432', sslmode='require')


        db = connection.cursor()
        total_data=0
        data_set=0
        # DB 데이터 호출
        if params.part_time==False: #부분 시간 학습 (ex 정규장이면 정규장만 불러오기)
            if params.backtest_or_forward=='back':
                db.execute(
                    f"SELECT NQ.datetime, {self.name2}.open AS open_, {self.name2}.close AS close_, {self.name2}.high AS high_, {self.name2}.low AS low_, {self.name2}.volume AS volume_ \
                        FROM (SELECT * FROM snowball.price_ai WHERE symbol='{self.name}' ORDER BY datetime DESC LIMIT {data_count}) AS NQ \
                        LEFT JOIN (SELECT * FROM snowball.price_ai WHERE symbol='{self.name2}') AS {self.name2} \
                        ON NQ.datetime = {self.name2}.datetime \
                        ORDER BY NQ.datetime ASC;")
                #db.execute(f"SELECT {self.name}.datetime, {self.name2}.open AS open_, {self.name2}.close AS close_, {self.name2}.high AS high_, {self.name2}.low AS low_, {self.name2}.volume AS volume_ FROM (SELECT * FROM snowball.price_ai WHERE symbol={self.name} ORDER BY datetime DESC LIMIT {data_count}) AS {self.name} LEFT JOIN (SELECT * FROM snowball.price_ai2 WHERE symbol={self.name2}) AS {self.name2} ON {self.name}.datetime = {self.name2}.datetime ORDER BY {self.name}.datetime ASC;")
                #db.execute(
                #    f"SELECT open,close,high,low,volume,datetime FROM (SELECT open,close,high,low,volume,datetime FROM snowball.price_ai WHERE symbol={self.name} ORDER BY datetime DESC limit {data_count}) as foo order by datetime asc;")
                total_data = db.fetchall()

            elif params.backtest_or_forward=='forward':

                if params.real_forward==False:
                    self.name= third_name
                    self.name2= fourth_name
                    #전진분석에서 자동으로data_count= params.test_data_count[0]으로 설정됨
                    db.execute(
                        f"SELECT NQ.datetime, GC.open AS open_, GC.close AS close_, GC.high AS high_, GC.low AS low_, GC.volume AS volume_ \
                            FROM (SELECT * FROM snowball_ai.price_ai2 WHERE symbol='{self.name}' ORDER BY datetime DESC LIMIT {data_count}) AS NQ \
                            LEFT JOIN (SELECT * FROM snowball_ai.price_ai2 WHERE symbol='{self.name2}') AS GC \
                            ON NQ.datetime = GC.datetime \
                            ORDER BY NQ.datetime ASC;")
                    total_data=db.fetchall()

                if params.real_forward==True: # 데이터 부족해서 학습구간도 뽑아와야할때
                    self.name=name
                    self.name2=second_name
                    data_count = params.test_data_count[1]  # 학습구간 데이터
                    # 학습구간 데이터 호출
                    db.execute(
                        f"SELECT NQ.datetime, {self.name2}.open AS open_, {self.name2}.close AS close_, {self.name2}.high AS high_, {self.name2}.low AS low_, {self.name2}.volume AS volume_ \
                                            FROM (SELECT * FROM snowball.price_ai WHERE symbol='{self.name}' ORDER BY datetime DESC LIMIT {data_count}) AS NQ \
                                            LEFT JOIN (SELECT * FROM snowball.price_ai WHERE symbol='{self.name2}') AS {self.name2} \
                                            ON NQ.datetime = {self.name2}.datetime \
                                            ORDER BY NQ.datetime ASC;")
                    train_data = db.fetchall()


                    # 실시간DB에서 데이터 호출
                    self.name= third_name
                    self.name2= fourth_name
                    data_count = params.test_data_count[0]  # 포함될 실시간 데이터 갯수
                    # order by asC로 앞에서부터 저장됐던 실시간데이터를 불러온다(여기서는 전체를 불러와야함)
                    '''''
                    db.execute(f"SELECT NQ.datetime, GC.open AS open_, GC.close AS close_, GC.high AS high_, GC.low AS low_, GC.volume AS volume_ \
                                                FROM (SELECT * FROM snowball_ai.price_ai2 WHERE symbol='{self.name}' ORDER BY datetime DESC LIMIT {data_count}) AS NQ \
                                                LEFT JOIN (SELECT * FROM snowball_ai.price_ai2 WHERE symbol='{self.name2}') AS GC \
                                                ON NQ.datetime = GC.datetime \
                                                ORDER BY NQ.datetime ASC;")
                    '''''
                    name2_alias = self.name2.replace("=", "_").replace("^",'_')

                    db.execute(
                        f"SELECT NQ.datetime, GC.open AS open_, GC.close AS close_, GC.high AS high_, GC.low AS low_, GC.volume AS volume_ \
                                                                    FROM (SELECT * FROM snowball_ai.price_ai2 WHERE symbol='{self.name}' ORDER BY datetime DESC LIMIT {data_count}) AS NQ \
                                                                    LEFT JOIN (SELECT * FROM snowball_ai.price_ai2 WHERE symbol='{self.name2}') AS GC \
                                                                    ON NQ.datetime = GC.datetime \
                                                                    ORDER BY NQ.datetime ASC;")

                    forward_data = db.fetchall()

                    total_data = train_data + forward_data # 최종 데이터

            else:
                print('backtest_or_forward 재설정 필요',params.backtest_or_forward,'현재 값')

        data_set =self.total_time_Frame(total_data,minute)

        #[0] 은 NQ 의 datetime이다 따라서 1 부터 호출
        open= pd.Series([float(t[1]) if t[1] is not None else np.nan for t in data_set]).fillna(method='ffill') #nan인 데이터는 이전 값으로 붙임
        close=pd.Series([float(t[2]) if t[2] is not None else np.nan for t in data_set]).fillna(method='ffill')
        high=pd.Series([float(t[3]) if t[3] is not None else np.nan for t in data_set]).fillna(method='ffill')
        low=pd.Series([float(t[4]) if t[4] is not None else np.nan for t in data_set]).fillna(method='ffill')
        vol=pd.Series([float(t[5]) if t[5] is not None else np.nan for t in data_set]).fillna(method='ffill')
        date=pd.Series([t[0] for t in data_set])


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

        data_=[close_, open_, high_, low_, vol_, close_s, open_s, high_s, low_s, vol_s, date]

        print('DB 데이터 호출 완료')

        return data_




    def ind_data_create(self,  # ind 에서 자체적으로 추가되는 종목 불러오는경우, Env 에서 인풋크레이트와 동일(여러종목 불러올때 시간일치 x, 한종목 불러올때 사용)
                    minute,
                    data_count,
                    coin_or_stock,
                    point_value,
                    name):

        self.name = name
        connection =psycopg2.connect(dbname='postgres', user='postgres', password='snowai**', host='172.30.1.96',
                                      port='5432', sslmode='require')


        db = connection.cursor()

        # DB 데이터 호출
        if params.part_time==False: #부분 시간 학습 (ex 정규장이면 정규장만 불러오기)
            db.execute(
                f"SELECT open,close,high,low,volume,datetime FROM (SELECT open,close,high,low,volume,datetime FROM snowball.price_ai WHERE symbol={self.name} ORDER BY datetime DESC limit {data_count}) as foo order by datetime asc;")

        if params.part_time==True:
            start_time=params.part_time_data[params.part_time_name][0]
            end_time=params.part_time_data[params.part_time_name][1]

            if start_time > end_time:  # ex 20시 이상 13시 이하 인경우
                db.execute(
                    f"SELECT open,close,high,low,volume,datetime FROM (SELECT open,close,high,low,volume,datetime FROM snowball.price_ai WHERE symbol= {self.name} ORDER BY datetime DESC limit {data_count} ) as foo WHERE extract(hour from datetime) <= {end_time} or extract(hour from datetime) >= {start_time}  order by datetime asc ;")
            else:  # ex(13시 이상 20시 이하인경우)
                db.execute(
                    f"SELECT open,close,high,low,volume,datetime FROM (SELECT open,close,high,low,volume,datetime FROM snowball.price_ai WHERE symbol= {self.name} ORDER BY datetime DESC limit {data_count} ) as foo WHERE extract(hour from datetime) <= {end_time} and extract(hour from datetime) >= {start_time}  order by datetime asc ;")

        data_set = db.fetchall()
        data_set = self.total_time_Frame(data_set, minute)

        open= pd.Series([float(t[0]) for t in data_set])
        close=pd.Series([float(t[1]) for t in data_set])
        high=pd.Series([float(t[2]) for t in data_set])
        low=pd.Series([float(t[3]) for t in data_set])
        vol=pd.Series([float(t[4]) for t in data_set])
        date=pd.Series([t[5] for t in data_set])

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

        data_=[close_, open_, high_, low_, vol_, close_s, open_s, high_s, low_s, vol_s, date]

        if params.real_train == True:  # 실시간DB데이터들과 과거 데이터DB 합치는경우(실시간 학습할때 사용)
            db_connect = psycopg2.connect(dbname='postgres', user='postgres', password='snowai**', host='172.30.1.96',
                                          port='5432', sslmode='require')

            db = db_connect.cursor()
            data_count2 = params.test_data_count[0]  # 포함될 실시간 데이터 갯수
            # order by asC로 앞에서부터 저장됐던 실시간데이터를 불러온다
            db.execute(f"SELECT * FROM (SELECT * FROM snowball_ai.price_ai2 WHERE symbol={params.stock_name_} ORDER BY datetime asc limit {data_count2}) as foo order by datetime asc;")
            data = db.fetchall()

            # 불러온 실시간 데이터 전처리
            res = [data[step * minute] for step in range(round(len(data) / minute))]  # 인터벌
            res = pd.DataFrame(res).reset_index()
            res.columns = ['index', 'symbol', 'Datetime', 'open', 'high', 'low', 'close', 'volume']
            for name in res.columns:
                if name == 'index' or name == 'symbol' or name == 'Datetime':
                    pass
                else:
                    data_1 = [float(value) for value in res[name]]  # Decimal 형태를 float으로 변경
                    res[name] = data_1

            total_close = pd.concat([data_[0], res['close'] ],ignore_index=True).reset_index()[0]
            total_open = pd.concat([data_[1], res['open'] ],ignore_index=True).reset_index()[0]
            total_high = pd.concat([data_[2], res['high'] ],ignore_index=True).reset_index()[0]
            total_low = pd.concat([data_[3], res['low'] ],ignore_index=True).reset_index()[0]
            total_vol =pd.concat([data_[4], res['volume'] ],ignore_index=True).reset_index()[0]
            total_date= pd.concat([data_[10], res['Datetime']],ignore_index=True).reset_index()[0]

            scaler = MinMaxScaler()
            close_s = scaler.fit_transform(total_close.values.reshape(-1, 1))
            open_s = scaler.fit_transform(total_open.values.reshape(-1, 1))
            high_s = scaler.fit_transform(total_high.values.reshape(-1, 1))
            low_s = scaler.fit_transform(total_low.values.reshape(-1, 1))
            vol_s = scaler.fit_transform(total_vol.values.reshape(-1, 1))

            total_data = [total_close, total_open, total_high, total_low, total_vol, close_s, open_s, high_s, low_s,
                          vol_s, total_date]
            data_ = total_data


        print('DB 데이터 호출 완료')

        return data_

    # 타임프레임  (토탈에서)

    def total_time_Frame(self, data, minute):  # Env에서의 타임프레임 함수와 동일

        price_data = pd.Series(data)
        price_data.dropna(inplace=True)

        if len(data) % minute == 0:
            index_data = [step * minute for step in range(int(np.trunc(len(data) / minute)))]
        else:
            index_data = [step * minute for step in range(int(np.trunc(len(data) / minute)) + 1)]  # 인터벌

        res = price_data[index_data].reset_index()[0]

        return res

    def STD(self, price,period):
        _,Avg= self.MA(price,period)
        SumSqrt_data=[]

        SumSqrt = 0
        price_=price[-len(Avg):]

        for counter in range(len(Avg)):
            SumSqrt += (price_[counter]-Avg[counter])*price_[counter]-Avg[counter]
            SumSqrt_data.append(SumSqrt)

        ori_std=np.sqrt(SumSqrt_data/len(price)).reshape(-1)
        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_std = scaler.fit_transform(pd.Series(ori_std).values.reshape(-1, 1))

        return scale_std, ori_std

    def STD(self, price, period):
        std_list = np.zeros(len(price))
        for i in range(period - 1, len(price)):
            avg = np.mean(price[i - period + 1:i + 1])
            std = np.std(price[i - period + 1:i + 1])
            std_list[i] = std

    def STD(self, price, period):
        std_list = [0] * len(price)
        for i in range(period - 1, len(price)):
            avg = sum(price[i - period + 1:i + 1]) / period
            sumSqrt = sum((price[i - period + j + 1] - avg) ** 2 for j in range(period))
            std = math.sqrt(sumSqrt / period)
            std_list[i] = std



        std_list = np.array(std_list).reshape(-1)
        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_std = scaler.fit_transform(pd.Series(std_list).values.reshape(-1, 1))

        return scale_std , std_list

    def STD(self,price, period):
        std_list = np.zeros(len(price))
        for i in range(period - 1, len(price)):
            std = np.std(price[i - period + 1:i + 1])
            std_list[i] = std


        std_list = np.array(std_list).reshape(-1)
        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_std = scaler.fit_transform(pd.Series(std_list).values.reshape(-1, 1))


        return scale_std , std_list


    def log_return(self, close, period):  # 종가의 로그리턴을 구하는 함수
        # log_return= log(Pt+1)-log(Pt)

        series = pd.DataFrame(close)
        log_re = [0]  # 다른지표와 인덱스를 맞추기위해 초기에 0을 넣는다

        for t in range(len(close) - period):

            return_ = np.log(series.iloc[t + period]) - np.log(series.iloc[t])
            log_re.append(round(float(return_), 5))
        log_re = pd.Series(log_re)
        ori_log_re = log_re.rolling(1).mean().dropna().reset_index()[0].values

        scaler= MinMaxScaler(feature_range=(0.1,1))
        log_re = scaler.fit_transform(pd.Series(ori_log_re).values.reshape(-1, 1))
        log_re = np.array(log_re).reshape(-1)

        return log_re,ori_log_re




    def denoise1(self, close, period):
        res = scipy.signal.savgol_filter(close, period)
        res = torch.Tensor(res)
        ori_res=pd.Series(res).values
        return res,ori_res



    def NCO_up(self, close, period, period2):
        NCO_up = close.iloc[period - 1:] - close.rolling(period).min()
        NCO_up = NCO_up.rolling(period2).mean()
        NCO_up = NCO_up.dropna().reset_index()[0]

        ori_res=NCO_up.values
        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(NCO_up.values.reshape(-1, 1))
        res = res.reshape(-1)
        res = pd.Series(res).values.reshape(-1)

        return res, ori_res


    def NCO_down(self, close, period, period2):

        NCO_down = close.rolling(period).max() - close.iloc[period - 1:]
        NCO_down = NCO_down.rolling(period2).mean()
        NCO_down = NCO_down.dropna().reset_index()[0]

        ori_res= NCO_down.values
        scaler = MinMaxScaler(feature_range=(0.1,1))

        res = scaler.fit_transform(NCO_down.values.reshape(-1, 1))
        res = res.reshape(-1)
        res = pd.Series(res).values.reshape(-1)

        return res, ori_res

    def CCI_origin(self, data, period):  # CCI 함수
        _,ma=self.MA(data, period)
        Aver = pd.Series(ma).dropna().reset_index()[0]
        nan_idx = -min(len(data), len(Aver))

        # nan 제거 (인덱스 일치시킴)

        Aver = Aver.iloc[nan_idx:].dropna().reset_index()[0]
        data = data.iloc[nan_idx:].dropna().reset_index()[0]

        # CCI 계산
        p = np.abs(data - Aver)

        MD = p.rolling(period).mean().dropna().reset_index()[0]
        nan_idx2 = -min(len(MD), len(data), len(Aver))
        MD = torch.Tensor(MD[nan_idx2:].tolist())
        data = torch.Tensor(data[nan_idx2:].values.tolist())
        Aver = torch.Tensor(Aver[nan_idx2:].values.tolist())

        CCI_data = ((data - Aver) / (0.015 * (MD + (1e-1))))
        CCI_data = pd.Series(CCI_data)

        scaler=MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(CCI_data.values.reshape(-1, 1))
        res = res.reshape(-1)
        res = pd.Series(res).values.reshape(-1)

        ori_res= CCI_data.values

        return res, ori_res


    def CCI(self, data, period):  # CCI 함수

        if len(data)==3:#high low close들어오는경우
            data= data[0]+data[1]+data[2]
        else:
            data=data*3 #open등 하나만 들어온경우그냥 3곱함

        _,ma=self.MA(data, period)
        Aver = pd.Series(ma).dropna().reset_index()[0]
        nan_idx = -min(len(data), len(Aver))

        # nan 제거 (인덱스 일치시킴)

        Aver = Aver.iloc[nan_idx:].dropna().reset_index()[0]
        data = data.iloc[nan_idx:].dropna().reset_index()[0]

        # CCI 계산 (빠른버전)
        CCI=[]
        self.MD=data.iloc[0]
        real_MD=[]
        AV2=[]

        MD = np.zeros(len(data) - period)
        for step in range(len(data) - period):
            MD[step] = np.abs(data.values[step:step + period] - Aver.values[step + period]).sum()
        real_MD = MD.copy()
        MD= np.array(real_MD)/period

        # MD와 data의 길이는 period만큼 data가 더길다. (MD는 period만큼 rolling된 효과)
        for step in range(len(data)-period):
            if MD[step]==0:
                CCI.append(0)
            else:
                CCI.append((data.iloc[step+period] - Aver.iloc[step+period])/(0.015*MD[step]))

        CCI_data=pd.Series(CCI).dropna().reset_index()[0]

        scaler=MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(CCI_data.values.reshape(-1, 1))
        res = res.reshape(-1)
        res = pd.Series(res).values.reshape(-1)

        ori_res= CCI_data.values
        '''''
        # Yes CCI 수식 풀어쓰면
        CCI = []
        self.MD = data.iloc[0]
        real_MD = []
        AV2 = []

        for step in range(len(data)-period):
            MD=[]
            self.MD=0
            for step_ in range(step,step+period):
                self.MD=self.MD + np.abs(data.iloc[step_] - Aver.iloc[step+period])
                MD.append(self.MD)
            real_MD.append(MD[-1])
            if step%1000==0:
                print(step,'/',len(data)-period,'CCI MD 계산 진행도')

        '''''


        return res, ori_res



    def MA(self, data, period):  # 이동 평균선 함수
        ma = pd.Series(data).rolling(period).sum().dropna().reset_index()[0]/period
        ori_ma=ma.values
        scaler=MinMaxScaler(feature_range=(0.1,1))
        ma=scaler.fit_transform(ma.values.reshape(-1,1))
        return ma,ori_ma

    def EMA(self, data, period): # 예스에서 사용하는 EMA
        k = 2 / (period + 1)
        ema = [data[0]]
        for i in range(len(data)):
            if i>0:
                ema.append(data[i] * k + ema[i - 1] * (1 - k))
            # EMA = 종가 x 배율 + EMA (전날) x (1 배율)
        ema=pd.Series(ema)

        scaler = MinMaxScaler(feature_range=(0.1,1))
        s_ema = scaler.fit_transform(ema.values.reshape(-1, 1))

        return s_ema,ema


    def EMA_noise(self,data,period): # 노이즈 심한 EMA
        a= 2/(period+1) #평활상수
        b= (a*data) + (1-a)*(period+1)
        ema= a*data +(1-a)*(b+1)

        scaler=MinMaxScaler(feature_range=(0.1,1))
        s_ema = scaler.fit_transform(ema.values.reshape(-1,1))

        return s_ema,ema



    def WMA(self,data,period):
        sum_data = []
        csum_data = []
        wma_data=[]

        price_data=[data[step:step+period] for step in range(len(data)-period)]
        for step in range(len(price_data)):
            #초기화
            sum_ = 0
            csum = 0
            batch_sum=[]
            batch_csum=[]

            for t in range(len(price_data[step])):
                sum_ += price_data[step].iloc[t]* (period-t)
                csum += period-t
                batch_sum.append(sum_)
                batch_csum.append(csum)

            sum_data.append(np.sum(batch_sum))
            csum_data.append(np.sum(batch_csum))

        for step in range(len(sum_data)):
            if csum_data[step]>0:
                wma = sum_data[step]/csum_data[step]
            else:
                wma=0

            wma_data.append(wma)
        ori_wma_data= np.array(wma_data).reshape(-1)
        wma=np.array(wma_data)

        scaler = MinMaxScaler(feature_range=(0.1,1))
        wma = scaler.fit_transform(wma.reshape(-1, 1))
        wma = np.array(wma).reshape(-1)

        return wma,ori_wma_data




    def slope_line(self, close, period):  # LRL
        #
        AccumValue = pd.Series(range(len(close)))  # 1부터 ~ 끝
        AccumValue = AccumValue.dropna().reset_index()[0]

        _,a=self.MA(AccumValue, period)
        _,b=self.MA(close, period)
        value1 = np.array(pd.Series(a).dropna().reset_index()[0])
        value2 = np.array(pd.Series(b).dropna().reset_index()[0])


        close = np.array(close.tolist())
        AccumValue = np.array(AccumValue)

        _,ori_1= self.MA(pd.Series((close * AccumValue).reshape(-1)), period)
        A = np.array(pd.Series(ori_1).dropna().reset_index()[0])
        B = value1 * value2

        _,ori_2= self.MA(pd.Series(AccumValue ** 2), period)
        C = np.array(pd.Series(ori_2).dropna().reset_index()[0])
        _,ori_3= self.MA(pd.Series(AccumValue), period)
        D = np.array((pd.Series(ori_3) ** 2).dropna().reset_index()[0])
        E = AccumValue[-len(value1):] - value1
        F = C-D

        LRL = (A - B) / F * E + value2 #inf값 발생

        ori_LRL = pd.Series(LRL.reshape(-1)).values


        scaler = MinMaxScaler(feature_range=(0.1,1))
        LRL = scaler.fit_transform(pd.Series(ori_LRL).values.reshape(-1, 1))
        LRL = np.array(LRL).reshape(-1)

        return LRL, ori_LRL


    def one_side(self, close, high_period, low_period, res2_period):  # 일목균형 지표

        value1 = ((close.rolling(high_period).max().dropna().reset_index()[0] +close.rolling(high_period).min().dropna().reset_index()[0]) / 2).dropna().reset_index()[0]
        value2 = ((close.rolling(low_period).max().dropna().reset_index()[0] +close.rolling(low_period).min().dropna().reset_index()[0]) / 2).dropna().reset_index()[0]
        res2_value = ((close.rolling(res2_period).max().dropna().reset_index()[0] +
                       close.rolling(res2_period).min().dropna().reset_index()[0]) / 2).dropna().reset_index()[0]
        start_index = -min(len(value1), len(value2))
        value1 = value1[start_index:].dropna().reset_index()[0]
        value2 = value2[start_index:].dropna().reset_index()[0]

        res1 = (value1 + value2) / 2
        res2 = res2_value

        ori_res1= np.array(res1)
        ori_res2= np.array(res2)

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res1 = scaler.fit_transform(res1.values.reshape(-1, 1))
        res2 = scaler.fit_transform(res2.values.reshape(-1, 1))
        res1 = np.array(res1).reshape(-1)
        res2= np.array(res2).reshape(-1)

        return res1, res2, ori_res1, ori_res2


    def LR_Long(self, close,vM1,vM2,vM3,vM4,vM5,vM6,vM7,vM8,acc_period):
        L1,ori_L1= self.slope_line(close,vM1)
        L2,ori_L2= self.slope_line(close,vM2)
        L3,ori_L3= self.slope_line(close,vM3)
        L4,ori_L4= self.slope_line(close,vM4)
        L5,ori_L5= self.slope_line(close,vM5)
        L6,ori_L6= self.slope_line(close,vM6)
        L7,ori_L7= self.slope_line(close,vM7)
        L8,ori_L8= self.slope_line(close,vM8)

        start_period = -min(len(ori_L1), len(ori_L2), len(ori_L3), len(ori_L4), len(ori_L5), len(ori_L6), len(ori_L7), len(ori_L8))
        L1 = pd.Series(ori_L1[start_period:])
        L2 = pd.Series(ori_L2[start_period:])
        L3 = pd.Series(ori_L3[start_period:])
        L4 = pd.Series(ori_L4[start_period:])
        L5 = pd.Series(ori_L5[start_period:])
        L6 = pd.Series(ori_L6[start_period:])
        L7 = pd.Series(ori_L7[start_period:])
        L8 = pd.Series(ori_L8[start_period:])

        MultipleL1= (L1+L2+L3+L4+L5+L6+L7+L8)/8
        LRLA= MultipleL1.rolling(acc_period).sum().values

        ori_res=LRLA
        scaler = MinMaxScaler(feature_range=(0.1,1))
        res= scaler.fit_transform(LRLA.reshape(-1,1))

        return res, ori_res


    def LR_short(self, close,vM1,vM2,vM3,vM4,vM5,vM6,vM7,vM8,acc_period):
        L1, ori_L1 = self.slope_line(close, vM1)
        L2, ori_L2 = self.slope_line(close, vM2)
        L3, ori_L3 = self.slope_line(close, vM3)
        L4, ori_L4 = self.slope_line(close, vM4)
        L5, ori_L5 = self.slope_line(close, vM5)
        L6, ori_L6 = self.slope_line(close, vM6)
        L7, ori_L7 = self.slope_line(close, vM7)
        L8, ori_L8 = self.slope_line(close, vM8)

        start_period = -min(len(ori_L1), len(ori_L2), len(ori_L3), len(ori_L4), len(ori_L5), len(ori_L6), len(ori_L7), len(ori_L8))
        L1 = pd.Series(ori_L1[start_period:])
        L2 = pd.Series(ori_L2[start_period:])
        L3 = pd.Series(ori_L3[start_period:])
        L4 = pd.Series(ori_L4[start_period:])
        L5 = pd.Series(ori_L5[start_period:])
        L6 = pd.Series(ori_L6[start_period:])
        L7 = pd.Series(ori_L7[start_period:])
        L8 = pd.Series(ori_L8[start_period:])

        MultipleL1= (L1+L2+L3+L4+L5+L6+L7+L8)/8
        LR8_short= MultipleL1.rolling(acc_period).sum().values

        ori_res=LR8_short
        scaler = MinMaxScaler(feature_range=(0.1,1))
        res= scaler.fit_transform(LR8_short.reshape(-1,1))

        return res,ori_res


    def LRLRA(self, close,vM1,vM2,vM3,vM4,vM5,vM6,vM7,vM8,period1,period2,acc_period):
        L1, ori_L1 = self.slope_line(close, vM1)
        L2, ori_L2 = self.slope_line(close, vM2)
        L3, ori_L3 = self.slope_line(close, vM3)
        L4, ori_L4 = self.slope_line(close, vM4)
        L5, ori_L5 = self.slope_line(close, vM5)
        L6, ori_L6 = self.slope_line(close, vM6)
        L7, ori_L7 = self.slope_line(close, vM7)
        L8, ori_L8 = self.slope_line(close, vM8)

        start_period = -min(len(ori_L1), len(ori_L2), len(ori_L3), len(ori_L4), len(ori_L5), len(ori_L6), len(ori_L7),
                            len(ori_L8))
        L1 = pd.Series(ori_L1[start_period:])
        L2 = pd.Series(ori_L2[start_period:])
        L3 = pd.Series(ori_L3[start_period:])
        L4 = pd.Series(ori_L4[start_period:])
        L5 = pd.Series(ori_L5[start_period:])
        L6 = pd.Series(ori_L6[start_period:])
        L7 = pd.Series(ori_L7[start_period:])
        L8 = pd.Series(ori_L8[start_period:])

        MultipleL1= (L1+L2+L3+L4+L5+L6+L7+L8)/8
        LR8_short= pd.Series(MultipleL1.rolling(acc_period).sum().dropna().reset_index()[0])

        _,AvgW1=self.WMA(LR8_short,period1)   # slope의 가중이평
        AvgW1=pd.Series(pd.Series(AvgW1).dropna().reset_index()[0])
        _,WLR= self.slope_line(AvgW1,period2)  #slope의 가중이평의 slope
        _,LRLR= self.slope_line(LR8_short,period2) # slope의 slope

        ori_LR8_short= LR8_short.values
        ori_WLR= WLR
        ori_LRLR= LRLR

        scaler = MinMaxScaler(feature_range=(0.1,1))
        LR8_short= scaler.fit_transform(LR8_short.values.reshape(-1,1))
        WLR= scaler.fit_transform(pd.Series(WLR).values.reshape(-1,1))
        LRLR= scaler.fit_transform(pd.Series(LRLR).values.reshape(-1,1))

        return LR8_short,WLR,LRLR, ori_LR8_short,ori_WLR,ori_LRLR



    def s_LRLRA(self,close,vM1,period1): # LRLRA 경량화 버전
        L1, ori_L1 = self.slope_line(close, vM1)
        start_period = -len(ori_L1)
        L1 = pd.Series(ori_L1[start_period:])

        _, AvgW1 = self.WMA(ori_L1, period1)    # slope의 가중이평
        LRLR = pd.Series(pd.Series(AvgW1).dropna().reset_index()[0])

        ori_LRLR=LRLR.values

        scaler = MinMaxScaler(feature_range=(0.1,1))
        LRLR = scaler.fit_transform(LRLR.values.reshape(-1, 1))

        return LRLR,ori_LRLR


    def LRS_CCI(self,close,vM1,vM2,vM3,vM4,vM5,vM6,vM7,vM8,acc_period):
        CCI1, ori_CCI1 = self.CCI(close, vM1)
        CCI2, ori_CCI2 = self.CCI(close, vM2)
        CCI3, ori_CCI3 = self.CCI(close, vM3)
        CCI4, ori_CCI4 = self.CCI(close, vM4)
        CCI5, ori_CCI5 = self.CCI(close, vM5)
        CCI6, ori_CCI6 = self.CCI(close, vM6)
        CCI7, ori_CCI7 = self.CCI(close, vM7)
        CCI8, ori_CCI8 = self.CCI(close, vM8)

        LR_C1, ori_LR_C1 = self.slope(ori_CCI1, vM1)  #LRS
        LR_C2, ori_LR_C2 = self.slope(ori_CCI2, vM2)
        LR_C3, ori_LR_C3 = self.slope(ori_CCI3, vM3)
        LR_C4, ori_LR_C4 = self.slope(ori_CCI4, vM4)
        LR_C5, ori_LR_C5 = self.slope(ori_CCI5, vM5)
        LR_C6, ori_LR_C6 = self.slope(ori_CCI6, vM6)
        LR_C7, ori_LR_C7 = self.slope(ori_CCI7, vM7)
        LR_C8, ori_LR_C8 = self.slope(ori_CCI8, vM8)

        start_period = -min(len(ori_LR_C1), len(ori_LR_C2), len(ori_LR_C3), len(ori_LR_C4), len(ori_LR_C5), len(ori_LR_C6), len(ori_LR_C7),
                            len(ori_LR_C8))

        LR_C1 = pd.Series(ori_LR_C1[start_period:])
        LR_C2 = pd.Series(ori_LR_C2[start_period:])
        LR_C3 = pd.Series(ori_LR_C3[start_period:])
        LR_C4 = pd.Series(ori_LR_C4[start_period:])
        LR_C5 = pd.Series(ori_LR_C5[start_period:])
        LR_C6 = pd.Series(ori_LR_C6[start_period:])
        LR_C7 = pd.Series(ori_LR_C7[start_period:])
        LR_C8 = pd.Series(ori_LR_C8[start_period:])

        signal_short = (LR_C1 + LR_C2 + LR_C3 + LR_C4 + LR_C5 + LR_C6 + LR_C7 + LR_C8) / 8
        sig_res = signal_short.rolling(acc_period).sum().values
        ori_res = sig_res

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(sig_res.reshape(-1, 1))

        return res,ori_res



    def LR_CCI(self,close,vM1,vM2,vM3,vM4,vM5,vM6,vM7,vM8,acc_period):  #slope의 CCI
        CCI1, ori_CCI1 = self.CCI(close, vM1)
        CCI2, ori_CCI2 = self.CCI(close, vM2)
        CCI3, ori_CCI3 = self.CCI(close, vM3)
        CCI4, ori_CCI4 = self.CCI(close, vM4)
        CCI5, ori_CCI5 = self.CCI(close, vM5)
        CCI6, ori_CCI6 = self.CCI(close, vM6)
        CCI7, ori_CCI7 = self.CCI(close, vM7)
        CCI8, ori_CCI8 = self.CCI(close, vM8)


        LR_C1, ori_LR_C1 = self.slope(ori_CCI1, vM1)  # LRS
        LR_C2, ori_LR_C2 = self.slope(ori_CCI2, vM2)
        LR_C3, ori_LR_C3 = self.slope(ori_CCI3, vM3)
        LR_C4, ori_LR_C4 = self.slope(ori_CCI4, vM4)
        LR_C5, ori_LR_C5 = self.slope(ori_CCI5, vM5)
        LR_C6, ori_LR_C6 = self.slope(ori_CCI6, vM6)
        LR_C7, ori_LR_C7 = self.slope(ori_CCI7, vM7)
        LR_C8, ori_LR_C8 = self.slope(ori_CCI8, vM8)

        start_period = -min(len(ori_LR_C1), len(ori_LR_C2), len(ori_LR_C3), len(ori_LR_C4), len(ori_LR_C5),
                            len(ori_LR_C6), len(ori_LR_C7),
                            len(ori_LR_C8))


        LR_C1 = pd.Series(ori_LR_C1[start_period:])
        LR_C2 = pd.Series(ori_LR_C2[start_period:])
        LR_C3 = pd.Series(ori_LR_C3[start_period:])
        LR_C4 = pd.Series(ori_LR_C4[start_period:])
        LR_C5 = pd.Series(ori_LR_C5[start_period:])
        LR_C6 = pd.Series(ori_LR_C6[start_period:])
        LR_C7 = pd.Series(ori_LR_C7[start_period:])
        LR_C8 = pd.Series(ori_LR_C8[start_period:])

        signal_short = (LR_C1 + LR_C2 + LR_C3 + LR_C4 + LR_C5 + LR_C6 + LR_C7 + LR_C8)/8
        sig_res = signal_short.rolling(acc_period).sum().values
        ori_res=sig_res

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(sig_res.reshape(-1, 1))

        return res,ori_res


    def s_LR_CCI(self,close,vM1,acc_period):

        CCI1, ori_CCI1 = self.CCI(close, vM1)
        LR_C1, ori_LR_C1 = self.slope_line(ori_CCI1, vM1)

        start_period = -len(ori_LR_C1)
        LR_C1 = pd.Series(ori_LR_C1[start_period:])

        signal_short = LR_C1
        sig_res = signal_short.rolling(acc_period).sum().values
        ori_res=sig_res

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(sig_res.reshape(-1, 1))

        return res,ori_res


    def s_LRS_CCI(self,close,vM1,acc_period):
        CCI1, ori_CCI1 = self.CCI(close, vM1)
        LR_C1, ori_LR_C1 = self.slope(ori_CCI1, vM1)

        start_period = -len(ori_LR_C1)
        LR_C1 = pd.Series(ori_LR_C1[start_period:])

        signal_short = LR_C1
        sig_res = signal_short.rolling(acc_period).sum().values
        ori_res = sig_res

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(sig_res.reshape(-1, 1))

        return res, ori_res

    def LR1m(self,close,vM1,vM2,vM3,vM4,vM5,vM6,vM7,vM8,acc_period):

        L1,ori_L1= self.slope_line(close,vM1)
        L2,ori_L2= self.slope_line(close,vM2)
        L3,ori_L3= self.slope_line(close,vM3)
        L4,ori_L4= self.slope_line(close,vM4)
        L5,ori_L5= self.slope_line(close,vM5)
        L6,ori_L6= self.slope_line(close,vM6)
        L7,ori_L7= self.slope_line(close,vM7)
        L8,ori_L8= self.slope_line(close,vM8)

        start_period= -min(len(ori_L1),len(ori_L2),len(ori_L3),len(ori_L4),len(ori_L5),len(ori_L6),len(ori_L7),len(ori_L8))
        L1= pd.Series(ori_L1[start_period:])
        L2= pd.Series(ori_L2[start_period:])
        L3= pd.Series(ori_L3[start_period:])
        L4= pd.Series(ori_L4[start_period:])
        L5= pd.Series(ori_L5[start_period:])
        L6= pd.Series(ori_L6[start_period:])
        L7= pd.Series(ori_L7[start_period:])
        L8= pd.Series(ori_L8[start_period:])


        MultipleL1 = (L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8) / 8
        LRLA= MultipleL1.rolling(acc_period).sum().dropna().reset_index()[0].values
        ori_res=LRLA

        scaler=MinMaxScaler(feature_range=(0.1,1))
        res= scaler.fit_transform(LRLA.reshape(-1,1))

        return res,ori_res


    def VR(self, vr_period, data, Vol_):  # 거래비율을 구하는 함수
        # volume Ratio 계산

        # VR=((N일간 상승일 거래량의 합)/(N일간 하락일 거래량의 합))*100
        period = vr_period
        series1 = data
        series2 = pd.DataFrame(series1)
        series = pd.Series(series2[0].values)
        series = series.diff().dropna()

        Volume1 = Vol_
        Volume2 = pd.DataFrame(Volume1)
        Volume = pd.Series(Volume2[0].values)

        ups = Volume * 0
        down = Volume * 0

        ups_index = series.index[series > 0]
        down_index = series.index[series < 0]

        ups[ups_index] = Volume[ups_index]
        down[down_index] = Volume[down_index]

        Volume_plus = ups.rolling(period).sum()
        Volume_minus1 = down.rolling(period).sum()
        Volume_minus = np.abs(Volume_minus1)
        Volume_minus[Volume_minus == 0] = 1e-5  # inf값을 방지
        VR = (Volume_plus / Volume_minus)

        VR_idx = VR.dropna()
        nan_idx = VR_idx.index[0]
        # nan값의 갯수가된다 즉 마지막nan값의 순번이다. 추후 지표들의 인덱스를 유지하며 nan을 효과적으로 제거하기위해 사용한다

        VR = VR.values.reshape(-1)

        ori_VR= VR

        scaler = MinMaxScaler(feature_range=(0.1,1))
        VR = scaler.fit_transform(VR.reshape(-1, 1))
        return VR,ori_VR


    def Min_span(self, close, high, low, period,period2):  # 일목 + slope_line 전략(초기값 period=1 , period2=200)

        scale_LRL,LRLv1 = self.slope_line(close,period2)

        전환 = (high.rolling(period * 9).max().dropna().reset_index() + low.rolling(
            period * 9).min().dropna().reset_index()) / 2
        전환 = 전환.dropna().reset_index()
        기준 = (high.rolling(period * 26).max().dropna().reset_index() + low.rolling(
            period * 26).min().dropna().reset_index()) / 2
        기준 = 기준.dropna().reset_index()
        후행 = close

        start_period = -min(len(전환), len(기준), len(후행), len(LRLv1))
        전환 = 전환[start_period:][0].dropna().reset_index()
        기준 = 기준[start_period:][0].dropna().reset_index()
        후행 = 후행[start_period:].dropna().reset_index()

        선행1 = (전환[0] + 기준[0]) / 2
        선행1 = 선행1.values
        선행2 = (high.rolling(period * 52).max().reset_index()[0] + low.rolling(period * 52).min().reset_index()[0]) / 2
        선행2= 선행2.dropna().reset_index()[0].values

        start_period2= -min(len(선행1),len(선행2),len(LRLv1))
        선행1 = 선행1[start_period2:]
        선행2= 선행2[start_period2:]
        LRLv1= LRLv1[start_period2:]

        #If CrossUP(LRLv1, 선행스팬1[25]) Then Buy("B1"); MIN strategy
        #If CrossDown(LRLv1, 선행스팬2[25]) Then Sell("S1");

        span1= LRLv1-선행1     #롱 크로스업 지표
        span2= LRLv1-선행2     #숏 크로스업 지표

        ori_span1=np.array(span1)
        ori_span2=np.array(span2)
        ori_span3=np.array(LRLv1)

        scaler= MinMaxScaler(feature_range=(0.1,1))
        span1= scaler.fit_transform(pd.Series(span1).values.reshape(-1,1))
        span2= scaler.fit_transform(pd.Series(span2).values.reshape(-1,1))

        return span1, span2, scale_LRL , ori_span1, ori_span2, ori_span3


    def slope(self, close, period):  # LRS
        if period == 0:
            LRS = 0
        Sumbars = period * (period - 1) * 0.5
        SumSqrBars = (period - 1) * period * (2 * period - 1) / 6

        SumY = close.rolling(period).sum().dropna().reset_index()[0]

        Sum1_data=[]
        rolling_close=[]
        for step in range(len(close)-period):
            rolling_close.append(close[step:step+period]) #rolling period 처럼 저장

        for step in range(len(rolling_close)):
            Sum1 = 0
            for step_ in range(len(rolling_close[step])):
                Sum1= Sum1 + step_*rolling_close[step][step_]
            Sum1_data.append(Sum1)


        Sum1=Sum1_data
        Sum2 = Sumbars * SumY

        Num1 = period * Sum1 - Sum2
        Num2 = Sumbars * Sumbars - period * SumSqrBars

        if Num2 != 0:
            LRS = Num1 / Num2
        else:
            print('slope 오류')
            LRS = 0
        ori_res=LRS.values
        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(LRS.values.reshape(-1, 1))

        return res,ori_res


    def bol_band(self,price,period,MultiD):

        STD,ori_STD=self.STD(price,period)
        ori_BBup= self.MA(price,period) + MultiD*ori_STD
        ori_BBdown= self.MA(price,period) - MultiD*ori_STD

        scaler= MinMaxScaler(feature_range=(0.1,1))
        BBup = scaler.fit_transform(ori_BBup.values.reshape(-1,1))
        BBdown = scaler.fit_transform(ori_BBdown.values.reshape(-1,1))

        return BBup,BBdown,ori_BBup,ori_BBdown


    def cross_wma_price(self,price,wma_period): #가중이동평균과 가격의 크로스업 or down
        wma,ori_wma=self.WMA(price,wma_period)
        price= price[-len(ori_wma):]

        res=ori_wma-price
        ori_res= np.array(res)

        scaler=MinMaxScaler(feature_range=(0.1,1))
        res=scaler.fit_transform(res.values.reshape(-1,1))
        res=np.array(res).reshape(-1)

        return res,ori_res


    def cross_bol_price(self,price,period,MultiD): #볼린저 밴드와 가격의 크로스
        BBup,BBdown,ori_BBup,ori_BBdown=self.bol_band(price,period,MultiD)
        price_=price[-len(ori_BBup):] # BB와 기간 맞춤

        up_cross= ori_BBup-price_
        down_cross= ori_BBdown-price_

        scaler =MinMaxScaler(feature_range=(0.1,1))
        up_cross_scale= scaler.fit_transform(up_cross.values.reshape(-1,1))
        down_cross_scale= scaler.fit_transform(down_cross.values.reshape(-1,1))

        return up_cross_scale,down_cross_scale,up_cross,down_cross




    def cross_wma_one(self,price,wma_period,high_period, low_period, res2_period):
        wma, ori_wma = self.WMA(price, wma_period)
        one1,one2,ori_one,ori_one2=self.one_side(price, high_period, low_period, res2_period)

        index_=-min(len(wma),len(one1),len(one2))

        wma=pd.Series(wma[index_:])
        one1=pd.Series(one1[index_:])
        one2=pd.Series(one2[index_:])

        res=np.array(wma-one1).reshape(-1)
        res2=np.array(wma-one2).reshape(-1)

        ori_res1= res
        ori_res2= res2

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(res.reshape(-1, 1))
        res = np.array(res).reshape(-1)

        res2 = scaler.fit_transform(res2.reshape(-1, 1))
        res2 = np.array(res2).reshape(-1)

        return res,res2,ori_res1,ori_res2

    def CCI_exercise(self,price,CCI_period, slope_period): #CCI_period= CCI 기간값 , slope_period= 기울기 기간값
        #추세의 운동량을 구한다

        #m= CCI (가격과 이평의 이격도가 커질수록 질량이 상승) 양의 이격도 인경우
        #v= 기울기
        #p = mv  운동량 공식

        #기울기가 커지고 , CCI 이격도가 커지면 변동성구간, 역추세 구간을 모두 찾게된다.

        scale_m,m= self.CCI(price,CCI_period)
        v= [(price[step+slope_period]-price[step])/slope_period for step in range(len(price)-slope_period)] #가속도 공식

        start_period=min(len(m),len(v))
        m=m[-start_period:]
        v=v[-start_period:]

        p= m*v

        # 추세가 꺾이면   기울기 음이 됨, 이격도는 작아짐   = 값 -
        # 상승추세 유지면  기울기 양 or + , 이격도는 일정? = 값 + 일정
        # 하락추세 유지면 기울기 -로 일정, 이격도 일정      = 값 -로 일정

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(p.reshape(-1, 1))
        res = np.array(res).reshape(-1)
        ori_res=p

        return res,ori_res



    def CCI_trend(self,price,CCI_period, log_period): #CCI기반 추세지표: 짧은 추세에서는 후행성이 있지만 긴추세에서는 선행하는 경향도 있다.

        scale_log_re, log_re= self.log_return(price,CCI_period)
        scale_c, cci = self.CCI(price, log_period)
        price_ma= price.rolling(CCI_period).mean().dropna()
        scale_price_ma = (price_ma-np.min(price_ma))/(-np.min(price_ma)+np.max(price_ma))
        # CCI와 로그수익률의 상승,하락 비율을 추세지표로 나타낸다 ( CCI와 주가가 변동할때 로그수익률이 얼마만큼 변동하는지 비율)
        # 변동성때문에 로그수익률이 급격히 상승하는 경우 이격도가 커지므로 CCI도 급격히 상승한다. 따라서 변동성에서는 변동비율이 거의 동일하므로 휩소가 어느정도는 상쇄된다.
        # 추세장 초기에서는 로그수익률이 낮은편이고 CCI의 이격도가 커지므로 지표값이 상승한다

        # 짧은 추세에서는 후행성이 있지만 긴추세 이전에는 선행하는 경향이 있다. (장기 하락 또는 장기 상승 전 방향과 관계없이 한번의 큰변동이 나오는경우가 많음)
        # 이 가정이 맞다면 장기 추세이전에 로그수익률이 더 민감하게 반응해서 지표의 변동이 미리 생긴다.


        start_period= min(len(scale_c),len(scale_log_re),len(scale_price_ma))

        price_ma=price_ma[-start_period:].reset_index()[0] #가격은 시리즈로 들어온다
        cci= pd.Series(cci[-start_period:]).reset_index(0)[0]
        log_re=pd.Series(log_re[-start_period:]).reset_index(0)[0]*max(cci*10)

        # LRL부분###################################

        close=cci-log_re
        period=int(round(log_period/CCI_period))

        AccumValue = pd.Series(range(len(close))) + 1  # 1부터 ~ 끝
        AccumValue = AccumValue.dropna().reset_index()[0]

        _, a = self.MA(AccumValue, period)
        _, b = self.MA(close, period)
        value1 = torch.Tensor(pd.Series(a).dropna().reset_index()[0])
        value2 = torch.Tensor(pd.Series(b).dropna().reset_index()[0])

        close = torch.Tensor(close.tolist())
        AccumValue = torch.Tensor(AccumValue)

        _, ori_1 = self.MA(pd.Series((close * AccumValue).view(-1)), period)
        A = torch.Tensor(pd.Series(ori_1).dropna().reset_index()[0])
        B = value1 * value2

        _, ori_2 = self.MA(pd.Series(AccumValue ** 2), period)
        C = torch.Tensor(pd.Series(ori_2).dropna().reset_index()[0])
        _, ori_3 = self.MA(pd.Series(AccumValue), period)
        D = torch.Tensor((pd.Series(ori_3) ** 2).dropna().reset_index()[0])

        E = AccumValue[-len(value1):] - value1

        LRL = (A - B) / (C - D + C.min()) * E + value2  # inf값방지
        ori_LRL = pd.Series(LRL.view(-1)).values

        scaler = MinMaxScaler(feature_range=(0.1,1))
        LRL = scaler.fit_transform(pd.Series(ori_LRL).values.reshape(-1, 1))
        LRL = np.array(LRL).reshape(-1)

        p= pd.Series(LRL)

        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(p.values.reshape(-1, 1))
        res = np.array(res).reshape(-1)
        ori_res= ori_LRL


        return res, ori_res



    def NCO_up_ma(self,price,period):   #CCI= 이평선과 주가의 이격도
        #이평선과 주가 ?(지지선,이평선의 이격)
        #현 지표 = 이평선과 지지선의 이격이 0 에 가까울수록 값이 0에 가까워짐

        '''''
        이전
        #CCI = 이평선과 주가의 이격
        #기존 사용 지지저항(NCO_up,NCO_down) = 주가와 지지선의 이격이 0에 가까울수록 값이 0에 가까워지는 역추세 지표
        '''''

        NCO_up = close.iloc[period - 1:] - close.rolling(period).min()
        NCO_up = NCO_up.rolling(period2).mean()
        NCO_up = NCO_up.dropna().reset_index()[0]

        ori_res = NCO_up.values
        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(NCO_up.values.reshape(-1, 1))
        res = res.reshape(-1)
        res = pd.Series(res).values.reshape(-1)



    def NCO_down_ma(self,price,period):
        pass


    def NNCO_up(self,close,period,period2):

        NCO_up = close.iloc[period-1:] - close.rolling(period).min()
        NCO_up = NCO_up.rolling(period2).mean() # period2 평균값
        NCO_up = NCO_up.dropna().reset_index()[0]

        MD= np.abs(NCO_up)
        MD= MD.rolling(period).mean().dropna().reset_index()[0]

        start_period=min(len(NCO_up),len(MD))
        ori_NCO_up=NCO_up[-start_period:].dropna().reset_index()[0]
        MD= MD[-start_period:].dropna().reset_index()[0]

        NCO_up = (ori_NCO_up / (0.015 * (MD + (1e-1))))

        ori_res = NCO_up.values
        scaler = MinMaxScaler(feature_range=(0.1,1))
        res = scaler.fit_transform(NCO_up.values.reshape(-1, 1))
        res = res.reshape(-1)
        res = pd.Series(res).values.reshape(-1)

        return res, ori_res


    def NNCO_down(self,close,period,period2):
        NCO_down = close.rolling(period).max() - close.iloc[period - 1:]
        NCO_down = NCO_down.rolling(period2).mean()
        NCO_down = NCO_down.dropna().reset_index()[0]

        MD = np.abs(NCO_down)
        MD = MD.rolling(period).mean().dropna().reset_index()[0]

        start_period=min(len(NCO_down), len(MD))
        ori_NCO_down= NCO_down[-start_period:].dropna().reset_index()[0]
        MD= MD[-start_period:].dropna().reset_index()[0]

        NCO_down = (ori_NCO_down / (0.015 * (MD+ (1e-1))))

        ori_res = NCO_down.values
        scaler = MinMaxScaler(feature_range=(0.1,1))

        res = scaler.fit_transform(NCO_down.values.reshape(-1, 1))
        res = res.reshape(-1)
        res = pd.Series(res).values.reshape(-1)

        return res, ori_res

    def LRC3m(self,close,vM1, vM2,
            cci_period1,cci_period2,cci_period3, cci_period4,cci_period5, cci_period6, cci_period7,
            cci_period8,cci_period11,cci_period12,cci_period13,cci_period14,
            cci_period15,cci_period16, cci_period17, cci_period18
            ,acc_period1, acc_period2):

        CCI1, ori_CCI1 = self.CCI(close, cci_period1)
        CCI2, ori_CCI2 = self.CCI(close, cci_period2)
        CCI3, ori_CCI3 = self.CCI(close, cci_period3)
        CCI4, ori_CCI4 = self.CCI(close, cci_period4)
        CCI5, ori_CCI5 = self.CCI(close, cci_period5)
        CCI6, ori_CCI6 = self.CCI(close, cci_period6)
        CCI7, ori_CCI7 = self.CCI(close, cci_period7)
        CCI8, ori_CCI8 = self.CCI(close, cci_period8)

        CCI11, ori_CCI11 = self.CCI(close, cci_period11)
        CCI12, ori_CCI12 = self.CCI(close, cci_period12)
        CCI13, ori_CCI13 = self.CCI(close, cci_period13)
        CCI14, ori_CCI14 = self.CCI(close, cci_period14)
        CCI15, ori_CCI15 = self.CCI(close, cci_period15)
        CCI16, ori_CCI16 = self.CCI(close, cci_period16)
        CCI17, ori_CCI17 = self.CCI(close, cci_period17)
        CCI18, ori_CCI18 = self.CCI(close, cci_period18)

        start_period= min(len(ori_CCI1), len(ori_CCI2), len(ori_CCI3), len(ori_CCI4), len(ori_CCI5),
                          len(ori_CCI6), len(ori_CCI7), len(ori_CCI8),
                          len(ori_CCI11), len(ori_CCI12), len(ori_CCI13), len(ori_CCI14), len(ori_CCI15)
                          ,len(ori_CCI16), len(ori_CCI17), len(ori_CCI18)
                            )

        ori_CCI1 = ori_CCI1[-start_period:]
        ori_CCI2 = ori_CCI2[-start_period:]
        ori_CCI3 = ori_CCI3[-start_period:]
        ori_CCI4 = ori_CCI4[-start_period:]
        ori_CCI5 = ori_CCI5[-start_period:]
        ori_CCI6 = ori_CCI6[-start_period:]
        ori_CCI7 = ori_CCI7[-start_period:]
        ori_CCI8 = ori_CCI8[-start_period:]
        ori_CCI11 = ori_CCI11[-start_period:]
        ori_CCI12 = ori_CCI12[-start_period:]
        ori_CCI13 = ori_CCI13[-start_period:]
        ori_CCI14 = ori_CCI14[-start_period:]
        ori_CCI15 = ori_CCI15[-start_period:]
        ori_CCI16 = ori_CCI16[-start_period:]
        ori_CCI17 = ori_CCI17[-start_period:]
        ori_CCI18 = ori_CCI18[-start_period:]


        vCCI1, v_ori_CCI1 = self.slope_line(ori_CCI1, vM1)
        vCCI2, v_ori_CCI2 = self.slope_line(ori_CCI2, vM1)
        vCCI3, v_ori_CCI3 = self.slope_line(ori_CCI3, vM1)
        vCCI4, v_ori_CCI4 = self.slope_line(ori_CCI4, vM1)
        vCCI5, v_ori_CCI5 = self.slope_line(ori_CCI5, vM1)
        vCCI6, v_ori_CCI6 = self.slope_line(ori_CCI6, vM1)
        vCCI7, v_ori_CCI7 = self.slope_line(ori_CCI7, vM1)
        vCCI8, v_ori_CCI8 = self.slope_line(ori_CCI8, vM1)

        vCCI11,v_ori_CCI11 = self.slope_line(ori_CCI11,vM2)
        vCCI12,v_ori_CCI12 = self.slope_line(ori_CCI12,vM2)
        vCCI13,v_ori_CCI13 = self.slope_line(ori_CCI13,vM2)
        vCCI14,v_ori_CCI14 = self.slope_line(ori_CCI14,vM2)
        vCCI15,v_ori_CCI15= self.slope_line(ori_CCI15,vM2)
        vCCI16,v_ori_CCI16= self.slope_line(ori_CCI16,vM2)
        vCCI17,v_ori_CCI17= self.slope_line(ori_CCI17,vM2)
        vCCI18,v_ori_CCI18 = self.slope_line(ori_CCI18,vM2)

        start_period2= min(len(v_ori_CCI1),len(v_ori_CCI2), len(v_ori_CCI3),len(v_ori_CCI4),len(v_ori_CCI5)
                          ,len(v_ori_CCI6),len(v_ori_CCI7),len(v_ori_CCI8),
                          len(v_ori_CCI11),len(v_ori_CCI12),len(v_ori_CCI13),len(v_ori_CCI14),len(v_ori_CCI15)
                          ,len(v_ori_CCI16),len(v_ori_CCI17),len(v_ori_CCI18))

        v_ori_CCI1 = v_ori_CCI1[-start_period2:]
        v_ori_CCI2 = v_ori_CCI2[-start_period2:]
        v_ori_CCI3 = v_ori_CCI3[-start_period2:]
        v_ori_CCI4 = v_ori_CCI4[-start_period2:]
        v_ori_CCI5 = v_ori_CCI5[-start_period2:]
        v_ori_CCI6 = v_ori_CCI6[-start_period2:]
        v_ori_CCI7 = v_ori_CCI7[-start_period2:]
        v_ori_CCI8 = v_ori_CCI8[-start_period2:]
        v_ori_CCI11 = v_ori_CCI11[-start_period2:]
        v_ori_CCI12 = v_ori_CCI12[-start_period2:]
        v_ori_CCI13 = v_ori_CCI13[-start_period2:]
        v_ori_CCI14 = v_ori_CCI14[-start_period2:]
        v_ori_CCI15 = v_ori_CCI15[-start_period2:]
        v_ori_CCI16 = v_ori_CCI16[-start_period2:]
        v_ori_CCI17 = v_ori_CCI17[-start_period2:]
        v_ori_CCI18 = v_ori_CCI18[-start_period2:]




        sign_short = (v_ori_CCI1 + v_ori_CCI2 + v_ori_CCI3 + v_ori_CCI4 + v_ori_CCI5 + v_ori_CCI6 + v_ori_CCI7
                      + v_ori_CCI8)/8
        sign_short2 = (v_ori_CCI11 + v_ori_CCI12 + v_ori_CCI13 + v_ori_CCI14 + v_ori_CCI15+ v_ori_CCI16
                       +v_ori_CCI17 + v_ori_CCI18)/8

        sign_short = pd.Series(sign_short).rolling(acc_period1).sum().dropna().reset_index()[0]
        sign_short2 = pd.Series(sign_short2).rolling(acc_period2).sum().dropna().reset_index()[0]

        start_period3=min(len(sign_short),len(sign_short2))
        sign_short= sign_short[-start_period3:]
        sign_short2= sign_short2[-start_period3:]

        scaler=MinMaxScaler(feature_range=(0.1,1))

        res=scaler.fit_transform(sign_short.values.reshape(-1,1))
        res2=scaler.fit_transform(sign_short2.values.reshape(-1,1))

        res=res.reshape(-1)
        res2=res2.reshape(-1)

        return  res, res2,sign_short, sign_short2


    def LRC7m(self, close, vM1, vM2, vM3,
              cci_period1, cci_period2, cci_period3, cci_period4, cci_period5, cci_period6, cci_period7,
              cci_period8, cci_period11, cci_period12, cci_period13, cci_period14,
              cci_period15, cci_period16, cci_period17, cci_period18,
              cci_period21, cci_period22,cci_period23, cci_period24, cci_period25, cci_period26, cci_period27, cci_period28
              , acc_period1, acc_period2,acc_period3):

        CCI1, ori_CCI1 = self.CCI(close, cci_period1)
        CCI2, ori_CCI2 = self.CCI(close, cci_period2)
        CCI3, ori_CCI3 = self.CCI(close, cci_period3)
        CCI4, ori_CCI4 = self.CCI(close, cci_period4)
        CCI5, ori_CCI5 = self.CCI(close, cci_period5)
        CCI6, ori_CCI6 = self.CCI(close, cci_period6)
        CCI7, ori_CCI7 = self.CCI(close, cci_period7)
        CCI8, ori_CCI8 = self.CCI(close, cci_period8)

        CCI11, ori_CCI11 = self.CCI(close, cci_period11)
        CCI12, ori_CCI12 = self.CCI(close, cci_period12)
        CCI13, ori_CCI13 = self.CCI(close, cci_period13)
        CCI14, ori_CCI14 = self.CCI(close, cci_period14)
        CCI15, ori_CCI15 = self.CCI(close, cci_period15)
        CCI16, ori_CCI16 = self.CCI(close, cci_period16)
        CCI17, ori_CCI17 = self.CCI(close, cci_period17)
        CCI18, ori_CCI18 = self.CCI(close, cci_period18)

        CCI21, ori_CCI21= self.CCI(close, cci_period21)
        CCI22, ori_CCI22= self.CCI(close, cci_period22)
        CCI23, ori_CCI23= self.CCI(close, cci_period23)
        CCI24, ori_CCI24= self.CCI(close, cci_period24)
        CCI25, ori_CCI25= self.CCI(close, cci_period25)
        CCI26, ori_CCI26= self.CCI(close, cci_period26)
        CCI27, ori_CCI27= self.CCI(close, cci_period27)
        CCI28, ori_CCI28= self.CCI(close, cci_period28)



        start_period = min(len(ori_CCI1), len(ori_CCI2), len(ori_CCI3), len(ori_CCI4), len(ori_CCI5),
                           len(ori_CCI6), len(ori_CCI7), len(ori_CCI8),
                           len(ori_CCI11), len(ori_CCI12), len(ori_CCI13), len(ori_CCI14), len(ori_CCI15)
                           , len(ori_CCI16), len(ori_CCI17), len(ori_CCI18)
                           ,len(ori_CCI21), len(ori_CCI22),len(ori_CCI23),len(ori_CCI24),len(ori_CCI25),
                           len(ori_CCI26),len(ori_CCI27),len(ori_CCI28)
                           )



        ori_CCI1 = ori_CCI1[-start_period:]
        ori_CCI2 = ori_CCI2[-start_period:]
        ori_CCI3 = ori_CCI3[-start_period:]
        ori_CCI4 = ori_CCI4[-start_period:]
        ori_CCI5 = ori_CCI5[-start_period:]
        ori_CCI6 = ori_CCI6[-start_period:]
        ori_CCI7 = ori_CCI7[-start_period:]
        ori_CCI8 = ori_CCI8[-start_period:]

        ori_CCI11 = ori_CCI11[-start_period:]
        ori_CCI12 = ori_CCI12[-start_period:]
        ori_CCI13 = ori_CCI13[-start_period:]
        ori_CCI14 = ori_CCI14[-start_period:]
        ori_CCI15 = ori_CCI15[-start_period:]
        ori_CCI16 = ori_CCI16[-start_period:]
        ori_CCI17 = ori_CCI17[-start_period:]
        ori_CCI18 = ori_CCI18[-start_period:]

        ori_CCI21 =ori_CCI21[-start_period:]
        ori_CCI22 =ori_CCI22[-start_period:]
        ori_CCI23 =ori_CCI23[-start_period:]
        ori_CCI24 =ori_CCI24[-start_period:]
        ori_CCI25 =ori_CCI25[-start_period:]
        ori_CCI26 =ori_CCI26[-start_period:]
        ori_CCI27 =ori_CCI27[-start_period:]
        ori_CCI28 =ori_CCI28[-start_period:]


        vCCI1, v_ori_CCI1 = self.slope_line(ori_CCI1, vM1)
        vCCI2, v_ori_CCI2 = self.slope_line(ori_CCI2, vM1)
        vCCI3, v_ori_CCI3 = self.slope_line(ori_CCI3, vM1)
        vCCI4, v_ori_CCI4 = self.slope_line(ori_CCI4, vM1)
        vCCI5, v_ori_CCI5 = self.slope_line(ori_CCI5, vM1)
        vCCI6, v_ori_CCI6 = self.slope_line(ori_CCI6, vM1)
        vCCI7, v_ori_CCI7 = self.slope_line(ori_CCI7, vM1)
        vCCI8, v_ori_CCI8 = self.slope_line(ori_CCI8, vM1)

        vCCI11, v_ori_CCI11 = self.slope_line(ori_CCI11, vM2)
        vCCI12, v_ori_CCI12 = self.slope_line(ori_CCI12, vM2)
        vCCI13, v_ori_CCI13 = self.slope_line(ori_CCI13, vM2)
        vCCI14, v_ori_CCI14 = self.slope_line(ori_CCI14, vM2)
        vCCI15, v_ori_CCI15 = self.slope_line(ori_CCI15, vM2)
        vCCI16, v_ori_CCI16 = self.slope_line(ori_CCI16, vM2)
        vCCI17, v_ori_CCI17 = self.slope_line(ori_CCI17, vM2)
        vCCI18, v_ori_CCI18 = self.slope_line(ori_CCI18, vM2)

        vCCI21, v_ori_CCI21= self.slope_line(ori_CCI21, vM3)
        vCCI22, v_ori_CCI22= self.slope_line(ori_CCI22, vM3)
        vCCI23, v_ori_CCI23=self.slope_line(ori_CCI23, vM3)
        vCCI24, v_ori_CCI24=self.slope_line(ori_CCI24, vM3)
        vCCI25, v_ori_CCI25=self.slope_line(ori_CCI25, vM3)
        vCCI26, v_ori_CCI26=self.slope_line(ori_CCI26, vM3)
        vCCI27, v_ori_CCI27=self.slope_line(ori_CCI27, vM3)
        vCCI28, v_ori_CCI28=self.slope_line(ori_CCI28, vM3)

        start_period2 = min(len(v_ori_CCI1), len(v_ori_CCI2), len(v_ori_CCI3), len(v_ori_CCI4), len(v_ori_CCI5)
                            , len(v_ori_CCI6), len(v_ori_CCI7), len(v_ori_CCI8),
                            len(v_ori_CCI11), len(v_ori_CCI12), len(v_ori_CCI13), len(v_ori_CCI14), len(v_ori_CCI15)
                            , len(v_ori_CCI16), len(v_ori_CCI17), len(v_ori_CCI18)
                            , len(v_ori_CCI21), len(v_ori_CCI22), len(v_ori_CCI23),len(v_ori_CCI24),len(v_ori_CCI25)
                            ,len(v_ori_CCI26),len(v_ori_CCI27),len(v_ori_CCI28)
                            )



        v_ori_CCI1 = v_ori_CCI1[-start_period2:]
        v_ori_CCI2 = v_ori_CCI2[-start_period2:]
        v_ori_CCI3 = v_ori_CCI3[-start_period2:]
        v_ori_CCI4 = v_ori_CCI4[-start_period2:]
        v_ori_CCI5 = v_ori_CCI5[-start_period2:]
        v_ori_CCI6 = v_ori_CCI6[-start_period2:]
        v_ori_CCI7 = v_ori_CCI7[-start_period2:]
        v_ori_CCI8 = v_ori_CCI8[-start_period2:]

        v_ori_CCI11 = v_ori_CCI11[-start_period2:]
        v_ori_CCI12 = v_ori_CCI12[-start_period2:]
        v_ori_CCI13 = v_ori_CCI13[-start_period2:]
        v_ori_CCI14 = v_ori_CCI14[-start_period2:]
        v_ori_CCI15 = v_ori_CCI15[-start_period2:]
        v_ori_CCI16 = v_ori_CCI16[-start_period2:]
        v_ori_CCI17 = v_ori_CCI17[-start_period2:]
        v_ori_CCI18 = v_ori_CCI18[-start_period2:]

        v_ori_CCI21 = v_ori_CCI21[-start_period2:]
        v_ori_CCI22 = v_ori_CCI22[-start_period2:]
        v_ori_CCI23 = v_ori_CCI23[-start_period2:]
        v_ori_CCI24 = v_ori_CCI24[-start_period2:]
        v_ori_CCI25 = v_ori_CCI25[-start_period2:]
        v_ori_CCI26 = v_ori_CCI26[-start_period2:]
        v_ori_CCI27 = v_ori_CCI27[-start_period2:]
        v_ori_CCI28 = v_ori_CCI28[-start_period2:]

        sign_short = (v_ori_CCI1 + v_ori_CCI2 + v_ori_CCI3 + v_ori_CCI4 + v_ori_CCI5 + v_ori_CCI6 + v_ori_CCI7
                      + v_ori_CCI8) / 8
        sign_short2 = (v_ori_CCI11 + v_ori_CCI12 + v_ori_CCI13 + v_ori_CCI14 + v_ori_CCI15 + v_ori_CCI16
                       + v_ori_CCI17 + v_ori_CCI18) / 8
        sign_short3 = (v_ori_CCI21 + v_ori_CCI22 + v_ori_CCI23 +v_ori_CCI24 + v_ori_CCI25 + v_ori_CCI26 + v_ori_CCI27 + v_ori_CCI28)/8

        sign_short = pd.Series(sign_short).rolling(acc_period1).sum().dropna().reset_index()[0]
        sign_short2 = pd.Series(sign_short2).rolling(acc_period2).sum().dropna().reset_index()[0]
        sign_short3 =pd.Series(sign_short3).rolling(acc_period3).sum().dropna().reset_index()[0]

        start_period3 = min(len(sign_short), len(sign_short2),len(sign_short3))
        sign_short = sign_short[-start_period3:]
        sign_short2 = sign_short2[-start_period3:]
        sign_short3 = sign_short3[-start_period3:]

        scaler = MinMaxScaler(feature_range=(0.1,1))

        res = scaler.fit_transform(sign_short.values.reshape(-1, 1))
        res2 = scaler.fit_transform(sign_short2.values.reshape(-1, 1))
        res3= scaler.fit_transform(sign_short3.values.reshape(-1,1))

        res = res.reshape(-1)
        res2 = res2.reshape(-1)
        res3= res3.reshape(-1)

        return res, res2, res3 , sign_short, sign_short2, sign_short3

    def LALA(self,close, cci_period1, cci_period2, cci_period3, cci_period4, cci_period5,cci_period6, cci_period7, cci_period8, acc_period):
        CCI1, ori_CCI1 = self.CCI(close, cci_period1)
        CCI2, ori_CCI2 = self.CCI(close, cci_period2)
        CCI3, ori_CCI3 = self.CCI(close, cci_period3)
        CCI4, ori_CCI4 = self.CCI(close, cci_period4)
        CCI5, ori_CCI5 = self.CCI(close, cci_period5)
        CCI6, ori_CCI6 = self.CCI(close, cci_period6)
        CCI7, ori_CCI7 = self.CCI(close, cci_period7)
        CCI8, ori_CCI8 = self.CCI(close, cci_period8)

        start_period = min(len(ori_CCI1),len(ori_CCI2),len(ori_CCI3),len(ori_CCI4),len(ori_CCI5),len(ori_CCI6),len(ori_CCI7),len(ori_CCI8))

        ori_CCI1=ori_CCI1[-start_period:]
        ori_CCI2=ori_CCI2[-start_period:]
        ori_CCI3=ori_CCI3[-start_period:]
        ori_CCI4=ori_CCI4[-start_period:]
        ori_CCI5=ori_CCI5[-start_period:]
        ori_CCI6=ori_CCI6[-start_period:]
        ori_CCI7=ori_CCI7[-start_period:]
        ori_CCI8=ori_CCI8[-start_period:]

        LR_CCI1, ori_LR_CCI1 = self.slope_line(ori_CCI1,cci_period1)
        LR_CCI2, ori_LR_CCI2 = self.slope_line(ori_CCI2,cci_period2)
        LR_CCI3, ori_LR_CCI3 = self.slope_line(ori_CCI3,cci_period3)
        LR_CCI4, ori_LR_CCI4 = self.slope_line(ori_CCI4,cci_period4)
        LR_CCI5, ori_LR_CCI5 = self.slope_line(ori_CCI5,cci_period5)
        LR_CCI6, ori_LR_CCI6 = self.slope_line(ori_CCI6,cci_period6)
        LR_CCI7, ori_LR_CCI7 = self.slope_line(ori_CCI7,cci_period7)
        LR_CCI8, ori_LR_CCI8 = self.slope_line(ori_CCI8,cci_period8)


        start_period = min(len(ori_LR_CCI1),len(ori_LR_CCI2),len(ori_LR_CCI3),len(ori_LR_CCI4),len(ori_LR_CCI5),len(ori_LR_CCI6),
                           len(ori_LR_CCI7),len(ori_LR_CCI8))


        ori_LR_CCI1= ori_LR_CCI1[-start_period:]
        ori_LR_CCI2= ori_LR_CCI2[-start_period:]
        ori_LR_CCI3= ori_LR_CCI3[-start_period:]
        ori_LR_CCI4= ori_LR_CCI4[-start_period:]
        ori_LR_CCI5= ori_LR_CCI5[-start_period:]
        ori_LR_CCI6= ori_LR_CCI6[-start_period:]
        ori_LR_CCI7= ori_LR_CCI7[-start_period:]
        ori_LR_CCI8= ori_LR_CCI8[-start_period:]

        sign_short= (ori_LR_CCI1 + ori_LR_CCI2 + ori_LR_CCI3 + ori_LR_CCI4 + ori_LR_CCI5 + ori_LR_CCI6 +ori_LR_CCI7 +ori_LR_CCI8)/8
        sign_short=pd.Series(sign_short).rolling(acc_period).sum()
        res=sign_short.dropna().reset_index()[0]

        scaler=MinMaxScaler(feature_range=(0.1,1))
        scale_res= scaler.fit_transform(pd.Series(res).values.reshape(-1,1))
        scale_res=scale_res.reshape(-1)

        return scale_res, res

    def LALA35(self, close ,open_,high_,low_, cci_period1, cci_period2, cci_period3, cci_period4, cci_period5, cci_period6, cci_period7,
             cci_period8, acc_period):

        close = close
        open = open_
        high = high_
        low = low_

        data = [high, low, close]

        CCI1, ori_CCI1 = self.CCI(data, cci_period1)
        CCI2, ori_CCI2 = self.CCI(data, cci_period2)
        CCI3, ori_CCI3 = self.CCI(data, cci_period3)
        CCI4, ori_CCI4 = self.CCI(data, cci_period4)
        CCI5, ori_CCI5 = self.CCI(data, cci_period5)
        CCI6, ori_CCI6 = self.CCI(data, cci_period6)
        CCI7, ori_CCI7 = self.CCI(data, cci_period7)
        CCI8, ori_CCI8 = self.CCI(data, cci_period8)

        start_period = min(len(ori_CCI1), len(ori_CCI2), len(ori_CCI3), len(ori_CCI4), len(ori_CCI5), len(ori_CCI6),
                           len(ori_CCI7), len(ori_CCI8))

        ori_CCI1 = ori_CCI1[-start_period:].reset_index()[0]
        ori_CCI2 = ori_CCI2[-start_period:].reset_index()[0]
        ori_CCI3 = ori_CCI3[-start_period:].reset_index()[0]
        ori_CCI4 = ori_CCI4[-start_period:].reset_index()[0]
        ori_CCI5 = ori_CCI5[-start_period:].reset_index()[0]
        ori_CCI6 = ori_CCI6[-start_period:].reset_index()[0]
        ori_CCI7 = ori_CCI7[-start_period:].reset_index()[0]
        ori_CCI8 = ori_CCI8[-start_period:].reset_index()[0]

        LR_CCI1, ori_LR_CCI1 = self.slope_line(ori_CCI1, cci_period1) #LRL
        LR_CCI2, ori_LR_CCI2 = self.slope_line(ori_CCI2, cci_period2)
        LR_CCI3, ori_LR_CCI3 = self.slope_line(ori_CCI3, cci_period3)
        LR_CCI4, ori_LR_CCI4 = self.slope_line(ori_CCI4, cci_period4)
        LR_CCI5, ori_LR_CCI5 = self.slope_line(ori_CCI5, cci_period5)
        LR_CCI6, ori_LR_CCI6 = self.slope_line(ori_CCI6, cci_period6)
        LR_CCI7, ori_LR_CCI7 = self.slope_line(ori_CCI7, cci_period7)
        LR_CCI8, ori_LR_CCI8 = self.slope_line(ori_CCI8, cci_period8)

        start_period = min(len(ori_LR_CCI1), len(ori_LR_CCI2), len(ori_LR_CCI3), len(ori_LR_CCI4), len(ori_LR_CCI5),
                           len(ori_LR_CCI6),
                           len(ori_LR_CCI7), len(ori_LR_CCI8))

        ori_LR_CCI1 = ori_LR_CCI1[-start_period:]
        ori_LR_CCI2 = ori_LR_CCI2[-start_period:]
        ori_LR_CCI3 = ori_LR_CCI3[-start_period:]
        ori_LR_CCI4 = ori_LR_CCI4[-start_period:]
        ori_LR_CCI5 = ori_LR_CCI5[-start_period:]
        ori_LR_CCI6 = ori_LR_CCI6[-start_period:]
        ori_LR_CCI7 = ori_LR_CCI7[-start_period:]
        ori_LR_CCI8 = ori_LR_CCI8[-start_period:]


        sign_short = (ori_LR_CCI1 + ori_LR_CCI2 + ori_LR_CCI3 + ori_LR_CCI4 + ori_LR_CCI5 + ori_LR_CCI6 + ori_LR_CCI7 + ori_LR_CCI8) / 8
        sign_short = pd.Series(sign_short).rolling(acc_period).sum()
        res = sign_short.dropna().reset_index()[0]

        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_res = scaler.fit_transform(pd.Series(res).values.reshape(-1, 1))
        scale_res = scale_res.reshape(-1)


        return scale_res, res

    def LRC15m(self,close,open_,high_,low_,vM1,cci_period1,cci_period2,cci_period3,cci_period4,cci_period5, cci_period6,cci_period7, cci_period8,
               acc_period1):
         # 15,100,200,400,600,800,1000,1200,1500,2,18,52,104
        close=close
        open=open_
        high=high_
        low=low_

        data=[high,low,close]

        CCI1,vData1_CCI= self.CCI(data,cci_period1)
        CCI2,vData1_CCI2=self.CCI(data,cci_period2)
        CCI3,vData1_CCI3=self.CCI(data,cci_period3)
        CCI4,vData1_CCI4=self.CCI(data,cci_period4)
        CCI5,vData1_CCI5=self.CCI(data,cci_period5)
        CCI6,vData1_CCI6=self.CCI(data,cci_period6)
        CCI7,vData1_CCI7=self.CCI(data,cci_period7)
        CCI8,vData1_CCI8=self.CCI(data,cci_period8)

        vCCI1,vvData1_CCI=self.slope_line(vData1_CCI,vM1)
        vCCI2,vvData1_CCI2=self.slope_line(vData1_CCI2,vM1)
        vCCI3,vvData1_CCI3=self.slope_line(vData1_CCI3,vM1)
        vCCI4,vvData1_CCI4=self.slope_line(vData1_CCI4,vM1)
        vCCI5,vvData1_CCI5=self.slope_line(vData1_CCI5,vM1)
        vCCI6,vvData1_CCI6=self.slope_line(vData1_CCI6,vM1)
        vCCI7,vvData1_CCI7=self.slope_line(vData1_CCI7,vM1)
        vCCI8,vvData1_CCI8=self.slope_line(vData1_CCI8,vM1)

        start_period=min(len(vvData1_CCI),len(vvData1_CCI2),len(vvData1_CCI3),len(vvData1_CCI4),
                        len(vvData1_CCI5),len(vvData1_CCI6),len(vvData1_CCI7),len(vvData1_CCI8))

        vvData1_CCI= vvData1_CCI[-start_period:]
        vvData1_CCI2=vvData1_CCI2[-start_period:]
        vvData1_CCI3=vvData1_CCI3[-start_period:]
        vvData1_CCI4=vvData1_CCI4[-start_period:]
        vvData1_CCI5=vvData1_CCI5[-start_period:]
        vvData1_CCI6=vvData1_CCI6[-start_period:]
        vvData1_CCI7=vvData1_CCI7[-start_period:]
        vvData1_CCI8=vvData1_CCI8[-start_period:]

        signal_short = (vvData1_CCI + vvData1_CCI2+ vvData1_CCI3+ vvData1_CCI4 + vvData1_CCI5 + vvData1_CCI6 + vvData1_CCI7 + vvData1_CCI8)/8
        SB_Signal=pd.Series(signal_short).rolling(acc_period1).sum()/acc_period1

        res = SB_Signal.dropna().reset_index()[0]


        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_res = scaler.fit_transform(pd.Series(res).values.reshape(-1, 1))
        scale_res = scale_res.reshape(-1)


        return scale_res,res


    def tanos(self,close,period):
        close=close

        MA1,ori_MA1 = self.MA(close,period)
        start_period= min(len(close),len(ori_MA1))

        close=close[-start_period:].reset_index()[0]
        ori_MA1=ori_MA1[-start_period:]

        tanos11ma = np.abs(ori_MA1-close)
        MA2, tanos11 = self.MA(tanos11ma,5)
        tanos11=pd.Series(tanos11)

        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_tanos = scaler.fit_transform(pd.Series(tanos11).values.reshape(-1, 1))
        scale_tanos = scale_tanos.reshape(-1)

        return scale_tanos, tanos11.values.reshape(-1)



    def span(self,close,high,low,전환선기간, 기준선기간, 스팬2기간):

        전환선 = (high.rolling(전환선기간) + low.rolling(전환선기간))/2
        기준선 = (high.rolling(기준선기간)+low.rolling(기준선기간))/2

        start_period= min(len(전환선),len(기준선))

        전환선= 전환선[-start_period:]
        기준선= 기준선[-start_period:]

        후행스팬 = close
        선행스팬1= (전환선 + 기준선)/2
        선행스팬2 = (high.rolling(스팬2기간) + low.rolling(스팬2기간))/2

        return 전환선,기준선,후행스팬,선행스팬1,선행스팬2


    def SBS_span(self,close,high,low,전환선기간,기준선기간,스팬2기간):

        close= close
        high = high
        low = low

        전환1= high.rolling(전환선기간).max()
        전환2= low.rolling(기준선기간).min()
        기준1= high.rolling(기준선기간).max()
        기준2= low.rolling(기준선기간).min()
        선행1= high.rolling(스팬2기간).max()
        선행2= low.rolling(스팬2기간).min()

        start_period=min(len(전환1),len(전환2))
        start_period2= min(len(기준1),len(기준2))



        전환1= 전환1[-start_period:].dropna().reset_index()[0]
        전환2= 전환2[-start_period:].dropna().reset_index()[0]
        기준1= 기준1[-start_period2:].dropna().reset_index()[0]
        기준2= 기준2[-start_period2:].dropna().reset_index()[0]


        전환선 = 전환1+전환2
        전환선 = 전환선/2

        기준선 = 기준1+기준2
        기준선 = 기준선/2

        start_period3=min(len(전환선),len(기준선))
        start_period4=min(len(선행1),len(선행2))

        전환선=전환선[-start_period3:].dropna().reset_index()[0]
        기준선=기준선[-start_period3:].dropna().reset_index()[0]
        선행1= 선행1[-start_period4:].dropna().reset_index()[0]
        선행2 = 선행2[-start_period4:].dropna().reset_index()[0]

        후행스팬 =close
        선행스팬1 = (전환선+ 기준선)/2
        선행스팬2 = (선행1+선행2)/2

        start_period5=min(len(선행스팬1),len(선행스팬2))
        선행스팬1 = 선행스팬1[-start_period5:].dropna().reset_index()[0]
        선행스팬2= 선행스팬2[-start_period5:].dropna().reset_index()[0]

        scaler= MinMaxScaler(feature_range=(0.1,1))
        scale_선행스팬1= scaler.fit_transform(pd.Series(선행스팬1).values.reshape(-1,1))
        scale_선행스팬2= scaler.fit_transform(pd.Series(선행스팬2).values.reshape(-1,1))

        scale_선행스팬1=scale_선행스팬1.reshape(-1)
        scale_선행스팬2=scale_선행스팬2.reshape(-1)

        return  scale_선행스팬1 , scale_선행스팬2 , 선행스팬1.values.reshape(-1), 선행스팬2.values.reshape(-1)


    def tanos_span(self,close,전환선기간,기준선기간,스팬2기간):

        close= close

        전환1= close.rolling(전환선기간).max()
        전환2= close.rolling(기준선기간).min()
        기준1= close.rolling(기준선기간).max()
        기준2= close.rolling(기준선기간).min()
        선행1= close.rolling(스팬2기간).max()
        선행2= close.rolling(스팬2기간).min()

        start_period=min(len(전환1),len(전환2))
        start_period2= min(len(기준1),len(기준2))

        전환1= 전환1[-start_period:].dropna().reset_index()[0]
        전환2= 전환2[-start_period:].dropna().reset_index()[0]
        기준1= 기준1[-start_period2:].dropna().reset_index()[0]
        기준2= 기준2[-start_period2:].dropna().reset_index()[0]


        전환선 = 전환1+전환2
        전환선 = 전환선/2

        기준선 = 기준1+기준2
        기준선 = 기준선/2

        start_period3=min(len(전환선),len(기준선))
        start_period4=min(len(선행1),len(선행2))

        전환선=전환선[-start_period3:].dropna().reset_index()[0]
        기준선=기준선[-start_period3:].dropna().reset_index()[0]
        선행1= 선행1[-start_period4:].dropna().reset_index()[0]
        선행2 = 선행2[-start_period4:].dropna().reset_index()[0]

        후행스팬 =close
        선행스팬1 = (전환선+ 기준선)/2
        선행스팬2 = (선행1+선행2)/2

        start_period5=min(len(선행스팬1),len(선행스팬2))
        선행스팬1 = 선행스팬1[-start_period5:].dropna().reset_index()[0]
        선행스팬2= 선행스팬2[-start_period5:].dropna().reset_index()[0]

        scaler= MinMaxScaler(feature_range=(0.1,1))
        scale_선행스팬1= scaler.fit_transform(pd.Series(선행스팬1).values.reshape(-1,1))
        scale_선행스팬2= scaler.fit_transform(pd.Series(선행스팬2).values.reshape(-1,1))

        scale_선행스팬1=scale_선행스팬1.reshape(-1)
        scale_선행스팬2=scale_선행스팬2.reshape(-1)

        return  scale_선행스팬1 , scale_선행스팬2 , 선행스팬1.values.reshape(-1), 선행스팬2.values.reshape(-1)


    def choco7m_xfive(self,close,open_,high_,low_,전환선기간, 기준선기간, 스팬2기간 ,cci기간값1,vSigPeriod): #초코는 기준데이터, SNP, AUD, Gold 총 4개의 데이터가 필요하다
        SNP = self.ind_data_create(params.minute, params.data_count, params.coin_or_stock,
                                    "'NQ'")
        AUD = self.ind_data_create(params.minute, params.data_count, params.coin_or_stock,
                                    "'NQ'")
        Gold= self.ind_data_create(params.minute, params.data_count, params.coin_or_stock,
                                    "'NQ'")

        SNP_close_, SNP_open_, SNP_high_, SNP_low_, SNP_vol_, SNP_close_scale, SNP_open_scale, SNP_high_scale, SNP_low_scale, SNP_vol_scale, SNP_date = SNP
        AUD_close_, AUD_open_, AUD_high_, AUD_low_, AUD_vol_, AUD_close_scale, AUD_open_scale, AUD_high_scale, AUD_low_scale, AUD_vol_scale, AUD_date = AUD
        Gold_close_, Gold_open_, Gold_high_, Gold_low_, Gold_vol_, Gold_close_scale, Gold_open_scale, Gold_high_scale, Gold_low_scale, Gold_vol_scale, Gold_date = Gold

        high= (high_).dropna().reset_index()[0]
        low= (low_).dropna().reset_index()[0]
        close = (close).dropna().reset_index()[0]

        SNP_close = (SNP_close_).dropna().reset_index()[0]
        SNP_low= (SNP_low_).dropna().reset_index()[0]
        SNP_high =(SNP_high_).dropna().reset_index()[0]

        AUD_close = (AUD_close_).dropna().reset_index()[0]
        AUD_low = (AUD_low_).dropna().reset_index()[0]
        AUD_high= (AUD_high_).dropna().reset_index()[0]

        Gold_close = (Gold_close_).dropna().reset_index()[0]
        Gold_low= (Gold_low_).dropna().reset_index()[0]
        Gold_high= (Gold_high_).dropna().reset_index()[0]

        전환선 = (high.rolling(전환선기간).max().dropna().reset_index()[0] + low.rolling(전환선기간).min().dropna().reset_index()[0])/2
        기준선 = (high.rolling(기준선기간).max().dropna().reset_index()[0]+ low.rolling(기준선기간).min().dropna().reset_index()[0])/2
        전환선 = 전환선.dropna().reset_index()[0]
        기준선 = 기준선.dropna().reset_index()[0]

        start_period =min(len(전환선),len(기준선))
        전환선= 전환선[-start_period:].dropna().reset_index()[0]
        기준선= 기준선[-start_period:].dropna().reset_index()[0]

        후행스팬 = close
        선행스팬1 = ((전환선 + 기준선)/2).dropna().reset_index()[0]
        선행스팬2 = (high.rolling(스팬2기간).max().dropna().reset_index()[0] + low.rolling(스팬2기간).min().dropna().reset_index()[0])/2

        CCI,data_CCI=self.CCI([high,low,close],cci기간값1)
        CCI,SNP_CCI=self.CCI([SNP_high,SNP_low,SNP_close],cci기간값1)
        CCI,AUD_CCI=self.CCI([AUD_high,AUD_low,AUD_close],cci기간값1)
        CCI,Gold_CCI=self.CCI([Gold_high,Gold_low,Gold_close],cci기간값1)

        start_period2=min(len(data_CCI),len(SNP_CCI),len(AUD_CCI),len(Gold_CCI))
        data_CCI=data_CCI[-start_period2:].dropna().reset_index()[0]
        SNP_CCI=SNP_CCI[-start_period2:].dropna().reset_index()[0]
        AUD_CCI=AUD_CCI[-start_period2:].dropna().reset_index()[0]
        Gold_CCI= Gold_CCI[-start_period2:].dropna().reset_index()[0]

        _,vsig1 = self.EMA(AUD_CCI,vSigPeriod)
        _,vsig2 = self.EMA(Gold_CCI,vSigPeriod)

        start_period3  = min(len(AUD_CCI),len(Gold_CCI),len(vsig1),len(vsig2))
        AUD_CCI= AUD_CCI[-start_period3:].reset_index()[0]
        Gold_CCI= Gold_CCI[-start_period3:].reset_index()[0]
        vsig1= vsig1[-start_period3:].reset_index()[0]
        vsig2= vsig2[-start_period3:].reset_index()[0]

        vgap= AUD_CCI-Gold_CCI
        vsiggap=vsig1-vsig2
        value99= np.abs(data_CCI-SNP_CCI)


        '''''
        plt.plot(close[-2000:])
        plt.show()
        plt.plot(AUD_close[-2000:])
        plt.show()
        plt.plot(vsig1[-2000:])
        plt.show()
        plt.plot(vsig2[-2000:])
        plt.show()
        plt.plot(data_CCI[-2000:])
        plt.show()
        plt.plot(SNP_CCI[-2000:])
        plt.show()
        plt.plot(AUD_CCI[-2000:])
        plt.show()
        plt.plot(vgap[-2000:])
        plt.show()
        plt.plot(vsiggap[-2000:])
        plt.show()
        plt.plot(value99[-2000:])
        plt.show()
        plt.plot(close[-2000:].dropna().reset_index()[0])
        plt.plot(선행스팬1[-2000:].dropna().reset_index()[0])
        plt.plot(선행스팬2[-2000:].dropna().reset_index()[0])
        plt.show()
        '''''

        scaler=MinMaxScaler(feature_range=(0.1,1))
        s_선행스팬1= scaler.fit_transform(pd.Series(선행스팬1).values.reshape(-1,1)).reshape(-1)
        s_선행스팬2= scaler.fit_transform(pd.Series(선행스팬2).values.reshape(-1,1)).reshape(-1)
        s_vgap= scaler.fit_transform(pd.Series(vgap).values.reshape(-1,1)).reshape(-1)
        s_vsiggap= scaler.fit_transform(pd.Series(vsiggap).values.reshape(-1,1)).reshape(-1)
        s_value99=scaler.fit_transform(pd.Series(value99).values.reshape(-1,1)).reshape(-1)

        return data_CCI,SNP_CCI,선행스팬1,선행스팬2, vgap, vsiggap, value99, s_선행스팬1, s_선행스팬2, s_vgap, s_vsiggap, s_value99



    def LALA2_3m_V4(self,close,open_,high_,low_,CCI기간값1,CCI기간값2, CCI기간값3,
                    CCI기간값4,CCI기간값5, CCI기간값6, CCI기간값7, CCI기간값8,vAccPeriod2,# 추세 기간값
                    전환선기간, 기준선기간, 스팬2기간,
                    qMultipleH1,qMultipleH2,qMultipleH3,qMultipleH4,qMultipleH5,qMultipleH6,qMultipleH7,qMultipleH8,
                    qCCI기간값1, qCCI기간값2, qCCI기간값3, qCCI기간값4, qCCI기간값5, qCCI기간값6,qCCI기간값7,
                    qCCI기간값8,qAccPeriod1):

        #####기본식 8시부터 21시까지 거래, 1시부터 8시까지 거래( 추세 신호)
        close = close   # 포인트 벨류 없엔다
        open = open_
        high = high_
        low = low_

        data = [high, low, close]

        CCI1, vData1_CCI = self.CCI(data, CCI기간값1)
        CCI2, vData1_CCI2 = self.CCI(data, CCI기간값2)
        CCI3, vData1_CCI3 = self.CCI(data, CCI기간값3)
        CCI4, vData1_CCI4 = self.CCI(data, CCI기간값4)
        CCI5, vData1_CCI5 = self.CCI(data, CCI기간값5)
        CCI6, vData1_CCI6 = self.CCI(data, CCI기간값6)
        CCI7, vData1_CCI7 = self.CCI(data, CCI기간값7)
        CCI8, vData1_CCI8 = self.CCI(data, CCI기간값8)

        vCCI1, vvData1_CCI = self.slope_line(vData1_CCI, CCI기간값1)
        vCCI2, vvData1_CCI2 = self.slope_line(vData1_CCI2, CCI기간값2)
        vCCI3, vvData1_CCI3 = self.slope_line(vData1_CCI3, CCI기간값3)
        vCCI4, vvData1_CCI4 = self.slope_line(vData1_CCI4, CCI기간값4)
        vCCI5, vvData1_CCI5 = self.slope_line(vData1_CCI5, CCI기간값5)
        vCCI6, vvData1_CCI6 = self.slope_line(vData1_CCI6, CCI기간값6)
        vCCI7, vvData1_CCI7 = self.slope_line(vData1_CCI7, CCI기간값7)
        vCCI8, vvData1_CCI8 = self.slope_line(vData1_CCI8, CCI기간값8)


        start_period=min(len(vvData1_CCI),len(vvData1_CCI2),len(vvData1_CCI3),len(vvData1_CCI4),len(vvData1_CCI5),len(vvData1_CCI6),len(vvData1_CCI7),len(vvData1_CCI8))

        vvData1_CCI=  pd.Series(vvData1_CCI)[-start_period:].dropna().reset_index()[0]
        vvData1_CCI2= pd.Series(vvData1_CCI2)[-start_period:].dropna().reset_index()[0]
        vvData1_CCI3 = pd.Series(vvData1_CCI3)[-start_period:].dropna().reset_index()[0]
        vvData1_CCI4 = pd.Series(vvData1_CCI4)[-start_period:].dropna().reset_index()[0]
        vvData1_CCI5 = pd.Series(vvData1_CCI5)[-start_period:].dropna().reset_index()[0]
        vvData1_CCI6 = pd.Series(vvData1_CCI6)[-start_period:].dropna().reset_index()[0]
        vvData1_CCI7 = pd.Series(vvData1_CCI7)[-start_period:].dropna().reset_index()[0]
        vvData1_CCI8 = pd.Series(vvData1_CCI8)[-start_period:].dropna().reset_index()[0]

        signal_short= (vvData1_CCI+vvData1_CCI2+vvData1_CCI3+ vvData1_CCI4+vvData1_CCI5+vvData1_CCI6+vvData1_CCI7+vvData1_CCI8)/8
        CCI_LRLA  = signal_short.rolling(vAccPeriod2).sum().dropna().reset_index()[0]

        scaler=MinMaxScaler(feature_range=(0.1,1))
        s_CCI_LRLA= scaler.fit_transform(pd.Series(CCI_LRLA).values.reshape(-1,1))
        s_CCI_LRLA = s_CCI_LRLA.reshape(-1)


        ########역추세 신호
        전환선 = (high.rolling(전환선기간).max().dropna().reset_index()[0] + low.rolling(전환선기간).min().dropna().reset_index()[0]) / 2
        기준선 = (high.rolling(기준선기간).max().dropna().reset_index()[0] + low.rolling(기준선기간).min().dropna().reset_index()[0]) / 2
        전환선 = 전환선.dropna().reset_index()[0]
        기준선 = 기준선.dropna().reset_index()[0]

        start_period = min(len(전환선), len(기준선))
        전환선 = 전환선[-start_period:].dropna().reset_index()[0]
        기준선 = 기준선[-start_period:].dropna().reset_index()[0]

        후행스팬 = close
        선행스팬1 = ((전환선 + 기준선) / 2).dropna().reset_index()[0]
        선행스팬2 = (high.rolling(스팬2기간).max().dropna().reset_index()[0] + low.rolling(스팬2기간).min().dropna().reset_index()[0]) / 2

        _,qL1= self.slope_line(close,qMultipleH1)
        _,qL2= self.slope_line(close,qMultipleH2)
        _,qL3= self.slope_line(close,qMultipleH3)
        _,qL4= self.slope_line(close,qMultipleH4)
        _,qL5= self.slope_line(close,qMultipleH5)
        _,qL6= self.slope_line(close,qMultipleH6)
        _,qL7= self.slope_line(close,qMultipleH7)
        _,qL8 = self.slope_line(close,qMultipleH8)


        start_period=min(len(qL1),len(qL2),len(qL3),len(qL4),len(qL5),len(qL6),len(qL7),len(qL8))
        qL1=qL1[-start_period:]
        qL2=qL2[-start_period:]
        qL3=qL3[-start_period:]
        qL4=qL4[-start_period:]
        qL5=qL5[-start_period:]
        qL6=qL6[-start_period:]
        qL7=qL7[-start_period:]
        qL8=qL8[-start_period:]

        qMultipleL1= (qL1+ qL2+ qL3+ qL4+ qL5+ qL6 + qL7 + qL8)/8

        CCI1, qData1_CCI = self.CCI(close, qCCI기간값1)
        CCI2, qData1_CCI2 = self.CCI(close, qCCI기간값2)
        CCI3, qData1_CCI3 = self.CCI(close, qCCI기간값3)
        CCI4, qData1_CCI4 = self.CCI(close, qCCI기간값4)
        CCI5, qData1_CCI5 = self.CCI(close, qCCI기간값5)
        CCI6, qData1_CCI6 = self.CCI(close, qCCI기간값6)
        CCI7, qData1_CCI7 = self.CCI(close, qCCI기간값7)
        CCI8, qData1_CCI8 = self.CCI(close, qCCI기간값8)

        vCCI1, qqData1_CCI = self.slope_line(qData1_CCI, qMultipleH1)
        vCCI2, qqData1_CCI2 = self.slope_line(qData1_CCI2, qMultipleH2)
        vCCI3, qqData1_CCI3 = self.slope_line(qData1_CCI3, qMultipleH3)
        vCCI4, qqData1_CCI4 = self.slope_line(qData1_CCI4, qMultipleH4)
        vCCI5, qqData1_CCI5 = self.slope_line(qData1_CCI5, qMultipleH5)
        vCCI6, qqData1_CCI6 = self.slope_line(qData1_CCI6, qMultipleH6)
        vCCI7, qqData1_CCI7 = self.slope_line(qData1_CCI7, qMultipleH7)
        vCCI8, qqData1_CCI8 = self.slope_line(qData1_CCI8, qMultipleH8)

        start_period2= min(len(qqData1_CCI), len(qqData1_CCI2), len(qqData1_CCI3),len(qqData1_CCI4),len(qqData1_CCI5),
                           len(qqData1_CCI6), len(qqData1_CCI7), len(qqData1_CCI8))

        qqData1_CCI= qqData1_CCI[-start_period2:]
        qqData1_CCI2= qqData1_CCI2[-start_period2:]
        qqData1_CCI3 = qqData1_CCI3[-start_period2:]
        qqData1_CCI4 = qqData1_CCI4[-start_period2:]
        qqData1_CCI5 = qqData1_CCI5[-start_period2:]
        qqData1_CCI6 = qqData1_CCI6[-start_period2:]
        qqData1_CCI7 = qqData1_CCI7[-start_period2:]
        qqData1_CCI8 = qqData1_CCI8[-start_period2:]

        qSignal_short = (qqData1_CCI + qqData1_CCI2 + qqData1_CCI3 + qqData1_CCI4 + qqData1_CCI5 + qqData1_CCI6 + qqData1_CCI7 + qqData1_CCI8)/8
        q8_SigshortA = pd.Series(qSignal_short).rolling(qAccPeriod1).sum()

        scaler=MinMaxScaler(feature_range=(0.1,1))
        s_q8_SigshortA = scaler.fit_transform(pd.Series(q8_SigshortA).values.reshape(-1,1))
        s_q8_SigshortA = s_q8_SigshortA.reshape(-1)


        return s_CCI_LRLA, CCI_LRLA, s_q8_SigshortA , q8_SigshortA, 선행스팬1, 선행스팬2


        #이프 CCI_LRLA >CCI_LRLA [step-1] 이면 COND=1
        #이프 CCI_LRLA< 이면 COND=1



    def LRC_SBSignal_15m_2(self,close,open_,high_,low_,vM1,cci_period1,cci_period2,cci_period3,cci_period4,cci_period5, cci_period6,cci_period7, cci_period8,
               acc_period1): #LRC 15분과 같은거(이름만 다름)

         # 15,100,200,400,600,800,1000,1200,1500,2,18,52,104
        close=close # 포인트 벨류 없엔다
        high=high_
        low=low_

        data=[high,low,close]

        CCI1,vData1_CCI= self.CCI(data,cci_period1)
        CCI2,vData1_CCI2=self.CCI(data,cci_period2)
        CCI3,vData1_CCI3=self.CCI(data,cci_period3)
        CCI4,vData1_CCI4=self.CCI(data,cci_period4)
        CCI5,vData1_CCI5=self.CCI(data,cci_period5)
        CCI6,vData1_CCI6=self.CCI(data,cci_period6)
        CCI7,vData1_CCI7=self.CCI(data,cci_period7)
        CCI8,vData1_CCI8=self.CCI(data,cci_period8)


        vCCI1,vvData1_CCI=self.slope_line(vData1_CCI,vM1)
        vCCI2,vvData1_CCI2=self.slope_line(vData1_CCI2,vM1)
        vCCI3,vvData1_CCI3=self.slope_line(vData1_CCI3,vM1)
        vCCI4,vvData1_CCI4=self.slope_line(vData1_CCI4,vM1)
        vCCI5,vvData1_CCI5=self.slope_line(vData1_CCI5,vM1)
        vCCI6,vvData1_CCI6=self.slope_line(vData1_CCI6,vM1)
        vCCI7,vvData1_CCI7=self.slope_line(vData1_CCI7,vM1)
        vCCI8,vvData1_CCI8=self.slope_line(vData1_CCI8,vM1)

        start_period=min(len(vvData1_CCI),len(vvData1_CCI2),len(vvData1_CCI3),len(vvData1_CCI4),
                        len(vvData1_CCI5),len(vvData1_CCI6),len(vvData1_CCI7),len(vvData1_CCI8))

        vvData1_CCI= vvData1_CCI[-start_period:]
        vvData1_CCI2=vvData1_CCI2[-start_period:]
        vvData1_CCI3=vvData1_CCI3[-start_period:]
        vvData1_CCI4=vvData1_CCI4[-start_period:]
        vvData1_CCI5=vvData1_CCI5[-start_period:]
        vvData1_CCI6=vvData1_CCI6[-start_period:]
        vvData1_CCI7=vvData1_CCI7[-start_period:]
        vvData1_CCI8=vvData1_CCI8[-start_period:]


        signal_short = (vvData1_CCI + vvData1_CCI2+ vvData1_CCI3+ vvData1_CCI4 + vvData1_CCI5 + vvData1_CCI6 + vvData1_CCI7 + vvData1_CCI8)/8
        SB_Signal=pd.Series(signal_short).rolling(acc_period1).sum()/acc_period1

        SB_signal = SB_Signal.dropna().reset_index()[0]

        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_SB_sig = scaler.fit_transform(pd.Series(SB_signal).values.reshape(-1, 1))
        scale_SB_sig = scale_SB_sig.reshape(-1)


        return scale_SB_sig,SB_signal



    def SBS_signal_Sigshort(self,close,open_,high_,low_,vM1,cci_period1,cci_period2,cci_period3,cci_period4,cci_period5, cci_period6,cci_period7, cci_period8,acc_period1):

        close = close   # 포인트 벨류 없엔다
        high = high_
        low = low_

        data = [high, low, close]

        CCI1, vData1_CCI = self.CCI(close, cci_period1)
        CCI2, vData1_CCI2 = self.CCI(close, cci_period2)
        CCI3, vData1_CCI3 = self.CCI(close, cci_period3)
        CCI4, vData1_CCI4 = self.CCI(close, cci_period4)
        CCI5, vData1_CCI5 = self.CCI(close, cci_period5)
        CCI6, vData1_CCI6 = self.CCI(close, cci_period6)
        CCI7, vData1_CCI7 = self.CCI(close, cci_period7)
        CCI8, vData1_CCI8 = self.CCI(close, cci_period8)

        MA1, MA_CCI1 = self.MA(vData1_CCI, cci_period1)
        MA2, MA_CCI2 = self.MA(vData1_CCI2, cci_period2)
        MA3, MA_CCI3 = self.MA(vData1_CCI3, cci_period3)
        MA4, MA_CCI4 = self.MA(vData1_CCI4, cci_period4)
        MA5, MA_CCI5 = self.MA(vData1_CCI5, cci_period5)
        MA6, MA_CCI6 = self.MA(vData1_CCI6, cci_period6)
        MA7, MA_CCI7 = self.MA(vData1_CCI7, cci_period7)
        MA8, MA_CCI8 = self.MA(vData1_CCI8, cci_period8)

        STD1,STD_CCI1 = self.STD(vData1_CCI,cci_period1)
        STD2,STD_CCI2 = self.STD(vData1_CCI2,cci_period2)
        STD3,STD_CCI3 = self.STD(vData1_CCI3,cci_period3)
        STD4,STD_CCI4 = self.STD(vData1_CCI4,cci_period4)
        STD5,STD_CCI5 = self.STD(vData1_CCI5,cci_period5)
        STD6,STD_CCI6 = self.STD(vData1_CCI6,cci_period6)
        STD7,STD_CCI7 = self.STD(vData1_CCI7,cci_period7)
        STD8,STD_CCI8 = self.STD(vData1_CCI8,cci_period8)

        equal_res = self.period_equal([vData1_CCI,vData1_CCI2,vData1_CCI3,vData1_CCI4,vData1_CCI5,vData1_CCI6,vData1_CCI7,vData1_CCI8,
                           MA_CCI1,MA_CCI2,MA_CCI3,MA_CCI4,MA_CCI5,MA_CCI6,MA_CCI7,MA_CCI8,
                           STD_CCI1, STD_CCI2, STD_CCI3, STD_CCI4, STD_CCI5, STD_CCI6, STD_CCI7, STD_CCI8])

        vData1_CCI = equal_res[0]
        vData1_CCI2 = equal_res[1]
        vData1_CCI3 = equal_res[2]
        vData1_CCI4 = equal_res[3]
        vData1_CCI5 = equal_res[4]
        vData1_CCI6 = equal_res[5]
        vData1_CCI7 = equal_res[6]
        vData1_CCI8 = equal_res[7]

        MA_CCI1 = equal_res[8]
        MA_CCI2 = equal_res[9]
        MA_CCI3 = equal_res[10]
        MA_CCI4 = equal_res[11]
        MA_CCI5 = equal_res[12]
        MA_CCI6 = equal_res[13]
        MA_CCI7 = equal_res[14]
        MA_CCI8 = equal_res[15]

        STD_CCI1 = equal_res[16]
        STD_CCI2 = equal_res[17]
        STD_CCI3 = equal_res[18]
        STD_CCI4 = equal_res[19]
        STD_CCI5 = equal_res[20]
        STD_CCI6 = equal_res[21]
        STD_CCI7 = equal_res[22]
        STD_CCI8 = equal_res[23]

        vData1_CCI_N1 = (vData1_CCI-MA_CCI1)/STD_CCI1
        vData1_CCI_N2 = (vData1_CCI2-MA_CCI2)/STD_CCI2
        vData1_CCI_N3 = (vData1_CCI3-MA_CCI3)/STD_CCI3
        vData1_CCI_N4 = (vData1_CCI4-MA_CCI4)/STD_CCI4
        vData1_CCI_N5 = (vData1_CCI5-MA_CCI5)/STD_CCI5
        vData1_CCI_N6 = (vData1_CCI6-MA_CCI6)/STD_CCI6
        vData1_CCI_N7 = (vData1_CCI7-MA_CCI7)/STD_CCI7
        vData1_CCI_N8 = (vData1_CCI8-MA_CCI8)/STD_CCI8

        vCCI1, vvData1_CCI = self.slope_line(vData1_CCI_N1, vM1)
        vCCI2, vvData1_CCI2 = self.slope_line(vData1_CCI_N2, vM1)
        vCCI3, vvData1_CCI3 = self.slope_line(vData1_CCI_N3, vM1)
        vCCI4, vvData1_CCI4 = self.slope_line(vData1_CCI_N4, vM1)
        vCCI5, vvData1_CCI5 = self.slope_line(vData1_CCI_N5, vM1)
        vCCI6, vvData1_CCI6 = self.slope_line(vData1_CCI_N6, vM1)
        vCCI7, vvData1_CCI7 = self.slope_line(vData1_CCI_N7, vM1)
        vCCI8, vvData1_CCI8 = self.slope_line(vData1_CCI_N8, vM1)

        Signal_Short = (vvData1_CCI + vvData1_CCI2 + vvData1_CCI3 + vvData1_CCI4 + vvData1_CCI5 + vvData1_CCI6 + vvData1_CCI7 +vvData1_CCI8)/8
        S8_SigShortA = self.AccumN(Signal_Short,acc_period1)
        S8_SigShortA = S8_SigShortA/acc_period1
        ori_SigshortA = S8_SigShortA.dropna().reset_index()[0]

        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_S8_sig = scaler.fit_transform(pd.Series(ori_SigshortA).values.reshape(-1, 1))
        scale_S8_sig = scale_S8_sig.reshape(-1)

        return scale_S8_sig, ori_SigshortA.values.reshape(-1)   #s8_sigshortA


    def SBS_signal_Sigshort_long2(self,close,open_,high_,low_,vM1,cci_period1,cci_period2,cci_period3,cci_period4,cci_period5, cci_period6,cci_period7, cci_period8,acc_period1):
        # long2 에 사용되는 Sigshort
        close = close  # 포인트 벨류 없엔다
        high = high_
        low = low_


        data = [high, low, close]

        CCI1, vData1_CCI = self.CCI(close, cci_period1)
        CCI2, vData1_CCI2 = self.CCI(close, cci_period2)
        CCI3, vData1_CCI3 = self.CCI(close, cci_period3)
        CCI4, vData1_CCI4 = self.CCI(close, cci_period4)
        CCI5, vData1_CCI5 = self.CCI(close, cci_period5)
        CCI6, vData1_CCI6 = self.CCI(close, cci_period6)
        CCI7, vData1_CCI7 = self.CCI(close, cci_period7)
        CCI8, vData1_CCI8 = self.CCI(close, cci_period8)

        equal_res = self.period_equal([vData1_CCI,vData1_CCI2,vData1_CCI3,vData1_CCI4,vData1_CCI5,vData1_CCI6,vData1_CCI7,vData1_CCI8])

        vData1_CCI = equal_res[0]
        vData1_CCI2 = equal_res[1]
        vData1_CCI3 = equal_res[2]
        vData1_CCI4 = equal_res[3]
        vData1_CCI5 = equal_res[4]
        vData1_CCI6 = equal_res[5]
        vData1_CCI7 = equal_res[6]
        vData1_CCI8 = equal_res[7]

        vData1_CCI_N1 = vData1_CCI
        vData1_CCI_N2 = vData1_CCI2
        vData1_CCI_N3 = vData1_CCI3
        vData1_CCI_N4 = vData1_CCI4
        vData1_CCI_N5 = vData1_CCI5
        vData1_CCI_N6 = vData1_CCI6
        vData1_CCI_N7 = vData1_CCI7
        vData1_CCI_N8 = vData1_CCI8

        vCCI1, vvData1_CCI = self.slope_line(vData1_CCI_N1, vM1)
        vCCI2, vvData1_CCI2 = self.slope_line(vData1_CCI_N2, vM1)
        vCCI3, vvData1_CCI3 = self.slope_line(vData1_CCI_N3, vM1)
        vCCI4, vvData1_CCI4 = self.slope_line(vData1_CCI_N4, vM1)
        vCCI5, vvData1_CCI5 = self.slope_line(vData1_CCI_N5, vM1)
        vCCI6, vvData1_CCI6 = self.slope_line(vData1_CCI_N6, vM1)
        vCCI7, vvData1_CCI7 = self.slope_line(vData1_CCI_N7, vM1)
        vCCI8, vvData1_CCI8 = self.slope_line(vData1_CCI_N8, vM1)

        Signal_Short = (vvData1_CCI + vvData1_CCI2 + vvData1_CCI3 + vvData1_CCI4 + vvData1_CCI5 + vvData1_CCI6 + vvData1_CCI7 +vvData1_CCI8)/8
        S8_SigShortA = self.AccumN(Signal_Short,acc_period1)
        ori_SigshortA = S8_SigShortA.dropna().reset_index()[0]

        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_S8_sig = scaler.fit_transform(pd.Series(ori_SigshortA).values.reshape(-1, 1))
        scale_S8_sig = scale_S8_sig.reshape(-1)


        return scale_S8_sig, ori_SigshortA.values.reshape(-1)   #s8_sigshortA



    def SBS_signal_NTRI(self,close,high,low,open,cci기간값1,cci기간값2, cci기간값3, cci기간값4, cci기간값5, cci기간값6,cci기간값7,cci기간값8,vM1,NTRIP1):


        data5=close
        Gold_open_=open
        Gold_high_=high
        Gold_low_=low

        data3=close
        VIX_open_=open
        VIX_high_=high
        VIX_low_=low

        data2=close
        USD_open_=open
        USD_high_=high
        USD_low_=low

        data2 = (data2 ).dropna().reset_index()[0]
        USD_high_=(USD_high_).dropna().reset_index()[0]
        USD_low_= (USD_low_).dropna().reset_index()[0]

        data3 = (data3 ).dropna().reset_index()[0]
        VIX_high_=(VIX_high_).dropna().reset_index()[0]
        VIX_low_ =(VIX_low_).dropna().reset_index()[0]

        data5 = (data5 ).dropna().reset_index()[0]
        Gold_high_=(Gold_high_).dropna().reset_index()[0]
        Gold_low_=(Gold_low_).dropna().reset_index()[0]

        data2 = [USD_high_,USD_low_,data2]
        data3 = [VIX_high_, VIX_low_,data3]
        data5 = [Gold_high_, Gold_low_, data5]

        _, vUSD_CCI= self.CCI(data2,cci기간값1) #달러
        _, vVIX_CCI= self.CCI(data3,cci기간값1) #vix
        _, vGL_CCI = self.CCI(data5,cci기간값1) #금


        _, vUSD_CCI2= self.CCI(data2,cci기간값2) # 달러
        _, vVIX_CCI2 = self.CCI(data3, cci기간값2)
        _, vGL_CCI2 = self.CCI(data5,cci기간값2)

        _, vUSD_CCI3 = self.CCI(data2, cci기간값3)  # 달러
        _, vVIX_CCI3 = self.CCI(data3, cci기간값3)
        _, vGL_CCI3= self.CCI(data5, cci기간값3)

        _, vUSD_CCI4 = self.CCI(data2, cci기간값4)  # 달러
        _, vVIX_CCI4 = self.CCI(data3, cci기간값4)
        _, vGL_CCI4 = self.CCI(data5, cci기간값4)

        _, vUSD_CCI5 = self.CCI(data2, cci기간값5)  # 달러
        _, vVIX_CCI5 = self.CCI(data3, cci기간값5)
        _, vGL_CCI5= self.CCI(data5, cci기간값5)

        _, vUSD_CCI6 = self.CCI(data2, cci기간값6)  # 달러
        _, vVIX_CCI6 = self.CCI(data3, cci기간값6)
        _, vGL_CCI6= self.CCI(data5, cci기간값6)

        _, vUSD_CCI7 = self.CCI(data2, cci기간값7)  # 달러
        _, vVIX_CCI7 = self.CCI(data3, cci기간값7)
        _, vGL_CCI7= self.CCI(data5, cci기간값7)

        _, vUSD_CCI8 = self.CCI(data2, cci기간값8)  # 달러
        _, vVIX_CCI8 = self.CCI(data3, cci기간값8)
        _, vGL_CCI8 = self.CCI(data5, cci기간값8)


        _, STD_USD= self.STD(vUSD_CCI,cci기간값1)
        _, STD_VIX= self.STD(vVIX_CCI,cci기간값1)
        _, STD_GL= self.STD(vGL_CCI, cci기간값1)

        _, STD_USD2 = self.STD(vUSD_CCI2, cci기간값2)
        _, STD_VIX2 = self.STD(vVIX_CCI2, cci기간값2)
        _, STD_GL2 = self.STD(vGL_CCI2, cci기간값2)

        _, STD_USD3 = self.STD(vUSD_CCI3, cci기간값3)
        _, STD_VIX3 = self.STD(vVIX_CCI3, cci기간값3)
        _, STD_GL3 = self.STD(vGL_CCI3, cci기간값3)

        _, STD_USD4 = self.STD(vUSD_CCI4, cci기간값4)
        _, STD_VIX4 = self.STD(vVIX_CCI4, cci기간값4)
        _, STD_GL4 = self.STD(vGL_CCI4, cci기간값4)

        _, STD_USD5 = self.STD(vUSD_CCI5, cci기간값5)
        _, STD_VIX5 = self.STD(vVIX_CCI5, cci기간값5)
        _, STD_GL5 = self.STD(vGL_CCI5, cci기간값5)

        _, STD_USD6 = self.STD(vUSD_CCI6, cci기간값6)
        _, STD_VIX6 = self.STD(vVIX_CCI6, cci기간값6)
        _, STD_GL6 = self.STD(vGL_CCI6, cci기간값6)

        _, STD_USD7 = self.STD(vUSD_CCI7, cci기간값7)
        _, STD_VIX7 = self.STD(vVIX_CCI7, cci기간값7)
        _, STD_GL7 = self.STD(vGL_CCI7, cci기간값7)

        _, STD_USD8 = self.STD(vUSD_CCI8, cci기간값8)
        _, STD_VIX8 = self.STD(vVIX_CCI8, cci기간값8)
        _, STD_GL8 = self.STD(vGL_CCI8, cci기간값8)



        _, MA_USD = self.MA(vUSD_CCI,cci기간값1)
        _, MA_VIX = self.MA(vVIX_CCI,cci기간값1)
        _, MA_GL = self.MA(vGL_CCI, cci기간값1)

        _, MA_USD2 = self.MA(vUSD_CCI2, cci기간값2)
        _, MA_VIX2 = self.MA(vVIX_CCI2, cci기간값2)
        _, MA_GL2 = self.MA(vGL_CCI2, cci기간값2)

        _, MA_USD3 = self.MA(vUSD_CCI3, cci기간값3)
        _, MA_VIX3 = self.MA(vVIX_CCI3, cci기간값3)
        _, MA_GL3 = self.MA(vGL_CCI3, cci기간값3)

        _, MA_USD4 = self.MA(vUSD_CCI4, cci기간값4)
        _, MA_VIX4 = self.MA(vVIX_CCI4, cci기간값4)
        _, MA_GL4 = self.MA(vGL_CCI4, cci기간값4)

        _, MA_USD5 = self.MA(vUSD_CCI5, cci기간값5)
        _, MA_VIX5 = self.MA(vVIX_CCI5, cci기간값5)
        _, MA_GL5 = self.MA(vGL_CCI5, cci기간값5)

        _, MA_USD6 = self.MA(vUSD_CCI6, cci기간값6)
        _, MA_VIX6 = self.MA(vVIX_CCI6, cci기간값6)
        _, MA_GL6 = self.MA(vGL_CCI6, cci기간값6)

        _, MA_USD7 = self.MA(vUSD_CCI7, cci기간값7)
        _, MA_VIX7 = self.MA(vVIX_CCI7, cci기간값7)
        _, MA_GL7 = self.MA(vGL_CCI7, cci기간값7)

        _, MA_USD8 = self.MA(vUSD_CCI8, cci기간값8)
        _, MA_VIX8 = self.MA(vVIX_CCI8, cci기간값8)
        _, MA_GL8 = self.MA(vGL_CCI8, cci기간값8)



        equal_res= self.period_equal([vUSD_CCI,vUSD_CCI2,vUSD_CCI3,vUSD_CCI4,vUSD_CCI5,vUSD_CCI6,vUSD_CCI7,vUSD_CCI8
                           ,vVIX_CCI,vVIX_CCI2,vVIX_CCI3,vVIX_CCI4,vVIX_CCI5,vVIX_CCI6,vVIX_CCI7,vVIX_CCI8,
                           vGL_CCI,vGL_CCI2,vGL_CCI3,vGL_CCI4,vGL_CCI5,vGL_CCI6,vGL_CCI7,vGL_CCI8,
                                      MA_USD,MA_USD2,MA_USD3,MA_USD4,MA_USD5,MA_USD6,MA_USD7,MA_USD8,
                                      MA_VIX,MA_VIX2,MA_VIX3,MA_VIX4,MA_VIX5,MA_VIX6,MA_VIX7,MA_VIX8,
                                      MA_GL,MA_GL2,MA_GL3,MA_GL4,MA_GL5,MA_GL6,MA_GL7,MA_GL8,
                                      STD_USD,STD_USD2,STD_USD3,STD_USD4,STD_USD5,STD_USD6,STD_USD7,STD_USD8,
                                      STD_VIX,STD_VIX2,STD_VIX3,STD_VIX4,STD_VIX5,STD_VIX6,STD_VIX7,STD_VIX8,
                                      STD_GL,STD_GL2,STD_GL3,STD_GL4,STD_GL5,STD_GL6,STD_GL7,STD_GL8])

        vUSD_CCI, vUSD_CCI2, vUSD_CCI3, vUSD_CCI4, vUSD_CCI5, vUSD_CCI6, vUSD_CCI7, vUSD_CCI8,vVIX_CCI, vVIX_CCI2, vVIX_CCI3, vVIX_CCI4, vVIX_CCI5, vVIX_CCI6, vVIX_CCI7, vVIX_CCI8,vGL_CCI, vGL_CCI2, vGL_CCI3, vGL_CCI4, vGL_CCI5, vGL_CCI6, vGL_CCI7, vGL_CCI8,MA_USD, MA_USD2, MA_USD3, MA_USD4, MA_USD5, MA_USD6, MA_USD7, MA_USD8,MA_VIX, MA_VIX2, MA_VIX3, MA_VIX4, MA_VIX5, MA_VIX6, MA_VIX7, MA_VIX8,MA_GL, MA_GL2, MA_GL3, MA_GL4, MA_GL5, MA_GL6, MA_GL7, MA_GL8,STD_USD, STD_USD2, STD_USD3, STD_USD4, STD_USD5, STD_USD6, STD_USD7, STD_USD8,STD_VIX, STD_VIX2, STD_VIX3, STD_VIX4, STD_VIX5, STD_VIX6, STD_VIX7, STD_VIX8,STD_GL, STD_GL2, STD_GL3, STD_GL4, STD_GL5, STD_GL6, STD_GL7, STD_GL8 = equal_res

        USD_std = (vUSD_CCI-MA_USD) / STD_USD
        USD_std2 = (vUSD_CCI2 - MA_USD2) / STD_USD2
        USD_std3 = (vUSD_CCI3 - MA_USD3) / STD_USD3
        USD_std4 = (vUSD_CCI4 - MA_USD4) / STD_USD4
        USD_std5 = (vUSD_CCI5 - MA_USD5) / STD_USD5
        USD_std6 = (vUSD_CCI6 - MA_USD6) / STD_USD6
        USD_std7 = (vUSD_CCI7 - MA_USD7) / STD_USD7
        USD_std8 = (vUSD_CCI8 - MA_USD8) / STD_USD8

        VIX_std = (vVIX_CCI - MA_VIX) / STD_VIX
        VIX_std2 = (vVIX_CCI2 - MA_VIX2) / STD_VIX2
        VIX_std3 = (vVIX_CCI3 - MA_VIX3) / STD_VIX3
        VIX_std4 = (vVIX_CCI4 - MA_VIX4) / STD_VIX4
        VIX_std5 = (vVIX_CCI5 - MA_VIX5) / STD_VIX5
        VIX_std6 = (vVIX_CCI6 - MA_VIX6) / STD_VIX6
        VIX_std7 = (vVIX_CCI7 - MA_VIX7) / STD_VIX7
        VIX_std8 = (vVIX_CCI8 -  MA_VIX8) / STD_VIX8

        GL_std = (vGL_CCI - MA_GL) / STD_GL
        GL_std2 = (vGL_CCI2 - MA_GL2) / STD_GL2
        GL_std3 = (vGL_CCI3 - MA_GL3) / STD_GL3
        GL_std4 = (vGL_CCI4 - MA_GL4) / STD_GL4
        GL_std5 = (vGL_CCI5 - MA_GL5) / STD_GL5
        GL_std6 = (vGL_CCI6 - MA_GL6) / STD_GL6
        GL_std7 = (vGL_CCI7 - MA_GL7) / STD_GL7
        GL_std8 = (vGL_CCI8 - MA_GL8) / STD_GL8

        vFinalAvg = ((VIX_std + GL_std) + (USD_std*2))/4
        vFinalAvg2 = ((VIX_std2 + GL_std2) + (USD_std2 * 2)) / 4
        vFinalAvg3 = ((VIX_std3 + GL_std3) + (USD_std3 * 2)) / 4
        vFinalAvg4 = ((VIX_std4 + GL_std4) + (USD_std4 * 2)) / 4
        vFinalAvg5 = ((VIX_std5 + GL_std5) + (USD_std5 * 2)) / 4
        vFinalAvg6 = ((VIX_std6 + GL_std6) + (USD_std6 * 2)) / 4
        vFinalAvg7 = ((VIX_std7 + GL_std7) + (USD_std7 * 2)) / 4
        vFinalAvg8 = ((VIX_std8 + GL_std8) + (USD_std8 * 2)) / 4

        _, vTRI = self.slope_line(pd.Series(vFinalAvg), vM1)
        _, vTRI_2= self.slope_line(pd.Series(vFinalAvg2),vM1)
        _, vTRI_3 = self.slope_line(pd.Series(vFinalAvg3), vM1)
        _, vTRI_4 = self.slope_line(pd.Series(vFinalAvg4), vM1)
        _, vTRI_5 = self.slope_line(pd.Series(vFinalAvg5), vM1)
        _, vTRI_6 = self.slope_line(pd.Series(vFinalAvg6), vM1)
        _, vTRI_7 = self.slope_line(pd.Series(vFinalAvg7), vM1)
        _, vTRI_8 = self.slope_line(pd.Series(vFinalAvg8), vM1)

        NTRI = -(vTRI+ vTRI_2+vTRI_3 + vTRI_4 + vTRI_5 + vTRI_6 + vTRI_7 + vTRI_8)/8
        NTRIA = pd.Series(NTRI).rolling(NTRIP1).sum()
        NTRIA = NTRIA.dropna().reset_index()[0]


        scaler = MinMaxScaler(feature_range=(0.1,1))
        s_NTRIA = scaler.fit_transform(pd.Series(NTRIA).values.reshape(-1, 1))
        s_NTRIA = s_NTRIA.reshape(-1)


        return s_NTRIA, NTRIA.values.reshape(-1)



    def SBS_signal_NTGI(self,close,high,low, cci기간값1, cci기간값2, cci기간값3, cci기간값4, cci기간값5, cci기간값6, cci기간값7, cci기간값8, vM1,NTGIP1):

        data4=close
        SNP_open_=open
        SNP_high_=high
        SNP_low_=low

        data7=close
        AUD_open_=open
        AUD_high_=high
        AUD_low_=low

        data6=close
        Oil_open_=open
        Oil_high_=high
        Oil_low_=low


        data5=close
        Gold_open_=open
        Gold_high_=high
        Gold_low_=low

        data3=close
        VIX_open_=open
        VIX_high_=high
        VIX_low_=low

        data2=close
        USD_open_=open
        USD_high_=high
        USD_low_=low


        data2 = (data2 ).dropna().reset_index()[0]
        USD_high_ = (USD_high_ ).dropna().reset_index()[0]
        USD_low_ = (USD_low_ ).dropna().reset_index()[0]

        data3 = (data3 ).dropna().reset_index()[0]
        VIX_high_ = (VIX_high_ ).dropna().reset_index()[0]
        VIX_low_ = (VIX_low_ ).dropna().reset_index()[0]

        data5 = (data5 ).dropna().reset_index()[0]
        Gold_high_ = (Gold_high_ ).dropna().reset_index()[0]
        Gold_low_ = (Gold_low_ ).dropna().reset_index()[0]

        data2 = [USD_high_, USD_low_, data2]
        data3 = [VIX_high_, VIX_low_, data3]
        data5 = [Gold_high_, Gold_low_, data5]

        _, vUSD_CCI = self.CCI(data2, cci기간값1)  # 달러
        _, vVIX_CCI = self.CCI(data3, cci기간값1)  # vix
        _, vGL_CCI = self.CCI(data5, cci기간값1)  # 금

        _, vUSD_CCI2 = self.CCI(data2, cci기간값2)  # 달러
        _, vVIX_CCI2 = self.CCI(data3, cci기간값2)
        _, vGL_CCI2 = self.CCI(data5, cci기간값2)

        _, vUSD_CCI3 = self.CCI(data2, cci기간값3)  # 달러
        _, vVIX_CCI3 = self.CCI(data3, cci기간값3)
        _, vGL_CCI3 = self.CCI(data5, cci기간값3)

        _, vUSD_CCI4 = self.CCI(data2, cci기간값4)  # 달러
        _, vVIX_CCI4 = self.CCI(data3, cci기간값4)
        _, vGL_CCI4 = self.CCI(data5, cci기간값4)

        _, vUSD_CCI5 = self.CCI(data2, cci기간값5)  # 달러
        _, vVIX_CCI5 = self.CCI(data3, cci기간값5)
        _, vGL_CCI5 = self.CCI(data5, cci기간값5)

        _, vUSD_CCI6 = self.CCI(data2, cci기간값6)  # 달러
        _, vVIX_CCI6 = self.CCI(data3, cci기간값6)
        _, vGL_CCI6 = self.CCI(data5, cci기간값6)

        _, vUSD_CCI7 = self.CCI(data2, cci기간값7)  # 달러
        _, vVIX_CCI7 = self.CCI(data3, cci기간값7)
        _, vGL_CCI7 = self.CCI(data5, cci기간값7)

        _, vUSD_CCI8 = self.CCI(data2, cci기간값8)  # 달러
        _, vVIX_CCI8 = self.CCI(data3, cci기간값8)
        _, vGL_CCI8 = self.CCI(data5, cci기간값8)



        _,vData1_CCI= self.CCI(close,cci기간값1)
        _, vData1_CCI2 = self.CCI(close, cci기간값2)
        _, vData1_CCI3 = self.CCI(close, cci기간값3)
        _, vData1_CCI4 = self.CCI(close, cci기간값4)
        _, vData1_CCI5 = self.CCI(close, cci기간값5)
        _, vData1_CCI6 = self.CCI(close, cci기간값6)
        _, vData1_CCI7 = self.CCI(close, cci기간값7)
        _, vData1_CCI8 = self.CCI(close, cci기간값8)

        _, vSNP_CCI = self.CCI(data4, cci기간값1)  # SNP
        _, vOil_CCI = self.CCI(data6, cci기간값1)  # Oil
        _, vAUD_CCI = self.CCI(data7, cci기간값1)  # AUD 호주

        _, vSNP_CCI2 = self.CCI(data4, cci기간값2)  #
        _, vOil_CCI2 = self.CCI(data6, cci기간값2)
        _, vAUD_CCI2 = self.CCI(data7, cci기간값2)

        _, vSNP_CCI3 = self.CCI(data4, cci기간값3)  #
        _, vOil_CCI3 = self.CCI(data6, cci기간값3)
        _, vAUD_CCI3 = self.CCI(data7, cci기간값3)

        _, vSNP_CCI4 = self.CCI(data4, cci기간값4)  # 달러
        _, vOil_CCI4 = self.CCI(data6, cci기간값4)
        _, vAUD_CCI4 = self.CCI(data7, cci기간값4)

        _, vSNP_CCI5 = self.CCI(data4, cci기간값5)  # 달러
        _, vOil_CCI5 = self.CCI(data6, cci기간값5)
        _, vAUD_CCI5 = self.CCI(data7, cci기간값5)

        _, vSNP_CCI6 = self.CCI(data4, cci기간값6)  # 달러
        _, vOil_CCI6 = self.CCI(data6, cci기간값6)
        _, vAUD_CCI6 = self.CCI(data7, cci기간값6)

        _, vSNP_CCI7 = self.CCI(data4, cci기간값7)  # 달러
        _, vOil_CCI7 = self.CCI(data6, cci기간값7)
        _, vAUD_CCI7 = self.CCI(data7, cci기간값7)

        _, vSNP_CCI8 = self.CCI(data4, cci기간값8)  # 달러
        _, vOil_CCI8 = self.CCI(data6, cci기간값8)
        _, vAUD_CCI8 = self.CCI(data7, cci기간값8)



###################
        vSNP_CCI = vData1_CCI -vSNP_CCI
        vSNP_CCI2 = vData1_CCI2 - vSNP_CCI2
        vSNP_CCI3 = vData1_CCI3 - vSNP_CCI3
        vSNP_CCI4 = vData1_CCI4 - vSNP_CCI4
        vSNP_CCI5 = vData1_CCI5 - vSNP_CCI5
        vSNP_CCI6 = vData1_CCI6 - vSNP_CCI6
        vSNP_CCI7 = vData1_CCI7 - vSNP_CCI7
        vSNP_CCI8 = vData1_CCI8 - vSNP_CCI8

        vOil_CCI = vOil_CCI - vUSD_CCI
        vOil_CCI2 = vOil_CCI2 - vUSD_CCI2
        vOil_CCI3 = vOil_CCI3 - vUSD_CCI3
        vOil_CCI4 = vOil_CCI4 - vUSD_CCI4
        vOil_CCI5 = vOil_CCI5 - vUSD_CCI5
        vOil_CCI6 = vOil_CCI6 - vUSD_CCI6
        vOil_CCI7 = vOil_CCI7 - vUSD_CCI7
        vOil_CCI8 = vOil_CCI8 - vUSD_CCI8

        vAUD_CCI = vAUD_CCI - vGL_CCI
        vAUD_CCI2 = vAUD_CCI2 - vGL_CCI2
        vAUD_CCI3 = vAUD_CCI3 - vGL_CCI3
        vAUD_CCI4 = vAUD_CCI4 - vGL_CCI4
        vAUD_CCI5 = vAUD_CCI5 - vGL_CCI5
        vAUD_CCI6 = vAUD_CCI6 - vGL_CCI6
        vAUD_CCI7 = vAUD_CCI7 - vGL_CCI7
        vAUD_CCI8 = vAUD_CCI8 - vGL_CCI8



        _, STD_SNP = self.STD(vSNP_CCI, cci기간값1)
        _, STD_Oil = self.STD(vOil_CCI, cci기간값1)
        _, STD_AUD = self.STD(vAUD_CCI, cci기간값1)

        _, STD_SNP2 = self.STD(vSNP_CCI2, cci기간값2)
        _, STD_Oil2 = self.STD(vOil_CCI2, cci기간값2)
        _, STD_AUD2 = self.STD(vAUD_CCI2, cci기간값1)

        _, STD_SNP3 = self.STD(vSNP_CCI3, cci기간값3)
        _, STD_Oil3 = self.STD(vOil_CCI3, cci기간값3)
        _, STD_AUD3 = self.STD(vAUD_CCI3, cci기간값1)

        _, STD_SNP4 = self.STD(vSNP_CCI4, cci기간값4)
        _, STD_Oil4 = self.STD(vOil_CCI4, cci기간값4)
        _, STD_AUD4 = self.STD(vAUD_CCI4, cci기간값1)

        _, STD_SNP5 = self.STD(vSNP_CCI5, cci기간값5)
        _, STD_Oil5 = self.STD(vOil_CCI5, cci기간값5)
        _, STD_AUD5 = self.STD(vAUD_CCI5, cci기간값1)

        _, STD_SNP6 = self.STD(vSNP_CCI6, cci기간값6)
        _, STD_Oil6 = self.STD(vOil_CCI6, cci기간값6)
        _, STD_AUD6 = self.STD(vAUD_CCI6, cci기간값1)

        _, STD_SNP7 = self.STD(vSNP_CCI7, cci기간값7)
        _, STD_Oil7 = self.STD(vOil_CCI7, cci기간값7)
        _, STD_AUD7 = self.STD(vAUD_CCI7, cci기간값1)

        _, STD_SNP8 = self.STD(vSNP_CCI8, cci기간값8)
        _, STD_Oil8 = self.STD(vOil_CCI8, cci기간값8)
        _, STD_AUD8 = self.STD(vAUD_CCI8, cci기간값1)

        _, MA_SNP = self.MA(vSNP_CCI, cci기간값1)
        _, MA_Oil = self.MA(vOil_CCI, cci기간값1)
        _, MA_AUD = self.MA(vAUD_CCI, cci기간값1)

        _, MA_SNP2 = self.MA(vSNP_CCI2, cci기간값2)
        _, MA_Oil2 = self.MA(vOil_CCI2, cci기간값2)
        _, MA_AUD2 = self.MA(vAUD_CCI2, cci기간값1)

        _, MA_SNP3 = self.MA(vSNP_CCI3, cci기간값3)
        _, MA_Oil3 = self.MA(vOil_CCI3, cci기간값3)
        _, MA_AUD3 = self.MA(vAUD_CCI3, cci기간값1)

        _, MA_SNP4 = self.MA(vSNP_CCI4, cci기간값4)
        _, MA_Oil4 = self.MA(vOil_CCI4, cci기간값4)
        _, MA_AUD4 = self.MA(vAUD_CCI4, cci기간값1)

        _, MA_SNP5 = self.MA(vSNP_CCI5, cci기간값5)
        _, MA_Oil5 = self.MA(vOil_CCI5, cci기간값5)
        _, MA_AUD5 = self.MA(vAUD_CCI5, cci기간값1)

        _, MA_SNP6 = self.MA(vSNP_CCI6, cci기간값6)
        _, MA_Oil6 = self.MA(vOil_CCI6, cci기간값6)
        _, MA_AUD6 = self.MA(vAUD_CCI6, cci기간값1)

        _, MA_SNP7 = self.MA(vSNP_CCI7, cci기간값7)
        _, MA_Oil7 = self.MA(vOil_CCI7, cci기간값7)
        _, MA_AUD7 = self.MA(vAUD_CCI7, cci기간값1)

        _, MA_SNP8 = self.MA(vSNP_CCI8, cci기간값8)
        _, MA_Oil8 = self.MA(vOil_CCI8, cci기간값8)
        _, MA_AUD8 = self.MA(vAUD_CCI8, cci기간값1)

        equal_res = self.period_equal(
            [vSNP_CCI, vSNP_CCI2, vSNP_CCI3, vSNP_CCI4, vSNP_CCI5, vSNP_CCI6, vSNP_CCI7, vSNP_CCI8
                , vOil_CCI, vOil_CCI2, vOil_CCI3, vOil_CCI4, vOil_CCI5, vOil_CCI6, vOil_CCI7,
             vOil_CCI8,
             vAUD_CCI, vAUD_CCI2, vAUD_CCI3, vAUD_CCI4, vAUD_CCI5, vAUD_CCI6, vAUD_CCI7, vAUD_CCI8,
             MA_SNP, MA_SNP2, MA_SNP3, MA_SNP4, MA_SNP5, MA_SNP6, MA_SNP7, MA_SNP8,
             MA_Oil, MA_Oil2, MA_Oil3, MA_Oil4, MA_Oil5, MA_Oil6, MA_Oil7, MA_Oil8,
             MA_AUD, MA_AUD2, MA_AUD3, MA_AUD4, MA_AUD5, MA_AUD6, MA_AUD7, MA_AUD8,
             STD_SNP, STD_SNP2, STD_SNP3, STD_SNP4, STD_SNP5, STD_SNP6, STD_SNP7, STD_SNP8,
             STD_Oil, STD_Oil2, STD_Oil3, STD_Oil4, STD_Oil5, STD_Oil6, STD_Oil7, STD_Oil8,
             STD_AUD, STD_AUD2, STD_AUD3, STD_AUD4, STD_AUD5, STD_AUD6, STD_AUD7, STD_AUD8])

        vSNP_CCI, vSNP_CCI2, vSNP_CCI3, vSNP_CCI4, vSNP_CCI5, vSNP_CCI6, vSNP_CCI7, vSNP_CCI8, vOil_CCI, vOil_CCI2, vOil_CCI3, vOil_CCI4, vOil_CCI5, vOil_CCI6, vOil_CCI7, vOil_CCI8, vAUD_CCI, vAUD_CCI2, vAUD_CCI3, vAUD_CCI4, vAUD_CCI5, vAUD_CCI6, vAUD_CCI7, vAUD_CCI8, MA_SNP, MA_SNP2, MA_SNP3, MA_SNP4, MA_SNP5, MA_SNP6, MA_SNP7, MA_SNP8, MA_Oil, MA_Oil2, MA_Oil3, MA_Oil4, MA_Oil5, MA_Oil6, MA_Oil7, MA_Oil8, MA_AUD, MA_AUD2, MA_AUD3, MA_AUD4, MA_AUD5, MA_AUD6, MA_AUD7, MA_AUD8, STD_SNP, STD_SNP2, STD_SNP3, STD_SNP4, STD_SNP5, STD_SNP6, STD_SNP7, STD_SNP8, STD_Oil, STD_Oil2, STD_Oil3, STD_Oil4, STD_Oil5, STD_Oil6, STD_Oil7, STD_Oil8, STD_AUD, STD_AUD2, STD_AUD3, STD_AUD4, STD_AUD5, STD_AUD6, STD_AUD7, STD_AUD8 = equal_res


        SNP_std = (vSNP_CCI - MA_SNP) / STD_SNP
        SNP_std2 = (vSNP_CCI2 - MA_SNP2) / STD_SNP2
        SNP_std3 = (vSNP_CCI3 - MA_SNP3) / STD_SNP3
        SNP_std4 = (vSNP_CCI4 - MA_SNP4) / STD_SNP4
        SNP_std5 = (vSNP_CCI5 - MA_SNP5) / STD_SNP5
        SNP_std6 = (vSNP_CCI6 - MA_SNP6) / STD_SNP6
        SNP_std7 = (vSNP_CCI7 - MA_SNP7) / STD_SNP7
        SNP_std8 = (vSNP_CCI8 - MA_SNP8) / STD_SNP8

        Oil_std = (vOil_CCI - MA_Oil) / STD_Oil
        Oil_std2 = (vOil_CCI2 - MA_Oil2) / STD_Oil2
        Oil_std3 = (vOil_CCI3 - MA_Oil3) / STD_Oil3
        Oil_std4 = (vOil_CCI4 - MA_Oil4) / STD_Oil4
        Oil_std5 = (vOil_CCI5 - MA_Oil5) / STD_Oil5
        Oil_std6 = (vOil_CCI6 - MA_Oil6) / STD_Oil6
        Oil_std7 = (vOil_CCI7 - MA_Oil7) / STD_Oil7
        Oil_std8 = (vOil_CCI8 - MA_Oil8) / STD_Oil8

        AUD_std = (vAUD_CCI - MA_AUD) / STD_AUD
        AUD_std2 = (vAUD_CCI2 - MA_AUD2) / STD_AUD2
        AUD_std3 = (vAUD_CCI3 - MA_AUD3) / STD_AUD3
        AUD_std4 = (vAUD_CCI4 - MA_AUD4) / STD_AUD4
        AUD_std5 = (vAUD_CCI5 - MA_AUD5) / STD_AUD5
        AUD_std6 = (vAUD_CCI6 - MA_AUD6) / STD_AUD6
        AUD_std7 = (vAUD_CCI7 - MA_AUD7) / STD_AUD7
        AUD_std8 = (vAUD_CCI8 - MA_AUD8) / STD_AUD8

        vFinalAvg = ((Oil_std + AUD_std) + (SNP_std * 2)) / 3
        vFinalAvg2 = ((Oil_std2 + AUD_std2) + (SNP_std2 * 2)) / 3
        vFinalAvg3 = ((Oil_std3 + AUD_std3) + (SNP_std3 * 2)) / 3
        vFinalAvg4 = ((Oil_std4 + AUD_std4) + (SNP_std4 * 2)) / 3
        vFinalAvg5 = ((Oil_std5 + AUD_std5) + (SNP_std5 * 2)) / 3
        vFinalAvg6 = ((Oil_std6 + AUD_std6) + (SNP_std6 * 2)) / 3
        vFinalAvg7 = ((Oil_std7 + AUD_std7) + (SNP_std7 * 2)) / 3
        vFinalAvg8 = ((Oil_std8 + AUD_std8) + (SNP_std8 * 2)) / 3

        try:
            _, vTGI = self.slope_line(vFinalAvg, vM1)
            _, vTGI_2 = self.slope_line(vFinalAvg2, vM1)
            _, vTGI_3 = self.slope_line(vFinalAvg3, vM1)
            _, vTGI_4 = self.slope_line(vFinalAvg4, vM1)
            _, vTGI_5 = self.slope_line(vFinalAvg5, vM1)
            _, vTGI_6 = self.slope_line(vFinalAvg6, vM1)
            _, vTGI_7 = self.slope_line(vFinalAvg7, vM1)
            _, vTGI_8 = self.slope_line(vFinalAvg8, vM1)

            NTGI = -(vTGI + vTGI_2 + vTGI_3 + vTGI_4 + vTGI_5 + vTGI_6 + vTGI_7 + vTGI_8) / 8
            NTGIA = pd.Series(NTGI).rolling(NTGIP1).sum()
            NTGIA = NTGIA.dropna().reset_index()[0]

            scaler = MinMaxScaler(feature_range=(0.1,1))
            scale_NTGIA = scaler.fit_transform(pd.Series(NTGIA).values.reshape(-1,1))
            scale_NTGIA = scale_NTGIA.reshape(-1)
        except:
            scale_NTGIA = vFinalAvg8.fillna(0)
            NTGIA = vFinalAvg8.fillna(0)




        return scale_NTGIA, NTGIA.values.reshape(-1)


    def SBS_signal_xSB(self,close,open_,high_,low_,vM1,cci_period1,cci_period2,cci_period3,cci_period4,cci_period5, cci_period6,cci_period7, cci_period8,
               acc_period1): #LRC 15분과 같은거(이름만 다름)

         # 15,100,200,400,600,800,1000,1200,1500,2,18,52,104
        close=close # 포인트 벨류 없엔다
        low=low_
        high= high_

        data=[high,low,close]

        CCI1,vData1_CCI= self.CCI(data,cci_period1)
        CCI2,vData1_CCI2=self.CCI(data,cci_period2)
        CCI3,vData1_CCI3=self.CCI(data,cci_period3)
        CCI4,vData1_CCI4=self.CCI(data,cci_period4)
        CCI5,vData1_CCI5=self.CCI(data,cci_period5)
        CCI6,vData1_CCI6=self.CCI(data,cci_period6)
        CCI7,vData1_CCI7=self.CCI(data,cci_period7)
        CCI8,vData1_CCI8=self.CCI(data,cci_period8)



        vCCI1,vvData1_CCI=self.slope_line(vData1_CCI,vM1)
        vCCI2,vvData1_CCI2=self.slope_line(vData1_CCI2,vM1)
        vCCI3,vvData1_CCI3=self.slope_line(vData1_CCI3,vM1)
        vCCI4,vvData1_CCI4=self.slope_line(vData1_CCI4,vM1)
        vCCI5,vvData1_CCI5=self.slope_line(vData1_CCI5,vM1)
        vCCI6,vvData1_CCI6=self.slope_line(vData1_CCI6,vM1)
        vCCI7,vvData1_CCI7=self.slope_line(vData1_CCI7,vM1)
        vCCI8,vvData1_CCI8=self.slope_line(vData1_CCI8,vM1)

        start_period=min(len(vvData1_CCI),len(vvData1_CCI2),len(vvData1_CCI3),len(vvData1_CCI4),
                        len(vvData1_CCI5),len(vvData1_CCI6),len(vvData1_CCI7),len(vvData1_CCI8))

        vvData1_CCI= vvData1_CCI[-start_period:]
        vvData1_CCI2=vvData1_CCI2[-start_period:]
        vvData1_CCI3=vvData1_CCI3[-start_period:]
        vvData1_CCI4=vvData1_CCI4[-start_period:]
        vvData1_CCI5=vvData1_CCI5[-start_period:]
        vvData1_CCI6=vvData1_CCI6[-start_period:]
        vvData1_CCI7=vvData1_CCI7[-start_period:]
        vvData1_CCI8=vvData1_CCI8[-start_period:]


        signal_short = (vvData1_CCI + vvData1_CCI2+ vvData1_CCI3+ vvData1_CCI4 + vvData1_CCI5 + vvData1_CCI6 + vvData1_CCI7 + vvData1_CCI8)/8
        SB_Signal=pd.Series(signal_short).rolling(acc_period1).sum()/acc_period1

        SB_signal = SB_Signal.dropna().reset_index()[0]

        scaler = MinMaxScaler(feature_range=(0.1,1))
        scale_SB_sig = scaler.fit_transform(pd.Series(SB_signal).values.reshape(-1, 1))
        scale_SB_sig = scale_SB_sig.reshape(-1)

        return scale_SB_sig,SB_signal.values.reshape(-1)



    def CI_angle_Neo(self,close_,high_,open_,vM1 ,cci_기간값1,cci_기간값2,cci_기간값3, cci_기간값4, cci_기간값5, cci_기간값6, cci_기간값7,cci_기간값8):
        close_=close_
        Gold = self.ind_data_create_second(params.minute, params.data_count, params.coin_or_stock
                                , 'NQ' , 'GC','NQ=F','GC=F') #price_ai 기준종목이름 ,price_ai 선택종목이름, price_ai2 선택종목이름 ,
        VIX = self.ind_data_create_second(params.minute, params.data_count, params.coin_or_stock
                                , 'NQ' , 'DX','NQ=F','DX=F')
        USD = self.ind_data_create_second(params.minute, params.data_count, params.coin_or_stock
                                , 'NQ' , 'DX','NQ=F','DX=F')


        #없는데이터? 어떻게 되지 = 없는데이터에서는 지표 이어버림. 그러면 백테랑 전진 다른데?
        # 전진에서 하나가 안들어오면? 없는대로 계산함.다시 들어오면? 다시 계산됨
        # 그럼 백테에서도 없으면 없는대로 계산해야함.


        data5, Gold_open_, Gold_high_, Gold_low_, Gold_vol_, Gold_close_scale, Gold_open_scale, Gold_high_scale, Gold_low_scale, Gold_vol_scale, Gold_date = Gold
        data3, VIX_open_, VIX_high_, VIX_low_, VIX_vol_, VIX_close_scale, VIX_open_scale, VIX_high_scale, VIX_low_scale, VIX_vol_scale, VIX_date = VIX
        data2, USD_open_, USD_high_, USD_low_, USD_vol_, USD_close_scale, USD_open_scale, USD_high_scale, USD_low_scale, USD_vol_scale, USD_date = USD


        data = close_
        USD_data= data2
        VIX_data= data3
        GL_data= data5

        #NQ
        CCI1, vData1_CCI = self.CCI(data, cci_기간값1)
        CCI2, vData1_CCI2 = self.CCI(data, cci_기간값2)
        CCI3, vData1_CCI3 = self.CCI(data, cci_기간값3)
        CCI4, vData1_CCI4 = self.CCI(data, cci_기간값4)
        CCI5, vData1_CCI5 = self.CCI(data, cci_기간값5)
        CCI6, vData1_CCI6 = self.CCI(data, cci_기간값6)
        CCI7, vData1_CCI7 = self.CCI(data, cci_기간값7)
        CCI8, vData1_CCI8 = self.CCI(data, cci_기간값8)

        vCCI1,vvNQ_CCI=self.slope_line(vData1_CCI,vM1)
        vCCI2,vvNQ_CCI2=self.slope_line(vData1_CCI2,vM1)
        vCCI3,vvNQ_CCI3=self.slope_line(vData1_CCI3,vM1)
        vCCI4,vvNQ_CCI4=self.slope_line(vData1_CCI4,vM1)
        vCCI5,vvNQ_CCI5=self.slope_line(vData1_CCI5,vM1)
        vCCI6,vvNQ_CCI6=self.slope_line(vData1_CCI6,vM1)
        vCCI7,vvNQ_CCI7=self.slope_line(vData1_CCI7,vM1)
        vCCI8,vvNQ_CCI8=self.slope_line(vData1_CCI8,vM1)

        #USD
        CCI1, vData1_CCI = self.CCI(USD_data, cci_기간값1)
        CCI2, vData1_CCI2 = self.CCI(USD_data, cci_기간값2)
        CCI3, vData1_CCI3 = self.CCI(USD_data, cci_기간값3)
        CCI4, vData1_CCI4 = self.CCI(USD_data, cci_기간값4)
        CCI5, vData1_CCI5 = self.CCI(USD_data, cci_기간값5)
        CCI6, vData1_CCI6 = self.CCI(USD_data, cci_기간값6)
        CCI7, vData1_CCI7 = self.CCI(USD_data, cci_기간값7)
        CCI8, vData1_CCI8 = self.CCI(USD_data, cci_기간값8)

        vCCI1, vvUSD_CCI = self.slope_line(vData1_CCI, vM1)
        vCCI2, vvUSD_CCI2 = self.slope_line(vData1_CCI2, vM1)
        vCCI3, vvUSD_CCI3 = self.slope_line(vData1_CCI3, vM1)
        vCCI4, vvUSD_CCI4 = self.slope_line(vData1_CCI4, vM1)
        vCCI5, vvUSD_CCI5 = self.slope_line(vData1_CCI5, vM1)
        vCCI6, vvUSD_CCI6 = self.slope_line(vData1_CCI6, vM1)
        vCCI7, vvUSD_CCI7 = self.slope_line(vData1_CCI7, vM1)
        vCCI8, vvUSD_CCI8 = self.slope_line(vData1_CCI8, vM1)


        #VIX
        CCI1, vData1_CCI = self.CCI(VIX_data, cci_기간값1)
        CCI2, vData1_CCI2 = self.CCI(VIX_data, cci_기간값2)
        CCI3, vData1_CCI3 = self.CCI(VIX_data, cci_기간값3)
        CCI4, vData1_CCI4 = self.CCI(VIX_data, cci_기간값4)
        CCI5, vData1_CCI5 = self.CCI(VIX_data, cci_기간값5)
        CCI6, vData1_CCI6 = self.CCI(VIX_data, cci_기간값6)
        CCI7, vData1_CCI7 = self.CCI(VIX_data, cci_기간값7)
        CCI8, vData1_CCI8 = self.CCI(VIX_data, cci_기간값8)

        vCCI1, vvVIX_CCI = self.slope_line(vData1_CCI, vM1)
        vCCI2, vvVIX_CCI2 = self.slope_line(vData1_CCI2, vM1)
        vCCI3, vvVIX_CCI3 = self.slope_line(vData1_CCI3, vM1)
        vCCI4, vvVIX_CCI4 = self.slope_line(vData1_CCI4, vM1)
        vCCI5, vvVIX_CCI5 = self.slope_line(vData1_CCI5, vM1)
        vCCI6, vvVIX_CCI6 = self.slope_line(vData1_CCI6, vM1)
        vCCI7, vvVIX_CCI7 = self.slope_line(vData1_CCI7, vM1)
        vCCI8, vvVIX_CCI8 = self.slope_line(vData1_CCI8, vM1)


        #Gold
        CCI1, vData1_CCI = self.CCI(GL_data, cci_기간값1)
        CCI2, vData1_CCI2 = self.CCI(GL_data, cci_기간값2)
        CCI3, vData1_CCI3 = self.CCI(GL_data, cci_기간값3)
        CCI4, vData1_CCI4 = self.CCI(GL_data, cci_기간값4)
        CCI5, vData1_CCI5 = self.CCI(GL_data, cci_기간값5)
        CCI6, vData1_CCI6 = self.CCI(GL_data, cci_기간값6)
        CCI7, vData1_CCI7 = self.CCI(GL_data, cci_기간값7)
        CCI8, vData1_CCI8 = self.CCI(GL_data, cci_기간값8)

        vCCI1, vvGold_CCI = self.slope_line(vData1_CCI, vM1)
        vCCI2, vvGold_CCI2 = self.slope_line(vData1_CCI2, vM1)
        vCCI3, vvGold_CCI3 = self.slope_line(vData1_CCI3, vM1)
        vCCI4, vvGold_CCI4 = self.slope_line(vData1_CCI4, vM1)
        vCCI5, vvGold_CCI5 = self.slope_line(vData1_CCI5, vM1)
        vCCI6, vvGold_CCI6 = self.slope_line(vData1_CCI6, vM1)
        vCCI7, vvGold_CCI7 = self.slope_line(vData1_CCI7, vM1)
        vCCI8, vvGold_CCI8 = self.slope_line(vData1_CCI8, vM1)

        vvNQ_CCI, vvNQ_CCI2, vvNQ_CCI3, vvNQ_CCI4, vvNQ_CCI5, vvNQ_CCI6, vvNQ_CCI7, vvNQ_CCI8,vvUSD_CCI, vvUSD_CCI2, vvUSD_CCI3, vvUSD_CCI4, vvUSD_CCI5, vvUSD_CCI6, vvUSD_CCI7, vvUSD_CCI8,vvVIX_CCI, vvVIX_CCI2, vvVIX_CCI3, vvVIX_CCI4, vvVIX_CCI5, vvVIX_CCI6, vvVIX_CCI7, vvVIX_CCI8,vvGold_CCI, vvGold_CCI2, vvGold_CCI3, vvGold_CCI4, vvGold_CCI5, vvGold_CCI6, vvGold_CCI7, vvGold_CCI8=self.period_equal([vvNQ_CCI,vvNQ_CCI2,vvNQ_CCI3,vvNQ_CCI4, vvNQ_CCI5,vvNQ_CCI6,vvNQ_CCI7,vvNQ_CCI8,vvUSD_CCI,vvUSD_CCI2,vvUSD_CCI3,vvUSD_CCI4,vvUSD_CCI5,vvUSD_CCI6,vvUSD_CCI7,vvUSD_CCI8,vvVIX_CCI,vvVIX_CCI2,vvVIX_CCI3,vvVIX_CCI4,vvVIX_CCI5,vvVIX_CCI6,vvVIX_CCI7,vvVIX_CCI8,vvGold_CCI,vvGold_CCI2,vvGold_CCI3,vvGold_CCI4,vvGold_CCI5,vvGold_CCI6,vvGold_CCI7,vvGold_CCI8])

        Signal_NQ= (vvNQ_CCI + vvNQ_CCI2 + vvNQ_CCI3 + vvNQ_CCI4 + vvNQ_CCI5 + vvNQ_CCI6 + vvNQ_CCI7 + vvNQ_CCI8)/8
        Signal_USD = (vvUSD_CCI+ vvUSD_CCI2 + vvUSD_CCI3 + vvUSD_CCI4 + vvUSD_CCI5 + vvUSD_CCI6 + vvUSD_CCI7 + vvUSD_CCI8)/8
        Signal_VIX = (vvVIX_CCI+ vvVIX_CCI2 + vvVIX_CCI3 + vvVIX_CCI4 + vvVIX_CCI5 + vvVIX_CCI6 + vvVIX_CCI7 + vvVIX_CCI8)/8
        Signal_GL  = (vvGold_CCI + vvGold_CCI2 + vvGold_CCI3 + vvGold_CCI4 + vvGold_CCI5 + vvGold_CCI6 + vvGold_CCI7 + vvGold_CCI8)/8

        Signal_CI = (Signal_NQ * 4 + Signal_VIX * 3 + Signal_USD * 2 + Signal_GL) / 10;  #순방향 합
        Signal_CI2 = (Signal_NQ * 6 - Signal_VIX * 3 - Signal_USD * 2  - Signal_GL) / 12; #안전자산 역방향 합

        Signal_Gap=  Signal_NQ - Signal_CI   # 나스닥과 안전자산의 갭
        Signal_Gap2 = (Signal_Gap- Signal_CI2) # 나스닥과 안전자산의 갭과 안전자산 역방향과 차이
        Signal_Gap3 = (Signal_NQ- Signal_CI2) #나스닥과 안전자산 역방향과 차이

        Signal_Gap  = Signal_Gap.reshape(-1)
        Signal_Gap2 = Signal_Gap2.reshape(-1)
        Signal_Gap3 = Signal_Gap3.reshape(-1)

        scaler = MinMaxScaler(feature_range=(0.1,1))
        s_Gap = scaler.fit_transform(pd.Series(Signal_Gap).values.reshape(-1, 1))
        s_Gap = s_Gap.reshape(-1)

        s_Gap2 = scaler.fit_transform(pd.Series(Signal_Gap2).values.reshape(-1, 1))
        s_Gap2 = s_Gap2.reshape(-1)

        s_Gap3 = scaler.fit_transform(pd.Series(Signal_Gap3).values.reshape(-1, 1))
        s_Gap3 = s_Gap3.reshape(-1)

        s_Signal_NQ =  scaler.fit_transform(pd.Series(Signal_NQ).values.reshape(-1, 1))
        s_Signal_NQ = s_Signal_NQ.reshape(-1)

        s_Signal_CI = scaler.fit_transform(pd.Series(Signal_CI).values.reshape(-1, 1))
        s_Signal_CI = s_Signal_CI.reshape(-1)

        s_Signal_CI2 = scaler.fit_transform(pd.Series(Signal_CI2).values.reshape(-1, 1))
        s_Signal_CI2 = s_Signal_CI2.reshape(-1)


        return s_Gap, s_Gap2, s_Gap3,s_Signal_NQ,s_Signal_CI,s_Signal_CI2, Signal_Gap, Signal_Gap2, Signal_Gap3,Signal_NQ,Signal_CI,Signal_CI2

    def ATR(self, close_, high_, low_, period):  # 전일 종가 사용하기 때문에 가격 open으로 사용  가능
        True_High = high_[:-2].reset_index()[0].combine(close_[1:-1].reset_index()[0],
                                                        max)  # 2개이전 고가와 이전 종가 .shift(1)써도됨>이전종가
        True_low = low_[:-2].reset_index()[0].combine(close_[1:-1].reset_index()[0], min)  # 이전 분봉 저가와 이전 분봉 종가
        True_Range = True_High - True_low
        True_Range = True_Range.rolling(period).mean().dropna().reset_index()[0]
        ATR = True_Range[:-1].reset_index()[0] * (period - 1) + True_Range[1:].reset_index()[0]
        ATR = ATR / period

        scaler = MinMaxScaler(feature_range=(0.1,1))
        s_ATR = scaler.fit_transform(pd.Series(ATR).values.reshape(-1, 1))
        s_ATR = s_ATR.reshape(-1)

        return s_ATR, ATR



    def generate_filter(self, filter_size, input_dim):
        filter = np.zeros(filter_size)
        for i in range(filter_size[0]):
            for j in range(filter_size[1]):
                filter[i, j] = (-1) ** (i + j)
        return filter * (-1)

    def convolution(self, input_data, filter):
        input_shape = input_data.shape
        filter_shape = filter.shape
        output_shape = tuple(np.subtract(input_shape, filter_shape) + 1)
        output = np.zeros(output_shape)

        for index in np.ndindex(*output_shape):
            output[index] = np.sum(input_data[tuple(slice(i, i + dim) for i, dim in zip(index, filter_shape))] * filter)

        return pd.Series(output.reshape(-1))


    def ind_conv_fitter(self, ind_data, period):  # ind 데이터값 모음 리스트
        # 컨볼루션 필터

        dim = len(ind_data)
        ind_data = self.period_equal(ind_data)

        for step in range(len(ind_data)):
            ind_data[step] = pd.Series(ind_data[step]).dropna().reset_index()[0]

        filter = self.generate_filter((dim, period), dim)
        data = np.array(ind_data).reshape(dim, -1)

        result = self.convolution(data, filter)
        result = result.rolling(2).mean().dropna().reset_index()[0]

        scaler = MinMaxScaler(feature_range=(0.1,1))
        s_result = scaler.fit_transform(pd.Series(result).values.reshape(-1, 1))
        s_result = s_result.reshape(-1)

        plt.plot(result)
        plt.show()


        return s_result, result.values






































