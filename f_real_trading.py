# 실전매매
import time
import ccxt
import torch
import a_Env as env_
import e_train as params
import b_rule
import requests
import os
from binance.client import Client
import f_back_test as bk
import torch.multiprocessing as multiprocessing

from datetime import datetime, timedelta
import pandas as pd



env=env_.Env()
params.real_or_train='real'



def real_save_price_data(data,ind_data,data_name):
    data_name = data_name.replace('/', '_')
    data = [pd.Series(data[step],name='data').reset_index()['data'] for step in range(len(data))]

    data_ = pd.DataFrame(data[:-1])
    date = pd.DataFrame(data[-1])
    data_minute = pd.DataFrame([params.minute])

    date_name_ = str('save_real_date_'+data_name)
    data_name_ = str('save_real_price_'+data_name)
    minute_name = 'minute_data'

    date.to_csv(date_name_,index=False)
    data_.to_csv(data_name_,index=False)
    torch.save(ind_data,'save_real_ind_'+data_name)
    data_minute.to_csv(minute_name,index=False)





def real_load_price_data(data_name,real_data_count):  #불러온 데이터를 csv로 저장하고 동일한 날짜인경우 불러올때 csv를 호출함으로써 시간 절약
    res = 0
    data = 0
    data_name = data_name.replace('/', '_')
    ind_data_ = [0, 0, 0]

    try:
        csv_data=pd.read_csv('save_real_date_'+data_name).values
        ind_data = torch.load('save_real_ind_'+data_name)
        past_minute =pd.read_csv('minute_data').values[0][0] #과거 분봉

        if params.minute == past_minute and real_data_count[1] == csv_data[-1][0][:16]:
            res='csv'
            csv_data_ = pd.read_csv('save_real_price_'+data_name).values

            data = [pd.Series(csv_data_[step]) for step in range(len(csv_data_))]
            data.append(pd.Series(csv_data.reshape(-1))) # 날짜 추가
            ind_data_ = [pd.Series(ind_data[step]) for step in range(len(ind_data))]
        else :
            #API와 저장된 데이터의 불러올 날짜가 다르면 새로 API 호출
            print('불러온 데이터의 마지막 날짜:',csv_data[-1][0][:16],'         불러온 최근 시간:',real_data_count[1])
            res='API'


    except Exception as e:  # 예외 유형을 Exception으로 지정
        print('저장된 데이터 파일이 없습니다. 새로운 API 호출 실시')
        print(f'오류 메시지: {e}')  # 오류 메시지 출력
        res = 'API'

    return res,data,ind_data_



class MyBinance(ccxt.binance):
    def nonce(self):
        return self.milliseconds() - self.options['timeDifference']



class trading:
    def __init__(self,train_val_test,symbol_name,ori_symbol):
        self.binance = ccxt.binance({
            'enableRateLimit': True,  # 데이터 속도제한
            'apiKey': API_key,
            'secret': Secret_key,
            'timeout': 3000,  # milliseconds
            }
        )
        self.client = Client(API_key, Secret_key)

        self.is_back_testing=True
        self.symbol_name= symbol_name
        self.ori_symbol_name=ori_symbol

        # long short 2개 생성
        self.PV_data = {'long':[], 'short':[]}
        self.action_data = {'long':[], 'short':[]}
        self.buy_data = {'long':[], 'short':[]}  # 매수한 가격
        self.sell_data = {'long':[], 'short':[]}  # 매도한 가격
        self.buy_date = {'long':[], 'short':[]}
        self.sell_date = {'long':[], 'short':[]}
        self.price_data={'long':[],'short':[]} #가격 데이터
        self.date_data={'long':[], 'short':[]} # 날짜 데이터
        self.scale_input={'long':[],'short':[]}

        self.train_val_test = train_val_test
        self.main_ind_state= 'init'
        self.agent_data={}

        self.past_data_date=0

        # 데이터 호출
        env = env_.Env()
        is_First=True #처음인경우
        real_data_count, is_API_ = self.recent_period(is_First)  #처음 뽑으면 API 최신까지 호출
        print(real_data_count,'원시 데이터 호출 날짜 기간')
        data_ = env.coin_data_create(params.minute, real_data_count, params.real_or_train, params.coin_or_stock,
                                     params.point_value,self.symbol_name)  # 학습시 뽑은 history 데이터

        long_input_, short_input_, ori_ind_data= env.input_create(params.minute, params.ratio,real_data_count,
                                                               params.coin_or_stock, params.point_value,
                                                               params.short_ind_name, params.long_ind_name,
                                                               data_)  # ind에서 높은 값을 뽑음
        ind_data = [long_input_,short_input_,ori_ind_data]
        real_save_price_data(data_,ind_data, self.symbol_name)

        long_train_data,long_val_data,long_test_data,long_ori_close,long_total_input,long_date_data,long_total_date= long_input_

        self.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])
        self.long_scale_input = long_total_input
        self.long_date_data = long_total_date

        short_or_long ='long' #현물= 롱

        input_=self.long_scale_input
        self.input_dim = params.input_dim['long']
        ori_close=self.long_price_data
        date_data=self.long_date_data

        if params.train_stock_or_future =='future':
            cost= params.coin_future_cost

        elif params.train_stock_or_future =='coin':
            balance = self.binance.fetch_balance()
            cash = balance['USDT']['free']
            cost = params.stock_cost
        else:
            cost = params.stock_cost

        self.agent_data[short_or_long] = b_rule.Rule_agent(cash,  # 초기 보유 현금
                                                  cost, # 수수료
                                                  input_, #인풋데이터
                                                  ori_ind_data, #인풋의 오리지날(숏 롱)
                                                  ori_close, # 주가 데이터
                                                  date_data, #날짜 데이터
                                                  self.input_dim, # 피쳐 디멘션
                                                  params.deposit, # 증거금
                                                  params.backtest_slippage, # 슬리피지
                                                  short_or_long)


    def recent_period(self,is_First): #최신데이터를 적절한 구간만큼 minute 고려하여 불러온다
        real_trading_data_num = 1000  # 새로출력시 불러올 데이터 갯수(새로운 분인경우)

        ##############현재 최신 데이터 시간 불러오기
        time.sleep(1) #많은호출 방지

        for i in range(20):  # 최대 20번 재시도
            try:
                self.binance = ccxt.binance({
                    'enableRateLimit': True,  # 데이터 속도제한
                    'apiKey': API_key,
                    'secret': Secret_key,
                    'timeout': 3000,  # milliseconds
                }
                )

                ohlcv = self.binance.fetch_ohlcv(self.symbol_name, '1m', limit=10)  # 1분 봉 데이터를 가져옴
                break  # 데이터를 성공적으로 가져오면 for 루프를 빠져나옴

            except ccxt.NetworkError:  # 네트워크 오류가 발생하면
                if i < 19:  # 19번 이하로 시도했다면
                    time.sleep(2)
                    print(i+1,'번 재시도')
                    continue  # 다시

                else:  # 10번이 모두 실패했다면
                    raise  # 오류를 던짐

        latest_ohlcv = ohlcv[-1]  # 가장 최근의 봉 데이터를 선택
        open_time = latest_ohlcv[0] / 1000  # 밀리초를 초로 변환
        open_time = datetime.fromtimestamp(open_time).strftime('%Y-%m-%d %H:%M')  # 해당 종목의 최근 가격 시간

        if is_First==False:
            #########이전에 저장된 최신데이터 불러오기

            try:  #과거 저장됐던 데이터 호출
                csv_data = pd.read_csv('save_real_date_' + self.symbol_name.replace('/','_')).values
                ori_last_minute = csv_data[-1][0][:16] #저장된 데이터의 마지막 시각
            except:
                pass

            ########### minute 만큼 시간이 지났는지 확인
            time1 = datetime.strptime(open_time, '%Y-%m-%d %H:%M')
            time2 = datetime.strptime(ori_last_minute, '%Y-%m-%d %H:%M')
            time_diff = time1-time2
            time_minute = int(time_diff.total_seconds()/60)

            if round(time_minute / params.minute) < real_trading_data_num : # 뽑게될 데이터 갯수가 적으면 데이터를 더뽑음
                last_minute = datetime.strptime(ori_last_minute, '%Y-%m-%d %H:%M') - timedelta(minutes=params.minute * real_trading_data_num)  # 데이터 1000개만큼 출력되도록 시작점 설정
                last_minute = str(last_minute)[:16]
                time2 = datetime.strptime(last_minute, '%Y-%m-%d %H:%M')
                time_diff = time1 - time2
                time_minute = int(time_diff.total_seconds() / 60)


            ########## 백테스트 마지막 ~ 최신 데이터 호출 구간
            if time_minute%params.minute==0: # minute 만큼 시간이 지났을경우 새로운 데이터 호출
                is_API = 'API'
                real_data_count = [last_minute, open_time]
            else:
                is_API = 'csv'
                real_data_count = [last_minute, ori_last_minute]

            '''''
            print(round(time_minute / params.minute))
            print(time2, 'csv 의 마지막 시간')
            print(time_diff, '시간차이')
            print(open_time, '실시간')
            print(last_minute, '불러올 초기시간')
            print(time_minute, '지나온시간')
            print(is_API, 'API냐 csv냐')
            print(real_data_count,'불러오는 시간')
            '''''

        if is_First==True: #csv도 없는 첫상태, 초기 트레이딩 시작일때
            is_API='API'
            real_data_count = [params.data_count[0],open_time]  #종목의 최근까지 호출

        return real_data_count, is_API


    def reset(self):
        self.PV_data = {'long':[], 'short':[]}
        self.action_data = {'long':[], 'short':[]}
        self.buy_data = {'long':[], 'short':[]}  # 매수한 가격
        self.sell_data = {'long':[], 'short':[]}  # 매도한 가격
        self.buy_date={'long':[], 'short':[]}
        self.sell_date= {'long':[], 'short':[]}

    def mul_trade(self):  # 결과 추출

        long_res_=[]
        short_res_=[]
        res_dict= {}

        print('병렬 트레이딩 시작.', '코어수:', multiprocessing.cpu_count())

        if params.train_stock_or_future != 'future':
            params.short_or_long_data = ['long']  # 주식 또는 코인인경우 롱만진행

        for is_short_or_long in params.short_or_long_data:
            long_res,short_res=self.trading_start(is_short_or_long)

            if is_short_or_long == 'long':
                res_ = long_res[0]
            if is_short_or_long == 'short':
                res_ = short_res[0]

            res_dict[is_short_or_long] = res_[0]

        return res_dict


    def trading_start(self,is_short_or_long):  # 트레이딩
        #시뮬레이션

        self.agent=self.agent_data[is_short_or_long]
        self.agent.reset()  # 리셋 (리셋때 back testing=False 된다)
        self.agent.back_testing = True
        self.is_back_testing=True



        if is_short_or_long == 'long':  # 롱인경우 액션
            self.decide_action = self.agent.long_decide_action
            self.discrete_step = env.long_discrete_step

        if params.train_stock_or_future != 'future':
            self.decide_action = self.agent.SC_decide_action
            self.discrete_step = env.SC_discrete_step


        while True:
            #재 접속 시도
            for step in range(100):
                try:
                    self.binance = MyBinance({
                        'enableRateLimit': True,  # 데이터 속도제한
                        'apiKey': API_key,
                        'secret': Secret_key,
                        'timeout': 300000,  # milliseconds
                        'options': {
                            'timeDifference': self.binance.fetch_time() - self.binance.milliseconds(),
                        },
                    })
                    self.client = Client(API_key, Secret_key)
                    break

                except Exception as e:
                    print(e)
                    print('재접속 시도',step,'번째')



            is_First=False
            real_data_count, is_API_ = self.recent_period(is_First)  # 처음 뽑으면(= is First : True) API 최신까지 호출

            is_API, data_, ind_data = real_load_price_data(self.symbol_name, real_data_count)  # csv를 불러올지, api를 불러올지 선택
            long_input_ = ind_data[0]
            short_input_= ind_data[1]
            ori_ind_data = ind_data[2]

            #데이터 불러옴
            if is_API=='API':
                data_ = env.coin_data_create(params.minute, real_data_count, params.real_or_train, params.coin_or_stock,
                                             params.point_value, self.symbol_name)  # 학습시 뽑은 history 데이터

                long_input_, short_input_, ori_ind_data = env.input_create(params.minute, params.ratio, real_data_count,
                                                                           params.coin_or_stock, params.point_value,
                                                                           params.short_ind_name, params.long_ind_name,
                                                                           data_)  # ind에서 높은 값을 뽑음
            ind_data = [long_input_, short_input_, ori_ind_data]
            real_save_price_data(data_, ind_data, self.symbol_name)

            long_train_data, long_val_data, long_test_data, long_ori_close, long_total_input, long_date_data, long_total_date = long_input_

            self.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])
            self.long_scale_input = long_total_input
            self.long_date_data = long_total_date

            #바이낸스 호출
            balance = self.binance.fetch_balance()

            for i in range(20):  # 최대 20번 재시도
                try:
                    self.binance = ccxt.binance({
                        'enableRateLimit': True,  # 데이터 속도제한
                        'apiKey': API_key,
                        'secret': Secret_key,
                        'timeout': 3000,  # milliseconds
                    }
                    )

                    ohlcv = self.binance.fetch_ohlcv(self.symbol_name, '1m', limit=5)  # 1분 봉 데이터를 가져옴
                    break  # 데이터를 성공적으로 가져오면 for 루프를 빠져나옴
                except ccxt.NetworkError:  # 네트워크 오류가 발생하면
                    if i < 19:  # 19번 이하로 시도했다면
                        time.sleep(2)
                        print(i + 1, '번 재시도')
                        continue  # 다시 시도
                    else:  # 10번이 모두 실패했다면
                        raise  # 오류를 던짐

            latest_ohlcv = ohlcv[-1]  # 가장 최근의 봉 데이터를 선택

            #데이터 업데이트
            self.agent.long_ori_input = ori_ind_data[0]
            self.agent.price_data = self.long_price_data  # 가격 데이터

            #상태 API 업데이트
            step = len(self.agent.price_data)-1 # 마지막 스탭
            self.data_price = latest_ohlcv[1]   #시장의 x분봉 open 가격
            self.agent.price = self.agent.price_data[-1] #분봉 계산된 가격
            self.agent.cash= balance['USDT']['free']
            self.agent.stock = balance[self.ori_symbol_name]['free'] #거래대상의 물량(주식 수량)


            # trading
            action, unit = self.agent.Rule_base(is_short_or_long,step)  # 액션 선택
            action, reward, step_ = self.discrete_step(action, unit, step, self.agent)  # PV및 cash, stock 업데이트


            if action == 0: #매도
                # unit 조절 (풀매수로 설정돼있기 때문에 조정)
                unit_ = round(unit[0] , 5)
                if unit_ > self.agent.stock:
                    unit_ = self.agent.stock-0.0001

                if unit_ > 0:
                    self.action_data[is_short_or_long].append(0)
                    self.order = self.order_market_sell(
                        symbol=self.symbol_name.replace('/',''),
                        quantity=unit_
                    )

                    print('매도')
                    print(unit[0])
                else:
                    print('매도 , 유닛수부족으로 패스')

            elif action == 1: #관망
                pass
                #self.action_data[is_short_or_long].append(1)

            else: #매수
                # unit 조절 (풀매수로 설정돼있기 때문에 조정)
                unit_ = round(unit[2] , 3)

                print('매수')
                print(unit[2])

                if unit_ > 0:
                    self.action_data[is_short_or_long].append(2)
                    self.order = self.client.order_market_buy(
                        symbol=self.symbol_name.replace('/',''),
                        quantity=unit_
                    )

                else:
                    print('매수, 유닛수 부족으로 패스')

            # 데이터 저장
            #self.PV_data[is_short_or_long].append(self.agent.PV)

            if real_data_count[1] != self.past_data_date: #분봉 시간이 바꼈을때

                if action == 0:
                    print('진입 분봉 시간:',real_data_count[1],'      행동: 매도', '진입 갯수:', unit[0], '진입 가격:', self.agent.price)

                if action == 2:
                    print('진입 분봉 시간:',real_data_count[1],'      행동: 매수', '진입 갯수:', unit[2], '진입 가격:', self.agent.price)

                print('------------------------------------------------------------------------------------------------------------------------------------------')
                print('종목명:',self.symbol_name,'           현재액션:',action,'       종목매수 가치(달러) : ',float(self.agent.price)*self.agent.stock,'       포트폴리오 가치(달러) : ',self.agent.PV,'    현재가(달러):', float(self.agent.price), '   평단가:','None','    보유수량:',self.agent.stock)
                print('------------------------------------------------------------------------------------------------------------------------------------------')

                self.past_data_date = real_data_count[1]





import torch.multiprocessing as multiprocessing

from torch.multiprocessing import Manager, Process, Queue
from torch.multiprocessing import Event



class start_Mul():  #멀티프로세싱 클래스
    def __init__(self):
        self.idx = 0

    def reset(self):
        self.idx = 0

    def func_start(self,class_list):
        self.reset()
        proc_list=[]
        while True:
            if self.idx < len(class_list):
                class_name = class_list[self.idx]
                proc = Process(target=class_name.trading_start, args=(['long']))
                proc_list.append(proc)
            else:
                break
            self.idx += 1

        return proc_list

    # dict형태로 에이전트 이름받아서 함수 실행
    def start(self,class_list):
        start_event = Event()
        print('병렬 학습 시작.','코어수:',multiprocessing.cpu_count())

        if __name__ == '__main__':
            # net 호출
            process_list = self.func_start(class_list)
            for idx, process_ in enumerate(process_list):
                time.sleep(1)  # 각 프로세스의 시작을 일정하게 딜레이
                process_.start()

            start_event.set()

            for idx, process_ in enumerate(process_list):
                time.sleep(1)  # 각 프로세스의 시작을 일정하게 딜레이
                process_.join()






if __name__ == '__main__':

    Mul= start_Mul()
    class_list = []
    for step in range(len(params.API_coin_name)):

        symbol_n = params.API_data_name[step]
        ori_symbol = params.API_coin_name[step]

        trading_class = trading(params.back_train_val_test, symbol_n, ori_symbol)
        class_list.append(trading_class)


    Mul.start(class_list)

    #res_data = bk.mul_trade()
    #bk.res_plot(res_data)


