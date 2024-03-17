#전진분석.py


# 일별로 백테스트 하여 플랏
# 일별 손익 표시,


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as multiprocessing

import a_Env as env_
import e_train as params
import f_back_test as bk
import b_rule


env=env_.Env()



class forward_testing:
    def __init__(self,train_val_test,symbol_name):
        # 시뮬 data
        self.is_back_testing=True
        self.symbol_name= symbol_name

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

        # 데이터 호출
        env = env_.Env()

        is_API, data_ = env_.load_price_data(symbol_name) #csv를 불러올지, api를 불러올지 선택

        if is_API=='API':
            data_ = env.coin_data_create(params.minute, params.data_count,params.real_or_train, params.coin_or_stock,
                                         params.point_value,self.symbol_name)  # 학습시 뽑은 history 데이터
            env_.save_price_data(data_,symbol_name)

        long_input_, short_input_, ori_ind_data= env.input_create(params.minute, params.ratio, params.data_count,
                                                               params.coin_or_stock, params.point_value,
                                                               params.short_ind_name, params.long_ind_name,
                                                               data_)  # ind에서 높은 값을 뽑음

        long_train_data,long_val_data,long_test_data,long_ori_close,long_total_input,long_date_data,long_total_date= long_input_

        # APPO는 전체 넣음, PPO는 데이터셋 나눠서 넣음


        if train_val_test == 'train':
            self.long_price_data = long_ori_close[0]
            self.long_scale_input = long_train_data
            self.long_date_data = long_date_data[0]

            self.short_price_data = short_ori_close[0]
            self.short_scale_input = short_train_data
            self.short_date_data = short_date_data[0]



        elif train_val_test == 'val':
            self.long_price_data = long_ori_close[1]
            self.long_scale_input = long_val_data
            self.long_date_data = long_date_data[1]

            self.short_price_data = short_ori_close[1]
            self.short_scale_input = short_val_data
            self.short_date_data = short_date_data[1]


        elif train_val_test == 'test':
            self.long_price_data = long_ori_close[2]
            self.long_scale_input = long_test_data
            self.long_date_data = long_date_data[2]

            self.short_price_data = short_ori_close[2]
            self.short_scale_input = short_test_data
            self.short_date_data = short_date_data[2]



        elif train_val_test == 'total':
            self.long_price_data = torch.cat([long_ori_close[0], long_ori_close[1], long_ori_close[2]])
            self.long_scale_input = long_total_input
            self.long_date_data = long_total_date

            self.short_price_data = torch.cat([short_ori_close[0], short_ori_close[1], short_ori_close[2]])
            self.short_scale_input = short_total_input
            self.short_date_data = short_total_date




        # 에이전트 호출
        for short_or_long in params.short_or_long_data:

            #숏 or 롱 포지션 따라 인풋 정의
            if short_or_long=='short':
                input_=self.short_scale_input
                self.input_dim=params.input_dim['short']
                ori_close=self.short_price_data
                date_data=self.short_date_data

            else:
                input_=self.long_scale_input
                self.input_dim = params.input_dim['long']
                ori_close=self.long_price_data
                date_data=self.long_date_data

            agent_num = 0  # 글로벌 에이전트 넘버=0
            #에이전트 정의
            cost= 0
            if params.train_stock_or_future =='future':
                cost= params.coin_future_cost
            else:
                cost = params.stock_cost
            self.agent_data[short_or_long] = b_rule.Rule_agent(params.cash,  # 초기 보유 현금
                                                      cost, # 수수료
                                                      input_, #인풋데이터
                                                      ori_ind_data, #인풋의 오리지날(숏 롱)
                                                      ori_close, # 주가 데이터
                                                      date_data, #날짜 데이터
                                                      self.input_dim, # 피쳐 디멘션
                                                      params.deposit, # 증거금
                                                      params.backtest_slippage, # 슬리피지
                                                      short_or_long)


    def reset(self):
        self.PV_data = {'long':[], 'short':[]}
        self.action_data = {'long':[], 'short':[]}
        self.buy_data = {'long':[], 'short':[]}  # 매수한 가격
        self.sell_data = {'long':[], 'short':[]}  # 매도한 가격
        self.buy_date={'long':[], 'short':[]}
        self.sell_date= {'long':[], 'short':[]}


    def forward(self,is_short_or_long,long_res,short_res):  # 전진분석
        #시뮬레이션

        self.agent=self.agent_data[is_short_or_long]

        self.agent.reset()  # 리셋 (리셋때 back testing=False 된다)
        self.agent.back_testing = True
        self.is_back_testing=True

        self.agent.scale_input = self.agent.scale_input # 인풋 데이터
        self.agent.price_data = self.agent.price_data  # 종가 데이터


        if is_short_or_long == 'short':  # 숏인경우 숏가중치 저장
            self.decide_action = self.agent.short_decide_action
            self.discrete_step = env.short_discrete_step

        elif is_short_or_long == 'long':  # 롱인경우 롱가중치 저장
            self.decide_action = self.agent.long_decide_action
            self.discrete_step = env.long_discrete_step

        if params.train_stock_or_future != 'future':
            self.decide_action = self.agent.SC_decide_action
            self.discrete_step = env.SC_discrete_step

        # forward

        for step in range(len(self.agent.price_data)):

            self.agent.price = self.agent.price_data[step]  # 현재 주가업데이트
            self.agent.deposit= self.agent.price_data[step]  # 코인 선물에서는 deposit = 코인가격

            action, unit = self.agent.Rule_base(is_short_or_long,step)  # 액션 선택
            action, reward, step_ = self.discrete_step(action, unit, step, self.agent)  # PV및 cash, stock 업데이트


            if action == 0: #매도
                self.action_data[is_short_or_long].append(0)
                if unit[0] !=0 : #매매 유닛이 0이 아닌경우
                    self.sell_data[is_short_or_long].append(self.agent.price_data[step])
                    self.sell_date[is_short_or_long].append(step)

            elif action == 1: #관망
                self.action_data[is_short_or_long].append(1)

            else: #매수
                self.action_data[is_short_or_long].append(2)
                if unit[2] != 0:  # 매매 유닛이 0이 아닌경우
                    self.buy_data[is_short_or_long].append(self.agent.price_data[step])
                    self.buy_date[is_short_or_long].append(step)

            # 데이터 저장
            self.PV_data[is_short_or_long].append(self.agent.PV)

            date_data_set={'long': self.long_date_data,
                           'short': self.short_date_data}


            if step % 50 ==0 :  #실시간 출력값
                print(step + 1, '/', len(self.agent.price_data), '테스팅중..',  is_short_or_long + '_agent PV :', float(self.PV_data[is_short_or_long][-1]) )
                print(self.agent.stock,'보유수량','   ',action,'action',unit,'unit',self.agent.price,'price')


        long_PV_first = (1 * self.agent.deposit) + (self.agent.price[0] - self.agent.init_cash) * 1 + self.agent.init_cash
        long_PV_last = (1 * self.agent.deposit) + (self.agent.price[-1] - self.agent.init_cash) * 1 + self.agent.init_cash


        # 결과
        if params.multi_or_thread=='multi': #멀티프로세싱인경우 Queue에 저장
            if is_short_or_long=='long':
                long_res.append([self])
            else:
                short_res.append([self])
        else:
            if is_short_or_long=='long': #스레드인 경우 리스트 저장
                long_res.append(self)
            else:
                short_res.append(self)


        market_first = self.agent.price_data[0]
        market_last = self.agent.price_data[-1]

        self.date_data[is_short_or_long]=self.agent.date_data
        self.price_data[is_short_or_long]= self.agent.price_data
        self.scale_input[is_short_or_long]= self.agent.scale_input
        #print(len(self.PV_data),len(self.action_data),len(self.buy_data),len(self.buy_date),len(self.sell_date),len(self.sell_date),'aksfnkasnfklsnkf')

        print((((market_last / market_first) - 1) * 100).item(), ':Market ratio of long return')
        print(float(((self.PV_data[is_short_or_long][-1] / self.PV_data[is_short_or_long][0]) - 1) * 100),
              '% :' + is_short_or_long + '_agent PV return')
        if params.coin_or_stock == 'future':  # 선물인경우
            print(float((((self.PV_data[is_short_or_long][-1] - self.agent.init_cash) / self.agent.deposit)) * 100),
                  '% :' + is_short_or_long + '_agent 증거금 대비 PV return')

        return long_res,short_res





    def res_plot(self, res_data):  # 백테스팅 이후 실행
        ########갯수 잘못됨
        ind_diff={}

        if params.train_stock_or_future == 'future':
            if len(res_data['long'].PV_data['long']) > len(res_data['short'].PV_data['short']):  # 롱데이터수 더길면 (롱의 지표변수가 더짧다)
                ind_diff['long'] = np.abs(len(res_data['long'].PV_data['long']) - len(res_data['short'].PV_data['short']))  # 그래프 계산시 얼마나 빼야할지
                ind_diff['short'] = 0

                # 인덱스 갯수 줄이기(롱이 많은경우)
                res_data['long'].PV_data['long']= res_data['long'].PV_data['long'][ind_diff['long']:]

                res_data['long'].buy_date['long']=res_data['long'].buy_date['long'] - ind_diff['long']  # 인덱스 조절
                res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'] - ind_diff['long']  # 인덱스 조절

                res_data['long'].buy_data['long']=res_data['long'].buy_data['long'][len(res_data['long'].buy_date['long'][res_data['long'].buy_date['long']<0]):]
                res_data['long'].sell_data['long'] = res_data['long'].sell_data['long'][len(res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] < 0]):]

                res_data['long'].buy_date['long']=res_data['long'].buy_date['long'][res_data['long'].buy_date['long'] >= 0]
                res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] >= 0]

                res_data['long'].agent.price_data= self.price_data['long'][ind_diff['long']:]
                res_data['long'].agent.date_data=self.date_data['long'][ind_diff['long']:]

                long_ind_data=[]

                for dim in range(len(self.scale_input['long'])):
                    long_ind_data.append(self.scale_input['long'][dim][ind_diff['long']:])
                res_data['long'].agent.scale_input = long_ind_data


            elif len(res_data['long'].PV_data['long']) < len(res_data['short'].PV_data['short']):  # 숏이20 더길면 숏에 20이 들어감
                ind_diff['short'] = np.abs(len(res_data['long'].PV_data['long']) - len(res_data['short'].PV_data['short']))
                ind_diff['long'] = 0

                res_data['short'].PV_data['short'] = res_data['short'].PV_data['short'][ind_diff['short']:]
                res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'] - ind_diff['short']  # 인덱스 조절
                res_data['short'].sell_date['short']= res_data['short'].sell_date['short'] - ind_diff['short']

                res_data['short'].buy_data['short']=res_data['short'].buy_data['short'][len(res_data['short'].buy_date['short'][res_data['short'].buy_date['short']<0]):]
                res_data['short'].sell_data['short']=res_data['short'].sell_data['short'][len(res_data['short'].sell_date['short'][res_data['short'].sell_date['short']<0]):]

                res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'][res_data['short'].buy_date['short'] >= 0]
                res_data['short'].sell_date['short'] = res_data['short'].sell_date['short'][res_data['short'].sell_date['short'] >= 0]

                res_data['short'].agent.price_data= self.price_data['short'][ind_diff['short']:]
                res_data['short'].agent.date_data= self.date_data['short'][ind_diff['short']:]

                res_data['short'].agent.scale_input = self.scale_input['short'][ind_diff['short']:]

                short_ind_data=[]
                for dim in range(len(self.scale_input['short'])):
                    short_ind_data.append(self.scale_input['short'][dim][ind_diff['short']:])
                res_data['short'].agent.scale_input = short_ind_data



            else: #같으면
                ind_diff['short']=0
                ind_diff['long']=0



        if __name__ == '__main__':
            if self.is_back_testing == True:  # 백테스팅일경우 출력
                fig, ax = plt.subplots(4, 1, figsize=(10, 9))
                total_dim = len(params.short_or_long_data)

                for dim in range(total_dim):
                    is_short_or_long = params.short_or_long_data[dim]

                    #앞에 self 붙으면 class안에 중복된 이름있기 때문에 dict이 소실됨
                    agent = res_data[is_short_or_long].agent
                    PV_data = res_data[is_short_or_long].PV_data[is_short_or_long]
                    buy_data = res_data[is_short_or_long].buy_data[is_short_or_long]
                    buy_date = res_data[is_short_or_long].buy_date[is_short_or_long]
                    sell_data = res_data[is_short_or_long].sell_data[is_short_or_long]
                    sell_date = res_data[is_short_or_long].sell_date[is_short_or_long]
                    price_data= res_data[is_short_or_long].agent.price_data.view(-1)

                    if dim == 0:  # 처음에만 출력
                        if len(params.short_or_long_data) > 1 :  #헷지모드 on( 롱숏 둘다 백테스트 하며 각각 PV연산하여 합산)
                            long_agent = res_data['long'].agent
                            long_PV_data = res_data['long'].PV_data['long']

                            short_agent = res_data['short'].agent
                            short_PV_data = res_data['short'].PV_data['short']

                            long_data_date = long_agent.date_data
                            short_data_date = short_agent.date_data

                            # 길이 일치
                            PV_data_set = [long_PV_data, short_PV_data]
                            date_set = [long_data_date, short_data_date]

                            less_data = PV_data_set[np.argmin([len(long_PV_data), len(short_PV_data)])]  # 갯수가 더 적은 데이터
                            more_data = PV_data_set[np.argmax([len(long_PV_data), len(short_PV_data)])]

                            more_date = date_set[np.argmax([len(long_data_date), len(short_data_date)])]  # 갯수 더 많은 날짜 데이터

                            len_diff = np.abs(len(long_PV_data) - len(short_PV_data))  # 차이

                            less_data = torch.cat([torch.zeros(len_diff), torch.Tensor(less_data).view(-1)]).view(-1)  # 적었던 PV데이터 길이 일치
                            if len_diff ==0: #차이가 없는경우
                                more_data=long_PV_data
                                less_data=short_PV_data
                            res_PV = torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)  # PV 합
                            res_date = more_date

                            ax[0].set_ylabel('NI of short and long Agent')
                            ax[0].plot(res_date, (res_PV - (long_agent.init_cash + short_agent.init_cash)))

                            series_PV=pd.Series(res_PV-(long_agent.init_cash+short_agent.init_cash))
                            series_date=pd.Series(res_date)



                            day_count=int(round(1440/params.minute))
                            week_count=int(round(7200/params.minute))
                            month_count=int(round(31000/params.minute))

                            daily_date=pd.Series([series_date[step] for step in range(len(series_date)) if step%day_count==0 and step>1])  #약 1일
                            weekly_date=pd.Series([series_date[step] for step in range(len(series_date)) if step%week_count==0 and step>1])  #약 1주일
                            month_date=pd.Series([series_date[step] for step in range(len(series_date)) if step%month_count==0 and step>1])  #약 1달

                            daily_NI=pd.Series([series_PV[step]-series_PV[step-day_count] for step in range(len(series_PV)) if step%day_count==0 and step>1])
                            weekly_NI=pd.Series([series_PV[step]-series_PV[step-week_count] for step in range(len(series_PV)) if step% week_count==0 and step>1] )
                            monthly_NI = pd.Series([series_PV[step] - series_PV[step - month_count] for step in range(len(series_PV)) if step % month_count == 0 and step > 1])

                            daily_csv= pd.concat([daily_date,daily_NI],axis=1)
                            daily_csv.columns=['date','daily NI']
                            weekly_csv=pd.concat([weekly_date,weekly_NI],axis=1)
                            weekly_csv.columns=['date','weekly NI']
                            monthly_csv=pd.concat([month_date,monthly_NI],axis=1)
                            monthly_csv.columns = ['date', 'monthly NI']

                            daily_csv.to_csv('z_daily NI_backtest result')
                            weekly_csv.to_csv('z_weekly NI_backtest result')
                            monthly_csv.to_csv('z_monthly NI_backtest result')

                            print(weekly_csv)
                            print('APPO_LS total NI :',res_PV[-1] - (long_agent.init_cash + short_agent.init_cash))

                        elif len(params.short_or_long_data) == 1:  # short or long 인경우
                            ax[0].set_ylabel('NI_of_' + is_short_or_long + '_Agent')
                            ax[0].plot(np.array(agent.date_data).reshape(-1), (torch.Tensor(PV_data) - agent.init_cash))

                        ax[3].set_ylabel('input')
                        ax[3].plot(np.array(agent.date_data).reshape(-1), np.array(agent.long_ori_input[0][:]))
                        for d in range(len(agent.long_ori_input)):
                            if d > 0:  # 두번째 부터 출력(첫번째는 이전에 출력했음)
                                ax[3].plot(np.array(agent.long_ori_input[d][:]))


                    if is_short_or_long == 'long':  # 롱인경우
                        ax[1].set_ylabel(is_short_or_long + '_Rulebase')
                        try:
                            ax[1].scatter(buy_date, np.array(torch.cat(buy_data)), marker='v', color='red')
                        except:
                            pass
                        try:
                            ax[1].scatter(sell_date, np.array(torch.cat(sell_data)), marker='v', color='blue')
                        except:
                            pass
                        ax[1].plot(price_data.numpy())



                    if is_short_or_long == 'short':  # 숏인경우
                        ax[2].set_ylabel(is_short_or_long + '_Rulebase')
                        try:
                            ax[2].scatter(buy_date, np.array(torch.cat(buy_data)), marker='v', color='blue')
                        except:
                            pass
                        try:
                            ax[2].scatter(sell_date, np.array(torch.cat(sell_data)), marker='v', color='red')
                        except:
                            pass
                        ax[2].plot(price_data.numpy())

                plt.show()




    def mul_back_test(self):  # 병렬 백테스트( 숏 롱 )

        long_res_=[]
        short_res_=[]
        res_dict= {}

        print('병렬 백테스트 시작.', '코어수:', multiprocessing.cpu_count())

        if params.train_stock_or_future != 'future':
            params.short_or_long_data = ['long']  # 주식 또는 코인인경우 롱만진행

        for is_short_or_long in params.short_or_long_data:
            print(is_short_or_long,'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
            long_res,short_res=self.forward(is_short_or_long,long_res_,short_res_)

            if is_short_or_long == 'long':
                res_ = long_res[0]
            if is_short_or_long == 'short':
                res_ = short_res[0]

            res_dict[is_short_or_long] = res_[0]

        return res_dict



if __name__ == '__main__':
    for symbol_n in params.API_data_name:
        bk = forward_testing(params.back_train_val_test,symbol_n)
        res_data = bk.mul_back_test()
        bk.res_plot(res_data)

        long_res_ = []
        short_res_ = []
        for is_short_or_long in params.short_or_long_data:  # 스레드 시뮬레이션 ( 메모리 낮은경우 대비)
            long_res, short_res = bk.back_test(is_short_or_long, long_res_, short_res_)
            if is_short_or_long == 'long':
                res_ = long_res[0]
            if is_short_or_long == 'short':
                res_ = short_res[0]
            res_data[is_short_or_long] = res_


        if __name__ == '__main__': ######플랏
            if bk.is_back_testing == True:  # 백테스팅일경우 출력

                total_dim = len(params.short_or_long_data)


                #지표 길이 다를 경우 인덱스 값 맞춘다
                ind_diff = {}
                if len(res_data['long'].PV_data['long']) > len(res_data['long'].PV_data['short']):  # 롱데이터수 더길면 (롱의 지표변수가 더짧다)

                    ind_diff['long'] = np.abs(len(res_data['long'].PV_data['long']) - len(res_data['short'].PV_data['short']))  # 그래프 계산시 얼마나 빼야할지
                    ind_diff['short'] = 0

                    PV_data = res_data['long'].PV_data['long']
                    res_data['long'].PV_data['long'] = PV_data[ind_diff['long']:]

                    #매매date 인덱스 조절
                    res_data['long'].buy_date['long'] = res_data['long'].buy_date['long'] - ind_diff['long'] # 인덱스 조절
                    res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'] - ind_diff['long']  # 인덱스 조절

                    #사고 판 가격 인덱스 조절
                    res_data['long'].buy_data['long'] = res_data['long'].buy_data['long'][len(res_data['long'].buy_date['long'][res_data['long'].buy_date['long'] < 0]):]
                    res_data['long'].sell_data['long'] = res_data['long'].sell_data['long'][len(res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] < 0]):]

                    # 매매 date 인덱스 조절2
                    res_data['long'].buy_date['long'] = res_data['long'].buy_date['long'][res_data['long'].buy_date['long'] >= 0].tolist()
                    res_data['long'].sell_date['long'] = res_data['long'].sell_date['long'][res_data['long'].sell_date['long'] >= 0].tolist()

                    # 가격, 날짜 인덱스 조절
                    res_data['long'].agent.price_data = bk.price_data['long'][ind_diff['long']:]
                    res_data['long'].agent.date_data = bk.date_data['long'][ind_diff['long']:]


                elif len(res_data['long'].PV_data['long']) < len(res_data['long'].PV_data['short']):  # 숏이20 더길면 숏에 20이 들어감
                    ind_diff['short'] = np.abs(len(res_data['short'].PV_data['short']) - len(res_data['long'].PV_data['long']))
                    ind_diff['long'] = 0

                    PV_data = res_data['short'].PV_data['short']
                    res_data['short'].PV_data['short'] = PV_data[ind_diff['short']:]

                    # 매매date 인덱스 조절
                    res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'] - ind_diff['short']  # 인덱스 조절
                    res_data['short'].sell_date['short'] = res_data['short'].sell_date['short'] - ind_diff['short']  # 인덱스 조절

                    # 사고 판 가격 인덱스 조절
                    res_data['short'].buy_data['short'] = res_data['short'].buy_data['short'][len(
                        res_data['short'].buy_date['short'][res_data['short'].buy_date['short'] < 0]):]
                    res_data['short'].sell_data['short'] = res_data['short'].sell_data['short'][len(
                        res_data['short'].sell_date['short'][res_data['short'].sell_date['short'] < 0]):]

                    # 매매 date 인덱스 조절2
                    res_data['short'].buy_date['short'] = res_data['short'].buy_date['short'][
                        res_data['short'].buy_date['short'] >= 0].tolist()
                    res_data['short'].sell_date['short'] = res_data['short'].sell_date['short'][
                        res_data['short'].sell_date['short'] >= 0].tolist()

                    # 가격, 날짜 인덱스 조절
                    res_data['short'].agent.price_data = bk.price_data['short'][ind_diff['short']:]
                    res_data['short'].agent.date_data = bk.date_data['short'][ind_diff['short']:]




                else:  # 길이같으면
                    ind_diff['short'] = 0
                    ind_diff['long'] = 0


                for dim in range(total_dim):
                    is_short_or_long = params.short_or_long_data[dim]
                    # 앞에 self 붙으면 class안에 중복된 이름있기 때문에 dict이 소실됨
                    agent = res_data[is_short_or_long].agent
                    PV_data = res_data[is_short_or_long].PV_data[is_short_or_long]
                    action_data = res_data[is_short_or_long].action_data[is_short_or_long]
                    buy_data = res_data[is_short_or_long].buy_data[is_short_or_long]
                    buy_date = res_data[is_short_or_long].buy_date[is_short_or_long]
                    sell_data = res_data[is_short_or_long].sell_data[is_short_or_long]
                    sell_date = res_data[is_short_or_long].sell_date[is_short_or_long]
                    price_data = res_data[is_short_or_long].agent.price_data.view(-1)

                    total_date = 0
                    if is_short_or_long == 'long':
                        total_date = long_total_date
                    else:
                        total_date = short_total_date

                    day_index, week_index, daily_index_zip, weekly_index_zip = bk.PV_date_compute2(total_date,'18:00:00', 'False') #미국 시간


                    day_index= day_index - (ind_diff['short']+ ind_diff['long'])
                    week_index= week_index - (ind_diff['short']+ ind_diff['long'])
                    daily_index_zip = np.array(daily_index_zip) - (ind_diff['short']+ ind_diff['long'])
                    weekly_index_zip = np.array(weekly_index_zip) - (ind_diff['short']+ ind_diff['long'])

                    #weekly_index_zip = weekly_index_zip - (ind_diff['short']+ ind_diff['long'])
                    #daily_index_zip = daily_index_zip - (ind_diff['short'] +ind_diff['long']
                    #week _index = week_index -(ind_diff['short'] +ind_diff['long']

                    #PV를 뽑아서 계산한다.


                    init_PV = params.cash  # PV

                    if dim == 0:  # 처음 dim에 출력
                        if len(params.short_or_long_data) > 1:  # 헷지모드 on( 롱숏 둘다 백테스트 하며 각각 PV연산하여 합산)
                            long_agent = res_data['long'].agent
                            long_PV_data = res_data['long'].PV_data['long']

                            short_agent = res_data['short'].agent
                            short_PV_data = res_data['short'].PV_data['short']

                            long_data_date = long_agent.date_data
                            short_data_date = short_agent.date_data

                            # 길이 일치
                            PV_data_set = [long_PV_data, short_PV_data]
                            date_set = [long_data_date, short_data_date]

                            less_data = PV_data_set[np.argmin([len(long_PV_data), len(short_PV_data)])]  # 갯수가 더 적은 데이터
                            more_data = PV_data_set[np.argmax([len(long_PV_data), len(short_PV_data)])]
                            more_date = date_set[np.argmax([len(long_data_date), len(short_data_date)])]  # 갯수 더 많은 날짜 데이터
                            len_diff = np.abs(len(long_PV_data) - len(short_PV_data))  # 차이


                            if less_data==PV_data_set[0]: #데이터수 적은 에이전트
                                less_agent=long_agent
                                len_less_data=len(less_data)
                            else:
                                less_agent=short_agent
                                len_less_data=len(less_data)

                            less_data = torch.cat([torch.ones(len_diff)*(less_agent.init_cash), torch.Tensor(less_data).view(-1)]).view(-1)  # 적었던 PV데이터 0으로 채워서 길이 일치



                            if len_diff == 0:  # 차이가 없는경우
                                more_data = long_PV_data
                                less_data = short_PV_data

                            if less_agent==short_agent: # 데이터수 적은게 숏인경우
                                short_res_PV=torch.Tensor(more_data[-len_less_data:]).view(-1) + torch.Tensor(less_data[-len_less_data:]).view(-1)
                                long_res_PV= torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)

                            elif less_agent==long_agent: # 데이터수 적은게 롱인경우
                                short_res_PV=torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)
                                long_res_PV=torch.Tensor(more_data[-len_less_data:]).view(-1) + torch.Tensor(less_data[-len_less_data:]).view(-1)

                            res_PV = torch.Tensor(more_data).view(-1) + torch.Tensor(less_data).view(-1)  # PV 합
                            res_date = more_date

                            g1.set_ylabel('NI of short and long Agent')
                            g1.plot(res_date, res_PV[-len(res_date):] - (long_agent.init_cash + short_agent.init_cash))


                        elif len(params.short_or_long_data) == 1:  # short or long 인경우
                            g1.set_ylabel('NI_of_' + is_short_or_long + '_Agent')
                            g1.plot(agent.date_data, torch.Tensor(PV_data) - agent.init_cash)



                    if is_short_or_long == 'long':  # 롱인경우

                        plot_size = 3000

                        sell_plot_size = sell_date.index(np.array(sell_date)[np.array(sell_date) > len(long_res_PV) - plot_size][0])
                        diff_sell_value = len(long_res_PV)-plot_size


                        buy_plot_size = buy_date.index(np.array(buy_date)[np.array(buy_date) > len(long_res_PV) - plot_size][0])
                        diff_buy_value = len(long_res_PV)-plot_size

                        g2.set_ylabel(is_short_or_long + '_AI')
                        g2.scatter(buy_date, buy_data, marker='v', color='red')
                        g2.scatter(sell_date, sell_data, marker='v', color='blue')
                        g2.plot(price_data)

                        g4.set_ylabel(is_short_or_long + '_AI')
                        g4.scatter(np.array(buy_date[buy_plot_size:])-diff_buy_value, buy_data[buy_plot_size:], marker='v', color='red')
                        g4.scatter(np.array(sell_date[sell_plot_size:])-diff_sell_value, sell_data[sell_plot_size:], marker='v', color='blue')
                        g4.plot(price_data[-plot_size:])


                    if is_short_or_long == 'short':  #숏인경우
                        try:
                            plot_size = 3000
                            sell_plot_size = sell_date.index(np.array(sell_date)[np.array(sell_date) > len(short_res_PV) - plot_size][0])
                            diff_sell_value = len(short_res_PV)-plot_size

                            buy_plot_size = buy_date.index(np.array(buy_date)[np.array(buy_date) > len(short_res_PV) - plot_size][0])
                            diff_buy_value = len(short_res_PV)-plot_size

                            g3.set_ylabel(is_short_or_long + '_AI')
                            g3.scatter(buy_date, buy_data, marker='v', color='blue')
                            g3.scatter(sell_date, sell_data, marker='v', color='red')
                            g3.plot(price_data)

                            g5.set_ylabel(is_short_or_long + '_AI')
                            g5.scatter(np.array(buy_date[buy_plot_size:])-diff_buy_value, buy_data[buy_plot_size:], marker='v', color='blue')
                            g5.scatter(np.array(sell_date[sell_plot_size:])-diff_sell_value, sell_data[sell_plot_size:], marker='v', color='red')
                            g5.plot(price_data[-plot_size:])

                            plt.text(0.7, 0.7, 'Type E,  Long:up,down>LRCCI LRLRA up down    short:up,down,LRLRA,LRCCI')
                            plt.text(0.7, 0.2, 'Long Net income: ' + str(long_PV_data[-1] - long_PV_data[day_index]))
                            plt.text(0.7, 0.1, 'Short Net income: ' + str(short_PV_data[-1] - short_PV_data[day_index]))
                            plt.text(0.7, 0.6, 'daily Net income: ' + str(res_PV[-1] - res_PV[day_index]))

                        except:
                            pass

                print('APPO_LS total NI :', res_PV[-1] - (long_agent.init_cash + short_agent.init_cash))
                print('APPO_LS Daily Net income :', res_PV[-1] - res_PV[day_index])

                print('---------------------------------------------------------------------------------')
                daily_date=[]
                daily_total_NI=[]
                daily_short_NI=[]
                daily_long_NI=[]

                total_date=total_date[params.window-1:]
                for step in range(len(daily_index_zip)-1):
                    print(total_date[daily_index_zip[step+1]],'daily 순 수익금 :',res_PV[daily_index_zip[step+1]]-res_PV[daily_index_zip[step]])
                    daily_date.append(total_date[daily_index_zip[step+1]])
                    daily_total_NI.append((res_PV[daily_index_zip[step+1]]-res_PV[daily_index_zip[step]]).item())
                print('---------------------------------------------------------------------------------')

                for step in range(len(daily_index_zip)-1):
                    print(total_date[daily_index_zip[step+1]],'daily long 순 수익금 : ', long_PV_data[daily_index_zip[step+1]]-long_PV_data[daily_index_zip[step]])
                    daily_long_NI.append((long_PV_data[daily_index_zip[step+1]]-long_PV_data[daily_index_zip[step]]).item())
                print('---------------------------------------------------------------------------------')



                for step in range(len(daily_index_zip)-1):
                    print(total_date[daily_index_zip[step+1]],'daily short 순 수익금 :', short_PV_data[daily_index_zip[step+1]]-short_PV_data[daily_index_zip[step]])
                    daily_short_NI.append((PV_data[daily_index_zip[step+1]]-PV_data[daily_index_zip[step]]).item())
                print('---------------------------------------------------------------------------------')


                all_date = pd.Series(total_date[params.window-1:])
                all_PV= pd.Series(res_PV)
                all_PV=pd.concat([all_date,all_PV],axis=1)
                all_PV.to_csv('all_PV')

                daily_date=pd.Series(daily_date)
                daily_total_NI=pd.Series(daily_total_NI)
                daily_short_NI=pd.Series(daily_short_NI)
                daily_long_NI=pd.Series(daily_long_NI)

                daily_total_NI=pd.concat([daily_date,daily_total_NI],axis=1)
                daily_short_NI=pd.concat([daily_date,daily_short_NI],axis=1)
                daily_long_NI=pd.concat([daily_date,daily_long_NI],axis=1)

                daily_total_NI.columns=['date','NI']
                daily_short_NI.columns=['date','NI']
                daily_long_NI.columns=['date','NI']

                daily_total_NI.to_csv('za_realtime_total_NI')
                daily_short_NI.to_csv('za_realtime_short_NI')
                daily_long_NI.to_csv('za_realtime_long_NI')


                print('사용지표 Long:지지 저항    short:지지저항 학습 이후 LRCCI LRLRA 지지 저항')




            plt.show(block=False)
            plt.pause(300)
            plt.close()
            bk.reset()



