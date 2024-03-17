
import numpy as np
import random
import torch
#

#




seed=1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


trading_site = 
Binance_API_key =
Binance_Secret_key =




# 여러개 백테스트 기능 (끝) , 백테스트 완료 ,전진분석, 실전완료
# PV계산방식 확인해야함,   전진분석, 실전은 동시에 이뤄질수 있게끔 작업
# 실시간 PV띄우기, 전진 실시간 PV띄우기





API_coin_name=[] #종목명(실제 트레이딩시 사용)
API_data_name=[] #  API에서 불러올 이름####################################################

data_count=['2020-01-01 00:00', '2024-03-07 22:06']   #호출할 데이터 갯수[날짜 , count]  학습시 날짜이전 count 만큼 호출 #####################################시 분초 쓰지않음
api_data_count=99999900 #실시간 트레이딩 할때 불러올 API data 갯수( 회당 1000개정도가 한계)
minute=60   #데이터의 분봉 ex) 3
# 으로 설정하면 3분마다


coin_or_stock='coin' # 불러올 데이터
train_stock_or_future='coin' # 학습, 트레이딩 방식 coin or future  (코인 현물인지 선물인지)  - 현물이면 long 만 함
real_or_train='train' # train은 날짜로 뽑음 , 실시간으로 돌리면 자동으로 real이됨 변경 x


PV_reward=False
PV_reward_only=True


#기능 참 거짓 설정 파라미터
traj_print= False
data_insert= False #api 데이터 실시간으로 인서트 할건지
plot_print= True
insert_yf_data= True # 실시간 데이터 인서트 허용


#금융 파라미터 (선물 - 학습시 선물 방식)
slippage=5 #슬리피지 (달러)
backtest_slippage=5 #백테스트 슬리피지(달러)  #나스닥:5 ES:12.5 골드: 10

coin_future_cost=0.02 #수수료 달러 (선물 계산시)
stock_cost=0.001 # 설정값 계수0.001 = 0.1% (현물인경우)

point_value= 1300 # 달러 가치  GC:100 ES:50 NQ:20 ES:50 FDX(Dax):25 Dow(YM):5
deposit=1  # 증거금( 코인에서는 1로 고정)
cash=1000 #KRW 원 (초기금액.백테스트시적용)

#금융 파라미터( 주식 - 실전 트레이딩 주식으로)
#실전 트레이딩

limit_unit=1 #최대 보유 리미트 유닛

#학습시 파라미터
ratio=[2,8,1]  #train,val,test set 비율 train과 val만 넣는다
train_val_test='total'
device='cpu'



if device =='cuda':
    print(torch.cuda.is_available(),'쿠다 설정 확인')

#long_ind_name=['NNCO_up_L','NNCO_down_L','NNCO_up_S','NNCO_down_S','log_return','CCI_exercise','CCI_trend','tanos']
long_ind_name=['CCI_trend','NNCO_up_L','NNCO_down_L']
short_ind_name=['CCI','NNCO_up_S','NNCO_down_S']


input_dim={'short':len(short_ind_name),'long':len(long_ind_name)}  #Network input dim  숏과 롱의 dim을 dict로 넣는다
short_or_long_data=['long','short']###########################################



#백테스팅시 기능 파라미터
multi_or_thread='multi' #백테스팅 멀티프로세싱으로 할건지 ('multi', 'thread')
back_train_val_test='total'
if_real_time='True' # 실시간 데이터도 합쳐서 불러옴
