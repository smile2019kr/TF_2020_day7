import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas.io.parsers import read_csv
import numpy as np
import pandas as pd
import plotly.offline as offline
import plotly.graph_objs as go

class Trader:
    def __init__(self):
        pass
        self.code_df = pd.DataFrame({'name':[], 'code':[]})

    def krx_crawl(self):
        self.code_df = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',header=0)[0]
        self.code_df.종목코드 = self.code_df.종목코드.map('{:06d}'.format())
        self.code_df = self.code_df[['회사명','종목코드']]
        self.code_df = self.code_df.rename(columns={'회사명':'name', '종목코드':'code'})
    def code_df_head(self):
        print(self.code_df.head())
    def get_url(self, item_name, code_df):
        code = code_df.query("name=='{}'".format(item_name))['code'].to_string(index=False)
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}'.format(code='005930') # 'code'로 대체
        print('요청 URL = {}'.format(url))
        return url
    def test(self, code):
        # item_name = '삼성전자'
        # url = self.get_url(item_name, self.code_df)
        df = pd.DataFrame()
        for page in range(1, 21):
            pg_url = 'https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=code, page=page)
            df = df.append(pd.read_html(pg_url, header=0)[0], ignore_index=True)
        df.dropna(inplace = True) # na 결측값 있는 행 제거
        return df
    def rename_item_name(self, param):
        df = param.rename(columns = {'날짜':'date', '종가':'close','전일비':'diff',
                                           '시가':'open','고가':'high','저가':'low','거래량':'volumn'})
        df[['close','diff','open','high','low','volumn']] =\
            df[['close','diff','open','high','low','volumn']].astype(int)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by=['date'], ascending=True)
        return df


    def crawling(self):
        import time
        from datetime import datetime, timedelta
        import re
        import pandas as pd
        ts = time.time()
        """ 라이브러리 호출 """
        import urllib
        from bs4 import BeautifulSoup
        import requests
        """ 회사코드 및 조회기간 설정 """
        symbol = '005930'
        startTime = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')
        count = str(1000)
        """ url 설정 """
        url = 'https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&startTime={}&count={}&requestType=2'.format(
            symbol, startTime, count)
        """ 크롤링 & 전처리 """
        r = requests.get(url)
        html = r.content
        soup = BeautifulSoup(html, 'html.parser')
        tr = soup.find_all('item')
        cols = ['일자', '시가', '고가', '저가', '종가', '거래량']
        list = []
        for i in range(0, len(soup.find_all('item'))):
            list.append(re.search(r'"(.*)"', str(tr[i])).group(1).split('|'))
        df = pd.DataFrame(list, columns=cols)
        df['일자'] = pd.to_datetime(df['일자'].str[:4] + '-' + df['일자'].str[4:6] + '-' + df['일자'].str[6:])
        df.set_index(df['일자'], inplace=True)
        df = df.drop(columns='일자')
        print('작동소요시간 :', round(time.time() - ts, 1), '초')
        df.to_csv('samsungOHLC.csv')

    def model(self):
        tf.global_variables_initializer()
        data = read_csv('samsungOHLC.csv', sep=',')
        data = data.iloc[:,1:6] #iloc 로 필요한 부분만 읽어들이기
        print(data.head())
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:, 1:-1]
        print(x_data)
        y_data = xy[:, [3]]
        #데이터구조 확인
        print("type:{}".format(type(xy)))
        print("shape:{}, dimension: {}, dtype: {}".format(xy.shape, xy.ndim, xy.dtype))
        print("Array's data:\n", xy)
        # 이하는 3/28수업에서 수정 필요
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        # 뉴런( y = wx + b) 을 구성하는 4개 값을 지정.
        hypothesis = tf.matmul(X, W) + b  # tf가 이해할 수 있는 공식 (tf.matmul , tf.add , .... )
        cost = tf.reduce_mean(tf.square(hypothesis - Y))  # tf에서 cost(예측값과 실제값의 차이)가 최소화되는 값을 계산
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=0.000005)  # tf2에서는 optimizer를 Adam으로 사용했었음. 더 나은 것으로 조합하는것
        train = optimizer.minimize(cost)  # cost를 최소화하는 최적화 값을 학습
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 기록할때마다 이전의 값을 초기화
            for step in range(100000):
                cost_, hypo_, _ = sess.run([cost, hypothesis, train], {X: x_data, Y: y_data})
                if step % 500 == 0:  # cost 감소 경향을 확인하기 위함
                    print(f' step: {step}, cost: {cost_} ')
                    print(f' price: {hypo_}')
            saver = tf.train.Saver()
            saver.save(sess, 'trader.ckpt')  # ckpt :  체크포인트. 실제데이터를 계속 누적/저장


if __name__ == '__main__':
    print('>>>')
    m = Trader()
    def print_menu():
        print('0. EXIT\n'
              '1. 종목헤드\n'
              '2. 종목컬럼명 보기\n'
              '3. 전처리결과 보기\n'
              '4. 종목 크롤링 및 csv 저장\n'
              '5. 모델 생성\n')
        return input('CHOOSE ONE \n')


    m = Trader()
    while 1:
        menu = print_menu()
        print('MENU %s \n' % menu)
        if menu == '0':
            break
        elif menu == '1':
            m.code_df_head()
        elif menu == '2':
            print(m.test('005930'))
        elif menu == '3':
            print(m.rename_item_name(m.test('005930')))
        elif menu == '4':
            m.crawling()
        elif menu == '5':
            m.model()
