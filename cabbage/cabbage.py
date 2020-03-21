import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas.io.parsers import read_csv
import numpy as np
    # avgTemp,minTemp,maxTemp,rainFall,avgPrice
    #계산기능을  쓸때는 tf ver1.0으로 써야 함. ver2.0은 즉시실행이지만 여기서는 모델저장후 호출

class Cabbage:
    def model(self):
        tf.global_variables_initializer()
        data = read_csv('cabbage_price.csv', sep=',')
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:, 1:-1]
        y_data = xy[:, [-1]]
        # 위의 4행은 data science파트. 데이터 전처리에 해당하는 과정이므로 가장 시간 오래걸림.
        X = tf.placeholder(tf.float32, shape=[None, 4])
        Y = tf.placeholder(tf.float32, shape=[None, 1])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        # 뉴런( y = wx + b) 을 구성하는 4개 값을 지정.
        hypothesis = tf.matmul(X, W) + b   # tf가 이해할 수 있는 공식 (tf.matmul , tf.add , .... )
        cost = tf.reduce_mean(tf.square(hypothesis - Y))  # tf에서 cost(예측값과 실제값의 차이)가 최소화되는 값을 계산
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)  # tf2에서는 optimizer를 Adam으로 사용했었음. 더 나은 것으로 조합하는것
        train = optimizer.minimize(cost)  # cost를 최소화하는 최적화 값을 학습
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 기록할때마다 이전의 값을 초기화
            for step in range(100000):
                cost_, hypo_, _ = sess.run([cost, hypothesis, train], {X: x_data, Y: y_data})
                if step % 500 == 0:  # cost 감소 경향을 확인하기 위함
                    print(f' step: {step}, cost: {cost_} ')
                    print(f' price: {hypo_}')
            saver = tf.train.Saver()
            saver.save(sess, 'cabbage.ckpt')  # ckpt :  체크포인트. 실제데이터를 계속 누적/저장


#아래에서는 위의 모델에서 학습/저장한 것을 바탕으로 예측하도록 코딩. 위의 모델 작성 후 main절에서 cabbage 모델 실행/생성 후 진행

    def initialize(self, avgTemp, minTemp, maxTemp, rainFall):  #
        self.avgTemp = avgTemp
        self.minTemp = minTemp
        self.maxTemp = maxTemp
        self.rainFall = rainFall

    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        # Y값은 tf가 예측해서 산출되는 값이므로 초기화하지 않음
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())  # 세션 시작시에는 항상 초기화
            saver.restore(sess, 'cabbage/cabbage.ckpt')
            data = [[self.avgTemp, self.minTemp, self.maxTemp,  self.rainFall], ]
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X, W) + b, {X: arr[0:4]})  # 0부터 4 미만 총 4개의 값을 X에 넣어서 뉴런값예측
        return int(dict[0])


if __name__ == '__main__':
    cabbage = Cabbage()
    cabbage.model()
