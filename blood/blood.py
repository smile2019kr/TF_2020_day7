import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#      I,  the index;
#      A0, 1;
#      A1, the weight;
#      A2, the age;
#      B,  the blood fat content.

class Blood:
    def initialize(self, weight, age):
        self._weight = weight
        self._age = age

    @staticmethod
    def raw_data():
        tf.set_random_seed(777)  #랜덤값을 tf의 777번째의 시드값으로 주면 값이 흔들리지않음
        return np.genfromtxt('blood.txt', skip_header=36)

    @staticmethod
    def model(raw_data):  # model에 raw_data를 주입
        x_data = np.array(raw_data[:, 2:4], dtype=np.float32)
        y_data = np.array(raw_data[:, 4], dtype=np.float32)
        y_data = y_data.reshape(25, 1)
        print(x_data, y_data)

        X = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
        Y = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')
        W = tf.Variable(tf.random_normal([2, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X, W) + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        cost_history = []
        for step in range(2000):
            cost_val, hy_val, _ = sess.run([cost, hypothesis, train], {X: x_data, Y: y_data})
            if step % 500 == 0:  # cost 감소 경향을 확인하기 위함
                    print(f' step: {step}, cost: {cost_val} ')
                    cost_history.append(sess.run(cost, {X: x_data, Y: y_data}))
        saver = tf.train.Saver()
        saver.save(sess, 'blood.ckpt')


    def service(self):
        X = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
        W = tf.Variable(tf.random_normal([2, 1]), name='weight')
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, 'blood/blood.ckpt')
            # app.py실행시 저장명령. app.py의 입장에서 html화면에서 값을 입력받고 실행하는 것이므로 blood경로를 넣어줘야함

            #saver.restore(sess, 'blood.ckpt')
            # main에서 실행하는 경우 (blood.py 내부에서만 실행. 아래에 main을 넣었으므로 가능)는 경로를 없앰
            val = sess.run(tf.matmul(X, W) + b, {X: [[self._weight, self._age]]})  #값을 줄때는 텐서구조 [[]]로 줘야 함

        print(f'혈중 지방농도: {val}')
        if val < 150:
            result = '정상'
        elif 150 <= val < 200:
            result = '경계역 중성지방혈증'
        elif 200 <= val < 500:
            result = '고 중성지방혈증'
        elif 500 <= val < 1000:
            result = '초고 중성지방혈증'
        elif 1000 <= val:
            result = '췌장염 발병 가능성 고도화'
        print(result)
        return result

if __name__ == '__main__':
    blood = Blood()
    raw_data = blood.raw_data()
    #Blood.model(raw_data)
    #blood.initialize(100, 30)  # 100kg에 30살의 경우를 테스트용으로 입력
    blood.service()
