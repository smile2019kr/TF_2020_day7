import tensorflow.compat.v1 as tf   #계산기능을  쓸때는 tf ver1.0으로 써야 함. ver2.0은 즉시실행이지만 여기서는 모델저장후 호출
tf.disable_v2_behavior()
import os

class Calculator:
    @staticmethod
    def add_model():
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}  #더미값
        r = tf.add(w1, w2, name='op_add')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # 모델 저장
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'덧셈 결과 {result}')
        saver.save(sess, './calculator_add_model/model', global_step=1000) #세션안에 있는 모델을 ''에 저장한다는 것

    def sub_model():
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}  # 더미값. 의미없는 숫자이지만 초기값으로 제시한 것. 훈련용으로 더미값을 준것일뿐 (지도학습)
        r = tf.subtract(w1, w2, name='op_sub')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # 모델 저장
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'뺄셈 결과 {result}')
        saver.save(sess, './calculator_sub_model/model', global_step=1000)

    def mul_model():
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}  # 더미값
        r = tf.multiply(w1, w2, name='op_mul')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # 모델 저장
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'곱셈 결과 {result}')
        saver.save(sess, './calculator_mul_model/model', global_step=1000)

    def div_model():
        w1 = tf.placeholder(tf.float32, name='w1')
        w2 = tf.placeholder(tf.float32, name='w2')
        feed_dict = {'w1': 8.0, 'w2': 2.0}  # 더미값
        r = tf.divide(w1, w2, name='op_div')
        sess = tf.Session()
        _ = tf.Variable(initial_value='fake_variable')
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # 모델 저장
        result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
        print(f'나눗셈 결과 {result}')
        saver.save(sess, './calculator_div_model/model', global_step=1000)


    @staticmethod
    def service(num1, num2, opcode):
        print(f'{num1} {opcode} {num2}')
        tf.reset_default_graph()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph(f'calculator/calculator_{opcode}_model/model-1000.meta')
            saver.restore(sess, tf.train.latest_checkpoint(f'calculator/calculator_{opcode}_model'))
            # checkpoint에 있는 모델을 sess에 넣어야 76행에서 sess.run 실행
            graph = tf.get_default_graph()  # 5행위에서 reset시킨 디폴트값 가져옴
            w1 = graph.get_tensor_by_name('w1:0')
            w2 = graph.get_tensor_by_name('w2:0')
            feed_dict = {w1: float(num1), w2: float(num2)}
            for key in feed_dict.keys():
                print(key, ':', feed_dict[key])
            op_to_restore = graph.get_tensor_by_name(f'op_{opcode}:0')
            result = sess.run(op_to_restore, feed_dict)
            print(f'텐서가 계산한 결과: {result}')
        return result


if __name__ == '__main__':
    if not os.path.exists('calculator_add_model/checkpoint'):
        Calculator.add_model()
    # 네개의 모델을 한꺼번에 일괄실행해서는 안됨. 왜냐하면 세션이 전역이기 때문.
    # 덧셈만 실행해보고 나머지는 주석처리해서 각각 잘 돌아가는지 확인
    # 덧셈 제대로 실행된 이후에는 if문으로 설정해서 checkpoint 파일이 생성되어있으면 add_model이 실행되지않도록 설정
    if not os.path.exists('calculator_sub_model/checkpoint'):
        Calculator.sub_model()
    if not os.path.exists('calculator_mul_model/checkpoint'):
        Calculator.mul_model()
    if not os.path.exists('calculator_div_model/checkpoint'):
        Calculator.div_model()
