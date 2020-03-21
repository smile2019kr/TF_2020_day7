from flask import Flask
from flask import render_template, request, jsonify
import re
from calculator.ai_calculator import Calculator
from cabbage.cabbage import Cabbage
from blood.blood import Blood

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/move/<path>')  #<>를 주면 변수명으로 처리됨. 해당페이지로 이동
def move(path):
    return render_template('{}.html'.format(path))   # move(path)에서 입력받은 해당path를 {}안에 넣음

@app.route('/calculator')  # 내부에서 계산하라는 URL
def ui_calculator():  # 인공지능없는 그냥 계산기
    stmt = request.args.get('stmt', 'NONE')
    if(stmt == 'NONE'):
        print('넘어온 값이 없음')
    else:
        print(f'넘어온 식: {stmt}')
        patt = '[0-9]+'  # 숫자가 반드시 하나이상 있어야 한다는 것
        op = re.sub(patt, '', stmt)  # patt에서 stmt를 빼기
        nums = stmt.split(op)
        result = 0
        n1 = int(nums[0])
        n2 = int(nums[1])
        if op == '+': result = n1 + n2
        elif op == '-': result = n1 - n2
        elif op == '*': result = n1 * n2
        elif op == '/': result = n1 / n2

    return jsonify(result = result)  # 위에서 파이썬 값으로 만들어진 것을 자바스크립트 값으로 바꿔주어야 함

@app.route('/ai_calculator', methods=['POST'])
def ai_calculator():
    num1 = request.form['num1']
    num2 = request.form['num2']
    opcode = request.form['opcode']
    # ai_calculator가 계산한 값을 가져와야 하므로 ai_calculator의 Calculator 클래스를 불러와서 result값에 넣어야 함
    result = Calculator.service(num1, num2, opcode)
    render_params = {}
    render_params['result'] = int(result)  #텐서의 기본형은 float이므로 자연스러운 표기를 위해 int로 변경
    return render_template('ai_calculator.html', **render_params)
    # ai_calculator.html에 있는 값(num1, num2, opcode)을 가져와서 render_params에 있는 값을 뿌리는 것


@app.route('/cabbage', methods=['POST'])
def cabbage():
    avg_temp = request.form['avg_temp']
    min_temp = request.form['min_temp']
    max_temp = request.form['max_temp']
    rain_fall = request.form['rain_fall']
    cabbage = Cabbage()
    cabbage.initialize(avg_temp, min_temp, max_temp, rain_fall)
    result = cabbage.service()
    render_params = {}
    render_params['result'] = result
    return render_template('cabbage.html', **render_params)


@app.route('/blood', methods=['POST'])
def blood():
    weight = request.form['weight']
    age = request.form['age']
    blood = Blood()
    blood.initialize(weight, age)
    result = blood.service()

    render_params = {}
    render_params['result'] = result
    return render_template('blood.html', **render_params)


if __name__ == '__main__':
    app.run()
