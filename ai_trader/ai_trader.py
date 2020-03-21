import math
import numpy as np
import random
import pandas_datareader as data_reader
from collections import deque
from tqdm import tqdm
import tensorflow as tf
print(f'Tensorflow version : {tf.__version__}')
class Trader:
    def __init__(self, state_size, action_space=3, model_name="AiTrader"):
        self.state_size = state_size
        self.action_space = action_space
        self.model_name = model_name
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
    def model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.01))
        return model
    def training(self, model, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = model.predict(state)
        return np.argmax(actions[0])
    def batch_train(self, model, batch_size):
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])
        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(model.predict(next_state)[0])
            target = model.predict(state)
            target[0][action] = reward
            model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay
class Trading:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))
    @staticmethod
    def stock_price_format(n):
        if n < 0:
            return '-${0:2f}'.format(abs(n))
        else:
            return '${0:2f}'.format(n)
    @staticmethod
    def dataset_loader(stock_name):
        dataset = data_reader.DataReader(stock_name, data_source='yahoo')
        return dataset['Close']
    def state(self, data, timestep, window_size):
        starting_id = timestep - window_size + 1
        if starting_id >= 0:
            windowed_data = data[starting_id: timestep +1]
        else:
            windowed_data =- starting_id * [data[0]] + list(data[0: timestep+1])
        state = []
        for i in range(window_size -1):
            state.append(self.sigmoid(windowed_data[i + 1] - windowed_data[i]))
        return np.array([state])
    def transaction(self,target):
        stock_name = target
        data = self.dataset_loader(stock_name)
        window_size = 10
        episodes = 1000
        batch_size = 32
        data_samples = len(data) - 1
        trader = Trader(window_size)
        model = trader.model()
        print(f' Model Summary ')
        print(model.summary())
        for episode in range(1, episodes + 1):
            print(f'Episode: {episode}/ {episodes}')
            state = self.state(data, 0, window_size + 1)
            total_profit = 0
            trader.inventory = []
            for t in tqdm(range(data_samples)):
                action = trader.training(model, state )
                next_state = self.state(data, t + 1, window_size + 1)
                reward = 0
                if action == 1: # buying
                    trader.inventory.append(data[t])
                    print(f'AI 트레이더 매수: {self.stock_price_format(data[t])}')
                elif action == 2 and len(trader.inventory) > 0: # selling
                    buy_price = trader.inventory.pop(0)
                    reward = max(data[t] - buy_price, 0)
                    total_profit += data[t] - buy_price
                    print(f'AI 트레이더 매도: {self.stock_price_format(data[t])},'
                          f'이익: {self.stock_price_format(data[t] - buy_price)}')
                if t == data_samples - 1:
                    done = True
                else:
                    done = False
                trader.memory.append((state, action, reward, next_state, done))
                state = next_state
                if done:
                    print(' ########## ')
                    print(f' 총이익 : {total_profit}')
                if len(trader.memory) > batch_size:
                    trader.batch_train(model, batch_size)
            if episode % 10 == 0:
                trader.model.save(f'ai_trader_{episode}.ht')
if __name__ == '__main__':
    trading = Trading()
    trading.transaction('AAPL')


    