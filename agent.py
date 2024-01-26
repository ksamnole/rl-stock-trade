from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random
from collections import deque


class Agent:
    def __init__(self, n_features, use_exploration=False, name_model="", random_action_min=0.1,
                 random_action_decay=0.999995, n_neurons=64, future_reward_importance=0.95):
        self.memory = deque(maxlen=100000)
        self.model_name = name_model
        self.use_exploration = use_exploration
        self.actions = ['hold', 'buy', 'sell']
        self.action_size = len(self.actions)
        self.gamma = future_reward_importance 
        self.epsilon = 1.0
        self.epsilon_min = random_action_min  # we want the agent to explore at least this amount.
        self.epsilon_decay = random_action_decay  # we want to decrease the number of explorations as it gets good
        self.num_trains = 0
        self.num_neurons = n_neurons
        self.num_features = n_features
        self.model =  self._nn_old(name_model) if name_model != '' else self._nn_new(n_features, n_neurons)
        self.model.summary()

    def _nn_old(self, name_model=''):
        model = load_model("files/output/" + name_model);
        return model

    def _nn_new(self, n_features, n_neurons):
        model = Sequential()
        model.add(Dense(units=np.maximum(int(n_neurons/ 1), 1), activation="relu", input_dim=n_features))
        model.add(Dense(units=np.maximum(int(n_neurons/ 2), 1), activation="relu"))
        model.add(Dense(units=np.maximum(int(n_neurons/ 8), 1), activation="relu"))
        model.add(Dense(units=self.action_size, activation="linear"))
        model.compile(loss="mse", optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_best_action(self, state):
        if self.use_exploration == True:
            prob_exploit = np.random.rand()
            if prob_exploit < self.epsilon:
                random_action = random.randrange(self.action_size)
                return random_action

        pred = self.model.predict(state)
        best_action = np.argmax(pred[0])
        return best_action

    def experience_replay(self, batch_size):
        memory_batch = self.prepare_mem_batch(batch_size)

        for curr_state, action, reward, next_state, done in memory_batch:
            #print(f'curr_state={curr_state}, next_state={next_state}, reward={reward}, action ={action}')
            if not done:
                reward_pred = self.model.predict(next_state)
                target = reward + self.gamma * np.amax(reward_pred[0])
            else:
                target = reward
            y_f = self.model.predict(curr_state)
            y_f[0][action] = target
            self.model.fit(curr_state, y_f, epochs=1, verbose=0)
            self.num_trains += 1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def prepare_mem_batch(self, mini_batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - mini_batch_size, l):
            mini_batch.append(self.memory[i])
        return mini_batch
