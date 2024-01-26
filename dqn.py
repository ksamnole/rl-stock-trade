from typing import Tuple
from agent import Agent
from utils import *
import numpy as np


class Dqn:
    def __init__(self):
        self.open_orders = []
        self.model_name = ''

    def learn(self, data, name_model, n_episodes, n_features, batch_size, use_exploration,
              random_action_min=0.1, random_action_decay=0.99995, n_neurons=64, future_reward_importance=0.95):

        agent = Agent(n_features, use_exploration, name_model, random_action_min, random_action_decay, n_neurons,
                      future_reward_importance)
        l = len(data) - 1
        rewards_vs_episode = []
        profit_vs_episode = []
        trades_vs_episode = []
        epsilon_vs_episode = []
        for episode in range(1, n_episodes + 1):
            state = self.get_state(data, n_features, n_features)
            total_profits = 0
            total_holds = 0
            total_buys = 1
            total_sells = 0
            total_notvalid = 0
            self.open_orders = [data[0]]

            for t in range(n_features, l):

                action = agent.choose_best_action(state) 

                reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
                    self.execute_action(action, data[t], t, total_profits, total_holds, total_buys, total_sells,
                                        total_notvalid)

                done = True if t == l - 1 else False

                next_state = self.get_state(data, t + 1, n_features)

                if agent.actions[action] == 'buy':
                    immediate_reward = next_state[0][-1]
                elif agent.actions[action] == 'sell':
                    immediate_reward = -next_state[0][-1]
                else:
                    immediate_reward = 0

                reward = immediate_reward

                agent.remember(state, action, reward, next_state, done)
                state = next_state

                if done:
                    reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
                        self.execute_action(2, data[t+1], t+1, total_profits, total_holds, total_buys, total_sells,
                                            total_notvalid)
                    eps = np.round(agent.epsilon, 3)
                    print(f'Episode {episode}/{n_episodes} Total Profit: {formatPrice(total_profits * 100)},'
                          f' Total hold/buy/sell/notvalid trades: {total_holds} / {total_buys} / {total_sells} / {total_notvalid},'
                          f' probability of random action: {eps}')
                    print("---------------------------------------")
                    profit_vs_episode.append(np.round(total_profits, 4))
                    trades_vs_episode.append(total_buys)
                    epsilon_vs_episode.append(eps)

                if len(agent.memory) >= batch_size:     # if enough recorded memory available
                   agent.experience_replay(batch_size)  # fit

        model_name = "files/output/model_ep" + str(n_episodes)
        agent.model.save(model_name)
        print(f'{model_name} saved')
        return profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, agent.num_trains, agent.epsilon

    def execute_action(self, action, close_price, t, total_profits, total_holds, total_buys, total_sells,
                       total_notvalid) -> Tuple[float, float, int, int, int, int]:

        if action == 0:  # hold
            reward = 0
            total_holds += 1
        elif action == 1 and len(self.open_orders) == 0:  # buy
            self.open_orders.append(close_price)
            total_buys += 1
            reward = 0
        elif action == 2 and len(self.open_orders) > 0:  # sell
            bought_price = self.open_orders.pop(0)
            return_rate = close_price / bought_price
            log_return = np.log(return_rate)
            total_profits += log_return
            reward = log_return
            total_sells += 1
        else:
            reward = 0
            total_notvalid += 1

        # total_rewards += reward
        return reward, total_profits, total_holds, total_buys, total_sells, total_notvalid

    # returns a n-day state representation ending at time t of difference bw close prices. ex. [0.5,0.4,0.3,0.2]
    def get_state(self, data, to_ix, n_features):
        from_ix = to_ix - n_features
        data_block = data[from_ix:to_ix + 1]
        res = []
        for i in range(n_features):
            res.append(np.log(data_block[i + 1] / data_block[i]))
        return np.array([res])
