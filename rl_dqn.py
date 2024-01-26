import numpy as np
import pandas as pd
from dqn import Dqn
from utils import plot_barchart, record_run_time
from consts import INPUT_CSV_TEMPLATE




np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


@record_run_time
def run_dqn(name_asset, name_model, n_features, n_neurons, n_episodes, batch_size, random_action_decay, random_action_min, future_reward_importance, use_exploration):

    df = pd.read_csv(INPUT_CSV_TEMPLATE % name_asset)
    data = df['Close'].astype(float).tolist()
    l = len(data) - 1

    print(f'Running {n_episodes} episodes, on {name_asset} (has {l} rows), features={n_features}, '
          f'batch={batch_size}, random_action_decay={random_action_decay}')
    dqn = Dqn()
    profit_vs_episode, trades_vs_episode, epsilon_vs_episode, model_name, num_trains, eps = \
        dqn.learn(data, name_model, n_episodes, n_features, batch_size,
                  use_exploration,
                  random_action_min, random_action_decay, n_neurons, future_reward_importance)

    print(f'Learning completed. Backtest the model {model_name} on any stock')
    print('python backtest.py ')

    print(f'see plot of profit_vs_episode = {profit_vs_episode[:10]}')
    plot_barchart(profit_vs_episode, "episode vs profit", "episode vs profit", "total profit", "episode", 'green')

    print(f'see plot of trades_vs_episode = {trades_vs_episode[:10]}')
    plot_barchart(trades_vs_episode, "episode vs trades", "episode vs trades", "total trades", "episode", 'blue')

    text = f'{name_asset} ({l}), features={n_features}, nn={n_neurons},batch={batch_size}, ' \
           f'epi={n_episodes}({num_trains}), eps={np.round(eps, 1)}({np.round(random_action_decay, 5)})'
    print(f'see plot of epsilon_vs_episode = {epsilon_vs_episode[:10]}')
    plot_barchart(epsilon_vs_episode, "episode vs epsilon", "episode vs epsilon", "epsilon(probability of random action)", text, 'red')
    print(text)


if __name__ == "__main__":
    # python rl_dqn.py -na 'test_sinus' -ne 2000 -nf 20 -nn 64 -nb 20 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_asset'              , '-na', type=str, default='^SPX')  # ^GSPC_2001_2010  ^GSPC_1970_2018  ^GSPC_2011    test_sinus
    parser.add_argument('--name_model'              , '-nm', type=str, default='') # use empty string for new model or else use model_ep0, model_ep10, model_ep20, model_ep20000
    parser.add_argument('--n_episodes'              , '-ne', type=int, default=5)  # (int) > 0 ,minimum 20,000 episodes for good results. episode represent trade and learn on all data.
    parser.add_argument('--n_features'              , '-nf', type=int, default=20)  # (int) > 0
    parser.add_argument('--n_neurons'               , '-nn', type=int, default=64)  # (int) > 0
    parser.add_argument('--n_batch_size'            , '-nb', type=int, default=20)  # (int) > 0 size of a batched sampled from replay buffer for training
    parser.add_argument('--random_action_min'       , '-rm', type=float, default=0.05)  # (int) > 0 size of a batched sampled from replay buffer for training
    parser.add_argument('--random_action_decay'     , '-rd', type=float, default=0.99999)  # (float) 0-1
    parser.add_argument('--future_reward_importance', '-fr', type=float, default=0.05)  # (float) 0-1 aka decay or discount rate, determines the importance of future rewards

    args = parser.parse_args()
    name_asset               = args.name_asset
    name_model               = args.name_model
    num_features             = args.n_features
    num_neurons              = args.n_neurons
    num_episodes             = args.n_episodes
    n_batch_size             = args.n_batch_size
    random_action_min        = args.random_action_min
    random_action_decay      = args.random_action_decay
    future_reward_importance = args.future_reward_importance


    run_dqn(name_asset=name_asset, name_model=name_model, n_features=num_features, n_neurons=num_neurons, n_episodes=num_episodes,
            batch_size=n_batch_size, random_action_decay=random_action_decay, random_action_min=random_action_min,
            future_reward_importance=future_reward_importance, use_exploration=True)
