from keras.models import load_model
from agent import Agent
from dqn import Dqn
from utils import *
import argparse


def bt(data, n_features, use_exploration, name_model):
    dqn = Dqn()
    dqn.open_orders = [data[0]]
    agent = Agent(n_features, use_exploration, name_model)
    state = dqn.get_state(data, n_features, n_features)
    total_profits = 0
    total_holds = 0
    total_buys = 1
    total_sells = 0
    total_notvalid = 0
    l = len(data) - 1

    for t in range(n_features, l):

        action = agent.choose_best_action(state)

        reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
            dqn.execute_action(action, data[t], t, total_profits, total_holds, total_buys, total_sells, total_notvalid)

        done = True if t == l - 1 else False

        next_state = dqn.get_state(data, t + 1, n_features)
        state = next_state

        if done:
            reward, total_profits, total_holds, total_buys, total_sells, total_notvalid = \
                dqn.execute_action(2, data[t+1], t+1, total_profits, total_holds, total_buys, total_sells, total_notvalid)
            print("-----------------------------------------")
            print(f'Total Profit: {formatPrice(total_profits*100)} ,'
                  f' Total hold/buy/sell/notvalid trades: {total_holds} / {total_buys} / {total_sells} / {total_notvalid}')
            print("-----------------------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_asset', '-na', type=str, default='^SPX_2020_24') #^GSPC_2011  GSPC_2019 GSPC_1970_2019 GSPC_1970_2018
    parser.add_argument('--name_model', '-nm', type=str, default='model_ep5000') # #model_ep0, model_ep10, model_ep20, model_ep30
    parser.add_argument('--trading_fee', '-tf', type=float, default=0.)
    #python backtest.py -s 'test_sinus' -mn 'model_ep2' -tf 0.
    args = parser.parse_args()
    name_asset = args.name_asset
    name_model = args.name_model
    trading_fee = args.trading_fee
    if name_model == '' :
        print( 'error: you must use existing model name')
        exit(1)
    model_inst = load_model("files/output/" + name_model)
    num_features = model_inst.layers[0].input.shape.as_list()[1]
    use_exploration    = False#must use False
    data = getStockDataVec(name_asset)
    l = len(data) - 1

    print(f'starting back-testing model {name_model} on {name_asset} (file has {l} rows), features = {num_features} ')
    bt(data, num_features, use_exploration, name_model)
    print(f'finished back-testing model {name_model} on {name_asset} (file has {l} rows), features = {num_features} ')


if __name__ == "__main__":
    main()
