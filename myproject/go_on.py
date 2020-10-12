import time
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import ReadTimeoutError
import os
import django
from collections import OrderedDict
import warnings
multi_coin = True

warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Predictor, Material


def wait_until(time_stamp, secs=10, time_step=30):
    """
    :param time_stamp: coming from the server
    :param secs: how many seconds should we start before new minute starts
    :param time_step: how many minutes should we wait each time?
    :return:
    """
    time_stamp = int(time_stamp)
    print('timestamp:', time_stamp)
    sleep = int((60 * time_step) - ((time_stamp/(60 * time_step)) -
                                    int(time_stamp/(60 * time_step))) * (60 * time_step)) - secs
    print('sleep for:', sleep)
    print('=' * 70)
    if sleep > 0:

        time.sleep(sleep)
    else:
        time.sleep(sleep + (60 * time_step))
    return True


def re_sample(the_df, method='1H'):
    period = len(the_df)
    ids = pd.period_range('2014-12-01 00:00:00', freq='T', periods=period)
    # print(the_df['Close'].head())
    the_df[['Open', 'High', 'Close', 'Low']] = the_df[['Open', 'High', 'Close', 'Low']].astype(float)
    df_res = the_df.set_index(ids).resample(method).agg(OrderedDict([('Open', 'first'),
                                                                    ('High', 'max'),
                                                                    ('Low', 'min'),
                                                                    ('Close', ['mean', 'last'])]))
    return df_res


def do_the_job(first=True):
    user = User.objects.all()[0]
    finance = user.finance_set.all()[0]
    timestamp = finance.get_time()
    if first:
        wait_until(timestamp, secs=30)
    else:
        time.sleep(5)

    if multi_coin:
        print('starting multi coin trades')
        users = User.objects.all()
        for user in users:
            print('user works:', user)
            list_of_states = []
            active_predictors = []  # predictors which have nonzero state
            finances = user.finance_set.all()
            print(finances)
            for finance in finances:
                print(finance.symbol)
                material = Material.objects.get(name=finance.symbol)
                print(material)
                predictor = material.predictor_set.get(user_name=user.name)
                print(predictor)
                if predictor.state != 0:
                    active_predictors.append([predictor, finance])
            if len(active_predictors) == 1:
                print('user has active coin:', active_predictors)
                predictor = active_predictors[0][0]
                finance = active_predictors[0][1]
                trader = predictor.trader_set.get(user=user)
                df = finance.give_ohlcv(interval=predictor.time_frame, size=predictor.input_size)
                close = float(df.tail(1)['Close'])
                is_done, new_states = trader.trade(close, df, finance=finance, investigate_mode=False)
                if is_done:
                    predictor.state = new_states[0]
                    predictor.state_have_money = new_states[1]
                    predictor.state_last_price_set = new_states[2]
                    predictor.state_last_buy_price = new_states[3]
                    predictor.state_max_from_last = new_states[4]
                    predictor.state_min_from_last = new_states[5]
                    predictor.state_var1 = new_states[6]
                    predictor.state_var2 = new_states[7]
                    predictor.state_var3 = new_states[8]
                    predictor.state_last_sell_price = new_states[9]
                    predictor.save()
                    print('saved!')
                if new_states[0] == 0:
                    print('the coin is been sold and dis-activated')
                    for finance in finances:
                        material = Material.objects.get(name=finance.symbol)
                        predictor = material.predictor_set.get(user_name=user.name)[0]
                        trader = predictor.trader_set.get(user=user)
                        df = finance.give_ohlcv(interval=predictor.time_frame, size=predictor.input_size)
                        close = float(df.tail(1)['Close'])
                        prediction, states = trader.trade(close, df, finance=finance, investigate_mode=True)
                        print('prediction is:', prediction)
                        list_of_states.append([trader, close, df, finance, prediction, states, predictor])
                    list_of_states.sort(key=lambda x: ((x[4][1]['start_price'] - x[1]) /
                                                       pd.to_numeric(x[2]['Close']).std()))
                    trader, close, df, finance, prediction, states, predictor = tuple(list_of_states[0])
                    print('the best thing to trade looks:', finance.symbol, prediction)
                    is_done, new_states = trader.trade(close, df, finance=finance, investigate_mode=False)
                    if is_done:
                        predictor.state = new_states[0]
                        predictor.state_have_money = new_states[1]
                        predictor.state_last_price_set = new_states[2]
                        predictor.state_last_buy_price = new_states[3]
                        predictor.state_max_from_last = new_states[4]
                        predictor.state_min_from_last = new_states[5]
                        predictor.state_var1 = new_states[6]
                        predictor.state_var2 = new_states[7]
                        predictor.state_var3 = new_states[8]
                        predictor.state_last_sell_price = new_states[9]
                        predictor.save()
                        print('saved!')

            elif len(active_predictors) > 1:
                print('WE HAVE TWO DIFFERENT COIN ACTIVE FOR USER:', user)
                input('please make it one and then press inter to continue')

            else:
                print('we do not have any active coin lets search...')
                for finance in finances:
                    material = Material.objects.get(name=finance.symbol)
                    predictor = material.predictor_set.get(user_name=user.name)
                    trader = predictor.trader_set.get(user=user)
                    df = finance.give_ohlcv(interval=predictor.time_frame, size=predictor.input_size)
                    close = float(df.tail(1)['Close'])
                    prediction, states = trader.trade(close, df, finance=finance, investigate_mode=True)
                    list_of_states.append([trader, close, df, finance, prediction, states, predictor])
                list_of_states.sort(key=lambda x: ((x[4][1]['start_price'] - x[1]) /
                                                   pd.to_numeric(x[2]['Close']).std()))
                trader, close, df, finance, prediction, states, predictor = tuple(list_of_states[0])
                print('going to trader.trade trader and predictor is:', trader, predictor)
                is_done, new_states = trader.trade(close, df, finance=finance, investigate_mode=False)
                print('is done and new_states are:', is_done, new_states)
                if is_done:
                    predictor.state = new_states[0]
                    predictor.state_have_money = new_states[1]
                    predictor.state_last_price_set = new_states[2]
                    predictor.state_last_buy_price = new_states[3]
                    predictor.state_max_from_last = new_states[4]
                    predictor.state_min_from_last = new_states[5]
                    predictor.state_var1 = new_states[6]
                    predictor.state_var2 = new_states[7]
                    predictor.state_var3 = new_states[8]
                    predictor.state_last_sell_price = new_states[9]
                    predictor.save()
                    print('saved!')
        return True
    try:
        if not multi_coin:
            predictors = Predictor.objects.all()
            for predictor in predictors:
                df = finance.give_ohlcv(interval=predictor.time_frame, size=predictor.input_size)
                last = df.tail(1)
                close = float(last['Close'])
                traders = predictor.trader_set.all()
                for trader in traders:
                    output = trader.trade(close, df)
                    if output[0]:
                        """
                        (state, state_have_money, state_last_price_set, state_last_buy_price, state_max_from_last,
                        state_min_from_last, state_var1, state_var2, state_var3, state_last_sell_price)
                        """
                        predictor.state = output[1][0]
                        predictor.state_have_money = output[1][1]
                        predictor.state_last_price_set = output[1][2]
                        predictor.state_last_buy_price = output[1][3]
                        predictor.state_max_from_last = output[1][4]
                        predictor.state_min_from_last = output[1][5]
                        predictor.state_var1 = output[1][6]
                        predictor.state_var2 = output[1][7]
                        predictor.state_var3 = output[1][8]
                        predictor.state_last_sell_price = output[1][9]
                        predictor.save()
                        print('saved!')
                return True
    except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError):
        print('we got an error')
        do_the_job(first=False)
    finally:
        return True


if __name__ == '__main__':
    while True:
        do_the_job(first=True)
        time.sleep(1)
