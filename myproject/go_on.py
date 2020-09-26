from binance.client import Client
import time
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import ReadTimeoutError
import os
import django
import sys
from collections import OrderedDict
import time as manage_time
import warnings

warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Predictor


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
    predictors = Predictor.objects.all()
    client = Client(user.api_key, user.secret_key)
    finance = user.finance_set.all()[0]
    timestamp = finance.get_time()
    if first:
        wait_until(timestamp, secs=10)
    else:
        time.sleep(5)
    try:
        for predictor in predictors:
            df = finance.give_ohlcv(interval=predictor.time_frame, size=predictor.input_size)
            # print(df.tail()[['Close']])
            last = df.tail(1)
            close = float(last['Close'])
            # print('the price is:', close)
            traders = predictor.trader_set.all()
            # print('we have ', len(traders), ' traders')
            for trader in traders:
                the_user = trader.user
                output = trader.trade(close, df)
                if output[0]:
                    new_predictor_states = output[1]
                    """
                    (state, state_have_money, state_last_price_set, state_last_buy_price, state_max_from_last,
                    state_min_from_last, state_var1, state_var2, state_var3, state_last_sell_price)
                    """
                    # print('saving the state changes')
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
