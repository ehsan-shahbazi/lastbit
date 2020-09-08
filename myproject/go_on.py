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
    print('=' * 100)
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
            print('the price is:', close)
            traders = predictor.trader_set.all()
            for trader in traders:
                the_user = trader.user
                trader.trade(close, df)
        return True
    except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError):
        do_the_job(first=False)
    finally:
        return True


if __name__ == '__main__':
    while True:
        do_the_job(first=True)
        time.sleep(1)
