import time
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import ReadTimeoutError
import os
import django
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Predictor, Material, Finance


def wait_until(time_stamp, secs=10, time_step=15):
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
        wait_until(timestamp, secs=50)
    else:
        time.sleep(5)

    predictors = Predictor.objects.all()
    print(predictors)
    for predictor in predictors:
        try:
            user = User.objects.get(name=predictor.user_name)
            print(user)
            finance = Finance.objects.get(user=user)
            df = finance.give_ohlcv(interval=predictor.time_frame, size=predictor.input_size)
            last = df.tail(1)
            close = float(last['Close'])
            traders = predictor.trader_set.all()
            for trader in traders:
                trader.trade(close, df, finance)
        except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError):
            print('we got an error for user:', user)
            continue


if __name__ == '__main__':
    debug_mode = input('press d to debug_mode and n to normal trading mode:')
    if debug_mode == 'd':
        while True:
            input('press enter to make a move:')
            do_the_job(first=False)
            time.sleep(1)
    else:
        while True:
            do_the_job(first=True)
            time.sleep(1)

