import time
import pandas as pd
from requests.exceptions import ReadTimeout, ConnectionError, ConnectTimeout
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
from polls.models import User, Predictor, Finance, Material


def wait_until(time_stamp, secs=10, time_step=5):
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
        is_done = False
        try:
            user = User.objects.get(name=predictor.user_name)
            print(user)
            finance = Finance.objects.get(user=user)
            df = finance.give_ohlcv(interval=predictor.time_frame, size=predictor.input_size)
            last = df.tail(1)
            close = float(last['Close'])
            material = Material.objects.get(name=finance.symbol)
            material.price = close
            material.save()
            traders = predictor.trader_set.all()
            for trader in traders:
                is_done, new_states = trader.trade(close, df, finance)

        except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError, ConnectTimeout):
            print('we got an error for user:', user)
            continue

        finally:
            if not is_done:
                print('we got an unusual error for user:', user)
            else:
                print('every thing is good for user:', user)
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

