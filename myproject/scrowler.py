import time
from requests.exceptions import ReadTimeout, ConnectionError
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import ReadTimeoutError
import os
import django
import warnings
warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Material, Asset


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


def do_the_job(first=True):
    user = User.objects.all()[0]
    finance = user.finance_set.all()[0]
    timestamp = finance.get_time()

    if first:
        wait_until(timestamp, secs=600)
    else:
        time.sleep(5)

    try:
        user = User.objects.get(name='shahbazi')
        print('|!|!|!' * 10)
        print('user works:', user)
        finances = user.finance_set.all()

        for finance in finances:
            material = Material.objects.get(name=finance.symbol)
            first_time_stamp = material.save_new_signals(df=None, give_first_time_stamp=True)
            print('first time stamp is:', first_time_stamp)
            df = finance.give_historical_ohlcv(first_time_stamp=first_time_stamp - (999 * 1000 * 60))
            print(df.head())
            material.save_new_signals(df)
        print('all the assets and signals are saved')

    except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError):
        print('we got an error')
        do_the_job(first=False)
    finally:
        return True


if __name__ == '__main__':
    debug_mode = input('press d to debug_mode and n to normal monitoring mode:')
    if debug_mode == 'd':
        while True:
            input('press enter to make a move:')
            do_the_job(first=False)
            time.sleep(1)
    else:
        while True:
            do_the_job(first=True)
            time.sleep(1)
