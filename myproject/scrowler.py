import time
from requests.exceptions import ReadTimeout, ConnectionError
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import ReadTimeoutError
import os
import sys
import django
import warnings
warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Material, Signal


def do_the_job(first=True):
    if first:
        time.sleep(10)
    else:
        time.sleep(5)

    try:
        user = User.objects.get(name='shahbazi')
        print('|!|!|!' * 10)
        finances = user.finance_set.all()
        for finance in finances:
            material = Material.objects.get(name=finance.symbol)
            first_time_stamp = material.save_new_signals(df=None, give_first_time_stamp=True)
            if first_time_stamp != 0:
                df = finance.give_historical_ohlcv(first_time_stamp=first_time_stamp - (999 * 1000 * 60))
            else:
                df = finance.give_ohlcv()
            material.save_new_signals(df)
        print('all the assets and signals are saved')

    except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError):
        print('we got an error')
        do_the_job(first=False)


if __name__ == '__main__':
    debug_mode = input('press d to debug_mode and n to normal monitoring mode:')
    if debug_mode == 'd':
        while True:
            input('press enter to make a move: note that all the current signals will be deleted !!!')
            signals = Signal.objects.all()
            print('deleting signals:')
            tot = len(signals)
            for signal in signals:
                signal.delete()
            print('all the signals are deleted')
            do_the_job(first=False)
            time.sleep(1)
    else:
        while True:
            do_the_job(first=True)
            time.sleep(1)
