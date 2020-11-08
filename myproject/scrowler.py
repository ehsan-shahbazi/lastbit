import time
from requests.exceptions import ReadTimeout, ConnectionError
from binance.exceptions import BinanceAPIException
from urllib3.exceptions import ReadTimeoutError
import os
import django
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Material, Signal


def do_the_job(my_first_time_stamps, first=True):
    if first:
        time.sleep(10)
    else:
        time.sleep(5)

    try:
        user = User.objects.get(name='shahbazi')
        print('|!|!|!' * 10)
        finances = user.finance_set.all()
        for finance in finances:
            if finance.symbol in my_first_time_stamps.keys():
                material = Material.objects.get(name=finance.symbol)
                first_time_stamp = my_first_time_stamps[finance.symbol]
                if first_time_stamp != 0:
                    df = finance.give_historical_ohlcv(first_time_stamp=first_time_stamp - (999 * 1000 * 60))
                else:
                    df = finance.give_ohlcv()
                my_first_time_stamps[finance.symbol] = my_first_time_stamps[finance.symbol] - (999 * 1000 * 60)
                print(df.head())
                if len(df) < 100:
                    my_first_time_stamps.pop(finance.symbol)
                material.save_new_signals(df)
            print('all the assets and signals are saved')
            return my_first_time_stamps
    except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError):
        print('we got an error')
        do_the_job(my_first_time_stamps, first=False)


def save_csv_files():
    user = User.objects.get(name='shahbazi')
    print('|!|!|!' * 10)
    finances = user.finance_set.all()
    for finance in finances:
        material = Material.objects.get(name=finance.symbol)
        pdf = pd.DataFrame(list(material.signal_set.all().values()))
        pdf.to_csv('./data/{}'.format(str(finance.symbol) + '.csv'))


if __name__ == '__main__':
    debug_mode = input('press d to debug_mode and n to normal monitoring mode or press s to save the csv files:')
    first_time_stamps = {}
    my_user = User.objects.get(name='shahbazi')
    for my_finance in my_user.finance_set.all():
        my_material = Material.objects.get(name=my_finance.symbol)
        first_time_stamps[my_finance.symbol] = my_material.save_new_signals(df=None, give_first_time_stamp=True)

    if debug_mode == 'd':
        while True:
            input('press enter to make a move:')
            print('there is no more signals')
            first_time_stamps = do_the_job(first_time_stamps, first=False)
            time.sleep(1)

    elif debug_mode == 's':
        save_csv_files()

    else:
        while True:
            first_time_stamps = do_the_job(first_time_stamps, first=True)
            time.sleep(1)
