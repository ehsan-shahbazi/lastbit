from binance.client import Client
import time
import pandas as pd
import django.utils.timezone as timezone
import os
import django
import sys
from collections import OrderedDict
import time as manage_time
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Predictor


def wait_until(time_stamp, secs=5):
    time_stamp = int(time_stamp)
    print('timestamp:', timestamp)
    sleep = int(300 - ((time_stamp/300) - int(time_stamp/300)) * 300) - secs
    print('sleep for:', sleep)
    if sleep > 0:
        time.sleep(sleep)
    else:
        time.sleep(sleep + 300)
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


if __name__ == '__main__':
    while True:
        user = User.objects.all()[0]
        predictors = Predictor.objects.all()
        # print(predictors)
        client = Client(user.api_key, user.secret_key)
        finance = user.finance_set.all()[0]
        timestamp = finance.get_time()
        wait_until(timestamp)
        # timestamp = finance.get_time()
        # print(timestamp)
        # print(client.get_server_time())
        for predictor in predictors:
            df = finance.give_ohlcv(interval=predictor.time_frame, size=500)
            print(df.head())
            last = df.tail(1)
            close = float(last['Close'])
            print('the price is:', close)
            traders = predictor.trader_set.all()
            for trader in traders:
                the_user = trader.user
                trader.trade(close, df)
                print('DECISION DONE.')
        # print(df['Close'])
        # print(close)
