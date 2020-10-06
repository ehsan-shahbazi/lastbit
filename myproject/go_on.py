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
from polls.models import User, Predictor, Activity
is_simulation = True
testing = True


def wait_until(time_stamp, secs=10, time_step=30):
    """
    :param time_stamp: coming from the server
    :param secs: how many seconds should we start before new minute starts
    :param time_step: how many minutes should we wait each time?
    :return:
    """
    if is_simulation:
        return True
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
    if method == 'tot':
        period = len(the_df)
        ids = pd.period_range('2014-12-01 00:00:00', freq='T', periods=period)
        # print(the_df['Close'].head())
        the_df[['Open', 'High', 'Close', 'Low', 'Volume_(Currency)']] = \
            the_df[['Open', 'High', 'Close', 'Low', 'Volume_(Currency)']].astype(float)
        the_df = the_df.rename({'Volume_(Currency)': 'Volume'}, axis='columns')
        df_res = the_df.set_index(ids).resample('1D').agg(OrderedDict([('Open', 'first'),
                                                                       ('High', 'max'),
                                                                       ('Low', 'min'),
                                                                       ('Close', 'last'),
                                                                       ('Volume', 'sum')]))
        return df_res

    period = len(the_df)
    ids = pd.period_range('2014-12-01 00:00:00', freq='T', periods=period)
    the_df[['Open', 'High', 'Close', 'Low', 'Volume_(Currency)']] = \
        the_df[['Open', 'High', 'Close', 'Low', 'Volume_(Currency)']].astype(float)
    the_df = the_df.rename({'Volume_(Currency)': 'Volume'}, axis='columns')

    df_res = the_df.set_index(ids).resample(method).agg(OrderedDict([('Open', 'first'),
                                                                    ('High', 'max'),
                                                                    ('Low', 'min'),
                                                                    ('Close', 'last'),
                                                                    ('Volume', 'sum')]))
    return df_res


class Simulator:
    def __init__(self):
        self.step_size = 30             # Step time
        self.current_step = 1000000      # The first price in the historical data
        self.df_size = 7000             # It must be less than current step
        self.resample_method = '15Min'  # It can be '1H' , 'nMin' , ...
        self.bit_df = pd.read_csv('./data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv').dropna()
        self.user = User.objects.get(name='simulator')
        self.predictor = Predictor.objects.all()[0]
        self.trader = self.predictor.trader_set.all()[0]
        print('deleting related activities')
        for activity in self.trader.activity_set.all():
            activity.delete()

    def give_df(self):
        self.current_step += self.step_size
        min_df = self.bit_df.iloc[self.current_step - self.step_size - self.df_size:self.current_step]
        last_step_df = self.bit_df[self.current_step:self.current_step + self.step_size]
        return re_sample(min_df, method=self.resample_method), re_sample(last_step_df, method='tot')

    def do_simulation_job(self):
        print('starting simulation.')
        df, new_df = self.give_df()
        have_btc = 1
        tot = len(self.bit_df)

        while len(df > 30):

            print(self.current_step * 100 / tot, ' % completed.')
            df, new_df = self.give_df()
            close = df.tail(1)['Close'].item()
            # print('close is:', close)
            if have_btc not in [1, 0]:
                print('have_btc is', have_btc)
                input('ooppss!')
            is_done, states, my_activity_set = self.trader.trade(close, df, have_btc=have_btc)
            have_btc = self.make_activities(new_df, my_activity_set, have_btc=have_btc)

            if is_done:
                """
                (state, state_have_money, state_last_price_set, state_last_buy_price, state_max_from_last,
                state_min_from_last, state_var1, state_var2, state_var3, state_last_sell_price)
                """
                # print('saving the state changes')
                self.predictor.state = states[0]
                self.predictor.state_have_money = states[1]
                self.predictor.state_last_price_set = states[2]
                self.predictor.state_last_buy_price = states[3]
                self.predictor.state_max_from_last = states[4]
                self.predictor.state_min_from_last = states[5]
                self.predictor.state_var1 = states[6]
                self.predictor.state_var2 = states[7]
                self.predictor.state_var3 = states[8]
                self.predictor.state_last_sell_price = states[9]
                self.predictor.save()
                # print('saved!')

    def make_activities(self, new_df, my_activity_set, have_btc):
        real_activities = []
        for activity in my_activity_set:
            if activity[0] in ['buy', 'sell']:
                real_activities.append(activity)
            elif (activity[1] < new_df['High'][0]) & (activity[1] > new_df['Low'][0]):
                real_activities.append(activity)
        for activity in real_activities:
            # print(activity)
            # input('so we have a activity and we must save it...')
            if activity[0] == 'stop_buy':

                my_activity = Activity(trader=self.trader, action='buy', price=activity[1])
                my_activity.save()
            elif activity[0] == 'stop_sell':
                my_activity = Activity(trader=self.trader, action='sell', price=activity[1])
                my_activity.save()
            elif activity[0] in ['buy', 'sell']:
                my_activity = Activity(trader=self.trader, action=activity[0], price=activity[1])
                my_activity.save()
        if len(real_activities) > 0:
            if real_activities[-1][0] in ['stop_buy', 'buy']:
                have_btc = 1
            elif real_activities[-1][0] in ['sell', 'stop_sell']:
                have_btc = 0
        return have_btc


def do_the_job(first=True):
    user = User.objects.all()[0]
    predictors = Predictor.objects.all()
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
                    # print('saved!')
        return True
    except (ReadTimeout, ReadTimeoutError, BinanceAPIException, ConnectionError):
        print('we got an error')
        do_the_job(first=False)
    finally:
        return True


if __name__ == '__main__':
    if testing:
        simulator = Simulator()
        print(simulator.give_df())
        print(simulator.do_simulation_job())

    else:
        while True:
            do_the_job(first=True)
            time.sleep(1)
