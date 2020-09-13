from django.db import models
import pandas as pd
from django.utils import timezone
import joblib
import ta
import numpy as np
from binance.client import Client
from talib._ta_lib import *
from binance.enums import *
import time
# Create your models here.


def make_all_ta_things(df):
    my_open = np.array(df['Open'])
    close = np.array(df['Close'])
    high = np.array(df['High'])
    low = np.array(df['Low'])
    volume = np.array(df['Volume'])
    the_dict = dict()
    the_dict['Open'] = list(my_open)
    the_dict['Close'] = list(close)
    the_dict['High'] = list(high)
    the_dict['Low'] = list(low)
    the_dict['Volume'] = list(volume)
    the_dict['MACD3'] = list(MACD(close, fastperiod=12, slowperiod=26, signalperiod=1)[0])
    the_dict['MACD_SIGNAL3'] = list(MACD(close, fastperiod=12, slowperiod=26, signalperiod=1)[1])
    the_dict['MACD_HIST3'] = list(MACD(close, fastperiod=12, slowperiod=26, signalperiod=1)[2])
    here = ta.trend.MACD(pd.Series(close), 26, 12, 1).macd_signal()
    print('we have from ta:')
    print(here)
    print('-' * 50)
    ans = pd.DataFrame.from_dict(the_dict).fillna(0)
    print(ans)
    return ans


class Histogram:
    def __init__(self, df):
        """
        :param df: numeric OHLCV
        """
        self.state = 'DON\'T MOVE!'
        self.prices = list(df['Close'])
        self.highs = list(df['High'])
        self.lows = list(df['Low'])
        self.volumes = list(df['Volume'])
        self.strategy = [{'STOP': (-0.80, 0.80), 'BUY': (0.99, 1), 'SELL': (0, 0.5)}]

    def stop_loss(self, start_alpha, stop_alpha):
        hist = self.prices
        hist.sort()
        tot = len(hist)
        stop_index = int((tot + 1) * stop_alpha)
        start_index = int((tot + 1) * start_alpha)
        stop_price = hist[stop_index]
        start_price = hist[start_index]
        return {'stop_price': stop_price, 'start_price': start_price}

    def decision(self, state, state_var1, state_max_from_last, state_min_from_last, state_last_sell_price,
                 state_last_buy_price):
        hist = self.prices
        the_strategy = self.strategy
        price = hist[-1]
        """
        :param hist: [price1, price2, price3, ...]
        :param price: current price
        :param the_strategy: self.strategy
        :return: 'ACTION', (stop_minus, stop_plus)
        """
        pi = 0  # more than current price
        j = 0  # less than current price
        tot = len(hist)
        for index in range(tot):
            if hist[index] < price:
                pi += 1
            else:
                j += 1

        alpha = pi / tot
        output = []

        # make decision
        # set the last price set
        last_price_set = 0
        if state == 0:
            if alpha >= 0.98:
                output.append('BUY')
                output.append(self.stop_loss(0.98, 0.5))
                last_price_set = price
            else:
                output.append('DON\'T MOVE!')
                output.append(self.stop_loss(0.98, 0.5))
                last_price_set = self.stop_loss(0.98, 0.5)['start_price']

        elif state == 1:
            if alpha <= 0.5:
                output.append('SELL')
                output.append(self.stop_loss(0.98, 0.5))
                last_price_set = price
            elif self.stop_loss(0.98, 0.5)['stop_price'] >= state_var1 * state_max_from_last:
                output.append('DON\'T MOVE!')
                output.append(self.stop_loss(0.98, 0.5))
                last_price_set = self.stop_loss(0.98, 0.5)['stop_price']
            else:
                output.append('DON\'T MOVE!')
                output.append({'start_price': 1000000, 'stop_price': state_var1 * state_max_from_last})
                last_price_set = state_var1 * state_max_from_last

        elif state == 2:
            if alpha <= 0.5:
                output.append('SELL')
                output.append(self.stop_loss(0.98, 0.5))
                last_price_set = price
            elif price > state_last_sell_price:
                output.append('BUY')
                output.append(self.stop_loss(0.98, 0.5))
                last_price_set = price
            else:
                output.append('DON\'T MOVE!')
                output.append({'start_price': state_last_sell_price, 'stop_price': 0})
                last_price_set = state_last_sell_price
        """        
        for one_strategy in the_strategy:
            if (alpha <= one_strategy['BUY'][1]) & (alpha >= one_strategy['BUY'][0]):
                output.append('BUY')
            elif (alpha <= one_strategy['SELL'][1]) & (alpha >= one_strategy['SELL'][0]):
                output.append('SELL')
            else:
                output.append('DON\'T MOVE!')
        output.append(self.stop_loss(0.98, 0.5))
        """

        return output, last_price_set


class User(models.Model):
    name = models.CharField(max_length=100, name='name')
    account = models.IntegerField(default=0, name='account')
    speed = models.IntegerField(default=1, name='speed')
    phone = models.CharField(default='09125459232', name='phone', max_length=11)
    last_mail_date = models.DateTimeField(name='last_mail_date')
    api_key = models.CharField(max_length=100, name='api_key',
                               default='api_key')
    secret_key = models.CharField(max_length=100, name='secret_key',
                                  default='secret_key')

    def __str__(self):
        return self.name


class Finance(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    symbol = models.CharField(name='symbol', default='BTCUSDT', max_length=20)
    # todo: check the order responses to figure out the transactions are done perfectly

    def get_time(self):
        client = Client(self.user.api_key, self.user.secret_key)
        timestamp = client.get_server_time()
        return int(timestamp['serverTime']) / 1000

    def have_btc(self):
        """
        :return: if we have btc then returns true else false
        """
        client = Client(self.user.api_key, self.user.secret_key)
        usd = float(client.get_asset_balance(asset='USDT')['free'])
        btc = float(client.get_asset_balance(asset='BTC')['free'])
        price = self.get_price()
        return usd < (btc * price)

    def buy(self, price, percent=0.95):
        """
        :param price: the last price of bitcoin
        :param percent: how much of your budget do you want to buy? 100 mean all of that
        :return:
        """
        print('buy')
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='USDT')['free'])
        quantity = balance * percent / price
        quantity = round(quantity, 6)
        print(quantity)
        if quantity > 0.001:
            print('order sent')
            if not self.have_btc():
                order = client.order_market_buy(
                    symbol=self.symbol,
                    quantity=quantity)
        return True

    def sell(self, percent=0.95):
        """
        :param percent: how much of your asset do you want to sell? 100 mean all of that
        :return:
        """
        print('sell')
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='BTC')['free'])
        quantity = balance * percent
        quantity = round(quantity, 6)
        print(quantity)
        if quantity > 0.001:
            print('order sent')
            if self.have_btc():
                order = client.order_market_sell(
                    symbol=self.symbol,
                    quantity=quantity)
        return True

    def get_price(self):
        client = Client(self.user.api_key, self.user.secret_key)
        orders = client.get_all_tickers()
        for order in orders:
            if order['symbol'] == self.symbol:
                return float(order['price'])
        return False

    def buy_limit(self, limit, percent=0.95):
        """
        :param limit: the limit of bitcoin
        :param percent: how much of your budget do you want to buy? 100 mean all of that
        :return:
        """
        print('limit buy')
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='USDT')['free'])
        quantity = balance * percent / limit
        quantity = round(quantity, 6)
        print(quantity)
        if quantity > 0.001:
            print('order sent')
            if not self.have_btc():
                order = client.order_limit_buy(
                    symbol=self.symbol,
                    quantity=quantity,
                    price=str(limit))

        return

    def sell_limit(self, limit, percent=0.95):
        """
        :param limit: the limit of bitcoin
        :param percent: how much of your budget do you want to buy? 100 mean all of that
        :return:
        """
        print('limit sell')
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='BTC')['free'])
        quantity = balance * percent
        quantity = round(quantity, 6)
        print(quantity)
        if quantity > 0.001:
            if self.have_btc():
                print('order sent')
                order = client.order_limit_sell(
                    symbol=self.symbol,
                    quantity=quantity,
                    price=str(limit))
        return

    def buy_stop(self, stop, percent=0.95):
        """

        :param stop: the price of stop
        :param percent: how much of the budget? 100 means all of it
        :return:
        """
        print('buy stop')
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='USDT')['free'])
        quantity = balance * percent / stop
        quantity = round(quantity, 6)
        print(quantity)
        if quantity > 0.001:
            if not self.have_btc():
                print('order sent')
                order = client.create_order(
                    symbol=self.symbol,
                    type='STOP_LOSS_LIMIT',
                    side=SIDE_BUY,
                    timeInForce='GTC',
                    quantity=quantity,
                    stopPrice=str(stop),
                    price=str(stop))
            return True
        return False

    def sell_stop(self, stop, percent=0.95):
        """

        :param stop: the price of stop
        :param percent: how much of the asset? 100 means all of it
        :return:
        """
        print('sell stop')
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='BTC')['free'])
        quantity = balance * percent
        quantity = round(quantity, 6)
        print(quantity)
        if quantity > 0.001:
            if self.have_btc():
                print('order sent')
                order = client.create_order(
                    symbol=self.symbol,
                    side=SIDE_SELL,
                    type='STOP_LOSS_LIMIT',
                    quantity=quantity,
                    timeInForce='GTC',
                    stopPrice=str(stop),
                    price=str(stop))
            return True
        return False

    def cancel_all_orders(self):
        print('cancel all')
        client = Client(self.user.api_key, self.user.secret_key)
        orders = client.get_open_orders(symbol=self.symbol)
        for order in orders:
            if order['symbol'] == self.symbol:
                time.sleep(1)
                the_answer = client.cancel_order(symbol=self.symbol, orderId=order['orderId'])
        return True

    def give_ohlcv(self, interval='1m', size=12):
        """
        :param interval: it can be 1m for 1min and 1h for one hour if you want else, then you should define it.
        :param size: the size of the input size of predictor plus 1
        :return: a data_frame
        """
        client = Client(self.user.api_key, self.user.secret_key)
        if interval == '1h':
            my_interval = Client.KLINE_INTERVAL_1HOUR
            limit = (size + 1)
        elif interval == '1m':
            my_interval = Client.KLINE_INTERVAL_1MINUTE
            limit = (size + 1) * 60
        elif interval == '5m':
            my_interval = Client.KLINE_INTERVAL_5MINUTE
            limit = (size + 1)
        elif interval == '15m':
            my_interval = Client.KLINE_INTERVAL_15MINUTE
            limit = (size + 1)
        else:
            return False
        candles = client.get_klines(symbol=self.symbol, interval=my_interval, limit=limit)
        df = pd.DataFrame(candles, columns=["Open time", "Open", "High", "Low", "Close", "Volume", "Close time",
                                            "Quote asset volume", "Number of trades", "Taker buy base asset volume",
                                            "Taker buy quote asset volume", "Ignore"])
        return df


class Material(models.Model):
    name = models.CharField(name='name', max_length=100, default='')
    persian_name = models.CharField(name='persian_name', max_length=100, default='')
    price = models.FloatField(name="price")
    volume = models.FloatField(name='volume', default=0.0)
    update = models.DateTimeField(name='update')
    trading_fee = models.FloatField(name='trading_fee', default=0.01)
    state = models.BooleanField(name='state', default=True)
    hour_tendency = models.IntegerField(name='hour_tendency', default=0)
    hour_calc = models.DateTimeField(name='hour_calc', default=timezone.now)
    hour_conf = models.IntegerField(name='hour_conf', default=0)
    day_tendency = models.IntegerField(name='day_tendency', default=0)
    dat_calc = models.DateTimeField(name='day_calc', default=timezone.now)
    day_conf = models.IntegerField(name='day_conf', default=0)
    week_tendency = models.IntegerField(name='week_tendency', default=0)
    week_calc = models.DateTimeField(name='week_calc', default=timezone.now)
    week_conf = models.IntegerField(name='week_conf', default=0)

    def __str__(self):
        return self.name


class Predictor(models.Model):
    """
    each time you change the predictor change time frame and
    """
    material = models.ForeignKey(Material, on_delete=models.CASCADE)
    model_dir = models.CharField(name='model_dir', default='polls/trained/?.h5', max_length=100)
    i_scale = models.CharField(name='i_scale', default='polls/trained/I_scaler.gz', max_length=100)
    o_scale = models.CharField(name='o_scale', default='polls/trained/O_scaler.gz', max_length=100)
    time_frame = models.CharField(name='time_frame', default='1H', max_length=20)
    last_calc = models.DateTimeField(name='last_calc', default=timezone.now)
    input_size = models.IntegerField(name='input_size', default=1)  # if HIST how many time frame is between two
    # operations of this predictor for example if we sleep 20 minutes and the time frame is 10Min it should be 2
    type = models.CharField(name='type', default='RNN', max_length=20)
    unit = models.CharField(name='unit', default='dollar', max_length=20)
    upper = models.FloatField(name='upper', default=0)
    lower = models.FloatField(name='lower', default=0)
    state = models.IntegerField(name='state', default=0)
    state_have_money = models.BooleanField(name='state_have_money', default=True)
    state_last_price_set = models.FloatField(name='last_price_set', default=0) # the last stop loss sat for knowing the
    # price which we had in stop loss
    state_last_buy_price = models.FloatField(name='state_last_buy_price', default=0)
    state_max_from_last = models.FloatField(name='state_max_from_last', default=0)
    state_min_from_last = models.FloatField(name='state_min_from_last', default=10000000000)
    state_var1 = models.FloatField(name='state_var1', default=0)  # percent of selling in state 1 for example 0.99
    state_var2 = models.FloatField(name='state_var2', default=0)
    state_var3 = models.FloatField(name='state_var3', default=0)
    state_last_sell_price = models.FloatField(name='state_last_sell_price', default=0)

    def make_inputs(self, df, have_money=True):
        """

        :param have_money:true if we have money
        :param df: It should have just OCHLV and in string
        :return: a list which is input
        """
        print('have_money is:', have_money)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df['Close'] = pd.to_numeric(df['Close'])
        df['Open'] = pd.to_numeric(df['Open'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['High'] = pd.to_numeric(df['High'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        inputs = []
        if self.type == 'DT':
            df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close",
                                        volume="Volume", fillna=True)
            for index, data in df.iterrows():
                # print(list(data))
                inputs.append(list(data))
        elif self.type == 'HIST':
            # todo: we should find n and i just set it 2 for 30 minutes for sleep and time framing 15 min
            new_high = max(list(df['High'].tail(n=2)))
            new_low = min(list(df['Low'].tail(n=2)))
            # print('new low and new high are:', new_low, new_high)
            tree1 = Histogram(df)
            prices = tree1.stop_loss(0.98, 0.5)
            # print('prices are: ', prices)
            if prices['stop_price'] > new_low:
                self.state = 0

            if have_money != self.state_have_money:
                # make the state
                # make the last_buy_price and last_sell_price
                # make max_from_last and min_from_last
                if have_money:
                    if prices['stop_price'] > new_low:
                        self.state = 0
                        self.state_last_sell_price = prices['stop_price']
                        self.state_min_from_last = self.state_last_price_set
                        self.state_max_from_last = self.state_last_price_set

                    else:
                        self.state = 2
                        self.state_last_sell_price = float(self.state_max_from_last * 0.99)
                        self.state_min_from_last = float(self.state_last_price_set)
                        self.state_max_from_last = float(self.state_last_price_set)
                else:
                    print('im here in state 1')
                    self.state = 1
                    self.save()
                    print('saved')
                    last_buy = float(self.state_last_price_set)
                    self.state_last_buy_price = last_buy
                    self.save()
                    print('hi2')
                    self.state_min_from_last = float(self.state_last_price_set)
                    print('hi3')
                    self.state_max_from_last = float(self.state_last_price_set)
                    print('hi4')
                    self.save()
                    print('state 1 works finished')

            print('setting the maxes')
            self.state_max_from_last = float(max(self.state_max_from_last, new_high))
            self.state_min_from_last = float(min(self.state_min_from_last, new_low))
            print('setted')
            self.state_have_money = have_money
            print('the state is:', self.state)
            self.save()
            print(df.head())
            return df
        elif self.type == 'MAD':
            df = make_all_ta_things(df)
            print(df.tail())
            for data in (df['MACD_SIGNAL3'] - df['Close']) / df['Close']:
                inputs.append([data])

        # print('and inputs are:')
        # print(inputs[-20:])
        print('we had: ', len(inputs), ' inputs')
        return inputs

    def predict(self, df='', gamma=0.65, have_money=True):
        """
        Inter your code
        :param: df: the df should be appropriated for prediction in size and time framing
        :return:
        """
        if self.type != 'HIST':

            tree1 = joblib.load(self.model_dir)
            the_input = self.make_inputs(df)
            if self.type == 'MAD':
                print('MAD mode')
                predictions = tree1.predict(the_input)

            elif self.type == 'DT':
                print('DT mode')
                temp = tree1.predict_proba(the_input)
                classes = tree1.classes_
                predictions = []
                for the_i in temp:
                    pred_idx = np.argmax(the_i)
                    if max(the_i) >= gamma:
                        predictions.append(classes[pred_idx])
                    else:
                        predictions.append(0)

            else:
                print('NO mode')
                predictions = [0]
            print('all predictions are: ', predictions[-20:])
            print('we have ', len(predictions), ' predictions')
            print('current prediction is: ', predictions[-2])
            return predictions[-2]

        else:
            """
            We need a class called histogram witch has predict and 
            """
            print('HIST mode and have money is:', have_money)
            df = self.make_inputs(df, have_money=have_money)
            print(df.tail(1))
            tree1 = Histogram(df)
            # do decision based on the states
            # set state_last_price_set when you made any decision
            # save the state changes
            decision, self.state_last_price_set = tree1.decision(self.state, self.state_var1, self.state_max_from_last,
                                                                 self.state_min_from_last, self.state_last_sell_price,
                                                                 self.state_last_buy_price)
            self.save()

            return decision


class Trader(models.Model):
    predictor = models.ForeignKey(Predictor, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, related_name='user')
    type = models.CharField(name='type', max_length=100)
    start_asset = models.FloatField(name='start_asset', default=0)  # it should be in dollar
    simulated_asset = models.FloatField(name='simulated_asset', default=0)
    real_mat_asset = models.FloatField(name='real_mat_asset', default=0)
    real_budget = models.FloatField(name='real_budget', default=0)
    start_date = models.DateTimeField(name='start_date', default=timezone.now)
    active = models.BooleanField(name='active', default=False)
    asset_chart_file = models.CharField(name='asset_chart_file', default='polls/chart/?.txt', max_length=100)

    def calculate(self):
        self.asset_chart_file = 'polls/chart/' + str(self.user.name) + '.txt'
        self.save()
        activities = self.activity_set.all()
        self.simulated_asset = activities[-1].price * activities[-1].mat_amount + activities[-1].budget
        with open(str(self.asset_chart_file), 'w') as file:
            file.flush()
            real_budget = self.start_asset
            real_mat_asset = 0
            simulated_asset = 0
            file.write(str(self.start_date) + ', ' + str(self.real_budget) + ', ' + str(True) + '\n')
            for activity in activities:
                file.write(str(activity.date_time) + ', ' + str(activity.price * activity.mat_amount + activity.budget)
                           + ', ' + str(activity.real) + '\n')
                simulated_asset = activity.price * activity.mat_amount + activity.budget
                if activity.real:
                    if activity.action == 'buy':
                        if real_budget != 0:
                            real_mat_asset = (real_budget / activity.price) * (1 - self.predictor.material.trading_fee)
                            real_budget = 0
                    elif activity.action == 'sell':
                        if real_mat_asset != 0:
                            real_budget = (real_mat_asset * activity.price) * (1 - self.predictor.material.trading_fee)
                            real_mat_asset = 0
                    file.write(str(activity.date_time) + ', ' +
                               str(activity.price * real_mat_asset + real_budget) +
                               ', ' + str(activity.real) + '\n')
            self.simulated_asset = simulated_asset
            self.real_mat_asset = real_mat_asset
            self.real_budget = real_budget

    def __str__(self):
        return str(self.predictor.material.name) + '---' + str(self.predictor.time_frame) + '---' + str(self.user.name)

    def trade(self, close, df=''):
        speaker = self.user.finance_set.all()[0]
        print(speaker)
        prediction = self.predictor.predict(df, have_money=not speaker.have_btc())
        print('prediction is:', prediction)
        if self.predictor.type == 'HIST':
            if prediction[0] == 'BUY':
                if self.have_btc():
                    self.cancel_all()
                    self.stop_sell(prediction[1]['stop_price'])
                else:
                    self.buy(close)
                    self.cancel_all()
                    self.stop_sell(prediction[1]['stop_price'])
                print('BUY')
                return True
            elif prediction[0] == 'SELL':
                if self.have_btc():
                    self.sell(close)
                    self.cancel_all()
                    self.stop_buy(prediction[1]['start_price'])
                else:
                    self.cancel_all()
                    self.stop_buy(prediction[1]['start_price'])
                print('SELL')
                return True
            else:
                self.cancel_all()
                if self.have_btc():
                    self.stop_sell(prediction[1]['stop_price'])
                else:
                    self.stop_buy(prediction[1]['start_price'])
            return True

        if prediction > self.predictor.upper:
            self.buy(close)
            print('BUY')
        if prediction < self.predictor.lower:
            self.sell(close)
            print('SELL')

    def buy(self, close):
        speaker = self.user.finance_set.all()[0]
        price = speaker.get_price()
        if price:
            price = price
        else:
            price = close
        if self.active:
            if self.type == '1':
                print('the trader is active and type is 1')
                speaker.buy(price, percent=0.95)
        mat = self.predictor.material
        mat.price = price
        mat.save()
        self.real_mat_asset = self.real_mat_asset + ((self.real_budget / close) * (1 - mat.trading_fee))
        self.real_budget = 0
        self.save()
        record = Activity(trader=self, action='buy', date_time=timezone.now(), real=self.active, price=price,
                          budget=self.real_budget, mat_amount=self.real_mat_asset)
        record.save()
        print('record saved')

    def sell(self, close):
        speaker = self.user.finance_set.all()[0]
        price = speaker.get_price()
        if price:
            price = price
        else:
            price = close
        if self.active:
            if self.type == '1':
                print('the trader is active and type is 1')
                speaker.sell(percent=0.95)
        mat = self.predictor.material
        mat.price = price
        mat.save()
        self.real_mat_asset = self.real_mat_asset + ((self.real_budget / close) * (1 - mat.trading_fee))
        self.real_budget = 0
        self.save()
        record = Activity(trader=self, action='sell', date_time=timezone.now(), real=self.active, price=close,
                          budget=self.real_budget, mat_amount=self.real_mat_asset)
        record.save()

    def cancel_all(self):
        speaker = self.user.finance_set.all()[0]
        speaker.cancel_all_orders()
        return True

    def limit_buy(self, limit):
        speaker = self.user.finance_set.all()[0]
        price = limit
        if self.active:
            if self.type == '1':
                print('the trader is active and type is 1')
                speaker.buy_limit(price, percent=0.95)
        mat = self.predictor.material
        mat.price = price
        mat.save()

    def limit_sell(self, limit):
        speaker = self.user.finance_set.all()[0]
        price = limit
        if self.active:
            if self.type == '1':
                print('the trader is active and type is 1')
                speaker.sell_limit(price, percent=0.95)
        mat = self.predictor.material
        mat.price = price
        mat.save()

    def stop_buy(self, limit):
        speaker = self.user.finance_set.all()[0]
        price = limit
        if self.active:
            if self.type == '1':
                print('the trader is active and type is 1')
                speaker.buy_stop(price, percent=0.95)
        mat = self.predictor.material
        mat.price = price
        mat.save()

    def have_btc(self):
        speaker = self.user.finance_set.all()[0]
        return speaker.have_btc()

    def stop_sell(self, limit):
        speaker = self.user.finance_set.all()[0]
        price = limit
        if self.active:
            if self.type == '1':
                print('the trader is active and type is 1')
                speaker.sell_stop(price, percent=0.95)
        mat = self.predictor.material
        mat.price = price
        mat.save()


class Activity(models.Model):
    trader = models.ForeignKey(Trader, on_delete=models.CASCADE)
    action = models.CharField(name='action', max_length=3, default='buy')  # it should be 'buy' or 'sell'
    date_time = models.DateTimeField(name='date_time', default=timezone.now)
    real = models.BooleanField(name='real', default=False)
    price = models.FloatField(name='price', default=0)
    budget = models.FloatField(name='budget', default=0)
    mat_amount = models.FloatField(name='mat_amount', default=0)

    def __str__(self):
        return self.action + '---' + str(self.date_time) + '---' + str(self.price)


