from django.db import models
import pandas as pd
from django.utils import timezone
import joblib
import ta
import numpy as np
from binance.client import Client
# Create your models here.


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

    def buy(self, price, percent=0.9):
        """
        :param price: the last price of bitcoin
        :param percent: how much of your budget do you want to buy? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='USDT')['free'])
        quantity = balance * percent / price
        quantity = round(quantity, 6)
        if quantity > 0.001:
            order = client.order_market_buy(
                symbol=self.symbol,
                quantity=quantity)
        return

    def sell(self, percent=0.9):
        """
        :param percent: how much of your asset do you want to sell? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)
        balance = float(client.get_asset_balance(asset='BTC')['free'])
        quantity = balance * percent
        quantity = round(quantity, 6)
        if quantity > 0.001:
            order = client.order_market_sell(
                symbol=self.symbol,
                quantity=quantity)
        return

    def get_price(self):
        client = Client(self.user.api_key, self.user.secret_key)
        orders = client.get_all_tickers()
        for order in orders:
            if order['symbol'] == self.symbol:
                return float(order['price'])
        return False

    def buy_limit(self, limit, percent=100):
        """
        :param limit: in witch cost do you want to buy
        :param percent: how much of your budget do you want to buy? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.order_limit_buy(
            symbol=self.symbol,
            quantity=percent,
            price=str(limit))

    def sell_limit(self, limit, percent=100):
        """
        :param limit: in witch cost do you want to sell
        :param percent: how much of your asset do you want to sell? 100 mean all of that
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)

        order = client.order_limit_sell(
            symbol=self.symbol,
            quantity=percent,
            price=str(limit))

    def buy_stop(self, stop, percent=100):
        """

        :param stop: the price of stop
        :param percent: how much of the budget? 100 means all of it
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)
        """
        order = client.create_oco_order(
            symbol=self.symbol,
            side=SIDE_BUY,
            stopLimitTimeInForce=TIME_IN_FORCE_GTC,
            quantity=percent,
            stopPrice=str(stop),
            price=str(stop))"""
        return True

    def sell_stop(self, stop, percent=100):
        """

        :param stop: the price of stop
        :param percent: how much of the asset? 100 means all of it
        :return:
        """
        client = Client(self.user.api_key, self.user.secret_key)
        """
        order = client.create_oco_order(
            symbol=self.symbol,
            side=SIDE_SELL,
            stopLimitTimeInForce=TIME_IN_FORCE_GTC,
            quantity=percent,
            stopPrice=str(stop),
            price=str(stop))"""
        return True

    def cancel_stop(self):

        client = Client(self.user.api_key, self.user.secret_key)

        orders = client.get_open_orders(symbol=self.symbol)
        for order in orders:
            if order['symbol'] == self.symbol:
                the_answer = client.cancel_order(symbol=self.symbol, orderId=order['orderId'])
        return

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
    material = models.ForeignKey(Material, on_delete=models.CASCADE)
    model_dir = models.CharField(name='model_dir', default='polls/trained/?.h5', max_length=100)
    i_scale = models.CharField(name='i_scale', default='polls/trained/I_scaler.gz', max_length=100)
    o_scale = models.CharField(name='o_scale', default='polls/trained/O_scaler.gz', max_length=100)
    time_frame = models.CharField(name='time_frame', default='1H', max_length=20)
    last_calc = models.DateTimeField(name='last_calc', default=timezone.now)
    input_size = models.IntegerField(name='input_size', default=24)
    type = models.CharField(name='type', default='RNN', max_length=20)
    unit = models.CharField(name='unit', default='dollar', max_length=20)
    upper = models.FloatField(name='upper', default=0)
    lower = models.FloatField(name='lower', default=0)

    @staticmethod
    def make_inputs(df):
        """

        :param df: It should have just OCHLV and in string
        :return: a list which is input
        """
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df['Close'] = pd.to_numeric(df['Close'])
        df['Open'] = pd.to_numeric(df['Open'])
        df['Low'] = pd.to_numeric(df['Low'])
        df['High'] = pd.to_numeric(df['High'])
        df['Volume'] = pd.to_numeric(df['Volume'])
        df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close",
                                    volume="Volume", fillna=True)
        inputs = []
        for index, data in df.iterrows():
            # print(list(data))

            inputs.append(list(data))
        return inputs

    def predict(self, df='', gamma=0.65):
        """
        Inter your code
        :param: df: the df should be appropriated for prediction in size and time framing
        :return:
        """
        tree1 = joblib.load(self.model_dir)
        the_input = self.make_inputs(df)
        temp = tree1.predict_proba(the_input)
        classes = tree1.classes_
        predictions = []
        for i in temp:
            pred_idx = np.argmax(i)
            if max(i) >= gamma:
                predictions.append(classes[pred_idx])
            else:
                predictions.append(0)

        print('prediction is: \n', predictions[-1])
        return predictions[-1]


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
        prediction = self.predictor.predict(df)
        print('prediction is:', prediction)
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
                speaker.buy(price, percent=0.95)
        mat = self.predictor.material
        mat.price = price
        mat.save()
        self.real_mat_asset = self.real_mat_asset + ((self.real_budget / close) * (1 - mat.trading_fee))
        self.real_budget = 0
        self.save()
        record = Activity(trader=self, action='sell', date_time=timezone.now(), real=self.active, price=close,
                          budget=self.real_budget, mat_amount=self.real_mat_asset)
        record.save()


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


