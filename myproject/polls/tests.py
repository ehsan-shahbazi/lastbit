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
from polls.models import User, Material
from binance.client import Client
from binance.enums import *


def long_buy(symbol='BTCUSDT', price=18400, percent=1):
    user = User.objects.get(name='ehsan')
    print(user)

    client = Client(user.api_key, user.secret_key)
    coin_symbol = str(symbol).replace('USDT', '')
    balance = float(client.get_asset_balance(asset='USDT')['free'])
    print(balance)
    if balance > 0:
        transaction = client.transfer_spot_to_margin(asset='USDT', amount=str(balance * percent))
        print(transaction)

    balance = float(client.get_asset_balance(asset=coin_symbol)['free'])
    print(balance)
    if balance > 0:
        transaction = client.transfer_spot_to_margin(asset=coin_symbol, amount=str(balance * percent))
        print(transaction)

    price_info = client.get_margin_price_index(symbol='BTCUSDT')
    price = float(price_info['price'])
    print('price is:', price)
    details = client.get_max_margin_loan(asset='USDT')
    if float(details['amount']) != 0:
        transaction = client.create_margin_loan(asset='USDT', amount=str(details['amount']))
        print(transaction)
    asset_info = client.get_margin_account()
    print('margin account info is:', [x for x in asset_info['userAssets'] if x['asset'] == 'USDT'])
    material = Material.objects.get(name='BTCUSDT')
    asset = [float(x['free']) for x in asset_info['userAssets'] if x['asset'] == 'USDT'][0]
    print('asset is:', asset)
    quantity = round(asset * 0.99 / price, int(material.amount_digits))
    order = client.create_margin_order(
        symbol='BTCUSDT',
        side=SIDE_BUY,
        type=ORDER_TYPE_MARKET,
        quantity=str(quantity)
    )
    print(order)
    asset_info = client.get_margin_account()
    print('margin account info is:\n', [x for x in asset_info['userAssets']])


def round_down(num, digit):
    if round(num, int(digit)) > num:
        return round(num, int(digit)) - pow(0.1, int(digit))
    return round(num, int(digit))


def finish_margin():
    user = User.objects.get(name='ehsan')
    client = Client(user.api_key, user.secret_key)
    asset_info = client.get_margin_account()
    loan = [x for x in asset_info['userAssets'] if x['borrowed'] != '0']
    if len(loan) == 1:
        loan = loan[0]
        material = Material.objects.get(name='BTCUSDT')
        price_info = client.get_margin_price_index(symbol='BTCUSDT')
        price = float(price_info['price'])

        if loan['asset'] == 'USDT':
            asset = [float(x['free']) for x in asset_info['userAssets'] if x['asset'] == str('BTC')][0]
            quantity = round_down(asset, int(material.amount_digits))
            print([x for x in asset_info['userAssets'] if x['asset'] == str('BTC')][0])
            print(quantity)
            if quantity != 0:
                order = client.create_margin_order(
                    symbol='BTCUSDT',
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=str(quantity)
                )
                print(order)

        elif loan['asset'] == 'BTC':
            asset = [float(x['free']) for x in asset_info['userAssets'] if x['asset'] == 'USDT'][0]
            print('asset is:', asset)
            quantity = round_down(asset * 0.999 / price, int(material.amount_digits))
            if quantity != 0:
                order = client.create_margin_order(
                    symbol='BTCUSDT',
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=str(quantity)
                )
                print(order)

        transaction = client.repay_margin_loan(
            asset=loan['asset'],
            amount=str(float(loan['borrowed']) + float(loan['interest']))
        )
    elif len(loan) > 1:
        print('we have many loans please check the system')
        return False

    asset_info = client.get_margin_account()
    assets = [(x['asset'], x['netAsset']) for x in asset_info['userAssets'] if x['netAsset'] != '0']
    for asset in assets:
        transaction = client.transfer_margin_to_spot(asset=asset[0], amount=asset[1])
    return True


def short_sell(portion):
    """
    :param portion: what portion of your usdt do you want to use? 0:1
    important you should just keep usdt in your spot wallet. when doing margin trading
    important your usdt should be free and the margin wallet should be empty
    :return: true (if done) or false
    """
    user = User.objects.get(name='ehsan')
    symbol = 'BTCUSDT'
    client = Client(user.api_key, user.secret_key)
    coin_symbol = str(symbol).replace('USDT', '')
    balance = float(client.get_asset_balance(asset='USDT')['free'])
    print(balance)
    if balance > 0:
        transaction = client.transfer_spot_to_margin(asset='USDT', amount=str(balance * portion))
        print(transaction)

    balance = float(client.get_asset_balance(asset=coin_symbol)['free'])
    print(balance)
    if balance > 0:
        transaction = client.transfer_spot_to_margin(asset=coin_symbol, amount=str(balance * portion))
        print(transaction)

    details = client.get_max_margin_loan(asset=coin_symbol)
    if float(details['amount']) != 0:
        # making the loan
        transaction = client.create_margin_loan(asset=coin_symbol, amount=str(details['amount']))
        print(transaction)
    asset_info = client.get_margin_account()
    print('margin account info is:', [x for x in asset_info['userAssets'] if x['asset'] == coin_symbol])
    material = Material.objects.get(name='BTCUSDT')
    asset = [float(x['free']) for x in asset_info['userAssets'] if x['asset'] == coin_symbol][0]
    print('asset is:', asset)
    quantity = round_down(asset, int(material.amount_digits))
    order = client.create_margin_order(
        symbol=str(symbol),
        side=SIDE_SELL,
        type=ORDER_TYPE_MARKET,
        quantity=str(quantity)
    )
    print(order)
    return True


"""
finish_margin()
short_sell(1)
input('continue?')
finish_margin()
input('continue?')
long_buy()
input('continue?')
finish_margin()

"""

