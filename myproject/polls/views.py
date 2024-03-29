import json

from django.shortcuts import render
from .models import Activity, User, Predictor, Trader, Finance
from django.http import JsonResponse

# Create your views here.


def monitor(request):
    name_asset = []
    users_asset = []
    labels = []
    for user in User.objects.all():
        asset = 0
        for finance in user.finance_set.all():
            asset += finance.get_asset_in_usd()

        users_asset.append([str(user.name), str(asset)])
    context = {
        "name": 'Mahsa',
        "labels": labels,
        "final_asset": users_asset,
        "assets": name_asset
    }
    print(len(context['labels']), len(context['assets']))
    return render(request, 'polls/assets.html', context=context)


def home(request):
    print('hi')

    activities = list(Activity.objects.all())
    labels = [i for i in range(len(activities))]
    data = [act.price for act in activities]
    asset = []
    mat = 0
    budget = 0
    for act in activities:
        if act == activities[0]:
            asset.append(act.price)
            if act.action == 'buy':
                mat = 1
            else:
                budget = act.price
            continue
        if act.action == 'buy':
            if budget == 0:
                asset.append(mat * act.price)
            else:
                asset.append(budget * 0.999)
                mat = budget * 0.999 / act.price
                budget = 0
        else:
            if mat == 0:
                asset.append(budget)
            else:
                asset.append(mat * 0.999 * act.price)
                budget = mat * 0.999 * act.price
                mat = 0

    context = {
        "labels": labels,
        "data": data,
        "asset": asset
    }
    return render(request, 'polls/plots.html', context=context)


def signal(request):
    for predictor in Predictor.objects.all():
        if predictor.user_name == 'mahsa':
            if predictor.state == 1:
                return JsonResponse({'Signal': 'buy', 'stop_sell_price': predictor.state_last_price_set})
            else:
                return JsonResponse({'Signal': 'sell', 'stop_buy_price': predictor.state_last_price_set})


def active_predictors(request):
    user_name = request.GET.get('name', 'mahsa')
    print(f"username is: {user_name}")
    traders = {user_name: []}
    for trader in Trader.objects.all():
        if trader.user.name == user_name:
            predictor = trader.predictor.type
            traders[user_name].append([{'trader': trader.__str__(), 'predictor_type': predictor}])
    return JsonResponse(traders)


def all_predictors(request):
    predictors = {'predictors': []}
    for predictor in Predictor.objects.all():
        if predictor.type not in predictors['predictors']:
            predictors['predictors'].append(predictor.type)
    return JsonResponse(predictors)


def get_trades(request):
    user_name = request.GET.get('name', 'mahsa')
    user = User.objects.get(name=user_name)
    finance = Finance.objects.get(user=user)
    trades = finance.get_trades()
    return JsonResponse({'trades': trades})


def get_asset(request):
    user_name = request.GET.get('name', 'mahsa')
    user = User.objects.get(name=user_name)
    finance = Finance.objects.get(user=user)
    asset = finance.get_asset_in_usd()
    return JsonResponse({'asset': asset})
