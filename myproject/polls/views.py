from django.shortcuts import render
import numpy as np
from .models import Activity

# Create your views here.


def home(request):
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
