from django.shortcuts import render
import numpy as np
from .models import Trader

# Create your views here.


def home(request):
    traders = Trader.objects.all()
    the_trader = traders[0]
    # todo: find the correct user when we have multiple users
    activities = the_trader.activity_set.all()
    context = {
        "activity": activities
    }
    return render(request, 'polls/plot.html', context=context)
