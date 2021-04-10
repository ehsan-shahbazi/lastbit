from django.urls import path
from . import views

urlpatterns = [
    path('plot', views.home, name='home'),
    path('assets', views.monitor, name='monitor'),
    path('BTCSignal', views.signal, name='BTCSignal'),
    path('user_traders', views.active_predictors, name='user_traders'),
    path('predictors', views.all_predictors, name='predictors')
]

