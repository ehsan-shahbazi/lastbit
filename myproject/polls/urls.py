from django.urls import path
from . import views

urlpatterns = [
    path('plot', views.home, name='home'),
    path('assets', views.monitor, name='monitor'),
    path('BTCSignal', views.signal, name='BTCSignal'),
]

