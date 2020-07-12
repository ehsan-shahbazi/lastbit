from django.urls import path
from . import views

urlpatterns = [
    path('plot', views.home, name='home')
]

