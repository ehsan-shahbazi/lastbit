from django.contrib import admin
from .models import User, Finance, Material, Predictor, Trader, Activity, Asset, Signal
# Register your models here.


admin.site.register(User)
admin.site.register(Material)
admin.site.register(Predictor)
admin.site.register(Trader)
admin.site.register(Activity)
admin.site.register(Finance)
admin.site.register(Asset)
admin.site.register(Signal)
