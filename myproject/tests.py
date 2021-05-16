import json
import time
import os
import django
import warnings
warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Finance, Predictor
name = 'mahsa'
user = User.objects.get(name=name)
finance = Finance.objects.get(user=user)
print(finance)

# finance.finish_margin()
# finance.sell(price=finance.get_price())
# finance.buy(price=finance.get_price())
# finance.long_buy(portion=1)
print(finance.get_price())
print(finance.get_asset_in_usd())
predictor = Predictor.objects.all()[0]
print(predictor)
print(predictor.state)
predictor.state += 1
print(f"predictor's state is now: {predictor.state}")
predictor.save()
print(f"predictor's state is now after saving: {predictor.state}")
