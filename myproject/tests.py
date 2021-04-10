import json
import time
import os
import django
import warnings
warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Finance
name = 'mahsa'
user = User.objects.get(name=name)
finance = Finance.objects.get(user=user)
print(finance)

# finance.finish_margin()
# finance.sell(price=finance.get_price())
# finance.buy(price=finance.get_price())
# finance.long_buy(portion=1)
print(finance.get_price())
trades = finance.get_trades()
ans = json.load(trades)
print(ans)