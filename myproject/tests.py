import time
import os
import django
import warnings
warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Material, Finance
name = 'mahsa'
user = User.objects.get(name='name')
finance = Finance.objects.get(user=user)
finance.finish_margin()
finance.buy(price=finance.get_price())
finance.long_buy(portion=1)
print(finance)
