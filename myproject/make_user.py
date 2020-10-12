import os
import django
import warnings
warnings.filterwarnings("ignore")
file_location = ''
os.environ["DJANGO_SETTINGS_MODULE"] = 'myproject.settings'
django.setup()
from polls.models import User, Predictor, Material, Trader, Finance


def make_user(user_name, multi_coin):
    user = User(name=user_name)
    user.save()
    if multi_coin:
        materials = Material.objects.all()
        for material in materials:
            predictor = Predictor(material=material, user_name=user_name, time_frame='15m', input_size=466, type='HIST')
            predictor.save()
            trader = Trader(predictor=predictor, user=user, type='1', active=True)
            trader.save()
            finance = Finance(user=user, symbol=material.name)
            finance.save()
    else:
        material = Material.objects.get(name='BTCUSDT')[0]
        predictor = Predictor(material=material, user_name=user_name, time_frame='15m', input_size=466, type='HIST')
        predictor.save()
        trader = Trader(predictor=predictor, user=user, type='1', active=True)
        trader.save()
        finance = Finance(user=user, symbol=material.name)
        finance.save()

    return True


if __name__ == '__main__':
    the_user_name = input('please insert the new users name:')
    is_multi = input('press m if the user is multi-coin and o otherwise:')
    if is_multi == 'm':
        ans = make_user(the_user_name, True)
    else:
        ans = make_user(the_user_name, False)
    if ans:
        print('the user and related stuffs are made\nplease note that the secret key and api key should be inserted'
              ' manually')
    else:
        print('some thing went wrong!')
