from django.test import TestCase
import pandas as pd

a = [1, 2, 3, 4, 5, 3, 2]
b = pd.DataFrame({'a': a})
print(max(list(b['a'].tail(n=3))))

# Create your tests here.
