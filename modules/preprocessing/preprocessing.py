import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from constants import ROOT_DIR

register_matplotlib_converters()

data = pd.read_csv('{}/data/pd_speech_features.csv'.format(ROOT_DIR))
print(data.shape)
