import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

# Settings
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load dataset
path = 'C:/Users/pablo/Documents/TercerUAB/Aprenentatge Computacional/Pràctiques/Pràctica 1/'
dataset = pd.read_csv(path + 'Life Expectancy Data.csv')

# Correct column names
dataset = dataset.rename(
    columns={
        'Life expectancy ': 'Life expectancy',
        'infant deaths': 'Infant deaths',
        'percentage expenditure': 'Percentage expenditure',
        'Measles ': 'Measles',
        ' BMI ': 'BMI',
        'under-five deaths ': 'Under-five deaths',
        'Diphtheria ': 'Diphtheria',
        ' HIV/AIDS': 'HIV/AIDS',
        ' thinness  1-19 years': 'Thinness  1-19 years',
        ' thinness 5-9 years': 'Thinness 5-9 years'
        }
)

print(dataset.columns)
