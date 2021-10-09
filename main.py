import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
        ' thinness  1-19 years': 'Thinness 1-19 years',
        ' thinness 5-9 years': 'Thinness 5-9 years'
        }
)

# Clean dataset (NaN etc.)
# dataset = dataset.dropna()
# print(dataset.isnull().sum())

# dataset = dataset.drop(columns=['Population'])  # Nans & correlation
# dataset = dataset.dropna(subset=['Life expectancy']) Eliminar els nans de les columnes escrites


# Plots
# sns.heatmap(dataset.corr(), annot=True, linewidths=.5)


#substituimos valores nan por la media del resto de valores de ese atributo
for col in dataset:
     if (dataset[col].dtype != 'object'):
         dataset[col][np.isnan(dataset[col])] = dataset[col].mean()
# print(dataset.isnull().sum())

# plt.show()

# for idx, col in enumerate(dataset):
#     plt.clf()
#     plt.title(col)
#     plt.hist(dataset[col])
#     plt.savefig(f'C:/Users/pablo/Documents/TercerUAB/Aprenentatge Computacional/Pràctiques/Pràctica 1/plots/{idx}')

ls = []
corr_matrix = dataset.corr().abs()['Life expectancy']
for col, corr in corr_matrix.items():
    if corr > 0.5:
        ls.append(col)

# sns.pairplot(dataset[ls])
plt.show()

# sns.pairplot(dataset[['Life expectancy', 'Adult Mortality', 'BMI', 'Schooling', 'Income composition of resources']])
# plt.show()

# for col in dataset:
#     print(f'{col} = {dataset[col].dtype}')

#Mirar si els atributs són Gaussianes mirant la mitja, la mediana i la desviació estandard:
# for col in dataset:
#     if dataset[col].dtype != 'object':
#         print(col+': '+str(dataset[col].mean()) +' '+ str(dataset[col].median())+' '+str(dataset[col].std()))

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# print(dataset[dataset['Income composition of resources'] == 0])

#Apartat B

def mse(v1, v2):
    return ((v1 - v2)**2).mean()

def regression(x, y, normalize=False):
    # Creem un objecte de regressió de sklearn
    regr = LinearRegression(normalize=normalize)
    # Entrenem el model per a predir y a partir de x
    regr.fit(x, y)
    # Retornem el model entrenat
    return regr

def standarize(x_train):
    mean = x_train.mean(0)
    std = x_train.std(0)
    x_t = x_train - mean[None, :]
    x_t /= std[None, :]
    return x_t

data = dataset.values
x = data[:, range(4, 10)]
y = dataset['Life expectancy']

atribut1 = x[:,0].reshape(x.shape[0], 1)
reg = regression(atribut1, y, normalize=True)
pred = reg.predict(atribut1)
plt.figure()
ax = plt.scatter(x[:,0],y)
plt.plot(atribut1[:,0], pred, 'r')
plt.show()
#plt.plot(reg)

#x_t = standarize(x)

MSE = mse(y,pred)
r2 = r2_score(y, pred)

print("Mean squared error: ", MSE)
print("R score: ", r2)
