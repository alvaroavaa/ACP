import matplotlib 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import seaborn as sns

# Utilizando os dados do iris

iris = sns.load_dataset("iris")

#print(iris.head(4))

#convertendo os valores numéricos para numpy arrays
num = iris[['sepal_length','sepal_width','petal_length','petal_width']].values # separa os valores da matriz

num=scale(num) # Essa função  (scale) realiza a padronização dos dados, o que envolve a subtração da média e a divisão pelo desvio padrão de cada variável.


pca = PCA(n_components=4)
pca.fit(num) # Quando chamamos pca.fit(num), estamos realizando o processo de ajuste do modelo PCA aos dados num. Isso significa que o modelo está sendo treinado para aprender os componentes principais a partir desses dados.

var = pca.explained_variance_ratio_

#plt.plot(var)
#plt.show()

#variacia cumulativa 

var1= np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)
plt.plot(var1)
plt.show()
