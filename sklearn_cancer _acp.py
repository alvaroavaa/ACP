import matplotlib 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()


#Essa linha carrega o conjunto de dados sobre câncer de mama do módulo datasets do scikit-learn. O conjunto de dados carregado é armazenado na variável cancer.
df= pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df)

#O DataFrame terá as colunas nomeadas de acordo com as características do conjunto de dados.
scaler = StandardScaler()
scaler.fit(df) # normalização do dado

# scaler calcula a média e o desvio padrão dos dados para padronizá-los.
scaled_data = scaler.transform(df)

# o modelo PCA calcula os dois primeiros componentes principais que explicam a maior parte da variação nos dados.
pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data) #  aplica a projeção dos dados nos componentes principais calculados pelo PCA.

x_pca_2 = pd.DataFrame(x_pca) # Os dados são organizados em um formato tabular, onde cada coluna representa um componente principal

x_pca_2.columns = ['PC1','PC2'] # colunas do DataFrame como "PC1" e "PC2", que correspondem aos nomes dos dois componentes principai
print(x_pca_2)

print(pca.explained_variance_ratio_) # proporção da variância explicada por cada componente principal.

plt.figure(figsize=(10,6))
plt.scatter(x_pca[:,0], x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('primeira acp')
plt.ylabel('segunda acp')

plt.show()
