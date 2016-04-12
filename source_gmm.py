import numpy as np
import pandas as pd
from sklearn import mixture
import pickle


filenames=['PCA/running.csv','PCA/walking.csv','PCA/jogging.csv','PCA/boxing.csv',
           'PCA/handclapping.csv','PCA/handwaving.csv']

g=mixture.GMM(n_components=512)

thetas=np.empty(6)

for fName in filenames:
    f=pd.read_csv(fName,header=0)
    g.fit(f)
    pickle.dump(g, open('Params/'+fName+'.pickle', 'wb'))


