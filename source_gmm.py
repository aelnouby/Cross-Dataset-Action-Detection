# Source GMM :
# fit a GMM model to the KTH data, represented in (STIPs) HOG/HOFs
# that are reduced with a PCA
# Author : Alaa El-Nouby

import numpy as np
import pandas as pd
from sklearn import mixture
import pickle


filenames=['Models/PCA/running.csv','Models/PCA/walking.csv','Models/PCA/jogging.csv','Models/PCA/boxing.csv',
           'Models/PCA/handclapping.csv','Models/PCA/handwaving.csv']

g=mixture.GMM(n_components=512)

thetas=np.empty(6)

for fName in filenames:
    f=pd.read_csv(fName,header=0)
    g.fit(f)
    pickle.dump(g, open('Models/Theta_C/'+fName+'.pickle', 'wb'))
