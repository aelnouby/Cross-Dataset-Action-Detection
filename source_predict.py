# Source predict :
# Test the GMM model fitted to the KTH data, and measure the accuracy of recognition
# w.r.t every action.
# Alaa El-Nouby

import pickle
import pandas as pd
import numpy as np
from sklearn.covariance import log_likelihood
from sklearn.decomposition import PCA
from sklearn import mixture



filenames=['Models/Theta_C/running.pickle','Models/Theta_C/walking.pickle','Models/Theta_C/jogging.pickle','Models/Theta_C/boxing.pickle',
           'Models/Theta_C/handclapping.pickle','Models/Theta_C/handwaving.pickle']

modelNames=['Models/PCA/running.pickle','Models/PCA/walking.pickle','Models/PCA/jogging.pickle','Models/PCA/boxing.pickle',
           'Models/PCA/handclapping.pickle','Models/PCA/handwaving.pickle']

testNames=['Data/KTH/Testing/running_rslt.txt','Data/KTH/Testing/walking_rslt.txt','Data/KTH/Testing/jogging_rslt.txt','Data/KTH/Testing/boxing_rslt.txt','Data/KTH/Testing/handclapping_rslt.txt','Data/KTH/Testing/handwaving_rslt.txt']


pcaModels=[]
for mName in modelNames:
    pca = pickle.load(open(mName, 'rb'))
    pcaModels.append(pca)

paramsList=[]
for fName in filenames:
    params = pickle.load(open(fName, 'rb'))
    paramsList.append(params)

count=0
total=0


for x in range(0,6):
    #Test Case
    f=open(testNames[x],'rt')
    content=f.read()
    examples=content.split(',')
    total+=len(examples)
    for ex in examples:
        lines=ex.splitlines()
        pts=[]
        for i in range(len(lines)):
            p=[float(x) for x in lines[i].split()]
            pts.append(p)
        arr=np.array(pts)
        hogHof = arr[:,9:]
        scores=[]
        for i in range(0,6):
            test=pcaModels[i].transform(hogHof)
            score=np.sum(paramsList[i].score(test))
            scores.append(score)

        s = np.argmax(scores)
        print('s',s)
        print("score",scores[s]/np.sum(scores))
        print(scores)
        print('x',x)
        # scores=np.array(scores)
        if np.argmax(scores)==x:
            count+=1

        # print(np.argmax(scores))
        # print('++++++++++++++++++++++++++++++')

print(str(count/total))
