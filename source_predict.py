import pickle
import pandas as pd
import numpy as np
from sklearn.covariance import log_likelihood
from sklearn.decomposition import PCA
from sklearn import mixture



filenames=['Params/running.pickle','Params/walking.pickle','Params/jogging.pickle','Params/boxing.pickle',
           'Params/handclapping.pickle','Params/handwaving.pickle']

modelNames=['PCA/running.pickle','PCA/walking.pickle','PCA/jogging.pickle','PCA/boxing.pickle',
           'PCA/handclapping.pickle','PCA/handwaving.pickle']

testNames=['Testing/running_rslt.txt','Testing/walking_rslt.txt','Testing/jogging_rslt.txt','Testing/boxing_rslt.txt','Testing/handclapping_rslt.txt','Testing/handwaving_rslt.txt']


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

        # scores=np.array(scores)
        if np.argmax(scores)==x:
            count+=1

        # print(np.argmax(scores))
        # print('++++++++++++++++++++++++++++++')

print(str(count/total))
