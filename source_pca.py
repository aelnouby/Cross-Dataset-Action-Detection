#Source PCA :
# Read the features file and spereate the HOG/HOFs vector
# and applying PCA reduction to the features
# Author : Alaa El-Nouby

import numpy as np
from sklearn.decomposition import PCA
import pickle

filenames=['Data/KTH/Training/running_rslt.txt','Data/KTH/Training/walking_rslt.txt','Data/KTH/Training/jogging_rslt.txt','Data/KTH/Training/boxing_rslt.txt',
           'Data/KTH/Training/handclapping_rslt.txt','Data/KTH/Training/handwaving_rslt.txt']

# TODO: Find a way to find a better PCA Reduction
pca=PCA(n_components=130)
for fName in filenames:
    f=open(fName,'rt')
    content=f.read()
    print(f)
    featues=np.empty(162)
    examples=content.split(',')
    for ex in examples:
        lines=ex.splitlines()
        pts=[]
        for i in range(len(lines)):
            p=[float(x) for x in lines[i].split()]
            pts.append(p)
        arr=np.array(pts)
        hogHof = arr[:,9:]
        featues=np.vstack([featues,hogHof])

    # floats=[]
    # for i in range(len(examples)):
    #     no=[float(x) for x in examples[i].split()]
    #     floats.append(no)
    featues=featues[1:,:]
    decomposed_examples=pca.fit(featues)
    pickle.dump(decomposed_examples, open('Models/PCA/'+fName+'.pickle', 'wb'))
    print(pca.n_components_)
    print(pca.explained_variance_ratio_)
    print("==========================")
    np.savetxt("PCA/"+fName+".csv",decomposed_examples , delimiter=",")
    print("saved")
