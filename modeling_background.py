# Background Modeling :
# Fitting the Background GMM model for the MSR_Action_II and applying PCA
# reduction
# Author : Alaa El-Nouby

import numpy as np
from sklearn.decomposition import PCA
from sklearn import mixture
import time
import pickle

pca=PCA(n_components=130)
g=mixture.GMM(n_components=512)

with open("Data/MSR_Action_II/background.txt") as f:
    content = f.readlines()

floats=[]
print(len(content))

# TODO: Use the Whole Data instead of half
for i in range(int(len(content)/2)):
    f=[float(x) for x in content[i].split()]
    floats.append(f)


arr = np.array(floats)
arr = arr.astype(np.float16)

print(arr[5,5])

hogHof = arr[:,9:]

decomposed = pca.fit(hogHof)
pickle.dump(decomposed, open('background_pcad.pickle', 'wb'))
print("PCA model saved sucessfully")

tic = time.time()
print("Fitting GMM")
g.fit(pca.transform(hogHof))
print("Saving GMM model")
pickle.dump(g, open('background_model.pickle', 'wb'))
toc = time.time()
print((toc-tic)/60)
