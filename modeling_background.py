import numpy as np
from sklearn.decomposition import PCA
from sklearn import mixture
import time
import pickle

pca=PCA(n_components=130)
g=mixture.GMM(n_components=512)

with open("Target/target.txt") as f:
    content = f.readlines()

floats=[]

# TODO: Use the Whole Data instead of half
for i in range(int(len(content))):
    f=[float(x) for x in content[i].split()]
    floats.append(f)


arr = np.array(floats)
arr = arr.astype(np.float16)

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
