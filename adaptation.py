# Adaptation :
# Framework for adapting the GMM model trained on KTH dataset on the MSR_Action_II
# Dataset, mainly updating the means of the GMM 512 cluster.
# Author : Alaa El-Nouby

import numpy as np
import pickle
from sklearn import mixture
from sklearn.decomposition import PCA


# Load Background Model
theta_b = pickle.load(open('Models/Theta_B/theta_b.pickle', 'rb'))
# Load Action Models
theta_c1 = pickle.load(open('Models/Theta_C/updated_handclapping.pickle', 'rb'))
theta_c2 = pickle.load(open('Models/Theta_C/updated_handwaving.pickle', 'rb'))
theta_c3 = pickle.load(open('Models/Theta_C/updated_boxing.pickle', 'rb'))
theta = [theta_c1, theta_c2, theta_c3]

# Load PCA background model
pca_b = pickle.load(open('Models/PCA/background.pickle', 'rb'))
# Load PCA Action Model
pca_c1 = pickle.load(open('Models/PCA/handclapping.pickle', 'rb'))
pca_c2 = pickle.load(open('Models/PCA/handwaving.pickle', 'rb'))
pca_c3 = pickle.load(open('Models/PCA/boxing.pickle', 'rb'))
pca = [pca_c1, pca_c2, pca_c3]


# Read the Extracted Subvolumes
# =================================
with open('groundtruth.txt') as k:
    content = k.readlines()

floats = []

for i in range(int(len(content))):
    k = [float(x) for x in content[i].split()]
    floats.append(k)

vol = np.array(floats)
vol = vol.astype(np.float16)
# ===================================

# Read Examples
# ===================================
f = open('Data/MSR_Action_II/target.txt', 'rt')
data = f.read()
examples = data.split(',')
counter = 1

for ex in examples:
    lines = ex.splitlines()
    pts = []
    for i in range(len(lines)):
        p = [float(x) for x in lines[i].split()]
        pts.append(p)
    arr = np.array(pts)
    exVol = vol[vol[:, 6] == counter, :]
    for x in [1, 2, 3]:
        action = exVol[exVol[:, 7] == x, :]
        print(action,x)
        if(action.shape[0]==0): break
        ptsVol = np.empty(171)
        for a in action:
            conditionX = np.logical_and(arr[:, 5] >= a[0], arr[:, 5] <= (a[1]+a[0]))
            conditionY = np.logical_and(arr[:, 4] >= a[2], arr[:, 4] <= (a[3]+a[2]))
            conditionT = np.logical_and(arr[:, 6] >= a[4], arr[:, 6] <= (a[5]+a[4]))
            condition = np.logical_and(conditionX, np.logical_and(conditionY, conditionT))
            ptsVol = arr[condition, :]
            # ptsVol = ptsVol[1:,:]
            Q = ptsVol[:, 9:]

            means = theta[x-1].means_
            weights = theta[x-1].weights_
            alpha = 0.1

            _, posterior = theta[x-1].score_samples(pca[x-1].transform(Q))
            posterior[posterior==0] = 1e-150    #Avoiding divison by zero
            denominator = np.dot(np.exp(posterior), weights)[np.newaxis]

            p = (posterior * weights)/denominator.transpose()

            E = np.dot(pca[x-1].transform(Q).transpose(), p)/np.sum(p, axis=0)

            theta[x-1].means_ = alpha*E.transpose() + (1-alpha)*means

    counter += 1

pickle.dump(theta[0], open('Models/Theta_C/updated_handclapping.pickle', 'wb'))
pickle.dump(theta[1], open('Models/Theta_C/updated_handwaving.pickle', 'wb'))
pickle.dump(theta[2], open('Models/Theta_C/updated_boxing.pickle', 'wb'))
