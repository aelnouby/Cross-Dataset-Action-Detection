# Scoring :
# Using the adapted models (Theta_C)s and the Background model (theta_b)
# to score each STIP (q) to determine if it belongs to the action or the Background
# The output will be feed to the branch and bound alg. to determine the action Subvolumes
# Author : Alaa El-Nouby
# NOTE: The is a fault with Scoring right now

import numpy as np
import pickle
import math
import sys

# Load Background Model
theta_b = pickle.load(open('Models/Theta_B/theta_b.pickle', 'rb'),encoding='latin1')
# Load Action Models
theta_c1 = pickle.load(open('Models/Theta_C/updated_handclapping.pickle', 'rb'))
theta_c2 = pickle.load(open('Models/Theta_C/updated_handwaving.pickle', 'rb'))
theta_c3 = pickle.load(open('Models/Theta_C/updated_boxing.pickle', 'rb'))

# Load PCA background model
pca_b = pickle.load(open('Models/PCA/background.pickle', 'rb'),encoding='latin1')
# Load PCA Action Model
pca_c1 = pickle.load(open('Models/PCA/handclapping.pickle', 'rb'))
pca_c2 = pickle.load(open('Models/PCA/handwaving.pickle', 'rb'))
pca_c3 = pickle.load(open('Models/PCA/boxing.pickle', 'rb'))

f = open('Data/MSR_Action_II/target.txt','rt')
content = f.read()
examples = content.split(',')
counter = 1

print(len(examples))
for ex in examples:
    lines = ex.splitlines()
    pts = []
    for i in range(len(lines)):
        p = [float(x) for x in lines[i].split()]
        pts.append(p)
    arr = np.array(pts)

    output = open('Scores/example_'+str(counter)+'.txt', 'w')
    stip_number = arr.shape[0]
    frames_number = int(arr[stip_number-1][6])

    output.write(str(stip_number))
    output.write('\n')
    output.write(str(frames_number))
    output.write('\n')
    for q in arr:
        y = q[4]
        x = q[5]
        t = q[6]
        hogHof = q[9:]
        _, posterior_c1 = theta_c1.score_samples(pca_c1.transform(hogHof))
        _, posterior_c2 = theta_c2.score_samples(pca_c2.transform(hogHof))
        _, posterior_c3 = theta_c3.score_samples(pca_c3.transform(hogHof))
        _, posterior_b = theta_b.score_samples(pca_b.transform(hogHof))

        score_b = np.dot(np.exp(posterior_b),theta_b.weights_)
        score_c1 = np.dot(np.exp(posterior_c1),theta_c1.weights_)/score_b
        score_c2 = np.dot(np.exp(posterior_c2),theta_c2.weights_)/score_b
        score_c3 = np.dot(np.exp(posterior_c3), theta_c3.weights_)/score_b

# print(r1)
        # print(r2)
        # print(r3)
        # print(rb)
        # print(np.sum(np.dot(r1,theta_c1.weights_)))
        # print("=========================")

        output.write(str(int(x))+"\t")
        output.write(str(int(y))+"\t")
        output.write(str(int(t)) +"\t\t")
        output.write(str("%.3f" %(math.log(score_c1*score_c2*score_c3*score_b))) + "\t")
        output.write(str("%.3f" %(math.log(score_c1))) + "\t")
        output.write(str("%.3f" %(math.log(score_c2))) + "\t")
        output.write(str("%.3f" %(math.log(score_c3)))+"\t")
        output.write('\n')

    output.close()
    counter += 1
