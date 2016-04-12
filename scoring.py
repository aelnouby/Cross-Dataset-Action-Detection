import numpy as np
import pickle
import math

# Load Background Model
theta_b = pickle.load(open('Theta_B/theta_b.pickle', 'rb'))
# Load Action Models
theta_c1 = pickle.load(open('Theta_C/boxing.pickle', 'rb'))
theta_c2 = pickle.load(open('Theta_C/handwaving.pickle', 'rb'))
theta_c3 = pickle.load(open('Theta_C/handclapping.pickle', 'rb'))

# Load PCA background model
pca_b = pickle.load(open('background_pcad.pickle', 'rb'))
# Load PCA Action Model
pca_c1 = pickle.load(open('PCA/boxing.pickle', 'rb'))
pca_c2 = pickle.load(open('PCA/handwaving.pickle', 'rb'))
pca_c3 = pickle.load(open('PCA/handclapping.pickle', 'rb'))

f = open('Target/target.txt','rt')
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

    output = open('example_'+str(counter)+'.txt', 'w')
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
    break
    counter += 1






