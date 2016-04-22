# Cross Dataset Action Detection
** The Work is Stil in progress
#Abstract
In recent years, many research works have been carried out to recognize human actions from video clips. To
learn an effective action classifier, most of the previous approaches rely on enough training labels. When being re-
quired to recognize the action in a different dataset, these approaches have to re-train the model using new labels.
However, labeling video sequences is a very tedious and time-consuming task, especially when detailed spatial lo-
cations and time durations are required. In this paper, we propose an adaptive action detection approach which reduces the requirement of training labels and is able to handle the task of cross-dataset action detection with few or no
extra training labels. Our approach combines model adaptation and action detection into a Maximum a Posterior
(MAP) estimation framework, which explores the spatialtemporal coherence of actions and makes good use of the
prior information which can be obtained without supervision. Our approach obtains state-of-the-art results on KTH
action dataset using only 50% of the training labels in tradition approaches. Furthermore, we show that our approach
is effective for the cross-dataset detection which adapts the model trained on KTH to two other challenging datasets.

<img src="https://www.researchgate.net/profile/Liangliang_Cao2/publication/221361464/figure/fig1/Figure-1-The-framework-of-our-cross-dataset-action-detection-method.png">

##Adaptation Results : 

Using alpha = 0.1 in the mean adaptaion equation and executing `predict_target.py`, the results were as follows

|          Adaptation          |    Accuracy      |    
| ---------------------------- |:----------------:| 
| No Adaptaion (pure KTH mode) |      46.4%       | 
|        1st Iteration         |      80.6%       | 
|        2nd Iteration         |      92.6%       | 
|        3nd Iteration         |      94.62%      | 


#Datasets
I. Recognition of human actions, KTH. [[Data](http://www.nada.kth.se/cvap/actions/)]

II. MSR Action Dataset I [[Data](http://research.microsoft.com/en-us/um/people/zliu/actionrecorsrc/)]

#Software
Subvolume Branch-and-Bound Search binaries(WIN32) [[binaries](http://research.microsoft.com/en-us/um/people/zliu/actionrecorsrc/SubvolumeSearch.zip)]

In Linux use wine as follows :

`wine SubVolumeSearch.exe infile outfile numclasses width height penaltyValue classthreshold1 classthreshold2 ... classthresholdn`

##Refrences
[1] Cao, Liangliang, Zicheng Liu, and Thomas S. Huang. "Cross-dataset action detection." Computer vision and pattern recognition (CVPR), 2010 IEEE conference on. IEEE, 2010. [[PDF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.680.6094&rep=rep1&type=pdf)]

[2] Yuan, Junsong, Zicheng Liu, and Ying Wu. "Discriminative video pattern search for efficient action detection." Pattern Analysis and Machine Intelligence, IEEE Transactions on 33.9 (2011): 1728-1743.[[PDF](https://dr.ntu.edu.sg/bitstream/handle/10220/18133/Discriminative%20Video%20Pattern%20Search%20for%20Efficient%20Action%20Detection.pdf?sequence=1)]

[3] Project page by Liangliang Cao [[Page](http://www.ifp.illinois.edu/~cao4/crossdataset_action/)]
