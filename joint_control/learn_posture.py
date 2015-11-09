#group: Daniel Runge, Scott Viet Phong Nguyen
'''
This program was partly written before learn_posture.ipynb was pushed to the
DAInamite branch. It does exactly the same as learn_posture.ipynb: it reads
the posture data and the corresponding target classes, trains an svm and saves
it to a file robot_pose.pkl which is then to be loaded as the posture classifier
in recognize_posture.py
'''

import pickle
from os import listdir, path
import numpy as np
from sklearn import svm, metrics

def evaluate(expected, predicted):
    print "Classification report: \n%s\n" % metrics.classification_report(expected,predicted)

ROBOT_POSE_DATA_DIR = 'robot_pose_data'

#listdir lists the files in ROBOT_POSE_DATA_DIR in arbitrary fashion, so
#we should list them individually and consistent with how we list them in
# recognize_posture to make the classifier system-independant
classes = ["Back","Belly","Crouch","Frog","HeadBack","Knee","Left","Right","Sit","Stand","StandInit"]
        
data=[]
N_samples = [] # number of samples for each posture in the classifier
        
for i in classes:
    data1 = pickle.load(open(path.join(ROBOT_POSE_DATA_DIR,i))) #load i-th class
    k,l = np.asarray(data1).shape
    N_samples.append(k) #update number of samples for posture in question
    for j in data1:
        data.append(j) #append the read data
        
data = np.asarray(data)
N_samples = np.asarray(N_samples) #convert to np arrays for further analysis
        
total_sample_number = np.sum(N_samples) #number of all samples (=200)
#target_postures = np.zeros((total_sample_number,),dtype='|S9')
target_postures = np.zeros((total_sample_number,))
#list of the classes the individual samples belong to
k=0
for i,j in enumerate(N_samples):
    k+=j
    target_postures[(k-j):k] = i
        #write for each sample in data the corresponding posture into target_postures for machine learning training

#prepare shuffling: use 70% of data for training
permutation = np.random.permutation(total_sample_number)
training_data_sample_number = int(total_sample_number*0.9)

#shuffle training data and target classes
training_data_indices = permutation[:training_data_sample_number]
testing_data_indices = permutation[training_data_sample_number:]

training_data = [data[i] for i in training_data_indices]
training_postures = [target_postures[i] for i in training_data_indices]

testing_data = [data[i] for i in testing_data_indices]
testing_postures = [target_postures[i] for i in testing_data_indices]
       
clf = svm.SVC(gamma=0.001,C=100.) #load support vector classifier
       
classifier = clf.fit(training_data,training_postures) #train classifier on the given data

evaluate(classifier.predict(testing_data),testing_postures)

pickle.dump(classifier,open('robot_pose.pkl','w'))