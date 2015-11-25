#group: Daniel Runge, Scott Viet Phong Nguyen
'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
from keyframes import leftBackToStand
from keyframes import rightBackToStand
from keyframes import leftBellyToStand
import numpy as np
import pickle
from sklearn import svm
from os import listdir, path
from spark_agent import INVERSED_JOINTS


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        ROBOT_POSE_CLF = 'robot_pose.pkl'
        self.posture = 'unknown'
        self.joint_data=['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch']
        self.posture_classifier = pickle.load(open(ROBOT_POSE_CLF))
	self.lock= 0 # needed in standing_up
        
        ROBOT_POSE_DATA_DIR = 'robot_pose_data'
        self.classes = ["Back","Belly","Crouch","Frog","HeadBack","Knee","Left","Right","Sit","Stand","StandInit"]
        #associates numerical value for posture with its actual name
        

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        posture = 'unknown'
        
        measurement = np.zeros((10,))
        measurement[8] = perception.imu[0]
        measurement[9] = perception.imu[1] #insert values for AngleX and AngleY
        
        for i,j in enumerate(measurement[:8]):
            measurement[i] = perception.joint[self.joint_data[i]]
        
        posture = self.classes[int(self.posture_classifier.predict(measurement)[0])]
        #predict posture with classifier
        
            

        return posture

if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = leftBackToStand()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
