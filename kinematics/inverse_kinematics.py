'''In this exercise you need to implemente inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinemtatics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h25/joints_h25.html
       http://doc.aldebaran.com/2-1/family/nao_h25/links_h25.html
    2. use the results of inverse kinemtatics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinemtatics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        joint_angles = []
        # YOUR CODE HERE
	# implement the Cyclic Coordinate Descent algorithm
	target = (transform[0][3],transform[1][3],transform[2][3])
	joints = self.chains[effector_name]
	num_of_links = len(self.chains[effector_name])
	end_pos = self.jointLinks[joints[-1]]
	for i in range(1,num_of_links):
		curr_pos= self.jointLinks[joints[-i]]
		t= numpy.array(target)
		e= numpy.array(end_pos)
		c=numpy.array(curr_pos)
		a = (e-c)/numpy.linalg.norm(e-c)
		b = (t-c)/numpy.linalg.norm(t-c)
		teta = numpy.arccos(numpy.dot(a,b))
		# check the direction in which we have to rotate
		direction =numpy.cross(a,b)
		if direction[2] <0:
			teta= -1*teta
		joint_angles.append(teta)
        return joint_angles

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        angles = self.inverse_kinematics(effector_name,transform)
        # not sure how to set the keyframes 
        self.keyframes = ([], [], [])  # the result joint angles have to fill in

if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = 0.26
    agent.set_transforms('LLeg', T)
    agent.run()