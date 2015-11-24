'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import hello
from keyframes import wipe_forehead
from keyframes import leftBackToStand
import numpy as np
import matplotlib.pyplot as plt
from spark_agent import INVERSED_JOINTS

epsilon = 1e-6 #error margin for x to t conversion

done = 0 #data structures for debug plotting
interpolatedPoints = [[],[]]
keyframePoints = [[],[]]
testJoint = "LElbowYaw"


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])
        self.myTime = self.perception.time
        self.done = 0 # only relevant for debug plotting
        

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)
         
    '''
    Simple function implementing a cubic Hermite interpolation of two points.
    input:	x containing the two x-coordinates of the two interpolation points 
    		y containing the two y-coordinates of the two interpolation points
    		as well as the two values of the derivative of the polynomial
    		in x[0] and x[1] respectively
    
    output:	np.poly1d(coefficients) being the np polynomial p(x) satisfying
    		p(x[0])=y[0],
    		p(x[1])=y[1],
    		d/dx (p(x[0]))=y[2],
    		d/dx (p(x[1]))=y[3].
    '''
    def cubic_Hermite(self,x,y):
    	hermiteMatrix = np.array([
    	[x[0]**3,x[0]**2,x[0],1.],
    	[x[1]**3,x[1]**2,x[1],1.],
    	[3*x[0]**2,2*x[0],1.,0.],
    	[3*x[1]**2,2*x[1],1.,0.]
    	])
    	coefficients = np.linalg.solve(hermiteMatrix, y)
    	return np.poly1d(coefficients)

    def angle_interpolation(self, keyframes):
        target_joints = {}
        
        '''
        Unlike our first attempt using Bezier interpolation, this version
	employs cubic Hermite interpolation using only two data points
	which is guaranteed to at least deliver a continuously differentiable
	interpolation polynomial - unlike Bezier interpolation.
        '''
        
        # YOUR CODE HERE
        time = self.perception.time - self.myTime
        (names, times, keys) = keyframes
        
        for i, name in enumerate(names):
	  curTimes = times[i]
	  
	  if curTimes[-1]<time or time<curTimes[0]: #skip if time is not in time frame
	    
	    #plot interpolated data for testing
	    if (not self.done) and curTimes[-1]<time:
	      self.done = 1
	      plt.plot(interpolatedPoints[0],interpolatedPoints[1],"r")
	      plt.plot(keyframePoints[0],keyframePoints[1],"bo")
	      plt.title(testJoint)
	      plt.savefig('Hermite_interpolation.png')
	      plt.show()
	    
	    continue
	  
	  #getting relevant indices
	  eIndex = len([x for x in curTimes if x<time])
	  sIndex = eIndex - 1
	  
	  #get interpolation points
	  skey = keys[i][sIndex]
	  (p0x, p0y) = (curTimes[sIndex], skey[0])
	  #use the slope defined by the handle bars as the derivative in the
	  #interpolation points; denoted by dy_i for the derivative in point i
	  #where the points are enumerated in the same fashion as in the
	  #bezier version	  
	  (p1x, p1y) = (p0x + skey[2][1], p0y + skey[2][2]) #handle bar end to the right of p0
	  (p1bx,p1by) = (p0x + skey[1][1], p0y + skey[1][2]) #handle bar end to the left of p0
	  dy_0 = (p1y - p1by) / (p1x - p1bx) #all assuming that both handle bars have an actual offset, i.e. the denominator is !=0
	  ekey = keys[i][eIndex]
	  (p3x, p3y) = (curTimes[eIndex], ekey[0])
	  (p2x, p2y) = (p3x + ekey[1][1], p3y + ekey[1][2]) #handle bar end to the left of p3
	  (p2bx, p2by) = (p3x + ekey[2][1], p3y + ekey[2][2]) # handle bar end to the right of p3
	  dy_3 = (p2by - p2y) / (p2bx - p2x) 
	  
	  #expressing cubic Hermite interpolation in matrix form and solving
	  #for the coefficients for the interpolatig polynomial
	  #note that in np.poly1d the coefficients for the polynomial are read
	  #in descending order as opposed to the bezier version where the
	  #coefficients to be interpreted as a polynomial were read in ascending order
	  x = np.array([p0x,p3x])
	  y = np.array([p0y,p3y,dy_0,dy_3])
	  polynomial = self.cubic_Hermite(x,y)
	  
	  	  	  	    
	  #getting y value
	  result = polynomial(time)
	  target_joints[name] = result
	  
	  
	  #collecting plot data
	  if name == testJoint:
	    interpolatedPoints[0].append(time)
	    interpolatedPoints[1].append(result)
	    keyframePoints[0].append(p0x)
	    keyframePoints[1].append(p0y)
	    keyframePoints[0].append(p3x)
	    keyframePoints[1].append(p3y)
	  
        
        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
