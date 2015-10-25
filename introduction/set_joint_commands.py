'''
In this exercise you need to know how to set joint commands.

* Tasks:
    1. set stiffness of LShoulderPitch to 0
    2. set speed of HeadYaw to 0.1

* Hint: The commands are store in action (class Action in spark_agent.py)

'''

from spark_agent import SparkAgent


class MyAgent(SparkAgent):
    def think(self, perception):
        action = super(MyAgent, self).think(perception)
        # YOUR CODE HERE
        for key, value in action.stiffness.items():
 			action.stiffness["LShoulderPitch"] = 0
        for key, value in action.speed.items():
 			action.speed["HeadYaw"] = 0.1 					
        return action

if '__main__' == __name__:
    agent = MyAgent()
    agent.run()
