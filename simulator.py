import numpy as np
from PID import PIDcontroller
from numpy import sin,cos,pi



#邓乃铭
class SailBoatModel:
    def cal_accelerate():
        pass

    def cal_windforce(self,state,wind,sail):
        pass

    def wind_to_absolute(self,wind):
        return wind
    
class BaseSimulator:
    def __init__(self):
        self.dt=0.1
        #u,v,r,dheel,x,y,yaw,heel
        self.state=np.zeros(8)
        self.sailboat_model=SailBoatModel()
        self.sail_list=np.linspace(-pi,pi,360)
        self.twind=np.zeros(2)

    
    #邓乃铭
    def update_state(self):
        accelerate=self.sailboat_model.cal_accelerate()
        pass
    
class FleetRace(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.rudder_controller=PIDcontroller(1,0,0,self.dt)
    
    #廖柯、林若轩
    def planner(self):
        pass
    
    #秦操
    def control(self,target):
        e=target-self.state[6]
        rudder=self.rudder_controller.feedback(e)
        windforce=np.array([self.sailboat_model.cal_windforce(self.state,self.twind,s) for s in self.sail_list])
        if e>0:
            windforce=windforce[np.where(windforce[:,2]>0)[0]]
        else:
            windforce = windforce[np.where(windforce[:, 2] < 0)[0]]
        sail=self.sail_list[np.argmax(windforce[:,0])]
        return sail,rudder

    
    #廖柯、林若轩
    def run(self):
        pass
