#邓乃铭
class SailBoatModel:
    def cal_accelerate():
        pass
    
class BaseSimulator:
    def __init__(self):
        self.state=np.zeros(6)
        self.sailboat_model=SailBoatModel()
    
    #邓乃铭
    def update_state():
        accelerate=self.sailboat_model.cal_accelerate()
        pass
    
class FleetRace(BaseSimulator):
    def __init__(self):
        super().__init__()
        self.controller=PIDController()
    
    #廖柯、林若轩
    def planner():
        pass
    
    #秦操
    def control():
        pass
    
    #廖柯、林若轩
    def run():
        pass

#秦操
class PIDController:
    pass