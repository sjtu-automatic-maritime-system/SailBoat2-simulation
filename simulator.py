import numpy as np
from PID import PIDcontroller
from numpy import sin,cos,pi,degrees,radians,arctan2,sqrt



#邓乃铭
class SailBoatModel:
    def cal_accelerate(self):
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
        self.goal=np.array([[10,0],[50,50],[-40,-40]])
        self.k=0
        self.distance_thresh=5.0
        vpp_temp=np.load('vpp.npz')
        self.vpp=[vpp_temp['vpp3'],vpp_temp['vpp5'],vpp_temp['vpp7']]
    
    #廖柯、林若轩
    def plannar(self):
        vx=1
        vy=0
        awv=5
        awa=180
        yaw,r,x,y=self.state[6],self.state[2],self.state[4],self.state[5]

        def true_wind(awv, awa, vx, vy, yaw, r):
            airmar_x = 0.7
            airmar_vx = vx + sin(radians(yaw)) * r * airmar_x
            airmar_vy = vy + cos(radians(yaw)) * r * airmar_x
            true_wind_x = awv * cos(radians(awa + yaw + 180)) + airmar_vx
            true_wind_y = awv * sin(radians(awa + yaw + 180)) + airmar_vy
            twv = sqrt(pow(true_wind_x, 2) + pow(true_wind_y, 2))
            if true_wind_y < 0:
                twa = degrees(arctan2(true_wind_y, true_wind_x)) + 180
            else:
                twa = degrees(arctan2(true_wind_y, true_wind_x)) - 180
            return twv, twa

        def goal_derection(goal_x, goal_y, x, y, twa):  # 目标点坐标goal_x,goal_y，x,y为当前位置坐标
            goal_angle = degrees(arctan2((goal_y - y), (goal_x - x)))
            goal_wind_angle = goal_angle - twa  # 风来向右侧为正
            if goal_wind_angle < -180:
                goal_wind_angle = goal_wind_angle + 360
            if goal_wind_angle > 180:
                goal_wind_angle = goal_wind_angle - 360
            # print(goal_angle,goal_wind_angle,twa)
            return goal_wind_angle

        def vpp_selection(twv):  # vpp每个速度对应矩阵形式，每一行为对应角度的各个参数。所有vpp矩阵按速度大小放入vpp列表中,第一列为角度，第二列为速度
            d = []
            vpp_v = [3, 5, 7]  # 对应vpp的每个速度
            for i in range(len(vpp_v)):
                d.append(twv - vpp_v[i])
            vpp_positon = d.index(max(d))
            if max(d) < 0:
                vpp_goal = self.vpp[0] + d[0] * (self.vpp[1] - self.vpp[0])
            elif vpp_positon == len(vpp_v):
                vpp_goal = self.vpp[vpp_positon] + (twv - vpp_v[vpp_positon]) * (
                            self.vpp[vpp_positon] - self.vpp[vpp_positon - 1])
            else:
                vpp_goal = self.vpp[vpp_positon] + (twv - vpp_v[vpp_positon]) * (
                            self.vpp[vpp_positon + 1] - self.vpp[vpp_positon])
            return vpp_goal

        def v_selection(goal_wind_angle, vpp_goal):
            v_goalcompute = []
            for i in range(np.size(vpp_goal, 0)):
                vpp_angle_compute = vpp_goal[i, 0] - goal_wind_angle
                if vpp_angle_compute < -180:
                    vpp_angle_compute = vpp_angle_compute + 360
                if vpp_angle_compute > 180:
                    vpp_angle_compute = vpp_angle_compute - 360
                v_goalcompute.append(vpp_goal[i, 1] * cos(radians(vpp_goal[i, 0] - goal_wind_angle)))
            v_goal_positon = v_goalcompute.index(max(v_goalcompute))
            v_angle = vpp_goal[v_goal_positon, 0]
            v_goal = vpp_goal[v_goal_positon, 1]
            v_sail_angle = vpp_goal[v_goal_positon, 2]
            v_rudder_angle = vpp_goal[v_goal_positon, 3]
            v_heading = v_angle
            return v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading

        twv, twa = true_wind(awv, awa, vx, vy, yaw, r)
        goal_wind_angle = goal_derection(self.goal[self.k][0], self.goal[self.k][1], x, y, twa)
        vpp_goal = vpp_selection(twv)
        v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading = v_selection(goal_wind_angle, vpp_goal)
        #v_angle航向角 #v_goal稳定目标速度 #v_sail_angle稳定帆角 #v_rudder_angle稳定舵角
        return v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading
    
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
        while True:
            dpos=self.goal[self.k]-self.state[4:6]
            distance = sqrt(np.inner(dpos,dpos))
            if distance < self.distance_thresh:
                self.k = self.k + 1
            if self.k>=len(self.goal):
                break
            planner_result=self.plannar()



def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x


if __name__=="__main__":
    fr=FleetRace()
    v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading=fr.plannar()