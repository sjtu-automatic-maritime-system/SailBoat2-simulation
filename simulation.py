import numpy as np
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
        self.goal=[[10,10],[50,50],[-40,-40]]
        self.k=0
        vpp_temp=np.load('vpp.npz')
        self.vpp=[vpp_temp['vpp3'],vpp_temp['vpp5'],vpp_temp['vpp7']]
        self.controller=PIDController()
    
    #廖柯、林若轩
    def plannar(awv,awa,vx,vy,yaw,r,x,y):
        goal_x=self.goal[self.k][0]
        goal_y=self.goal[self.k][1]
        def true_wind(awv,awa,vx,vy,yaw,r):
            airmar_x=0.7
            airmar_vx=vx+math.sin(math.radians(yaw))*r*airmar_x
            airmar_vy=vy+math.cos(math.radians(yaw))*r*airmar_x
            true_wind_x=awv*math.cos(math.radians(awa+yaw+180))+airmar_vx
            true_wind_y=awv*math.sin(math.radians(awa+yaw+180))+airmar_vy
            twv=math.sqrt(pow(true_wind_x,2)+pow(true_wind_y,2))
            if true_wind_y<0:
                twa=math.degrees(math.atan2(true_wind_y,true_wind_x))+180
            else:
                twa=math.degrees(math.atan2(true_wind_y,true_wind_x))-180
            return twv,twa
           

        def goal_derection(goal_x,goal_y,x,y,twa):#目标点坐标goal_x,goal_y，x,y为当前位置坐标
            goal_angle=math.degrees(math.atan2((goal_y-y),(goal_x-x)))
            goal_wind_angle=goal_angle-twa#风来向右侧为正
            if goal_wind_angle<-180:
                goal_wind_angle=goal_wind_angle+360
            if goal_wind_angle>180:
                goal_wind_angle=goal_wind_angle-360
            #print(goal_angle,goal_wind_angle,twa)
            return goal_wind_angle

        def vpp_selection(twv):#vpp每个速度对应矩阵形式，每一行为对应角度的各个参数。所有vpp矩阵按速度大小放入vpp列表中,第一列为角度，第二列为速度
            d=[]
            vpp_v=[3,5,7]  #对应vpp的每个速度
            for i in range(len(vpp_v)):
                d.append(twv-vpp_v[i])
            vpp_positon=d.index(max(d))
            if max(d)<0:
                vpp_goal=self.vpp[0]+d[0]*(self.vpp[1]-self.vpp[0])
            elif vpp_positon==len(vpp_v):
                vpp_goal=self.vpp[vpp_positon]+(twv-vpp_v[vpp_positon])*(self.vpp[vpp_positon]-self.vpp[vpp_positon-1])   
            else:
                vpp_goal=self.vpp[vpp_positon]+(twv-vpp_v[vpp_positon])*(self.vpp[vpp_positon+1]-self.vpp[vpp_positon])
            return vpp_goal

        def v_selection(goal_wind_angle,vpp_goal):
            v_goalcompute=[]
            for i in range(np.size(vpp_goal,0)):
                vpp_angle_compute=vpp_goal[i,0]-goal_wind_angle
                if vpp_angle_compute<-180:
                    vpp_angle_compute=vpp_angle_compute+360
                if vpp_angle_compute>180:
                    vpp_angle_compute=vpp_angle_compute-360
                v_goalcompute.append(vpp_goal[i,1]*math.cos(math.radians(vpp_goal[i,0]-goal_wind_angle)))
            v_goal_positon=v_goalcompute.index(max(v_goalcompute))
            v_angle=vpp_goal[v_goal_positon,0]
            v_goal=vpp_goal[v_goal_positon,1]
            v_sail_angle=vpp_goal[v_goal_positon,2]
            v_rudder_angle=vpp_goal[v_goal_positon,3]
            v_heading=v_angle
            return v_angle,v_goal,v_sail_angle,v_rudder_angle,v_heading
        twv,twa=true_wind(awv,awa,vx,vy,yaw,r)
        goal_wind_angle=goal_derection(self.goal[self.k][0],self.goal[self.k][1],x,y,twa)
        vpp_goal=vpp_selection(twv)
        v_angle,v_goal,v_sail_angle,v_rudder_angle,v_heading=v_selection(goal_wind_angle,vpp_goal)

        return v_angle,v_goal,v_sail_angle,v_rudder_angle,v_heading
    #秦操
    def control():
        pass
    
    #廖柯、林若轩
    def run(self,awv,awa,vx,vy,yaw,r,x,y,distance_thresh=5):
        distance=math.sqrt(pow(goal[self.k][0]-self.x,2)+pow(goal[self.k][1]-self.f,2))
        if distance<distance_thresh:
            self.k=self.k+1
        v_angle,v_goal,v_sail_angle,v_rudder_angle,v_heading=self.plannar(awv,awa,vx,vy,yaw,r,x,y)


#秦操
class PIDController:
pass
