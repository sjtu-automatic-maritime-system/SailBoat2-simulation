import numpy as np
from PID import PIDcontroller
from numpy import sin,cos,pi,arctan2,sqrt
import matplotlib.pyplot as plt
from SailBoatModel import SailBoatModel


class BaseSimulator:
    def __init__(self):
        self.dt=0.1
        #u,v,r,dheel,x,y,yaw,heel
        self.state=np.zeros(8)
        self.sailboat_model=SailBoatModel(init_posx=self.state[4], init_posy=self.state[5], init_yaw=self.state[6], init_heel=self.state[7],\
                init_u=self.state[0], init_v=self.state[1], init_r=self.state[2], init_dheel=self.state[3])
        self.sail_list=np.linspace(-pi,pi,360)
        self.twind=np.zeros(2)

    #邓乃铭
    def update_state(self,sail,rudder):
        self.sailboat_model.update(sail, rudder, self.twind[0], self.twind[1])
        self.state[0] = self.sailboat_model.u
        self.state[1] = self.sailboat_model.v
        self.state[2] = self.sailboat_model.r
        self.state[3] = self.sailboat_model.dheel
        self.state[4] = self.sailboat_model.posx
        self.state[5] = self.sailboat_model.posy
        self.state[6] = self.sailboat_model.yaw
        self.state[7] = self.sailboat_model.heel
    
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
    def planner(self):
        vx=1
        vy=0
        awv=5
        awa=pi/4
        yaw,r,x,y=self.state[6],self.state[2],self.state[4],self.state[5]

        def true_wind(awv, awa, vx, vy, yaw, r): #awv相对风速 awa相对风向角（-pi，pi）船首来风为0,船尾来风pi，右舷为正。北东地坐标系，vx帆船对地x方向速度，vy帆船对地y方向速度，yaw帆船首向角，r帆船转首角速度
            airmar_x = 0.7
            airmar_vx = vx + sin(yaw) * r * airmar_x
            airmar_vy = vy - cos(yaw) * r * airmar_x
            true_wind_x = awv * cos(awa + yaw + pi) + airmar_vx
            true_wind_y = awv * sin(awa + yaw + pi) + airmar_vy
            twv = sqrt(pow(true_wind_x, 2) + pow(true_wind_y, 2))
            if true_wind_y < 0:
                twa = arctan2(true_wind_y, true_wind_x) + pi
            else:
                twa = arctan2(true_wind_y, true_wind_x) - pi
            return twv, twa#twv绝对风速 twa绝对风向角（-pi，pi），x正向来风为0,x负向来风为pi，顺时针为正

        def goal_derection(goal_x, goal_y, x, y, twa):  # 目标点坐标goal_x,goal_y，x,y为当前位置坐标
            goal_angle = arctan2((goal_y - y), (goal_x - x))
            goal_wind_angle = yawRange(goal_angle - twa)  # 风来向右侧为正
            # print(goal_angle,goal_wind_angle,twa)
            return goal_wind_angle #goal_wind_angle 目标方向与绝对风向角夹角，绝对风向角顺时针方向为正


        def vpp_selection(twv):  # vpp每个速度对应矩阵形式，每一行为对应角度的各个参数。所有vpp矩阵按速度大小放入vpp列表中,第一列为角度，第二列为速度，第三列为帆角，第四列为舵角
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
            return vpp_goal#vpp_goal差值得到当前风速的vpp

        def v_selection(goal_wind_angle, vpp_goal):
            v_goalcompute=vpp_goal[:, 1] * cos( vpp_goal[:, 0] - goal_wind_angle)
            v_goal_positon = np.argmax(v_goalcompute)
            v_angle = yawRange(vpp_goal[v_goal_positon, 0]+twa)#对地坐标航向角
            v_goal = vpp_goal[v_goal_positon, 1]
            v_sail_angle = vpp_goal[v_goal_positon, 2]
            v_rudder_angle = vpp_goal[v_goal_positon, 3]
            v_heading = v_angle
            return v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading #v_angle 对地坐标系下，vmg算法计算航向角，v_goal该航向对应稳态速度。v_sail_angle该行下对应稳态帆角。v_rudder_angle该航向对应稳态舵角，v_heading该航向对应首向角

        twv, twa = true_wind(awv, awa, vx, vy, yaw, r)
        goal_wind_angle = goal_derection(self.goal[self.k][0], self.goal[self.k][1], x, y, twa)
        vpp_goal = vpp_selection(twv)
        v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading = v_selection(goal_wind_angle, vpp_goal)
        #v_angle航向角 #v_goal稳定目标速度 #v_sail_angle稳定帆角 #v_rudder_angle稳定舵角
        # print(np.tan(v_angle))
        return v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading
    
    #秦操
    def control(self,target,target_sail):
        e=target-self.state[6]
        aws,awa=self.sailboat_model.tw2aw(self.twind[0],self.twind[1])
        rudder=self.rudder_controller.feedback(e)
        windforce=np.array([self.sailboat_model.cal_windforce(self.state[0],self.state[1],self.state[2],self.state[7],self.state[3],aws,awa,sa,) for s in self.sail_list])
        if e>0:
            sail_list=self.sail_list[np.where(windforce[:,2]>0)[0]]
        else:
            sail_list = self.sail_list[np.where(windforce[:, 2] < 0)[0]]
        sail=sail_list(np.argmin(np.abs(sail_list-target_sail)))
        return sail,rudder

    
    #廖柯、林若轩
    def run(self,fig_plot=True):


        while True:
            dpos=self.goal[self.k]-self.state[4:6]
            distance = sqrt(np.inner(dpos,dpos))
            if distance < self.distance_thresh:
                self.k = self.k + 1
            if self.k>=len(self.goal):
                break
            v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading=self.planner()
            for i in range(5):
                sail,rudder=self.control(v_angle,v_sail_angle)
                for j in range(10):
                    self.update_state(sail,rudder)
                if fig_plot:
                    fig = plt.gca()
                    fig.set_xlabel('E/m')
                    fig.set_ylabel('N/m')
                    fig.plot(self.goal[:, 1], self.goal[:, 0], "or", markersize=5)
                    try:
                        for plot_line in plot_lines:
                            fig.lines.remove(plot_line[0])
                    except:
                        pass

                    plot_lines = []
                    plot_lines.append(fig.plot(self.state[5], self.state[4], "ob", markersize=5))
                    plt.pause(0.0001)




def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x


if __name__=="__main__":
    fr=FleetRace()
    v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading=fr.planner()