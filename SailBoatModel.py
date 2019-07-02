import math
import numpy as np 
import itertools

class SailBoatModel(object):
	def __init__(self):
		self.loa = 1.5
		self.lwl = 1.3115
		self.bwl = 0.3638
		self.b = 0.4763
		self.dc = 0.0688
		self.cb = 0.4326
		self.m = 15.3
		self.mx = self.m*0.01*(0.398+11.97*self.cb*(1+3.73*self.dc/self.b)-2.89*self.cb*self.loa/self.b*(1+1.13*self.dc/self.b)+0.175*self.cb*(self.loa/self.b)\
					*(self.loa/self.b)*(1+0.541*self.dc/self.b)-1.107*self.loa*self.dc/self.b*self.b)
		self.my = self.m*(0.882-0.54*self.cb*(1-1.6*self.dc/self.b)-0.156*self.loa/self.b*(1-0.673*self.cb)+0.826*self.dc*self.loa/self.b*self.b*(1-0.678*self.dc/self.b)\
					-0.638*self.cb*self.dc*self.loa/self.b*self.b*(1-0.669*self.dc/self.b))
		self.angle_NACA = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 
							85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

		self.cl_NACA = 	[0.0000, 0.0842, 0.1879, 0.2861, 0.3800, 0.4687, 0.5486, 0.6209, 0.6745, 0.7148, 0.7374, 0.7443, 0.7363, 0.7255, 0.6993, 0.6487, 0.6098, 0.5920,\
						0.6023, 0.6664, 0.8550, 0.9800, 1.0350, 1.0500, 1.0200, 0.9550, 0.8750, 0.7600, 0.6300, 0.5000, 0.3650, 0.2300, 0.0900, -0.0500, -0.1850, -0.3200,\
						-0.4500, -0.5750, -0.6700, -0.7600, -0.8500, -0.9300, -0.9800, -0.9000, -0.7700, -0.6700, -0.6350, -0.6800, -0.8500, -0.5600, 0.0000]

		self.cd_NACA = 	[0.0139, 0.0140, 0.0143, 0.0148, 0.0155, 0.0163, 0.0174, 0.0187, 0.0204, 0.0222, 0.0243, 0.0266, 0.0292, 0.0860, 0.1580, 0.1960, 0.2380, 0.2820, \
						0.3290, 0.4050, 0.5700, 0.7450, 0.9200, 1.0750, 1.2150, 1.3450, 1.4700, 1.5750, 1.6650, 1.7350, 1.7800, 1.8000, 1.8000, 1.7800, 1.7500, 1.7000, 1.6350,\
						1.5550, 1.4650, 1.3500, 1.2250, 1.0850, 0.9250, 0.7555, 0.5750, 0.4200, 0.3200, 0.2300, 0.1400, 0.0550, 0.0250]

		self.ad = 1.205
		self.sarea = 0.27
		self.x_s1 = 0.3911
		self.z_s1 = 0.7082
		self.x_s2 = -0.1089
		self.z_s2 = 0.7082
		self.wd = 1025
		self.wcv = 1.005e-3
		self.lk = 0.225
		self.lr = 0.0875
		self.kc = 0.09
		self.kk = 0.215
		self.kr = 0.252
		self.cm = 0.791
		self.wac = 0.4931
		self.wak = 0.0672
		self.war = 0.0234
		self.m_apo = self.m/0.5/self.wd/(self.loa*self.loa)/self.dc
		self.mx_apo = self.mx/0.5/self.wd/(self.loa*self.loa)/self.dc
		self.lamda = 2*self.dc/self.loa
		self.yv = -(math.pi*self.lamda/2+1.4*self.cb*self.b/self.loa)
		self.yr = self.m_apo+self.mx_apo-1.5*self.cb*self.b/self.loa
		self.yvv = -187
		self.yrr = 0.343*self.cb*self.dc/self.b-0.07
		self.yvrr = -5.95*(1-self.cb)*self.dc/self.b
		self.yvvr = 1.5*self.cb*self.dc/self.b-0.65
		self.nv = -self.lamda
		self.nr = -self.lamda*(0.54-self.lamda)
		self.nvv = 0.96*(1-self.cb)*self.dc/self.b-0.066
		self.nrr = 0.5*self.cb*self.b/self.loa-0.09
		self.nvrr = 0.5*self.cb*self.b/self.loa-0.05
		self.nvvr = -57.5*(self.cb*self.b/self.loa)*(self.cb*self.b/self.loa)+18.4*(self.cb*self.b/self.loa)-1.6
		self.buoyancy = 150
		self.gm = 0.1323
		self.Ndheel = -0.8
		self.hr = 0.2795
		self.br = 0.0875
		self.ar = self.hr*self.br
		self.x_r = -0.6069


	def HullModel(self, u, v, r, heel, dheel):
		acs = np.sqrt(u*u+v*v)
		aca = np.arctan2(v, u)
		L = np.array([self.lwl, self.lk, self.lr])
		Re = np.abs((self.wd*acs/self.wcv)*L)
		Cf = 0.075/(np.log(Re)-2)
		k = [self.kc+1, self.kk+1, self.kr+1]
		wa = [self.wac, self.wak, self.war]
		Rv = self.wd*acs*Cf*wa*k
		Fr = acs/np.sqrt(9.8*self.lwl)
		Rheel = 0.5*self.wd*acs*acs*self.wac*np.abs(heel)*(0.000891*Fr*self.bwl/self.dc+0.004267*heel*heel*self.bwl/self.dc-0.003142)

		yv = self.yv*0.5*self.wd*self.lwl*self.dc*acs*acs
		yr = self.yr*0.5*self.wd*self.lwl*self.dc*acs*acs
		yvv = self.yvv*0.5*self.wd*self.lwl*self.dc*acs*acs
		yrr = self.yrr*0.5*self.wd*self.lwl*self.dc*acs*acs
		yvrr = self.yvrr*0.5*self.wd*self.lwl*self.dc*acs*acs
		yvvr = self.yvvr*0.5*self.wd*self.lwl*self.dc*acs*acs
		Ykijima = yv*v+yr*r+yvv*np.abs(v)*v+yrr*np.abs(r)*r+yvrr*v*r*r+yvvr*v*v*r

		nv = self.nv*0.5*self.wd*self.lwl*self.dc*acs*acs
		nr = self.nr*0.5*self.wd*self.lwl*self.dc*acs*acs
		nvv = self.nvv*0.5*self.wd*self.lwl*self.dc*acs*acs
		nrr = self.nrr*0.5*self.wd*self.lwl*self.dc*acs*acs
		nvrr = self.nvrr*0.5*self.wd*self.lwl*self.dc*acs*acs
		nvvr = self.nvvr*0.5*self.wd*self.lwl*self.dc*acs*acs
		Nkijima = nv*v+nr*r+nvv*np.abs(v)*v+nrr*np.abs(r)*r+nvrr*v*r*r+nvvr*v*v*r

		Mheel = -self.buoyancy*self.gm*np.sin(heel)
		Mdheel = self.Ndheel*dheel

		XH = -np.cos(aca)*np.sum(Rv)-np.cos(aca)*Rheel
		YH = -np.sin(aca)*np.sum(Rv)-np.sin(aca)*Rheel
		NH = Nkijima
		MH = Mheel+Mdheel

		return XH, YH, NH, MH

	def SailModel(self, aws, awa, sa):
		angle = awa-sa
		if angle >= math.pi:
			angle = angle-2*math.pi
		if angle < -math.pi:
			angle = angle+2*math.pi

		cl = np.interp(np.abs(angle)*180/math.pi, self.angle_NACA, self.cl_NACA)
		cd = np.interp(np.abs(angle)*180/math.pi, self.angle_NACA, self.cd_NACA)

		L = cl*0.5*self.ad*aws*aws*self.sarea
		D = cd*0.5*self.ad*aws*aws*self.sarea

		if angle >= 0:
		    X = -D*np.cos(angle)+L*np.sin(angle)
		    Y = -D*np.sin(angle)-L*np.cos(angle)
		if angle < 0:
		    X = -D*np.cos(angle)-L*np.sin(angle)
		    Y = -D*np.sin(angle)+L*np.cos(angle)

		XS = 2*(X*np.cos(sa)+Y*np.cos(sa+math.pi/2))
		YS = 2*(X*np.sin(sa)+Y*np.sin(sa+math.pi/2))
		NS = YS/2*(self.x_s1+self.x_s2)
		MS = YS/2*(self.z_s1+self.z_s2)

		return XS, YS, NS, MS

	def RudderModel(self, u, v, ra):
		bs = np.sqrt(u*u+v*v)
		fn = 0.5*self.wd*self.ar*6.13*self.hr/self.br/(2.25+self.hr/self.br)*bs*bs
		XR = (0.28*self.cb+0.55)*fn*np.sin(ra)*np.sin(ra)
		YR = -(0.627*self.cb+1.153)*fn*np.sin(ra)*np.cos(ra)
		NR = YR*self.x_r

		return XR, YR, NR

	def cal_force(self, u, v, r, heel, dheel, aws, awa, sa, ra):
		XH, YH, NH, MH = self.HullModel(u, v, r, heel, dheel)
		XS, YS, NS, MS = self.SailModel(aws, awa, sa)
		XR, YR, NR = self.RudderModel(u, v, ra)
		X = XH+XS+XR+(self.m+self.mx)*r*v
		Y = YH+YS+YR-(self.m+self.my)*r*u
		N = NH+NS+NR
		M = MS+MH
		return X, Y, N, M

def test_model():
	ship = SailBoatModel()
	# Unit: m/s
	current_u = 1
	current_v = 0.1
	# Unit: rad/s
	current_r = 0.2
	# Unit: rad
	current_heel = math.pi/6
	# Unit: rad/s
	current_dheel = 0.1

	# Unit: m/s
	current_aws = 5
	# Unit: rad
	current_awa = 0

	# Unit: rad
	current_sa = math.pi/4
	current_ra = math.pi/10

	X, Y, N, M = ship.cal_force(current_u, current_v, current_r, current_heel, current_dheel, current_aws, current_awa, current_sa, current_ra)
	print('Force --x direction:', X, 'N')
	print('Force --y direction:', Y, 'N')
	print('Moment --z direction:', N, 'Nm')
	print('Moment --x direction:', M, 'Nm')

# 计算扭计，@邓乃铭 完成
def cal_torque(sail,rudder,*args):
    torque=sail+rudder
    return torque


def get_sail_rudder(sail_list,rudder_list,target_torque,target_sail,target_rudder,topk=20):
    sail_rudder_set=np.array(list(itertools.product(sail_list,rudder_list)))  # 帆舵组合
    torque_set=np.array([cal_torque(s[0],s[1]) for s in sail_rudder_set])    # 组合对应的扭矩
    topk_sail_rudder=sail_rudder_set[np.argpartition(np.abs(torque_set-target_torque),topk)[:topk]]  # 与目标扭矩最接近的topk个帆舵组合
    sail_diff_abs=np.abs(topk_sail_rudder[:,0]-target_sail)   # 帆角与目标帆角的差的绝对值
    rudder_diff_abs=np.abs(topk_sail_rudder[:,1]-target_rudder)  # 舵角与目标帆角的差的绝对值
    final_sail_rudder=topk_sail_rudder[np.argmin(sail_diff_abs+1.0*rudder_diff_abs)]  # 最终的帆舵组合
    return final_sail_rudder


def main():
    sail_list=np.linspace(-100,100,50)
    rudder_list=np.linspace(-100,100,50)
    target_torque = 50
    target_sail=50
    target_rudder=-50

    final_sail_rudder=get_sail_rudder(sail_list,rudder_list,target_torque,target_sail,target_rudder)
    print(final_sail_rudder)

if __name__=="__main__":
    # main()
    test_model()

