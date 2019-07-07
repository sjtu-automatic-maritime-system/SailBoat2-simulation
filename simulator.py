import numpy as np
from PID import PIDcontroller
from numpy import sin,cos,pi,arctan2,sqrt


class SailBoatModel(object):
    def __init__(self, init_posx=0, init_posy=0, init_yaw=0, init_heel=0,\
                init_u=0, init_v=0, init_r=0, init_dheel=0, dt=0.1):
        self.posx = init_posx
        self.posy = init_posy
        self.yaw = init_yaw
        self.heel = init_heel
        self.u = init_u
        self.v = init_v
        self.r = init_r
        self.dheel = init_dheel
        self.velocity_x = self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)
        self.velocity_y = self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)
        self.dt = dt

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
        
        self.Izz = 0.8*self.m*(0.4*self.b)*(0.4*self.b)
        self.Jzz = 0.2*self.m*(0.4*self.b)*(0.4*self.b)
        self.Ixx = 0.8*self.m*(0.4*self.b)*(0.4*self.b)
        self.Jxx = 0.2*self.m*(0.4*self.b)*(0.4*self.b)

        self.angle_NACA = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 
                            85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

        self.cl_NACA =  [0.0000, 0.0842, 0.1879, 0.2861, 0.3800, 0.4687, 0.5486, 0.6209, 0.6745, 0.7148, 0.7374, 0.7443, 0.7363, 0.7255, 0.6993, 0.6487, 0.6098, 0.5920,\
                        0.6023, 0.6664, 0.8550, 0.9800, 1.0350, 1.0500, 1.0200, 0.9550, 0.8750, 0.7600, 0.6300, 0.5000, 0.3650, 0.2300, 0.0900, -0.0500, -0.1850, -0.3200,\
                        -0.4500, -0.5750, -0.6700, -0.7600, -0.8500, -0.9300, -0.9800, -0.9000, -0.7700, -0.6700, -0.6350, -0.6800, -0.8500, -0.5600, 0.0000]

        self.cd_NACA =  [0.0139, 0.0140, 0.0143, 0.0148, 0.0155, 0.0163, 0.0174, 0.0187, 0.0204, 0.0222, 0.0243, 0.0266, 0.0292, 0.0860, 0.1580, 0.1960, 0.2380, 0.2820, \
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
        self.yv = -(np.pi*self.lamda/2+1.4*self.cb*self.b/self.loa)
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

    def angle_limit(self, angle):
        if angle < -np.pi:
            angle += 2*np.pi
        if angle >= np.pi:
            angle -= 2*np.pi
        return angle

    def tw2aw(self, tws, twa):
        velocity_x = self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)
        velocity_y = self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)
        twx = tws*np.cos(twa)
        twy = tws*np.sin(twa)
        awx = twx+velocity_x
        awy = twy+velocity_y
        aws = np.sqrt(awx*awx+awy*awy)
        awa = math.atan2(awy, awx)
        return aws, awa

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
        if angle >= np.pi:
            angle = angle-2*np.pi
        if angle < -np.pi:
            angle = angle+2*np.pi

        cl = np.interp(np.abs(angle)*180/np.pi, self.angle_NACA, self.cl_NACA)
        cd = np.interp(np.abs(angle)*180/np.pi, self.angle_NACA, self.cd_NACA)

        L = cl*0.5*self.ad*aws*aws*self.sarea
        D = cd*0.5*self.ad*aws*aws*self.sarea

        if angle >= 0:
            X = -D*np.cos(angle)+L*np.sin(angle)
            Y = -D*np.sin(angle)-L*np.cos(angle)
        if angle < 0:
            X = -D*np.cos(angle)-L*np.sin(angle)
            Y = -D*np.sin(angle)+L*np.cos(angle)

        XS = 2*(X*np.cos(sa)+Y*np.cos(sa+np.pi/2))
        YS = 2*(X*np.sin(sa)+Y*np.sin(sa+np.pi/2))
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

    def state_equation(self, sa, ra, tws, twa, mean_du=0, mean_dv=0, mean_dr=0, mean_ddheel=0,\
                    std_du=5.0e-2, std_dv=5.0e-2, std_dr=1.0e-2, std_ddheel=1.0e-2):


        aws, awa = self.tw2aw(tws, twa)
        X, Y, N, M = self.cal_force(self.u, self.v, self.r, self.heel, self.dheel, aws, awa, sa, ra)
        du = X/(self.m+self.mx)+mean_du+std_du*np.random.normal()
        dv = Y/(self.m+self.my)+mean_dv+std_dv*np.random.normal()
        dr = N/(self.Izz+self.Jzz)+mean_dr+std_dr*np.random.normal()
        ddheel = M/(self.Ixx+self.Jxx)+mean_ddheel+std_ddheel*np.random.normal()

        u = self.u + du*self.dt
        v = self.v + dv*self.dt
        r = self.r + dr*self.dt
        dheel = self.dheel + ddheel*self.dt

        yaw = self.angle_limit(self.yaw + (self.r+r)*self.dt/2)

        self.posx = self.posx+(self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)+(u*np.cos(yaw)-v*np.sin(yaw)))*self.dt/2
        self.posy = self.posy+(self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)+(u*np.sin(yaw)+v*np.cos(yaw)))*self.dt/2
        self.yaw = yaw
        self.heel = self.angle_limit(self.heel + (self.dheel+dheel)*self.dt/2)

        self.u = u
        self.v = v
        self.r = r
        self.dheel = dheel

        self.velocity_x = self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)
        self.velocity_y = self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)

    def update(self, sa, ra, tws, twa, interval=5):
        for i in range(interval):
            self.state_equation(sa=sa, ra=ra, tws=tws, twa=twa)

#邓乃铭
class SailBoatModel(object):
    def __init__(self, init_posx=0, init_posy=0, init_yaw=0, init_heel=0,\
                init_u=0, init_v=0, init_r=0, init_dheel=0, dt=0.1):
        self.posx = init_posx
        self.posy = init_posy
        self.yaw = init_yaw
        self.heel = init_heel
        self.u = init_u
        self.v = init_v
        self.r = init_r
        self.dheel = init_dheel
        self.velocity_x = self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)
        self.velocity_y = self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)
        self.dt = dt

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
        
        self.Izz = 0.8*self.m*(0.4*self.b)*(0.4*self.b)
        self.Jzz = 0.2*self.m*(0.4*self.b)*(0.4*self.b)
        self.Ixx = 0.8*self.m*(0.4*self.b)*(0.4*self.b)
        self.Jxx = 0.2*self.m*(0.4*self.b)*(0.4*self.b)

        self.angle_NACA = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 
                            85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180]

        self.cl_NACA =  [0.0000, 0.0842, 0.1879, 0.2861, 0.3800, 0.4687, 0.5486, 0.6209, 0.6745, 0.7148, 0.7374, 0.7443, 0.7363, 0.7255, 0.6993, 0.6487, 0.6098, 0.5920,\
                        0.6023, 0.6664, 0.8550, 0.9800, 1.0350, 1.0500, 1.0200, 0.9550, 0.8750, 0.7600, 0.6300, 0.5000, 0.3650, 0.2300, 0.0900, -0.0500, -0.1850, -0.3200,\
                        -0.4500, -0.5750, -0.6700, -0.7600, -0.8500, -0.9300, -0.9800, -0.9000, -0.7700, -0.6700, -0.6350, -0.6800, -0.8500, -0.5600, 0.0000]

        self.cd_NACA =  [0.0139, 0.0140, 0.0143, 0.0148, 0.0155, 0.0163, 0.0174, 0.0187, 0.0204, 0.0222, 0.0243, 0.0266, 0.0292, 0.0860, 0.1580, 0.1960, 0.2380, 0.2820, \
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
        self.yv = -(np.pi*self.lamda/2+1.4*self.cb*self.b/self.loa)
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

    def angle_limit(self, angle):
        if angle < -np.pi:
            angle += 2*np.pi
        if angle >= np.pi:
            angle -= 2*np.pi
        return angle

    def tw2aw(self, tws, twa):
        velocity_x = self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)
        velocity_y = self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)
        twx = tws*np.cos(twa)
        twy = tws*np.sin(twa)
        awx = twx+velocity_x
        awy = twy+velocity_y
        aws = np.sqrt(awx*awx+awy*awy)
        awa = math.atan2(awy, awx)
        return aws, awa

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
        if angle >= np.pi:
            angle = angle-2*np.pi
        if angle < -np.pi:
            angle = angle+2*np.pi

        cl = np.interp(np.abs(angle)*180/np.pi, self.angle_NACA, self.cl_NACA)
        cd = np.interp(np.abs(angle)*180/np.pi, self.angle_NACA, self.cd_NACA)

        L = cl*0.5*self.ad*aws*aws*self.sarea
        D = cd*0.5*self.ad*aws*aws*self.sarea

        if angle >= 0:
            X = -D*np.cos(angle)+L*np.sin(angle)
            Y = -D*np.sin(angle)-L*np.cos(angle)
        if angle < 0:
            X = -D*np.cos(angle)-L*np.sin(angle)
            Y = -D*np.sin(angle)+L*np.cos(angle)

        XS = 2*(X*np.cos(sa)+Y*np.cos(sa+np.pi/2))
        YS = 2*(X*np.sin(sa)+Y*np.sin(sa+np.pi/2))
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

    def cal_windforce(self, tws, twa, sa):

    	aws, awa = self.tw2aw(tws, twa)

        XS, YS, NS, MS = self.SailModel(aws, awa, sa)

        X = XS*np.cos(self.yaw)-YS*np.sin(self.yaw)
        Y = XS*np.sin(self.yaw)+YS*np.cos(self.yaw)
        N = NS
        M = MS
        return X, Y, N, M

    def state_equation(self, sa, ra, tws, twa, mean_du=0, mean_dv=0, mean_dr=0, mean_ddheel=0,\
                    std_du=5.0e-2, std_dv=5.0e-2, std_dr=1.0e-2, std_ddheel=1.0e-2):


        aws, awa = self.tw2aw(tws, twa)
        X, Y, N, M = self.cal_force(self.u, self.v, self.r, self.heel, self.dheel, aws, awa, sa, ra)
        du = X/(self.m+self.mx)+mean_du+std_du*np.random.normal()
        dv = Y/(self.m+self.my)+mean_dv+std_dv*np.random.normal()
        dr = N/(self.Izz+self.Jzz)+mean_dr+std_dr*np.random.normal()
        ddheel = M/(self.Ixx+self.Jxx)+mean_ddheel+std_ddheel*np.random.normal()

        u = self.u + du*self.dt
        v = self.v + dv*self.dt
        r = self.r + dr*self.dt
        dheel = self.dheel + ddheel*self.dt

        yaw = self.angle_limit(self.yaw + (self.r+r)*self.dt/2)

        self.posx = self.posx+(self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)+(u*np.cos(yaw)-v*np.sin(yaw)))*self.dt/2
        self.posy = self.posy+(self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)+(u*np.sin(yaw)+v*np.cos(yaw)))*self.dt/2
        self.yaw = yaw
        self.heel = self.angle_limit(self.heel + (self.dheel+dheel)*self.dt/2)

        self.u = u
        self.v = v
        self.r = r
        self.dheel = dheel

        self.velocity_x = self.u*np.cos(self.yaw)-self.v*np.sin(self.yaw)
        self.velocity_y = self.u*np.sin(self.yaw)+self.v*np.cos(self.yaw)

    def update(self, sa, ra, tws, twa, interval=5):
        for i in range(interval):
            self.state_equation(sa=sa, ra=ra, tws=tws, twa=twa)

    
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
        awv,awa=self.sailboat_model.tw2aw(self.twind[0],self.twind[1])
        yaw,u,v,r,x,y=self.state[6],self.state[0],self.state[1],self.state[2],self.state[4],self.state[5]
        vx=u*cos(yaw)-v*sin(yaw)
        vy=u*sin(yaw)+v*cos(yaw)

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
        rudder=self.rudder_controller.feedback(e)
        windforce=np.array([self.sailboat_model.cal_windforce(self.twind,s) for s in self.sail_list])
        if e>0:
            sail_list=self.sail_list[np.where(windforce[:,2]>0)[0]]
        else:
            sail_list = self.sail_list[np.where(windforce[:, 2] < 0)[0]]
        sail=sail_list(np.argmin(np.abs(sail_list-target_sail)))
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
            v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading=self.planner()
            for i in range(5):
                sail,rudder=self.control(v_angle,v_sail_angle)
                for j in range(10):
                    self.update_state(sail,rudder)




def yawRange(x):
    if x > pi:
        x = x - 2 * pi
    elif x < -pi:
        x = x + 2 * pi
    return x


if __name__=="__main__":
    fr=FleetRace()
    v_angle, v_goal, v_sail_angle, v_rudder_angle, v_heading=fr.planner()