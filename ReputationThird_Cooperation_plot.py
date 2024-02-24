# Who is a leader in the leading eight? --indirect reciprocity under private assessment--
# output Fig. 4A and Fig. A2A from data by "ReputationThird_Cooperation.py" and "ReputationThird_IndividualBased.py"


import numpy as np
import matplotlib.pyplot as plt


N = 800
e1 = 0.03
Li = 1
xmax = 0.5
experimental_plot = 1 # 1 for marker plot


ymax_vc = [5, 10, 5, 5, 10, 30, 5, 30]
ymax = ymax_vc[Li-1]
ARSN_vc = [[3,1,14],[3,3,14],[4,1,13],[4,2,13],[4,3,13],[4,4,13],[4,2,14],[4,4,14]] # from L1 ~ L8
[AR, SNC, SND] = ARSN_vc[Li-1]

# AR_vc: iD->(iR,iD) = (G,G), (G,B), (B,G), (B,B)
AR_vc = [1-int((AR-1)/8),1-int((AR-1)%8/4),1-int((AR-1)%4/2),1-int((AR-1)%2)]
AR_vc = (1-2*e1)*np.array(AR_vc)+e1

dataT = np.loadtxt('filenameLX.txt') # [:,0]=e2, [:,1]=pWW, [:,2]=sW, [:,3]=pMWALLC, [:,4]=pMWALLD
e2 = dataT[:,0]
pWW, sW = dataT[:,1], dataT[:,2]
pMW1, pMW2 = dataT[:,3], dataT[:,4]
CWW_T = np.dot(np.array([pWW*sW,pWW*(1-sW),(1-pWW)*sW,(1-pWW)*(1-sW)]).T,AR_vc)
CMW1_T = np.dot(np.array([pMW1*sW,pMW1*(1-sW),(1-pMW1)*sW,(1-pMW1)*(1-sW)]).T,AR_vc)
CMW2_T = np.dot(np.array([pMW2*sW,pMW2*(1-sW),(1-pMW2)*sW,(1-pMW2)*(1-sW)]).T,AR_vc)
if experimental_plot == 1:
    data01ALLC = np.loadtxt('filenameALLC_1.txt') # [:,0]=pWW, [:,1]=pMWALLC, [:,2]=sW
    pWW_01C, pMW_01C, sW_01C = data01ALLC[Li-1,0], data01ALLC[Li-1,1], data01ALLC[Li-1,2]
    CWW_01C = np.dot(np.array([pWW_01C*sW_01C,pWW_01C*(1-sW_01C),(1-pWW_01C)*sW_01C,(1-pWW_01C)*(1-sW_01C)]).T,AR_vc)
    CMW_01C = np.dot(np.array([pMW_01C*sW_01C,pMW_01C*(1-sW_01C),(1-pMW_01C)*sW_01C,(1-pMW_01C)*(1-sW_01C)]).T,AR_vc)
    data01ALLD = np.loadtxt('filenameALLD_1.txt') # [:,0]=pWW, [:,1]=pMWALLD, [:,2]=sW
    pWW_01D, pMW_01D, sW_01D = data01ALLD[Li-1,0], data01ALLD[Li-1,1], data01ALLD[Li-1,2]
    CWW_01D = np.dot(np.array([pWW_01D*sW_01D,pWW_01D*(1-sW_01D),(1-pWW_01D)*sW_01D,(1-pWW_01D)*(1-sW_01D)]).T,AR_vc)
    CMW_01D = np.dot(np.array([pMW_01D*sW_01D,pMW_01D*(1-sW_01D),(1-pMW_01D)*sW_01D,(1-pMW_01D)*(1-sW_01D)]).T,AR_vc)
    data02ALLC = np.loadtxt('filenameALLC_2.txt') # [:,0]=pWW, [:,1]=pMWALLC, [:,2]=sW
    pWW_02C, pMW_02C, sW_02C = data02ALLC[Li-1,0], data02ALLC[Li-1,1], data02ALLC[Li-1,2]
    CWW_02C = np.dot(np.array([pWW_02C*sW_02C,pWW_02C*(1-sW_02C),(1-pWW_02C)*sW_02C,(1-pWW_02C)*(1-sW_02C)]).T,AR_vc)
    CMW_02C = np.dot(np.array([pMW_02C*sW_02C,pMW_02C*(1-sW_02C),(1-pMW_02C)*sW_02C,(1-pMW_02C)*(1-sW_02C)]).T,AR_vc)
    data02ALLD = np.loadtxt('filenameALLD_2.txt') # [:,0]=pWW, [:,1]=pMWALLD, [:,2]=sW
    pWW_02D, pMW_02D, sW_02D = data02ALLD[Li-1,0], data02ALLD[Li-1,1], data02ALLD[Li-1,2]
    CWW_02D = np.dot(np.array([pWW_02D*sW_02D,pWW_02D*(1-sW_02D),(1-pWW_02D)*sW_02D,(1-pWW_02D)*(1-sW_02D)]).T,AR_vc)
    CMW_02D = np.dot(np.array([pMW_02D*sW_02D,pMW_02D*(1-sW_02D),(1-pMW_02D)*sW_02D,(1-pMW_02D)*(1-sW_02D)]).T,AR_vc)

ALP1 = 0.8
DRK = 1.0
L1color = (65./255*DRK,111./255*DRK,165./255*DRK,ALP1)
L2color = (167./255*DRK,66./255*DRK,63./255*DRK,ALP1)
L3color = (133./255*DRK,163./255*DRK,74./255*DRK,ALP1)
L4color = (110./255*DRK,84./255*DRK,140./255*DRK,ALP1)
L5color = (60./255*DRK,149./255*DRK,173./255*DRK,ALP1)
L6color = (217./255*DRK,128./255*DRK,56./255*DRK,ALP1)
L7color = (144./255*DRK,167./255*DRK,203./255*DRK,ALP1)
L8color = (205./255*DRK,141./255*DRK,140./255*DRK,ALP1)
Licolor = [L1color, L2color, L3color, L4color, L5color, L6color, L7color, L8color]
MainColor = Licolor[Li-1]
ColorBack = 'whitesmoke'

CWWave = np.dot(np.array([pWW*sW,pWW*(1-sW),(1-pWW)*sW,(1-pWW)*(1-sW)]).T,AR_vc)
CMW1ave = np.dot(np.array([pMW1*sW,pMW1*(1-sW),(1-pMW1)*sW,(1-pMW1)*(1-sW)]).T,AR_vc)
CMW2ave = np.dot(np.array([pMW2*sW,pMW2*(1-sW),(1-pMW2)*sW,(1-pMW2)*(1-sW)]).T,AR_vc)
CWM1ave, CMM1ave = 1-e1, 1-e1
CWM2ave, CMM2ave = e1, e1
threshold_ALLC = (CWWave-CWM1ave)/(CWWave-CMW1ave+10**-12)
threshold_ALLD = (CWWave-CWM2ave)/(CWWave-CMW2ave+10**-12)

det = 1
if det == 1:
    plt.figure(figsize=(4,3))
    plt.xlabel('e2')
    plt.plot(e2, CWW_T, color=MainColor, lw=3, ls='-')
    plt.plot(e2, CMW1_T, color=MainColor, lw=3, ls='--')
    plt.plot(e2, CMW2_T, color=MainColor, lw=3, ls=':')
    if experimental_plot == 1:
        plt.scatter([0.1,0.2], [CWW_01C,CWW_02C], marker='x', s=480, color='k', alpha=.7, zorder=5, linewidth=3)
        plt.scatter([0.1,0.2], [CMW_01C,CMW_02C], marker='^', s=480, color='k', alpha=.7, zorder=5)
        plt.scatter([0.1,0.2], [CWW_01D,CWW_02D], marker='+', s=640, color='k', alpha=.7, zorder=5, linewidth=3)
        plt.scatter([0.1,0.2], [CMW_01D,CMW_02D], marker='v', s=480, color='k', alpha=.7, zorder=5)
    plt.xlim(0.0,xmax)
    plt.ylim(0.0,1.0)
    plt.xticks([0.0,0.1,0.2,0.3,0.4,0.5], fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('[AR, SNC, SND] = ' + str([AR, SNC, SND]) + ', N = ' + str(N) + ', e1 = ' + str(e1), fontsize=10)
    plt.savefig('figurename.png', format='png', dpi=400)
    plt.show()



