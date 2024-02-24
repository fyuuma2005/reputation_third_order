# Who is a leader in the leading eight? --indirect reciprocity under private assessment--
# output Fig. 4B and Fig. A2B from data by "ReputationThird_Cooperation.py"


import numpy as np
import matplotlib.pyplot as plt


N = 800
e1 = 0
Li = 1 # L1-L8
xmin, xmax = 0.03, 0.4

ymax_vc = [5, 10, 5, 5, 10, 30, 5, 30]
ymax = ymax_vc[Li-1]
ARSN_vc = [[3,1,14],[3,3,14],[4,1,13],[4,2,13],[4,3,13],[4,4,13],[4,2,14],[4,4,14]] # from L1 ~ L8
[AR, SNC, SND] = ARSN_vc[Li-1]

# AR_vc: iD->(iR,iD) = (G,G), (G,B), (B,G), (B,B)
AR_vc = [1-int((AR-1)/8),1-int((AR-1)%8/4),1-int((AR-1)%4/2),1-int((AR-1)%2)]
AR_vc = (1-2*e1)*np.array(AR_vc)+e1

data = np.loadtxt('filenameLX.txt') # [:,0]=e2, [:,1]=pWW, [:,2]=sW, [:,3]=pMWALLC, [:,4]=pMWALLD
e2 = data[:45,0]
pWW, sW = data[:45,1], data[:45,2]
pMW1, pMW2 = data[:45,3], data[:45,4]

ALP1 = 0.8
ALP2 = 1.0
DRK = 1.0
L1color = (65./255*DRK,111./255*DRK,165./255*DRK,ALP1)
L1colorthin = (65./255,111./255,165./255,ALP2)
L2color = (167./255*DRK,66./255*DRK,63./255*DRK,ALP1)
L2colorthin = (167./255,66./255,63./255,ALP2)
L3color = (133./255*DRK,163./255*DRK,74./255*DRK,ALP1)
L3colorthin = (133./255,163./255,74./255,ALP2)
L4color = (110./255*DRK,84./255*DRK,140./255*DRK,ALP1)
L4colorthin = (110./255,84./255,140./255,ALP2)
L5color = (60./255*DRK,149./255*DRK,173./255*DRK,ALP1)
L5colorthin = (60./255,149./255,173./255,ALP2)
L6color = (217./255*DRK,128./255*DRK,56./255*DRK,ALP1)
L6colorthin = (217./255,128./255,56./255,ALP2)
L7color = (144./255*DRK,167./255*DRK,203./255*DRK,ALP1)
L7colorthin = (144./255,167./255,203./255,ALP2)
L8color = (205./255*DRK,141./255*DRK,140./255*DRK,ALP1)
L8colorthin = (205./255,141./255,140./255,ALP2)
MainColor_vc = [L1color, L2color, L3color, L4color, L5color, L6color, L7color, L8color]
ColorThin_vc = [L1colorthin, L2colorthin, L3colorthin, L4colorthin, L5colorthin, L6colorthin, L7colorthin, L8colorthin]
MainColor = MainColor_vc[Li-1]
ColorThin = ColorThin_vc[Li-1]
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
    plt.figure(figsize=(6,4))
    plt.xlabel('e2')
    where_fill = np.where(threshold_ALLC > threshold_ALLD)[0]
    if len(where_fill) == 0:
        plt.fill_between([e1,xmax], [1,1], [ymax,ymax], color=ColorBack)
    else:
        nfin = where_fill[-1]
        if nfin != len(e2)-1:
            x1, x2, yC1, yC2, yD1, yD2 = e2[nfin], e2[nfin+1], threshold_ALLC[nfin], threshold_ALLC[nfin+1], threshold_ALLD[nfin], threshold_ALLD[nfin+1]
            e2fin = (yC1-yD1)*(x2-x1)/(yD2-yD1-yC2+yC1)+x1
            thresholdfin = (yC2-yC1)/(x2-x1)*(e2fin-x1)+yC1
            e2_copy, threshold_ALLD_copy, threshold_ALLC_copy = np.append(e2[where_fill], e2fin), np.append(threshold_ALLD[where_fill], thresholdfin), np.append(threshold_ALLC[where_fill], thresholdfin)
            plt.fill_between(e2_copy, threshold_ALLD_copy, threshold_ALLC_copy, color=MainColor)
            plt.fill_between(e2_copy, np.ones(len(e2_copy)), threshold_ALLD_copy, color=ColorBack)
            plt.fill_between(e2_copy, threshold_ALLC_copy, np.ones(len(e2_copy))*ymax, color=ColorBack)
            plt.fill_between([e2_copy[-1],xmax], [1,1], [ymax,ymax], color=ColorBack)
        else:
            plt.fill_between(e2, threshold_ALLD, threshold_ALLC, color=MainColor)
            plt.fill_between(e2, np.ones(len(e2)), threshold_ALLD, color=ColorBack)
            plt.fill_between(e2, threshold_ALLC, np.ones(len(e2))*ymax, color=ColorBack)
    plt.xlim(0.03,xmax)
    plt.ylim(1.0,ymax)
    plt.xticks([xmin,0.1,0.2,0.3,0.4], fontsize=15)
    if Li == 1 or Li == 3 or Li == 4 or Li == 7:
        plt.yticks([1,2,3,4,5], fontsize=15)
    else:
        plt.yticks(fontsize=15)
    plt.title('[AR, SNC, SND] = ' + str([AR, SNC, SND]) + ', N = ' + str(N) + ', e1 = ' + str(e1), fontsize=10)
    plt.savefig("figurename.png", format="png", dpi=400)
    plt.show()



