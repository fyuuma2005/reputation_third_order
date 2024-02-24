# Who is a leader in the leading eight? --indirect reciprocity under private assessment--
# output Fig. 3 and Fig. A1


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation


Tmin, Tmax = 50, 2050
N, Nbins = 800, 800 # Nbins = N is default
e1, e2 = 0.03, 0.1
Li = 1 # L1-L8

ARSN_vc = [[3,1,14],[3,3,14],[4,1,13],[4,2,13],[4,3,13],[4,4,13],[4,2,14],[4,4,14]] # from L1 ~ L8
[AR, SNC, SND] = ARSN_vc[Li-1]
#AR, SNC, SND = 3, 1, 14 # 1 ~ 16 for all ARs and SNs
image_mt = np.ones((N,N)) # (+1,-1) is (G,B)
RNchoice_vc = np.random.randint(N, size=(N*Tmax+1,2))
RNaction_vc = np.random.choice([1,-1],N*Tmax+1,p=[1-e1,e1])


# AR_vc: iD->(iR,iD) = (+1,+1), (+1,-1), (-1,+1), (-1,-1)
AR_vc = [1-int((AR-1)/8),1-int((AR-1)%8/4),1-int((AR-1)%4/2),1-int((AR-1)%2)]
AR_vc = np.array(AR_vc)*2-1

# definition: CMAP
if SNC == 1: # CCCC
    def CMAP(IR_vc, ID_vc, vcL):
        return(np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 2: # CCCD
    def CMAP(IR_vc, ID_vc, vcL):
        return(-((IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 3: # CCDC
    def CMAP(IR_vc, ID_vc, vcL):
        return(-((IR_vc-1)*(-ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 4: # CCDD
    def CMAP(IR_vc, ID_vc, vcL):
        return(IR_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 5: # CDCC
    def CMAP(IR_vc, ID_vc, vcL):
        return(-((-IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 6: # CDCD
    def CMAP(IR_vc, ID_vc, vcL):
        return(ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 7: # CDDC
    def CMAP(IR_vc, ID_vc, vcL):
        return(IR_vc*ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 8: # CDDD
    def CMAP(IR_vc, ID_vc, vcL):
        return(((IR_vc+1)*(ID_vc+1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 9: # DCCC
    def CMAP(IR_vc, ID_vc, vcL):
        return(-((IR_vc+1)*(ID_vc+1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 10: # DCCD
    def CMAP(IR_vc, ID_vc, vcL):
        return(-IR_vc*ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 11: # DCDC
    def CMAP(IR_vc, ID_vc, vcL):
        return(-ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 12: # DCDD
    def CMAP(IR_vc, ID_vc, vcL):
        return(((-IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 13: # DDCC
    def CMAP(IR_vc, ID_vc, vcL):
        return(-IR_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 14: # DDCD
    def CMAP(IR_vc, ID_vc, vcL):
        return(((IR_vc-1)*(-ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SNC == 15: # DDDC
    def CMAP(IR_vc, ID_vc, vcL):
        return(((IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
else: # DDDD
    def CMAP(IR_vc, ID_vc, vcL):
        return(-np.random.choice([1,-1], vcL, p=[1-e2,e2]))

# definition: DMAP
if SND == 1: # CCCC
    def DMAP(IR_vc, ID_vc, vcL):
        return(np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 2: # CCCD
    def DMAP(IR_vc, ID_vc, vcL):
        return(-((IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 3: # CCDC
    def DMAP(IR_vc, ID_vc, vcL):
        return(-((IR_vc-1)*(-ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 4: # CCDD
    def DMAP(IR_vc, ID_vc, vcL):
        return(IR_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 5: # CDCC
    def DMAP(IR_vc, ID_vc, vcL):
        return(-((-IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 6: # CDCD
    def DMAP(IR_vc, ID_vc, vcL):
        return(ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 7: # CDDC
    def DMAP(IR_vc, ID_vc, vcL):
        return(IR_vc*ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 8: # CDDD
    def DMAP(IR_vc, ID_vc, vcL):
        return(((IR_vc+1)*(ID_vc+1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 9: # DCCC
    def DMAP(IR_vc, ID_vc, vcL):
        return(-((IR_vc+1)*(ID_vc+1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 10: # DCCD
    def DMAP(IR_vc, ID_vc, vcL):
        return(-IR_vc*ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 11: # DCDC
    def DMAP(IR_vc, ID_vc, vcL):
        return(-ID_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 12: # DCDD
    def DMAP(IR_vc, ID_vc, vcL):
        return(((-IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 13: # DDCC
    def DMAP(IR_vc, ID_vc, vcL):
        return(-IR_vc*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 14: # DDCD
    def DMAP(IR_vc, ID_vc, vcL):
        return(((IR_vc-1)*(-ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
elif SND == 15: # DDDC
    def DMAP(IR_vc, ID_vc, vcL):
        return(((IR_vc-1)*(ID_vc-1)-2)/2*np.random.choice([1,-1], vcL, p=[1-e2,e2]))
else: # DDDD
    def DMAP(IR_vc, ID_vc, vcL):
        return(-np.random.choice([1,-1], vcL, p=[1-e2,e2]))

PhiG_vc, PhiB_vc = np.zeros(Nbins), np.zeros(Nbins)
for T in range(0,N*Tmax+1):
    [iR,iD] = RNchoice_vc[T]
    nAR = int((1-image_mt[iD,iR])+(1-image_mt[iD,iD])/2)
    if AR_vc[nAR]*RNaction_vc[T] == 1: # CMAP
        image_mt[:,iD] = CMAP(image_mt[:,iR], image_mt[:,iD], N)
    else: # DMAP
        image_mt[:,iD] = DMAP(image_mt[:,iR], image_mt[:,iD], N)
    if T%N == 0:
        if T/N > Tmin:
            where_selfG = np.where(np.diag(image_mt)==+1)[0]
            where_selfB = np.where(np.diag(image_mt)==-1)[0]
            goodG_vc = ((np.sum(image_mt[:,where_selfG], axis=0)+N)/2-1)/(N-1)
            goodB_vc = (np.sum(image_mt[:,where_selfB], axis=0)+N)/2/(N-1)
            PhiG_vc += np.histogram(goodG_vc, bins=Nbins, range=(-0.5/(Nbins-1),1+0.5/(Nbins-1)))[0]
            PhiB_vc += np.histogram(goodB_vc, bins=Nbins, range=(-0.5/(Nbins-1),1+0.5/(Nbins-1)))[0]
            #PhiG_vc += np.histogram(goodG_vc, bins=Nbins+1, range=(0,1))[0]
            #PhiB_vc += np.histogram(goodB_vc, bins=Nbins+1, range=(0,1))[0]
    if (T/N)%100 == 0:
        print(T/N)
PhiG_vc = PhiG_vc/(Tmax-Tmin)/N*(Nbins-1)
PhiB_vc = PhiB_vc/(Tmax-Tmin)/N*(Nbins-1)


phi1_vc = np.ones(Nbins)/2
phi0_vc = np.ones(Nbins)/2
state_vc = np.linspace(0,1,Nbins)
sigsig = e2*(1-e2)/N

# AR_vc: iD->(iR,iD) = (G,G), (G,B), (B,G), (B,B)
AR_vc = [1-int((AR-1)/8),1-int((AR-1)%8/4),1-int((AR-1)%4/2),1-int((AR-1)%2)]
AR_vc = (1-2*e1)*np.array(AR_vc)+e1
# SNC_vc, SND_vc
SNC_vc = [1-int((SNC-1)/8),1-int((SNC-1)%8/4),1-int((SNC-1)%4/2),1-int((SNC-1)%2)]
SND_vc = [1-int((SND-1)/8),1-int((SND-1)%8/4),1-int((SND-1)%4/2),1-int((SND-1)%2)]
SNC_vc, SND_vc = (1-2*e2)*np.array(SNC_vc)+e2, (1-2*e2)*np.array(SND_vc)+e2

# C-map and D-map
Cmap_vc = np.reshape(np.outer(state_vc,state_vc)*SNC_vc[0]+np.outer(state_vc,1-state_vc)*SNC_vc[1]+np.outer(1-state_vc,state_vc)*SNC_vc[2]+np.outer(1-state_vc,1-state_vc)*SNC_vc[3], [Nbins**2])
Dmap_vc = np.reshape(np.outer(state_vc,state_vc)*SND_vc[0]+np.outer(state_vc,1-state_vc)*SND_vc[1]+np.outer(1-state_vc,state_vc)*SND_vc[2]+np.outer(1-state_vc,1-state_vc)*SND_vc[3], [Nbins**2])
Cmap_mt = np.exp(-(np.outer(state_vc,np.ones(Nbins**2))-Cmap_vc)**2/(2*sigsig))/(2*np.pi*sigsig)**(0.5)
Dmap_mt = np.exp(-(np.outer(state_vc,np.ones(Nbins**2))-Dmap_vc)**2/(2*sigsig))/(2*np.pi*sigsig)**(0.5)

dmax, T = 1, 0
while dmax > 10**-6:
    q1_vc = np.reshape(np.outer(phi1_vc+phi0_vc,phi1_vc)/Nbins**2, [Nbins**2])
    q0_vc = np.reshape(np.outer(phi1_vc+phi0_vc,phi0_vc)/Nbins**2, [Nbins**2])
    CG1_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*AR_vc[0]*SNC_vc[0] + np.outer(1-state_vc,np.ones(Nbins))*AR_vc[2]*SNC_vc[2], [Nbins**2])
    CB1_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*AR_vc[0]*(1-SNC_vc[0]) + np.outer(1-state_vc,np.ones(Nbins))*AR_vc[2]*(1-SNC_vc[2]), [Nbins**2])
    DG1_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*(1-AR_vc[0])*SND_vc[0] + np.outer(1-state_vc,np.ones(Nbins))*(1-AR_vc[2])*SND_vc[2], [Nbins**2])
    DB1_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*(1-AR_vc[0])*(1-SND_vc[0]) + np.outer(1-state_vc,np.ones(Nbins))*(1-AR_vc[2])*(1-SND_vc[2]), [Nbins**2])
    CG0_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*AR_vc[1]*SNC_vc[1] + np.outer(1-state_vc,np.ones(Nbins))*AR_vc[3]*SNC_vc[3], [Nbins**2])
    CB0_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*AR_vc[1]*(1-SNC_vc[1]) + np.outer(1-state_vc,np.ones(Nbins))*AR_vc[3]*(1-SNC_vc[3]), [Nbins**2])
    DG0_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*(1-AR_vc[1])*SND_vc[1] + np.outer(1-state_vc,np.ones(Nbins))*(1-AR_vc[3])*SND_vc[3], [Nbins**2])
    DB0_vc = np.reshape(np.outer(state_vc,np.ones(Nbins))*(1-AR_vc[1])*(1-SND_vc[1]) + np.outer(1-state_vc,np.ones(Nbins))*(1-AR_vc[3])*(1-SND_vc[3]), [Nbins**2])
    nphi1_vc = np.sum((CG1_vc*q1_vc+CG0_vc*q0_vc)*Cmap_mt+(DG1_vc*q1_vc+DG0_vc*q0_vc)*Dmap_mt, axis=1)/Nbins
    nphi0_vc = np.sum((CB1_vc*q1_vc+CB0_vc*q0_vc)*Cmap_mt+(DB1_vc*q1_vc+DB0_vc*q0_vc)*Dmap_mt, axis=1)/Nbins
    sumphin = np.sum(nphi1_vc)+np.sum(nphi0_vc)
    nphi1_vc, nphi0_vc = nphi1_vc/sumphin*Nbins, nphi0_vc/sumphin*Nbins
    dmax = np.sum((nphi1_vc+nphi0_vc-phi1_vc-phi0_vc)**2)**0.5/Nbins
    phi1_vc, phi0_vc = nphi1_vc, nphi0_vc
    T += 1
    if T%10 == 0:
        print(T, dmax)

ALP1 = 0.6
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
Licolor = [L1color, L2color, L3color, L4color, L5color, L6color, L7color, L8color]
Licolorthin = [L1colorthin, L2colorthin, L3colorthin, L4colorthin, L5colorthin, L6colorthin, L7colorthin, L8colorthin]
MainColor = Licolor[Li-1]
ColorThin = Licolorthin[Li-1]

det = 1
if det == 1:
    fig, ax1 = plt.subplots(figsize=(6,4))
    ax2 = ax1.twinx()
    plt.xlabel('goodness')
    ax1.set_ylabel('frequency')
    ax2.set_ylabel('ratio')
    plt.title('[AR, SNC, SND] = ' + str([AR, SNC, SND]) + ', logN = ' + str([e2,round(np.log10(N),2)]) + ', [e1,e2] = ' + str([e1,e2]), fontsize=10)
    xaxis = np.linspace(-0.5/(Nbins-1),1.0+0.5/(Nbins-1),Nbins)
    ax1.bar(xaxis, PhiG_vc+PhiB_vc, width=1.0/(Nbins-1), align='center', color=ColorThin, linewidth=0)
    where_manysample = np.where(PhiG_vc+PhiB_vc>0.2)[0]
    ax2.scatter(xaxis[where_manysample], (PhiG_vc/(PhiG_vc+PhiB_vc+10**-10))[where_manysample], color=MainColor, s=12, marker='x')
    xaxis = np.linspace(-0.5/(Nbins-1),1.0+0.5/(Nbins-1),Nbins)
    phiTH_vc = phi1_vc+phi0_vc
    ax1.plot(xaxis, phiTH_vc, color='k', alpha=.8, lw=2)
    where_manysample = np.where(phiTH_vc>0.2)[0]
    L = len(where_manysample)
    where_manysample_mt = []
    where_manysample_vc = []
    xbefore = where_manysample[0]
    for j in range(0,L):
        xnow = where_manysample[j]
        if np.abs(xnow-xbefore)<20:
            where_manysample_vc.append(where_manysample[j])
        else:
            where_manysample_mt.append(where_manysample_vc)
            where_manysample_vc = [xnow]
        xbefore = xnow
    where_manysample_mt.append(where_manysample_vc)
    for i in range(0,len(where_manysample_mt)):
        where_manysample_mt[i] = np.array(where_manysample_mt[i])
        ax2.plot(xaxis[where_manysample_mt[i]], (phi1_vc/(phiTH_vc+10**-8))[where_manysample_mt[i]], color='k', alpha=.7, lw=2, ls='--')
    ax1.spines['left'].set_color(ColorThin)
    ax1.spines['right'].set_color(MainColor)
    ax2.spines['left'].set_color(ColorThin)
    ax2.spines['right'].set_color(MainColor)
    ax1.tick_params(axis='y', colors=ColorThin)
    ax2.tick_params(axis='y', colors =MainColor)
    ax2.set_ylim(-0.05,1.05)
    ax1.set_xlim(0,1)
    ax2.set_xlim(0,1)
    ax1.tick_params(labelsize=15)
    ax2.tick_params(labelsize=15)
    plt.savefig("figurename.png", format="png", dpi=400)
    plt.show()


