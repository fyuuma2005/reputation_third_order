# Who is a leader in the leading eight? --indirect reciprocity under private assessment--
# output data for Fig. A3


import numpy as np
np.random.seed(0)


Tmax = 50
N, Nbins = 800, 800 # Nbins = N is default
e1, e2 = 0.03, 0.1
Li = 1 # L1-L8
Nsample = 100

ARSN_vc = [[3,1,14],[3,3,14],[4,1,13],[4,2,13],[4,3,13],[4,4,13],[4,2,14],[4,4,14]] # from L1 ~ L8
[AR, SNC, SND] = ARSN_vc[Li-1]
#AR, SNC, SND = 3, 1, 14 # 1 ~ 16 for all ARs and SNs

# AR_vc: iD->(iR,iD) = (G,G), (G,B), (B,G), (B,B)
AR_vc = [1-int((AR-1)/8),1-int((AR-1)%8/4),1-int((AR-1)%4/2),1-int((AR-1)%2)]
AR_vc = (1-2*e1)*np.array(AR_vc)+e1
# SNC_vc, SND_vc
SNC_vc = [1-int((SNC-1)/8),1-int((SNC-1)%8/4),1-int((SNC-1)%4/2),1-int((SNC-1)%2)]
SND_vc = [1-int((SND-1)/8),1-int((SND-1)%8/4),1-int((SND-1)%4/2),1-int((SND-1)%2)]
SNC_vc, SND_vc = (1-2*e2)*np.array(SNC_vc)+e2, (1-2*e2)*np.array(SND_vc)+e2

# output textfile
txt = open('filename.txt', 'w')
txt.write('#time: Tmax = ' + str(Tmax) + '\n')
txt.write('#populations: [N,Nbins] = ' + str([N,Nbins]) + '\n')
txt.write('#error rates: [e1,e2] = ' + str([e1,e2]) + '\n')
txt.write('#social norm: L' + str(Li) + '\n')
txt.write('#sample number [1], time [1], phi(p,1) [Nbins], phi(p,0) [Nbins]' + '\n')

for i in range(0,Nsample):
    RN3 = np.random.random(3)
    phi1_vc = 2*RN3[0]*np.linspace(RN3[1],1-RN3[1],Nbins)
    phi0_vc = 2*(1-RN3[0])*np.linspace(RN3[2],1-RN3[2],Nbins)
    state_vc = np.linspace(0,1,Nbins)
    sigsig = e2*(1-e2)/N

    # C-map and D-map
    Cmap_vc = np.reshape(np.outer(state_vc,state_vc)*SNC_vc[0]+np.outer(state_vc,1-state_vc)*SNC_vc[1]+np.outer(1-state_vc,state_vc)*SNC_vc[2]+np.outer(1-state_vc,1-state_vc)*SNC_vc[3], [Nbins**2])
    Dmap_vc = np.reshape(np.outer(state_vc,state_vc)*SND_vc[0]+np.outer(state_vc,1-state_vc)*SND_vc[1]+np.outer(1-state_vc,state_vc)*SND_vc[2]+np.outer(1-state_vc,1-state_vc)*SND_vc[3], [Nbins**2])
    Cmap_mt = np.exp(-(np.outer(state_vc,np.ones(Nbins**2))-Cmap_vc)**2/(2*sigsig))/(2*np.pi*sigsig)**(0.5)
    Dmap_mt = np.exp(-(np.outer(state_vc,np.ones(Nbins**2))-Dmap_vc)**2/(2*sigsig))/(2*np.pi*sigsig)**(0.5)

    for T in range(0,Tmax+1):
        txt.write(str(i)+'\t'+str(T)+'\t')
        for l in range(0,Nbins):
            txt.write(str(phi1_vc[l])+'\t')
        for l in range(0,Nbins-1):
            txt.write(str(phi0_vc[l])+'\t')
        txt.write(str(phi0_vc[-1])+'\n')

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
    
    print(i, 'finished')

txt.close()




