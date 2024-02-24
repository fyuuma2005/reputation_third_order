# Who is a leader in the leading eight? --indirect reciprocity under private assessment--
# output data for Fig. 4 and Fig. A2


import numpy as np
np.random.seed(0)


mindphi = 10**-6
N, Nbins = 800, 800
e1 = 0.1
e2min, e2max, Nde2 = 0.01, 0.5, 50
AR, SNC, SND = 4, 1, 13 # 1 ~ 16
# L1:3,1,14, L2:3,3,14, L3:4,1,13, L4:4,2,13, L5:4,3,13, L6:4,4,13, L7:4,2,14, L8:4,4,14
#AR, SNC, SND = np.random.randint(16, size=(3))+1 # random


# output textfile
txt = open('filename.txt', 'w')
txt.write('#population [N,Nbins] = ' + str([N,Nbins]) + '\n')
txt.write('#errors [e1,e2max,Nde2] = ' + str([e1,e2max,Nde2]) + '\n')
txt.write('#accuracy [mindphi] = ' + str([mindphi]) + '\n')
txt.write('#norm [AR,SNC,SND] = ' + str([AR,SNC,SND]) + '\n')
txt.write('#norm [AR,SNC,SND] = ' + str([AR,SNC,SND]) + '\n')
txt.write('e2, pWW, sW, pMW_ALLC, pMW_ALLD, T' + '\n')


# AR_vc: iD->(iR,iD) = (G,G), (G,B), (B,G), (B,B)
AR_vc = [1-int((AR-1)/8),1-int((AR-1)%8/4),1-int((AR-1)%4/2),1-int((AR-1)%2)]
AR_vc = (1-2*e1)*np.array(AR_vc)+e1


# main simulation
e2_vc = np.linspace(e2min, e2max, Nde2)
for e2 in e2_vc:
    sigsig = e2*(1-e2)/N
    phi1_vc = np.ones(Nbins)/2
    phi0_vc = np.ones(Nbins)/2
    state_vc = np.linspace(0.5/Nbins,1-0.5/Nbins,Nbins)
    # SNC_vc, SND_vc
    SNC_vc = [1-int((SNC-1)/8),1-int((SNC-1)%8/4),1-int((SNC-1)%4/2),1-int((SNC-1)%2)]
    SND_vc = [1-int((SND-1)/8),1-int((SND-1)%8/4),1-int((SND-1)%4/2),1-int((SND-1)%2)]
    SNC_vc, SND_vc = (1-2*e2)*np.array(SNC_vc)+e2, (1-2*e2)*np.array(SND_vc)+e2
    # C-map and D-map
    Cmap_vc = np.reshape(np.outer(state_vc,state_vc)*SNC_vc[0]+np.outer(state_vc,1-state_vc)*SNC_vc[1]+np.outer(1-state_vc,state_vc)*SNC_vc[2]+np.outer(1-state_vc,1-state_vc)*SNC_vc[3], [Nbins**2])
    Dmap_vc = np.reshape(np.outer(state_vc,state_vc)*SND_vc[0]+np.outer(state_vc,1-state_vc)*SND_vc[1]+np.outer(1-state_vc,state_vc)*SND_vc[2]+np.outer(1-state_vc,1-state_vc)*SND_vc[3], [Nbins**2])
    Cmap_mt = np.exp(-(np.outer(state_vc,np.ones(Nbins**2))-Cmap_vc)**2/(2*sigsig))/(2*np.pi*sigsig)**(0.5)
    Dmap_mt = np.exp(-(np.outer(state_vc,np.ones(Nbins**2))-Dmap_vc)**2/(2*sigsig))/(2*np.pi*sigsig)**(0.5)

    T, dphi = 0, 1
    while dphi >  mindphi:
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
        dphi = np.sum((nphi1_vc+nphi0_vc-phi1_vc-phi0_vc)**2)**0.5/Nbins
        phi1_vc, phi0_vc = nphi1_vc, nphi0_vc
        if T%10 == 0:
            print(round(e2,2), T)
        T += 1
    
    phi_vc = phi1_vc+phi0_vc
    goodself_vc = phi1_vc/(phi_vc+10**-10)
    pWW = np.mean(phi_vc*state_vc)
    sW = np.mean(phi_vc*goodself_vc)
    pMW1 = ((1-e1)*(SNC_vc[1]*pWW+SNC_vc[3]*(1-pWW))+e1*(SND_vc[1]*pWW+SND_vc[3]*(1-pWW)))/(1-(1-e1)*((SNC_vc[0]-SNC_vc[1])*pWW+(SNC_vc[2]-SNC_vc[3])*(1-pWW))-e1*((SND_vc[0]-SND_vc[1])*pWW+(SND_vc[2]-SND_vc[3])*(1-pWW)))
    pMW2 = (e1*(SNC_vc[1]*pWW+SNC_vc[3]*(1-pWW))+(1-e1)*(SND_vc[1]*pWW+SND_vc[3]*(1-pWW)))/(1-e1*((SNC_vc[0]-SNC_vc[1])*pWW+(SNC_vc[2]-SNC_vc[3])*(1-pWW))-(1-e1)*((SND_vc[0]-SND_vc[1])*pWW+(SND_vc[2]-SND_vc[3])*(1-pWW)))
    txt.write(str(round(e2,2))+'\t'+str(pWW)+'\t'+str(sW)+'\t'+str(pMW1)+'\t'+str(pMW2)+'\t'+str(T)+'\n')

txt.close()



