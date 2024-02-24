# Who is a leader in the leading eight? --indirect reciprocity under private assessment--
# output data for Fig. 4A and Fig. A2A.


import numpy as np
np.random.seed(0)


Tmin, Tmax = 50, 1050
NW, NM = 9700, 300
N = NW + NM
e1, e2 = 0.03, 0.1
ARSN_vc = [[3,1,14], [3,3,14], [4,1,13], [4,2,13], [4,3,13], [4,4,13], [4,2,14], [4,4,14]]
MUTANT = 1 # ALLC: MUTANT=1, ALLD: MUTANT=0
# L1:3,1,14, L2:3,3,14, L3:4,1,13, L4:4,2,13, L5:4,3,13, L6:4,4,13, L7:4,2,14, L8:4,4,14
txt = open('filename.txt', 'w')
txt.write('#population [NW,NM,N] = ' + str([NW,NM,N]) + '\n')
txt.write('#errors [e1,e2] = ' + str([e1,e2]) + '\n')
txt.write('#time [Tmin,Tmax] = ' + str([Tmin,Tmax]) + '\n')
for Li in range(1,9):
    AR, SNC, SND = ARSN_vc[Li-1]
    image_mt = np.ones((N,N)) # (+1,-1) is (G,B)
    RNchoice_vc = np.random.randint(N, size=(N*Tmax+1,2))
    RNaction_vc = np.random.choice([1,-1],N*Tmax+1,p=[1-e1,e1])

    # AR_vc: iD->(iR,iD) = (+1,+1), (+1,-1), (-1,+1), (-1,-1)
    AR_vc = [1-int((AR-1)/8),1-int((AR-1)%8/4),1-int((AR-1)%4/2),1-int((AR-1)%2)]

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

    pWW, pMW, sW = 0, 0, 0
    for T in range(0,N*Tmax+1):
        [iR,iD] = RNchoice_vc[T]
        if iD > NW-1:
            if MUTANT == 1:
                image_mt[:,iD] = CMAP(image_mt[:,iR], image_mt[:,iD], N)
            if MUTANT == 0:
                image_mt[:,iD] = DMAP(image_mt[:,iR], image_mt[:,iD], N)
        else:
            nAR = int((1-image_mt[iD,iR])+(1-image_mt[iD,iD])/2)
            if AR_vc[nAR]*RNaction_vc[T] == 1: # CMAP
                image_mt[:,iD] = CMAP(image_mt[:,iR], image_mt[:,iD], N)
            else: # DMAP
                image_mt[:,iD] = DMAP(image_mt[:,iR], image_mt[:,iD], N)
        if T%N == 0:
            if T/N > Tmin:
                pWW += (np.mean(image_mt[:NW,:NW])+1)/2
                pMW += (np.mean(image_mt[:NW,NW:])+1)/2
                sW += (np.mean(np.diag(image_mt)[:NW])+1)/2
        if (T/N)%100 == 0:
            print(T/N)
    pWW /= Tmax-Tmin
    pMW /= Tmax-Tmin
    sW /= Tmax-Tmin
    txt.write(str(pWW)+'\t'+str(pMW)+'\t'+str(sW)+'\n')

txt.close()


