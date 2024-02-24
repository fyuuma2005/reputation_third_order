# Who is a leader in the leading eight? --indirect reciprocity under private assessment--
# output Fig. A3 from data by "ReputationThird_TimeSeries.py"


import numpy as np
import matplotlib.pyplot as plt


filenumber_vc = ['01', '02', '03', '04', '05', '06', '07', '08'] # use "filenameXX.txt"
Li_vc = [1, 2, 3, 4, 5, 6, 7, 8]
N, Nbins = 800, 800 # Nbins = N is default
Tmax = 50
Nsample = 100

Tmax_plot = 40

######### define color #########
ALP1 = 0.6
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
####################################

plt.figure(figsize=(6,4))
xaxis = np.linspace(0,1,Nbins)

for n in range(8):
    filenumber = filenumber_vc[n]
    Li = Li_vc[n]
    MainColor = Licolor[n]

    data = np.loadtxt('filename'+filenumber+'.txt') # [:,0]-sample, [:,1]-time, [:,2~Nbins+2]-phi1, [:,Nbins+3,2*Nbins+3]-phi0 
    phi1_mt, phi0_mt = data[:,2:Nbins+2], data[:,Nbins+2:]

    phi1eq = np.mean(phi1_mt[Tmax::Tmax+1,:], axis=0)
    phi0eq = np.mean(phi0_mt[Tmax::Tmax+1,:], axis=0)
    dphi1_mt, dphi0_mt = np.abs(phi1_mt-phi1eq), np.abs(phi0_mt-phi0eq)
    dphi1_mt, dphi0_mt = (phi1_mt-phi1eq)**2, (phi0_mt-phi0eq)**2
    dst_vc = []
    for i in range(0,Tmax+1):
        phi1eq = np.mean(dphi1_mt[i::Tmax+1,:])
        phi0eq = np.mean(dphi0_mt[i::Tmax+1,:])
        dst = np.mean(phi1eq+phi0eq)**(1/2)
        
        dst_vc.append(dst)
    
    T_vc = np.linspace(0,Tmax_plot,Tmax_plot+1)
    plt.plot(T_vc, np.log10(dst_vc)[:Tmax_plot+1], lw=2, color=MainColor, label='L'+str(n+1))

    print(n+1, '/', 8, 'finished')

plt.xlim(-1,Tmax_plot+1)
plt.legend()
plt.savefig("figurename.png", format="png", dpi=400, transparent=True)
plt.show()





