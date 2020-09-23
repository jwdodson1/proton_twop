from scipy import stats
import os
import numpy as np
import math


def qcd_fold(data, time, configs):
#folds the data around t/2
    output = np.zeros((configs, int(time/2) + 1))
    for k in range(configs):
        for t in range(math.floor(time/2) + 1):
            if t == 0:#Don't average t=0 with anything
                output[k][t] = data[k][t]
            elif t == int(time/2):
                output[k][t] = data[k][t]
            else: #note that this also doesn't average t/2 with anything
                output[k][t] = .5*(data[k][t] + data[k][time - t])
    return output





unpoltwop = np.zeros((1315,64,4,4))



directory = os.fsencode("/gpfs/projects/qcd/forJack/data/data_p3_q_020_xi0")
n = 0
for dirname in os.listdir(directory): #Loop through each directory, reading the data from the desired file
    dirname2 = dirname.decode("utf-8")
    dataname = "/gpfs/projects/qcd/forJack/data/data_p3_q_020_xi0/" + dirname2 + "/twop_proton_mom_+0_-1_+3_zeta_-0.6000.dat"
    data = np.loadtxt(dataname)
    for k in range(len(data)): #loop over the lines in the file
        for p in range(2,8,2): #loop over the entries in the line
            unpoltwop[n][int(data[k][0])][int(data[k][1])][int((p-2)/2)] = data[k][p] #the imaginary part averages out to zero anyway after trace, so only take real part
    n = n + 1

poltwop = np.zeros((n,64)) #This is going to Hold the "Polarized Twop", a single number for every time
for k in range(n):
    for t in range(64): # for each time take Trace of .25*(I+g0)S
        poltwop[k][t] = .25*(unpoltwop[k][t][0][0] + unpoltwop[k][t][1][1] + unpoltwop[k][t][2][2] + unpoltwop[k][t][3][3] - (unpoltwop[k][t][2][0] + unpoltwop[k][t][0][2] + unpoltwop[k][t][1][3] + unpoltwop[k][t][3][1]))


#Resample the data, excluding on index k each time
twopex = np.zeros((1315,63))
for t in range(63):
    for k in range(n): #pick the excluded configuration
        for j in range(n): #loop over the rest
            if (k-j):
                twopex[k][t] = twopex[k][t] + poltwop[j][t]/(n-1)
            else:
                pass

twop = qcd_fold(twopex, 63, n)


M = np.zeros((n,32))
for k in range(n):
    for t in range(15): #The mass started to be imaginary
        if (not t):
            M[k][t] = math.log(twop[k][t]/twop[k][t+1])
        else: #The formula for folded data 
            M[k][t] = .5*math.log((twop[k][t-1] + math.sqrt(twop[k][t-1]*twop[k][t-1] - twop[k][31]*twop[k][31]))/(twop[k][t+1] + math.sqrt(twop[k][t+1]*twop[k][t+1] - twop[k][31]*twop[k][31])))

yint = np.zeros(n)
fiterror = np.zeros(n)
x = np.linspace(8,13,num=5)
for k in range(n):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,M[k,8:13])
    yint[k]=intercept
    fiterror[k]=std_err

Mass_Avg = np.average(yint)
fiterror = np.average(fiterror)

C2pt = np.zeros(15)
C2ptErr = np.zeros(15)
Mass = np.zeros(15)
MassErr = np.zeros(15)
#calculate average over the bins
for t in range(15):
    for k in range(n):
        Mass[t] = Mass[t] + (M[k][t]/n)
        C2pt[t] = C2pt[t] + twop[k][t]/n
    diff = 0
    diff2 = 0
    for j in range(n): #calulate error
        diff = diff + (M[j][t] - Mass[t])*(M[j][t] - Mass[t])
        diff2 = diff2 + (twop[j][t] - C2pt[t])*(twop[j][t] - C2pt[t])
    MassErr[t] = math.sqrt(diff)*math.sqrt((n-1)/n)
    C2ptErr[t] = math.sqrt(diff2)*math.sqrt((n-1)/n)

f = open("ProtonMass_mom_+0_-1_+3_zeta_-0.6000.txt", "w") #Make a file to do data visualization locally
for t in range(15):
    f.write(str(t) + " " + str(Mass[t]) + " " + str(MassErr[t]) + " " + str(C2pt[t]) + " " + str(C2ptErr[t]) + "\n")

f.write(str(0) + " " + str(Mass_Avg) + " " + str(fiterror) + " " + str(0) + " " + str(0))

f.close()
