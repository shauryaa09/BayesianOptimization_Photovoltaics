from os.path import exists
import matlab.engine

import pandas
import datetime
import sys
from numpy import savetxt
import sklearn.gaussian_process as gp
import numpy as np
from numpy import genfromtxt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
from sklearn.gaussian_process.kernels import ExpSineSquared

##training set
# N is no. of dimensions of the problem
print("Warning: Input only integers")
print("Input the total number of layers")
N = int(input())

##Reading the upper limit and lower limit thicknesses of the layers from the csv file and also the total number of points one wants in the scanning or testing dataset
df_n = pandas.read_csv('data_frame_tandem.csv')
df_n.columns = ['start', 'end', 'total points']

##initial set - just one point
X = np.empty(N, dtype=object)
for i in range(N):
    X[i] = np.random.randint(df_n.iloc[i, 0], df_n.iloc[i, 1], 1).astype(int)
X[8] = X[8] * 10000  # for c_si it only makes sense experimentally to have multiples of 10000

# Copying the set
X_new = X.astype(int)

##Now making a dataframe of the first point with each layer as a column
df = pandas.DataFrame()
for i in range(N):
    df[i] = X[i]

##preprocessing - a method to treat data in order to make the program faster where one converts the set of numbers into a normal distribution
df = df.astype(float)

for i in range(N):
    d = df[i].to_numpy()
    d = d.reshape(-1, 1)
    scaler_x = preprocessing.StandardScaler().fit(d)
    df[i] = scaler_x.transform(d)


##scanning space set- this is a set of points which is used by GPR to fit and then the results of the fit are used by acquisition function to go to the next point
def scan_set_gen():
    dftest = pandas.DataFrame()
    temp = np.empty(N, dtype=object)
    for i in range(N):
        temp[i] = np.random.randint(df_n.iloc[i, 0], df_n.iloc[i, 1], df_n.iloc[i, 2]).astype(int)
    temp[8] = temp[8] * 10000
    for i in range(N):
        dftest[i] = temp[i].flatten()
    dftest = dftest.astype(float)
    scaler = np.empty(N, dtype=object)
    # preprocessing
    for i in range(N):
        d = dftest[i].to_numpy()
        d = d.reshape(-1, 1)
        scaler[i] = preprocessing.StandardScaler().fit(d)
        dftest[i] = scaler[i].transform(d)
    return dftest, scaler


c = 0  # counter for convergence
iter = 5000  # total iterations
maximum_tracker = np.zeros((iter))

start_time = time.time()

Current_density_perovskite = []
Current_density_silicon = []
Current_density_tandem = []
F_train = []
dat_frame = []
print("Enter run name")
rname = str(input())
for t in range(iter):
    ## making string for the input file name
    st = '_'
    for i in range(N):
        st = st + str(X_new[i]) + "_"
    original = "PK_Si_tandem_DST_reference.txt"
    ##creating input file directly in the folder where simulator/CROWM can pick it and simulate it
    f = open(r"C:\Users\Shaurya\Crowm_sims_BO_shaurya\In\PK_Si_tandem" + st + ".txt", "w")
    ct = 0
    for line in open(original):
        ct = ct + 1
        li = line.strip()
        if ct == 47:
            li = "Base name of the output files:         PK_Si_tandem" + st
            f.writelines(li + "\n")
        else:
            f.writelines(li + "\n")

    for i in range(N):
        l = str(X_new[i]) + "\n"
        f.write(l)

    l = str(0)
    f.write(l)
    f.close()

    ##checking if simulator/CROWM returned the file
    file_exists = False
    print("Waiting...for", st, datetime.datetime.now())
    while (file_exists == False):
        path_to_file = r"C:\Users\Shaurya\Crowm_sims_BO_shaurya\Out\PK_Si_tandem" + st + "_Jsc.txt"
        file_exists = exists(path_to_file)
        if file_exists == True:
            print("File found")
            break
        else:
            time.sleep(10)

    ## Copying current value from the output file coming from CROWM/simulator
    ct = 0
    f1 = open(r"C:\Users\Shaurya\Crowm_sims_BO_shaurya\Out\PK_Si_tandem" + st + "_Jsc.txt", "r")
    for line in f1:
        ct = ct + 1
        if (ct == 9):
            l1 = float(line[:11])
        if (ct == 13):
            l2 = float(line[:11])
    ## Matlab engine started for electrical simulation 
    eng = matlab.engine.start_matlab()
    thickness = float(X_new[1])
    finger_no = float(2)
    Jsc_Pk = float(l1 * 0.001)
    Jsc_Si = float(l2 * 0.001)
    print("Jsc_Si", "Jsc_pk", "thickness", Jsc_Si * 1e3, Jsc_Pk * 1e3, thickness)
    Jsc_device, Jsc_CROWM, Jsc_total_unshadowed, Jsc_total_shadowed, Jsc_top, Jsc_bot, Voc, FF, Eff, V_mpp, J_mpp = eng.Tandem3Diode_v2(
        thickness, finger_no, Jsc_Pk, Jsc_Si, nargout=11)
    eng.quit()


    Current_density_perovskite.append(l1)
    Current_density_silicon.append(l2)
    Current_density_tandem.append(Eff)

    ## Saving both X(thickness combination) and Y (current values- all three) into text files for later
    savetxt("Jsc_perovskite" + rname + ".txt", Current_density_perovskite)
    savetxt("Jsc_silicon" + rname + ".txt", Current_density_silicon)
    savetxt("Jsc_tandem" + rname + ".txt", Current_density_tandem)
    F_train = np.append(F_train, Eff)
    X_new_copy = X_new
    dat_frame.append(X_new_copy)
    savetxt("df" + rname + ".csv", dat_frame)
    savetxt("F_train" + rname + ".txt", F_train)

    ##implementing algorithm
    f_max = np.max(F_train)
    dftest, scaler = scan_set_gen()
    ## GPR fit   ##Refer to sklearn library to understand the arguments sent into the function
    gpr = gp.GaussianProcessRegressor(kernel=None, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=10, alpha=1e-3,
                                      normalize_y=True, random_state=2).fit(df, F_train)

    # Predicting mu and sigma
    mu, sigma = gpr.predict(dftest, return_std=True)

    ## Acquisiton function evaluation
    p = (mu - f_max) / (sigma)
    CDF = norm.cdf(p)
    PDF = norm.pdf(p)

    EI = (mu - f_max) * CDF + sigma * PDF

    EI[np.where(sigma < 1e-4)] = 0

    # Locating and storing the next query point
    newsample = dftest.iloc[np.argmax(EI)]

    ## Converting preprocessed data back to normal values in order to print on the terminal
    X_new = np.empty(N, dtype=int)
    for i in range(N):
        X_new[i] = scaler[i].inverse_transform((newsample[i]).reshape(-1, 1))

    ##appending
    df.loc[len(df)] = newsample

    print(Eff, "         Eff at   ", X_new_copy)

    ##Checking if maxima has been reached and convergence achieved by making sure each new value of efficiency is 1e-3 close to the previous value continuously 15 times
    if (abs(Eff - f_max)) < 1e-3:
        tick = True
        print("Location         " + "Maxima")
        print(*X_new_copy, "         ", Eff)
        last_iter = t

    else:
        tick = False
    if tick == True:
        c = c + 1
    else:
        c = 0
    if c == 15:
        break
    # notes the maximum value at t iterations
    maximum_tracker[t] = np.max(F_train)

# plots maximum_tracker array vs iteration
plt.plot(maximum_tracker[0:last_iter], marker='+')
plt.title("Z_value vs iterations")
plt.xlabel("Iterations")
plt.ylabel("Z_value")
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))

