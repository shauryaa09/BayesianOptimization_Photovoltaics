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

# import sys
#np.set_printoptions(threshold=sys.maxsize)
##training set
# "N is no. of dimensions of the problem"
print("Warning: Input only integers")
print("Input the total number of layers")
N = int(input())
offset = []
low_lim = np.zeros(N, dtype=int)
upp_lim = []
points = []
# for i in range(N):
#     print('Enter starting thickness in nm for layer '+str(i+1))
#     nx = int(input())
#     print('Enter ending thickness in nm for layer '+ str(i+1))
#     mx = int(input())
#     offset.append(nx)
#     upp_lim.append(mx)
#     pt=int((mx-nx+1)/5)
#     points.append(pt)

# df_n=pandas.DataFrame()
# df_n[0]=low_lim
# df_n[1]=upp_lim
# df_n[2]=points
# df_n[3]=offset

df_n=pandas.read_csv('data_frame_tandem.csv')
df_n.columns=['start','end','total points']

##training set
X=np.empty(N,dtype=object)
for i in range(N):
    X[i]=np.random.randint(df_n.iloc[i,0],df_n.iloc[i,1],1).astype(int)
X[8]=X[8]*10000 # for c_si it only makes sense experimentally to have multiples of 10000

#Copying the set
X_new=X.astype(int)

df=pandas.DataFrame()
for i in range(N):
    df[i]=X[i]


##scanning space set

# print(tracemalloc.get_traced_memory())
##preprocessing
df=df.astype(float)

for i in range(N):
    d = df[i].to_numpy()
    d = d.reshape(-1,1)
    scaler_x = preprocessing.StandardScaler().fit(d)
    df[i] = scaler_x.transform(d)


def scan_set_gen():
    dftest=pandas.DataFrame()
    temp=np.empty(N, dtype=object)
    for i in range(N):
        temp[i] = np.random.randint(df_n.iloc[i, 0], df_n.iloc[i, 1], df_n.iloc[i, 2]).astype(int)
    temp[8] = temp[8] * 10000
    for i in range (N):
        dftest[i]=temp[i].flatten()
    dftest = dftest.astype(float)
    scaler = np.empty(N, dtype=object)
    for i in range(N):
        d = dftest[i].to_numpy()
        d = d.reshape(-1, 1)
        scaler[i] = preprocessing.StandardScaler().fit(d)
        dftest[i] = scaler[i].transform(d)
    return dftest,scaler

c=0
iter=5000
maximum_tracker=np.zeros((iter))

start_time = time.time()
kernel = ExpSineSquared(length_scale=0.5, periodicity=1)

Current_density_perovskite = []
Current_density_silicon = []
Current_density_tandem = []
F_train = []
dat_frame = []
print("Enter run name")
rname = str(input())
for t in range(iter):
    st = '_'
    for i in range(N):
        st = st + str(X_new[i]) + "_"
    original = "PK_Si_tandem_DST_reference.txt"
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

    file_exists=False
    print("Waiting...for",st, datetime.datetime.now())
    while(file_exists==False):
        path_to_file = r"C:\Users\Shaurya\Crowm_sims_BO_shaurya\Out\PK_Si_tandem"+st+"_Jsc.txt"
        file_exists = exists(path_to_file)
        if file_exists==True:
            print("File found")
            break
        else:
            time.sleep(10)

    ct = 0
    f1 = open(r"C:\Users\Shaurya\Crowm_sims_BO_shaurya\Out\PK_Si_tandem"+st+"_Jsc.txt", "r")
    for line in f1:
        ct= ct + 1
        if (ct == 9):
            l1 = float(line[:11])
        if (ct == 13):
            l2 = float(line[:11])

    curr = min (l1, l2)
    Current_density_perovskite.append(l1)
    Current_density_silicon.append(l2)
    Current_density_tandem.append(curr)

    savetxt("Jsc_perovskite"+rname+".txt",Current_density_perovskite)
    savetxt("Jsc_silicon"+rname+".txt",Current_density_silicon)
    savetxt("Jsc_tandem"+rname+".txt",Current_density_tandem)
    F_train = np.append(F_train, curr)
    X_new_copy = X_new
    dat_frame.append(X_new_copy)

    savetxt("df" + rname + ".csv", dat_frame)
    savetxt("F_train" + rname + ".txt", F_train)
    f_max = np.max(F_train)

    # mu = np.zeros((np.prod(df_n['total points'])))
    # sigma = np.zeros((np.prod(df_n['total points'].to_numpy())))
    # EI = np.zeros((np.prod(df_n['total points'].to_numpy())))
      # max of z in the z value set
    dftest, scaler = scan_set_gen()
    gpr = gp.GaussianProcessRegressor(kernel=None, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=10, alpha=1e-3,
                                      normalize_y=True, random_state=2).fit(df, F_train)


    mu, sigma = gpr.predict(dftest, return_std=True)

    p = (mu - f_max) / (sigma)
    CDF = norm.cdf(p)
    PDF = norm.pdf(p)


    EI = (mu - f_max) * CDF + sigma * PDF


    EI[np.where(sigma < 1e-4)] = 0

    newsample = dftest.iloc[np.argmax(EI)]

    X_new=np.empty(N, dtype=int)
    for i in range(N):
        X_new[i]=scaler[i].inverse_transform((newsample[i]).reshape(-1,1))
    ##appending

    df.loc[len(df)] = newsample

    #F_train = np.append(F_train, (F[tuple(X_new[i] for i in range(len(X_new)))]))
    print(curr, "          curr at   ", X_new_copy)
    if (abs(curr - f_max)/f_max) < 1e-4:
        tick = True
        print("Location         "+"Maxima")
        print(*X_new_copy, "         ", curr)
        last_iter = t

    else:
        tick= False
    if  tick==True:
        c =c+1
    else:
        c=0
    if  c==15:
        break

    maximum_tracker[t] = np.max(F_train)


plt.plot(maximum_tracker[0:last_iter], marker='+')
plt.title("Z_value vs iterations")
plt.xlabel("Iterations")
plt.ylabel("Z_value")
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
