import numpy as np
import pandas as pd

### encode CSV To feature Matrix
def get_feature_matrix(file_path):
    #imports
    import time
    def date2x(x):
        k=(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))-time.mktime(time.strptime("2016-01-11 17:00:00", "%Y-%m-%d %H:%M:%S")))/600
        k=k%1008
        return int(k)
    def date2k(x):
        k=(time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S"))-time.mktime(time.strptime("2016-01-11 17:00:00", "%Y-%m-%d %H:%M:%S")))/600
        k=k/1008
        return int(k)

    data=pd.read_csv(file_path)
    data['x']=data['date'].apply(date2x)
    data['k']=data['date'].apply(date2k)

    sigma=2
    num_f=1008
    k_len=20
    mus=range(1,1009)
    mat=data[['x']].as_matrix()
    ks=data[['k']].as_matrix()
    conn=np.concatenate
    phix=np.zeros((len(mat),20161))
    phix[:,0]=ks.flatten()
    phi=ks
    for j in range(num_f):
        x=mat[:,0].reshape((mat.shape[0],1))
        x=((x-mus[j])**2)/(2*sigma**2)
        x=np.exp(-x)
        phi=np.hstack((phi,x))
    print
    for i in range(len(mat)):
        k=int(phix[i][0])
        k1=np.zeros((k*num_f))
        k2=np.zeros(((k_len-k-1)*num_f))
        k3=conn((np.array([k]),k1))
        k4=conn((phi[i][1:],k2))
        phix[i]=conn((k3,k4))
    return phix

## Encode Target Data to numpy array from csv
def get_output(file_path):
    data=pd.read_csv(file_path)
    return data[['Output']].as_matrix()


## Function to train models in Interval
def get_wt(feature_matrix, output, lambda_reg, p):
    phi=feature_matrix
    y=output
    learning_rate=0.02
    max_iter=50
    def power(x):
        if x==0:
            return 0
        else:
            return x**(p-2)

    def mod_w_p(w):
        x=np.absolute(w)
        vfunc = np.vectorize(power)
        x=vfunc(x)
        return x

    def lasso(w):
        for i in range(len(w)):
            if(w[i]<0):
                w[i]=-1
            if(w[i]>0):
                w[i]=1
        return w

    def rmse(x,y,w):
        err=0.0
        err=((y-np.dot(x,w))**2).sum()/y.size
        err=err**.5
        return err

    w=np.zeros(phi.shape[1]).reshape((phi.shape[1],1))
    errs=[]
    for i in range(max_iter):
        t1=np.dot(phi.T,y)
        t2=np.dot(np.dot(phi.T,phi),w)
        if(p>1):
            t3=lambda_reg*p*np.dot(w.T,mod_w_p(w))
        if(p<=1):
            t3=lambda_reg*lasso(w)
        t=t1-t2-t3
        w=w+learning_rate*t
        errs.append(rmse(phi,y,w))
    print("RMSE : ",errs[-1])
    return w


## to get weight vectr given feature_matrix , target , lambda and P
def get_weight_vector(feature_matrix,output,lambda_reg,p):
    num_f=1008
    k_len=20
    fxs=[None]*k_len
    ys=[None]*k_len
    for i in range(len(feature_matrix)):
        k=int(feature_matrix[i][0])
        if fxs[k] is None:
            fxs[k]=np.array([feature_matrix[i][k*num_f+1:k*num_f+1009]])

            ys[k]=np.array([output[i]])
        else:
            fxs[k]=np.vstack((fxs[k],feature_matrix[i][k*num_f+1:k*num_f+1009]))
            ys[k]=np.vstack((ys[k],output[i]))

    wts=[]
    for i in range(k_len):
        print('Model ',i+1)
        wts.append(get_wt(fxs[i],ys[i],lambda_reg,p))
    print(wts[0].shape)
    wtss=np.array([0])
    wtss.shape=(1,1)
    wtss=np.concatenate((wtss,wts[0]))
    for i in range(1,k_len):
        wtss=np.concatenate((wtss,wts[i]))
    return wtss

## get best weight vector
def get_my_best_weight_vector():
    wts=np.load('best_wts.pikl')
    return wts
