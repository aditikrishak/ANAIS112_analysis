#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 12:14:28 2019

@author: aditi
"""


import numpy as np 
from scipy import optimize , stats
import os
import nestle
import matplotlib.pyplot as plt
plt.style.use('ggplot')

f="fig2_16.dat"
path=os.path.expanduser("~")+"/Desktop/anais112/data/"+f
t=open(path, "r")
data1 = np.loadtxt(t)                    
t.close()
data1=np.transpose(data1)

f="fig2_26.dat"
path=os.path.expanduser("~")+"/Desktop/anais112/data/"+f
t=open(path, "r")
data2 = np.loadtxt(t)                    
t.close()
data2=np.transpose(data2)

#-----------------MODULATION-------------------

def fit_cosine(x,S,w,t_0):
    	return S*np.cos(w*(x+t_0))

def log_likelihood_cosine(P,DATA):
    y_fit=fit_cosine(DATA[0],P[0],P[1],P[2])
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


def chi2_val_cosine(P,DATA):
        sigma=DATA[2]
        y_fit=fit_cosine(DATA[0],P[0],P[1],P[2])
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#degrees of freedom
def dof_val(P,DATA):
  return len(DATA[0]) - len(P)

#chi squared likelihood function
def chi2L_cosine(P,DATA):
  chi2 = chi2_val_cosine(P,DATA)
  dof = dof_val(P,DATA)
  return stats.chi2(dof).pdf(chi2)


#=========================constant function==========================
def fit_const(x,k):
    	return k[0]*x**0

def log_likelihood_const(k,DATA):
    y_fit=fit_const(DATA[0],k)
    return sum(stats.norm.logpdf(*args) for args in zip(DATA[1],y_fit,DATA[2]))


def chi2_val_const(k,DATA):
        sigma=DATA[2]
        y_fit=fit_const(DATA[0],k)
        r=(DATA[1]-y_fit)/sigma
        return np.sum(r**2)

#chi squared likelihood function
def chi2L_const(k,DATA):
  chi2 = chi2_val_const(k,DATA)
  dof = dof_val(k,DATA)
  return stats.chi2(dof).pdf(chi2)

def frequentist(P,k,DATA):
    cc=chi2_val_cosine(P,DATA)
    ck=chi2_val_const(k,DATA)
    dofc=dof_val(P,DATA)
    dofk=dof_val(k,DATA)
    
    print("\nCosine : Amplitude=",'%.4f'%P[0],";  w=",'%.4f'%P[1],"/days ","; time period",'%.2f'%(2.0*np.pi/P[1]),"days",";  initial phase= ",'%.2f'%P[2]," days")
    print("Constant k= ",'%.4f'%k[0])
    print("\nCosine :  Chi-Square likelihood:" , '%.4f'%chi2L_cosine(P,DATA)," ; Chi square value=",'%.4f'%cc,   " ;  Chi2/dof=",'%.4f'%cc,'/',dofc,
          "\nConstant :  Chi-Square likelihood:" , '%.4f'%chi2L_const(k,DATA)," ; Chi square value=",'%.4f'%ck ,"  ;  chi2/dof=",'%.4f'%ck,'/',dofk)
    d=np.abs(cc-ck)
    print("difference in chi square values = ",'%.4f'%d)
    p=stats.chi2(dofk-dofc).sf(d)
    print ("p value=",'%.4f'%p)
    print("Confidence level : ",'%.4f'%stats.norm.isf(p),'\u03C3','\n')
    
def AIC(P,k,DATA):
    aic_const=-2*log_likelihood_const(k,DATA) + 2
    aic_cosine=-2*log_likelihood_cosine(P,DATA) + 2*3
    del_aic= np.abs(aic_const-aic_cosine)
    print("AIC cosine=",'%.2f'%aic_cosine,", AIC const=",'%.2f'%aic_const)
    print ("diff in AIC values = ",'%.2f'%del_aic)

    
def BIC(P,k,DATA):
    bic_const=-2*log_likelihood_const(k,DATA) + np.log(len(DATA[0]))
    bic_cosine=-2*log_likelihood_cosine(P,DATA)  + 3*np.log(len(DATA[0]))
    del_bic= np.abs(bic_const-bic_cosine)
    print("BIC cosine=",'%.2f'%bic_cosine,", BIC const=",'%.2f'%bic_const)
    print ("diff in BIC values = ",'%.2f'%del_bic,'\n')

def prior_transform_const(k,data):
    if np.abs(np.min(data[1]))>np.abs(np.max(data[1])): A_lim=np.abs(np.min(data[1]))
    else: A_lim=np.abs(np.max(data[1]))
    return A_lim*k

def nestle_const(k,DATA):
    f = lambda k: log_likelihood_const(k, DATA)
    prior = lambda k: prior_transform_const(k,DATA)
    res = nestle.sample(f, prior, 1, method='multi',
                    npoints=2000)
    """print(res.summary())
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)
    nweights = res.weights/np.max(res.weights)
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    samples_nestle = res.samples[keepidx,:]
    print(np.mean(samples_nestle[:,0])) # mean of m samples
    print(np.std(samples_nestle[:,0]) )# standard deviation of m samples"""
    return res.logz

def prior_transform_cosine(P,data):
    if np.abs(np.min(data[1]))>np.abs(np.max(data[1])): A_lim=np.abs(np.min(data[1]))
    else: A_lim=np.abs(np.max(data[1]))
    return np.array([A_lim*P[0] , 2*np.pi*P[1]/(365.25) , P[2]*2*np.pi/P[1]])

def nestle_cosine(P,DATA):
    f = lambda P: log_likelihood_cosine(P, DATA)
    prior = lambda P: prior_transform_cosine(P,DATA)
    res = nestle.sample(f, prior, 3, method='multi',
                    npoints=2000)
    """print(res.summary())
    pm, covm = nestle.mean_and_cov(res.samples, res.weights)
    nweights = res.weights/np.max(res.weights)
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    samples_nestle = res.samples[keepidx,:]
    print(np.mean(samples_nestle[:,0]))      # mean of m samples
    print(np.std(samples_nestle[:,0])      )# standard deviation of m samples
    print( np.mean(samples_nestle[:,1])    )  # mean of c samples
    print( np.std(samples_nestle[:,1])    )  # standard deviation of c samples
    print( np.mean(samples_nestle[:,2])    )  # mean of c samples
    print( np.std(samples_nestle[:,2])    )  # standard deviation of c samples"""
    return res.logz
  
def bayesian(P,k,data):
    Zc=nestle_cosine(P,data)
    Zk=nestle_const(k,data)
    Z= np.exp(Zc-Zk)
    print('Cosine logz=',Zc,', Const logz=',Zk,'\nBayes Factor: ',Z)
    
def plot(P1,P2,k1,k2):
       
    fig,ax=plt.subplots(nrows=2,ncols=1,figsize=(10,4))
    
    
    plt.subplot(211)
    plt.xlim(-20,650)
    p1 = np.linspace(data1[0].min(),data1[0].max(),10000)   
    plt.plot(p1, fit_cosine(p1,P1[0],P1[1],P1[2]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data1[0],data1[1],c='black',s=20)
    plt.plot(p1, np.ones(10000)*k1[0],color='dodgerblue',lw=1,linestyle='--',label='$H_0$' ,linewidth=1.75)
    plt.grid(color='w')
    plt.errorbar(data1[0],data1[1],yerr = data1[2],fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title='1-6keV',title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    
    
    plt.subplot(212)
    plt.xlim(-20,650)
    plt.ylim(-0.1,0.1)
    p2 = np.linspace(data2[0].min(),data2[0].max(),10000)
    plt.plot(p2, fit_cosine(p2,P2[0],P2[1],P2[2]), color = 'red',label='$H_1$',linewidth=1.6)
    plt.scatter(data2[0],data2[1],c='black',s=20)
    plt.plot(p2, np.ones(10000)*k2[0], color='dodgerblue',lw=1,linestyle='--',label='$H_0$',linewidth=1.75 )
    plt.grid(color='w')
    plt.errorbar(data2[0],data2[1],yerr = data2[2], 
                     fmt='none',alpha=0.6,c='black')
    plt.legend(loc='upper right',fontsize=13,title='2-6keV',title_fontsize=13)
    plt.tick_params(axis='both',labelsize=14)
    
    #plt.savefig('try0fig.png')


#==============================================================================

g=[0]

k1 = optimize.minimize(chi2_val_const, g,args=data1,method='BFGS')
k2 = optimize.minimize(chi2_val_const, g,args=data2,method='BFGS')

guess=[0,0.0172,0]

cos1 = optimize.minimize(chi2_val_cosine, guess,args=data1,method='BFGS')
cos2 = optimize.minimize(chi2_val_cosine, guess,args=data2,method='BFGS')    

print("H_0 : No modulation, constant fit")
print("H_1 : Cosine modulation")

print("\n1-6keV")
frequentist(cos1.x, k1.x, data1)
AIC(cos1.x, k1.x, data1)
BIC(cos1.x, k1.x, data1)
#bayesian(cos1.x, k1.x, data1)

print("========================================================================")
print("\n2-6keV")
frequentist(cos2.x, k2.x, data2)
AIC(cos2.x, k2.x, data2)
BIC(cos2.x, k2.x, data2)
#bayesian(cos2.x, k2.x, data2)

plot(cos1.x, cos2.x, k1.x, k2.x)
