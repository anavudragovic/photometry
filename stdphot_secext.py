#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:24:19 2021

@author: anavudragovic
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import gridspec
import seaborn as sns
from scipy import stats
from scipy.odr import ODR, Model, Data, RealData
import fnmatch
from matplotlib import rc
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import normaltest
import statsmodels.api as sm    
from scipy.stats import anderson

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 13}

rc('font', **font)
rc('axes', titlesize=13)
rc('axes', labelsize=13)   
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',color='black')
    
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

def linearSlope(x,slope):
    y = slope * x
    return y

def unireg(beta,x):
    return beta[0] + beta[1] * x

def mixedreg(beta,x):
    return beta[0] + beta[1] * x[0] + beta[2] * x[1] + beta[3] * x[0] * x[1] 

def multireg(beta,x):
    return beta[0] + beta[1] * x[0] + beta[2] * x[1] 



df = pd.read_csv('/Users/anavudragovic/monika/filter_mags_all_formated.txt',header=None, sep = " ")
df.columns=['Name','RAJ2000','DECJ2000','Vcat','Bcat','Rcat','Icat','VcatErr','BcatErr','RcatErr','IcatErr','FilterV','FluxV','FluxVErr','MagV','MagVErr','AirmassV','FilterB','FluxB','FluxBErr','MagB','MagBErr','AirmassB','FilterR','FluxR','FluxRErr','MagR','MagRErr','AirmassR','FilterI','FluxI','FluxIErr','MagI','MagIErr','AirmassI']

filters=["B","V","R","I"] 
# Exclude outliers
df = df[(df['MagB']-df['MagV'] < 3) & (df['Bcat']-df['Vcat']>0) & (df['MagB']-df['Bcat']<-20.8) & (df["MagB"]<0) ]

# Three models are tested: 
#   (1) simple: m_std = m_i - k X + m_zp
#   (2) multi: m_std = m_i - k * X + + c * colour + m_zp
#   (3) mixed: m_std = m_i - k' * X + c * colour + k'' * X * colour +  m_zp

models=["simple","multi","mixed"]   
cols = 4 # number of params: zeropoint, extinction, colour term, secondary extinction
rows = 4 # number of filters
simple = [[0. for i in range(cols)] for j in range(rows)]
simpleErr = [[0. for i in range(cols)] for j in range(rows)]
multi = [[0. for i in range(cols)] for j in range(rows)] 
multiErr = [[0. for i in range(cols)] for j in range(rows)]
mixed = [[0. for i in range(cols)] for j in range(rows)] 
mixedErr = [[0. for i in range(cols)] for j in range(rows)]     
n = len(df) # dim of calibrated magnitude = =length of input data
nm = 3 # number of models
# For each of three models one calibrated magnitude Bcal/Vcal/Rcal/Icat should exist
Bcal = [[0. for i in range(n)] for j in range(nm)]
BcalErr = [[0. for i in range(n)] for j in range(nm)]
Vcal = [[0. for i in range(n)] for j in range(nm)]
VcalErr = [[0. for i in range(n)] for j in range(nm)]
Rcal = [[0. for i in range(n)] for j in range(nm)]
RcalErr = [[0. for i in range(n)] for j in range(nm)]
Ical = [[0. for i in range(n)] for j in range(nm)]
IcalErr = [[0. for i in range(n)] for j in range(nm)]
# For each of three models one array with residuals Bresi/Vresi/Rresi/Iresi should exist
# BVRIresi = BVRIcal - BVRIstd
Bresi = [[0. for i in range(n)] for j in range(nm)]
Vresi = [[0. for i in range(n)] for j in range(nm)]
Rresi = [[0. for i in range(n)] for j in range(nm)]
Iresi = [[0. for i in range(n)] for j in range(nm)]

for i in range(0,len(filters)):
    if (fnmatch.fnmatch(filters[i], 'B')):
           print("--------------- B ----------------")
           X = (df['AirmassB']+df['AirmassV'])/2
           XErr = X*0+.00001
           boja = df['Bcat'] - df['Vcat']
           bojaErr=np.sqrt(df['BcatErr']**2+df['VcatErr']**2)+0.00001
           y = df['Bcat'] - df['MagB']
           yErr = np.sqrt(df['MagBErr']**2+df['BcatErr']**2)
           for m in range(0,len(models)):
               print('Model: ******',models[m],"*********")
               if (fnmatch.fnmatch(models[m], 'simple')):
                   x = X
                   xErr = XErr
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   simple_model = Model(unireg)
                   odrfit = ODR(data, simple_model, beta0=[21., -.1])
                   odrres = odrfit.run()
                   odrres.pprint()
                   simple[0][0:2] = odrres.beta[0:2]
                   simpleErr[0][0:2] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:2]
                   Bcal[0] = unireg(odrres.beta, x)
                   BcalErr[0] = np.sqrt(unireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))                   
                   Bresi[0] = sorted(Bcal[0] + df['MagB'] - df['Bcat'])
               elif (fnmatch.fnmatch(models[m], 'multi')):
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   multi_model = Model(multireg)
                   odrfit = ODR(data, multi_model, beta0=[21., -.1, 0.2])
                   odrres = odrfit.run()
                   Bcal[1] = multireg(odrres.beta, x)
                   BcalErr[1] = np.sqrt(multireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Bresi[1] = sorted(Bcal[1] + df['MagB'] - df['Bcat'])
                   odrres.pprint()
                   multi[0][0:3] = odrres.beta[0:3]
                   multiErr[0][0:3] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:3]
               else: 
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   mixed_model = Model(mixedreg)
                   odrfit = ODR(data, mixed_model, beta0=[21., -.1, 0.2, 0.2])
                   odrres = odrfit.run()
                   Bcal[2] = mixedreg(odrres.beta, x)
                   BcalErr[2] = np.sqrt(mixedreg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Bresi[2] = sorted(Bcal[2] + df['MagB'] - df['Bcat'])
                   odrres.pprint()                  
                   mixed[0][0:4] = odrres.beta
                   mixedErr[0][0:4] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))
    elif fnmatch.fnmatch(filters[i], 'V'):
           print("--------------- V ----------------")
           X = (df['AirmassB']+df['AirmassV'])/2
           XErr = X*0+.00001
           boja = df['Bcat'] - df['Vcat']
           bojaErr=np.sqrt(df['BcatErr']**2+df['VcatErr']**2)+0.00001
           y = df['Vcat'] - df['MagV']
           yErr = np.sqrt(df['MagVErr']**2+df['VcatErr']**2)
           for m in range(0,len(models)):
               print('Model: ******',models[m],"*********")
               if (fnmatch.fnmatch(models[m], 'simple')):
                   x = X
                   xErr = XErr
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   simple_model = Model(unireg)
                   odrfit = ODR(data, simple_model, beta0=[21., -.1])
                   odrres = odrfit.run()
                   Vcal[0] = unireg(odrres.beta, x)
                   VcalErr[0] = np.sqrt(unireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Vresi[0] = sorted(Vcal[0] + df['MagV'] - df['Vcat'])
                   odrres.pprint()
                   simple[1][0:2] = odrres.beta[0:2]
                   simpleErr[1][0:2] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:2]
                   
               elif (fnmatch.fnmatch(models[m], 'multi')):
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   multi_model = Model(multireg)
                   odrfit = ODR(data, multi_model, beta0=[21., -.1, 0.2])
                   odrres = odrfit.run()
                   Vcal[1] = multireg(odrres.beta, x)
                   VcalErr[1] = np.sqrt(multireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Vresi[1] = sorted(Vcal[1] + df['MagV'] - df['Vcat'])
                   odrres.pprint()
                   multi[1][0:3] = odrres.beta[0:3]
                   multiErr[1][0:3] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:3]
               else: 
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   mixed_model = Model(mixedreg)
                   odrfit = ODR(data, mixed_model, beta0=[21., -.1, 0.2, 0.2])
                   odrres = odrfit.run()
                   Vcal[2] = mixedreg(odrres.beta, x)
                   VcalErr[2] = np.sqrt(mixedreg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Vresi[2] = sorted(Vcal[2] + df['MagV'] - df['Vcat'])
                   odrres.pprint()                  
                   mixed[1][0:4] = odrres.beta
                   mixedErr[1][0:4] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))

    elif fnmatch.fnmatch(filters[i], 'R'):
        print("--------------- R ----------------")
        X = (df['AirmassV']+df['AirmassR'])/2
        XErr = X*0+.00001
        boja = df['Vcat'] - df['Rcat']
        bojaErr=np.sqrt(df['VcatErr']**2+df['RcatErr']**2)+0.00001           
        y = df['Rcat'] - df['MagR']
        yErr = np.sqrt(df['MagRErr']**2+df['RcatErr']**2)
        for m in range(0,len(models)):
               print('Model: ******',models[m],"*********")
               if (fnmatch.fnmatch(models[m], 'simple')):
                   x = X
                   xErr = XErr
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   simple_model = Model(unireg)
                   odrfit = ODR(data, simple_model, beta0=[21., -.1])
                   odrres = odrfit.run()
                   Rcal[0] = unireg(odrres.beta, x)
                   RcalErr[0] = np.sqrt(unireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Rresi[0] = sorted(Rcal[0] + df['MagR'] - df['Rcat'])
                   odrres.pprint()
                   simple[2][0:2] = odrres.beta[0:2]
                   simpleErr[2][0:2] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:2]
                   
               elif (fnmatch.fnmatch(models[m], 'multi')):
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   multi_model = Model(multireg)
                   odrfit = ODR(data, multi_model, beta0=[21., -.1, 0.2])
                   odrres = odrfit.run()
                   Rcal[1] = multireg(odrres.beta, x)
                   RcalErr[1] = np.sqrt(multireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Rresi[1] = sorted(Rcal[1] + df['MagR'] - df['Rcat'])
                   odrres.pprint()
                   multi[2][0:3] = odrres.beta[0:3]
                   multiErr[2][0:3] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:3]
               else: 
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   mixed_model = Model(mixedreg)
                   odrfit = ODR(data, mixed_model, beta0=[21., -.1, 0.2, 0.2])
                   odrres = odrfit.run()
                   Rcal[2] = mixedreg(odrres.beta, x)
                   RcalErr[2] = np.sqrt(mixedreg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Rresi[2] = sorted(Rcal[2] + df['MagR'] - df['Rcat'])
                   odrres.pprint()                  
                   mixed[2][0:4] = odrres.beta
                   mixedErr[2][0:4] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))

    else:
        print("--------------- I ----------------")
        X = (df['AirmassV']+df['AirmassI'])/2
        XErr = X*0+.00001
        boja = df['Vcat'] - df['Icat']
        bojaErr=np.sqrt(df['VcatErr']**2+df['IcatErr']**2)+0.00001
        y = df['Icat'] - df['MagI']
        yErr = np.sqrt(df['MagIErr']**2+df['IcatErr']**2)
        for m in range(0,len(models)):
               print('Model: ******',models[m],"*********")
               if (fnmatch.fnmatch(models[m], 'simple')):
                   x = X
                   xErr = XErr
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   simple_model = Model(unireg)
                   odrfit = ODR(data, simple_model, beta0=[21., -.1])
                   odrres = odrfit.run()
                   Ical[0] = unireg(odrres.beta, x)
                   IcalErr[0] = np.sqrt(unireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Iresi[0] = sorted(Ical[0] + df['MagI'] - df['Icat'])
                   odrres.pprint()
                   simple[3][0:2] = odrres.beta[0:2]
                   simpleErr[3][0:2] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:2]
                   
               elif (fnmatch.fnmatch(models[m], 'multi')):
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   multi_model = Model(multireg)
                   odrfit = ODR(data, multi_model, beta0=[21., -.1, 0.2])
                   odrres = odrfit.run()
                   Ical[1] = multireg(odrres.beta, x)
                   IcalErr[1] = np.sqrt(multireg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Iresi[1] = sorted(Ical[1] + df['MagI'] - df['Icat'])
                   odrres.pprint()
                   multi[3][0:3] = odrres.beta[0:3]
                   multiErr[3][0:3] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))[0:3]
               else: 
                   x = np.row_stack( (X, boja) )
                   xErr = np.row_stack( (XErr,bojaErr) )
                   data = RealData(x, y, 1/xErr**2, 1/yErr**2)
                   mixed_model = Model(mixedreg)
                   odrfit = ODR(data, mixed_model, beta0=[21., -.1, 0.2, 0.2])
                   odrres = odrfit.run()
                   Ical[2] = mixedreg(odrres.beta, x)
                   IcalErr[2] = np.sqrt(mixedreg(np.diag(odrres.cov_beta * odrres.res_var),x**2))
                   Iresi[2] = sorted(Ical[2] + df['MagI'] - df['Icat'])
                   odrres.pprint()                  
                   mixed[3][0:4] = odrres.beta
                   mixedErr[3][0:4] = np.sqrt(np.diag(odrres.cov_beta * odrres.res_var))

        

'-----------------------------------------------'
'                     PLOTS                     '
'-----------------------------------------------'
# For each model create a plot of comparison between calibrated and standard magnitude 
# In the B-band:
title = ["Extinction correction","Extinction + colour correction", "Extinction + secondary extinction + colour correction"]
cols = ['C0','C1','C2', 'C3']
for m in range(0,nm):
    print('Output figure: B'+str(m)+'.pdf')
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=True))
    plt.subplots_adjust(hspace=.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.set_title(title[m])
    ax2.set_xlabel(r'$B_{\rm cal}$')
    ax1.set_ylabel('B')
    ax2.set_ylabel(r'$\Delta B$')
    ax1.errorbar(Bcal[m]+df['MagB'],df['Bcat'],xerr=np.sqrt(BcalErr[m]**2+df['MagBErr']**2),yerr=df['BcatErr'], fmt='o', fillstyle='none',ms=5,ecolor=cols[0],mfc=cols[0],mec=cols[0]) 
    ax1.plot(Bcal[m]+df['MagB'],Bcal[m]+df['MagB'],color='k')
    ax2.set_ylim(-0.32,0.32)
    ax2.errorbar(Bcal[m]+df['MagB'],Bcal[m]+df['MagB']-df['Bcat'],xerr=np.sqrt(BcalErr[m]**2+df['MagBErr']**2),yerr=np.sqrt(df['BcatErr']**2+BcalErr[m]**2+df['MagBErr']**2), fmt='o', fillstyle='none',ms=3,ecolor=cols[0],mfc=cols[0],mec=cols[0]);
    ax2.axhline(y=0,color='k',lw=1,ls='--')
    plt.savefig('B'+str(m)+'.pdf',format='pdf')
# In the V-band:
for m in range(0,nm):
    print('Output figure: V'+str(m)+'.pdf')
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=True))
    plt.subplots_adjust(hspace=.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.set_title(title[m])
    ax2.set_xlabel(r'$V_{\rm cal}$')
    ax1.set_ylabel('V')
    ax2.set_ylabel(r'$\Delta V$')
    ax1.errorbar(Vcal[m]+df['MagV'],df['Vcat'],xerr=np.sqrt(VcalErr[m]**2+df['MagVErr']**2),yerr=df['VcatErr'], fmt='o', fillstyle='none',ms=5,ecolor=cols[1],mfc=cols[1],mec=cols[1],alpha=0.7) 
    ax1.plot(Vcal[m]+df['MagV'],Vcal[m]+df['MagV'],color='k')
    ax2.set_ylim(-0.32,0.32)
    ax2.errorbar(Vcal[m]+df['MagB'],Vcal[m]+df['MagV']-df['Vcat'],xerr=np.sqrt(VcalErr[m]**2+df['MagVErr']**2),yerr=np.sqrt(df['VcatErr']**2+VcalErr[m]**2+df['MagVErr']**2), fmt='o', fillstyle='none',ms=3,ecolor=cols[1],mfc=cols[1],mec=cols[1],alpha=0.7);
    ax2.axhline(y=0,color='k',lw=1,ls='--')
    plt.savefig('V'+str(m)+'.pdf',format='pdf')
# In the R-band:
for m in range(0,nm):
    print('Output figure: R'+str(m)+'.pdf')
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=True))
    plt.subplots_adjust(hspace=.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.set_title(title[m])
    ax2.set_xlabel(r'$R_{\rm cal}$')
    ax1.set_ylabel('R')
    ax2.set_ylabel(r'$\Delta R$')
    ax1.errorbar(Rcal[m]+df['MagR'],df['Rcat'],xerr=np.sqrt(RcalErr[m]**2+df['MagRErr']**2),yerr=df['RcatErr'], fmt='o', fillstyle='none',ms=5,ecolor=cols[2],mfc=cols[2],mec=cols[2],alpha=0.7) 
    ax1.plot(Rcal[m]+df['MagR'],Rcal[m]+df['MagR'],color='k')
    ax2.set_ylim(-0.32,0.32)
    ax2.errorbar(Rcal[m]+df['MagR'],Rcal[m]+df['MagR']-df['Rcat'],xerr=np.sqrt(RcalErr[m]**2+df['MagRErr']**2),yerr=np.sqrt(df['RcatErr']**2+RcalErr[m]**2+df['MagRErr']**2), fmt='o', fillstyle='none',ms=3,ecolor=cols[2],mfc=cols[2],mec=cols[2],alpha=0.7);
    ax2.axhline(y=0,color='k',lw=1,ls='--')
    plt.savefig('R'+str(m)+'.pdf',format='pdf')
# In the I-band:
for m in range(0,nm):
    print('Output figure: I'+str(m)+'.pdf')
    fig = plt.figure()
    # set height ratios for subplots
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    fig, (ax1,ax2) = plt.subplots(nrows=2, sharex=True, subplot_kw=dict(frameon=True))
    plt.subplots_adjust(hspace=.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax1.set_title(title[m])
    ax2.set_xlabel(r'$I_{\rm cal}$')
    ax1.set_ylabel('I')
    ax2.set_ylabel(r'$\Delta I$')
    ax1.errorbar(Ical[m]+df['MagI'],df['Icat'],xerr=np.sqrt(IcalErr[m]**2+df['MagIErr']**2),yerr=df['IcatErr'], fmt='o', fillstyle='none',ms=5,ecolor=cols[3],mfc=cols[3],mec=cols[3],alpha=0.7) 
    ax1.plot(Ical[m]+df['MagI'],Ical[m]+df['MagI'],color='k')
    ax2.set_ylim(-0.32,0.32)
    ax2.errorbar(Ical[m]+df['MagI'],Ical[m]+df['MagI']-df['Icat'],xerr=np.sqrt(IcalErr[m]**2+df['MagIErr']**2),yerr=np.sqrt(df['IcatErr']**2+IcalErr[m]**2+df['MagIErr']**2), fmt='o', fillstyle='none',ms=3,ecolor=cols[3],mfc=cols[3],mec=cols[3],alpha=0.7);
    ax2.axhline(y=0,color='k',lw=1,ls='--')
    plt.savefig('I'+str(m)+'.pdf',format='pdf')

for i in range(0,len(filters)):
    for m in range(0,nm):
        if (fnmatch.fnmatch(filters[i], 'B')):
            outfile ='B'+str(m)+'_resi.pdf'
            residuals = Bresi[m]
            xlab = r'$\Delta B$'
            y1 = np.max(stats.norm.pdf(Bresi[0], np.mean(Bresi[0]), np.std(Bresi[0])))
            y2 = np.max(stats.norm.pdf(Bresi[1], np.mean(Bresi[1]), np.std(Bresi[1])))
            y3 = np.max(stats.norm.pdf(Bresi[2], np.mean(Bresi[2]), np.std(Bresi[2])))            
            ylimit = np.max([y1,y2,y3])+.5
            xlim=[-0.3,0.3]
            colplot = cols[0]
        elif (fnmatch.fnmatch(filters[i], 'V')):
            outfile ='V'+str(m)+'_resi.pdf'
            residuals = Vresi[m]
            xlab = r'$\Delta V$'
            y1 = np.max(stats.norm.pdf(Vresi[0], np.mean(Vresi[0]), np.std(Vresi[0])))
            y2 = np.max(stats.norm.pdf(Vresi[1], np.mean(Vresi[1]), np.std(Vresi[1])))
            y3 = np.max(stats.norm.pdf(Vresi[2], np.mean(Vresi[2]), np.std(Vresi[2])))            
            ylimit = np.max([y1,y2,y3])+.5
            colplot = cols[1]
            xlim=[-0.3,0.3]

        elif (fnmatch.fnmatch(filters[i], 'R')):
            outfile ='R'+str(m)+'_resi.pdf'
            residuals = Rresi[m]
            xlab = r'$\Delta R$'
            y1 = np.max(stats.norm.pdf(Rresi[0], np.mean(Rresi[0]), np.std(Rresi[0])))
            y2 = np.max(stats.norm.pdf(Rresi[1], np.mean(Rresi[1]), np.std(Rresi[1])))
            y3 = np.max(stats.norm.pdf(Rresi[2], np.mean(Rresi[2]), np.std(Rresi[2])))            
            ylimit = np.max([y1,y2,y3])+1
            colplot=cols[2]
            xlim=[-0.3,0.3]
        else: 
            outfile ='I'+str(m)+'_resi.pdf'
            residuals = Iresi[m]
            xlab = r'$\Delta I$'
            y1 = np.max(stats.norm.pdf(Iresi[0], np.mean(Iresi[0]), np.std(Iresi[0])))
            y2 = np.max(stats.norm.pdf(Iresi[1], np.mean(Iresi[1]), np.std(Iresi[1])))
            y3 = np.max(stats.norm.pdf(Iresi[2], np.mean(Iresi[2]), np.std(Iresi[2])))            
            ylimit = np.max([y1,y2,y3])+1.5
            colplot=cols[3]
            xlim=[-0.3,0.3]
          
        normal_distribution = stats.norm.pdf(residuals, np.mean(residuals), np.std(residuals))
        x = residuals
        fig = plt.subplots(figsize=(6, 4))
        ax = sns.kdeplot(x, shade=False, color=colplot)
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        middle = np.mean(x)
        sdev = np.std(x)
        left = middle - sdev
        right = middle + sdev
        ax.set_xlabel(xlab)
        ax.set_ylabel('Counts')
        #ax.set_title(title[m])
        ax.fill_between(xs, 0, ys, facecolor=colplot, alpha=0.2)
        ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), interpolate=True, facecolor=colplot, alpha=0.2)
        ax.plot(residuals, normal_distribution,  color=colplot, ls='--')
        ax.vlines(middle, 0, np.interp(middle, xs, ys), color=colplot, ls=':')
        ax.set_xlim(-0.3,0.3)
        ax.set_ylim(0,ylimit)
        plt.savefig(outfile,format='pdf')
        # Test for the normality of residuals' distributions using the Shapiro-Wilk test 
        # implemented in Python via shapiro() SciPy function 
        stat, p = shapiro(np.array(residuals))
        # D’Agostino’s K^2 Test via the normaltest() SciPy function
        statn, pn = normaltest(np.array(residuals))
        # Anderson-Darling Test via the anderson() SciPy function
        result = anderson(np.array(residuals))
        print('-----------------------',filters[i],models[m],'-----------------------')
        print('Stdev = ',np.std(x))
        print('Shapiro: Statistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p > alpha:
            print('     Sample looks Gaussian (fail to reject H0)')
        else:
            print('     Sample does not look Gaussian (reject H0)')
        print('D\'Agostino\'s K^2: Statistics=%.3f, p=%.3f' % (statn, pn))
        if pn > alpha:
	         print('     Sample looks Gaussian (fail to reject H0)')
        else:
	         print('     Sample does not look Gaussian (reject H0)')
        print('Anderson-Darling: Statistic: %.3f' % result.statistic)
        # Interpret statistical results
        p = 0
        for j in range(len(result.critical_values)):
            sl, cv = result.significance_level[j], result.critical_values[j]
            if result.statistic < result.critical_values[j]:
                print('     %.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
            else: 
                print('     %.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
        # QQ Plot:
        #fig = plt.figure(figsize=(6, 4))
        #pp = sm.ProbPlot(np.array(residuals), fit=True)
        #qq = pp.qqplot(marker='o', markerfacecolor=colplot, markeredgecolor=colplot,alpha=0.5)
        #sm.qqline(qq.axes[0], line='45', fmt='k--')
        #plt.savefig('qq'+outfile)
        # 'crimson'