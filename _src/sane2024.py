# Derived from code associated with:
# "Internal Versus Forced Variability Metrics for General Circulation Models Using Information Theory"
# Aakash Sane, Baylor Fox-Kemper, Dave Ullman
# Source: Zenodo Version v2, https://zenodo.org/records/7992844
# License: GPL-3.0 for the copied code portion, according to the archive README
# (see the original archive for the authoritative terms).

import numpy as np
import scipy as sp

def ent_1(x, bins1, range_x):
    a=np.min(x); b=a+range_x # testing constant bin ranges
    px=np.histogram(x, bins=bins1, density=True, range=(a,b)),
    rel=np.zeros(np.size(px[0][0]))
    p=px[0][0]#/np.size(x)
    binarea=px[0][1][1]-px[0][1][0]
    p=p*binarea
    for i in range(np.size(p)):
        if p[i]<1e-04:
            rel[i]=0
        else:
            rel[i]=p[i]*np.log2(1/p[i])

    ent=np.sum(rel)#*binarea
    return ent#, binarea

def mutualinfo(x,y,bins1,range_x):
    a=np.min(np.concatenate([x,y])); b=a+range_x; #np.max(np.concatenate([x,y]))
    H1,a,b=np.histogram2d(x,y,bins=(bins1,bins1),density=True,range=((a,b),(a,b)))
    histshape=np.shape(H1)      #gives shape of hist/jpdf
    binarea=(a[2]-a[1])*(b[2]-b[1]) #bin area
    H=H1*binarea   #got the jpdf
    mpdfx=np.zeros(histshape[0]);   #mrpdfx - marginal pdf in x
    mpdfy=np.zeros(histshape[1]);   #mrpdfy - marginal pdf in y
    mpdfx=np.sum(H,axis=1)
    mpdfy=np.sum(H,axis=0)

    mut_ent=np.zeros(histshape)
    for i in range(histshape[0]):
        for j in range(histshape[1]):
            if H[i,j]<1e-04:
                mut_ent[i,j]=0;
            else:
                mut_ent[i,j]=H[i,j]*np.log2(H[i,j]/(mpdfx[i]*mpdfy[j]))
    ent=np.sum(mut_ent)
  #  return ent, H1, H, mpdfx, mpdfy, binarea, mut_ent
    return ent

def optbins(data, maxM):

    # Knuth's OPTBIN algorithm. For reference, check the manuscript.
    # this function gives exactly the same output as MATLAB code optbins. :)

    N = len(data) # data has to be one-dimensional.
    logp = np.zeros(maxM-2)  # settings min bins to be considered as 2
    bin_array = np.arange(2,maxM)

    i=0
    for M in range(2,maxM):
        px=np.histogram(data, bins=M)

        counts=px[0]

        #sp.special.loggamma(px[0])  # same as gammaln from MATLAB

        logp[i] = N*np.log(M) + sp.special.loggamma(M/2) -  sp.special.loggamma(N+M/2)
        logp[i] = logp[i] - M*sp.special.loggamma(0.5) + np.sum(sp.special.loggamma(px[0]+0.5))

        i=i+1

    maxxx = np.max(logp)
    ind = np.where(logp==maxxx)
    optM = bin_array[ind]

    return optM[0]


def infofromy(y,M,range_x,bins=60):
    f=np.concatenate(y); g=np.tile(np.mean(y,axis=0),M);
    ent=ent_1(f,bins,range_x)
    mi=mutualinfo(f,g,bins,range_x)
    sigi=np.mean(np.std(y,axis=0))
    sigf=np.std(np.mean(y,axis=0))

    gam=1-(mi/ent); sstd=(sigi)/np.sqrt(sigi**2 + sigf**2);
    return gam, sstd
