import numpy as np
from numba import njit, prange, int64
import numba as nb
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

np.random.seed(123456)

redshift = 2.0
zstring = '2.000'

ngrid = 256
lbox = 500.

lambdath = 0. # Eigenalues threshold for cosmic web classification

dm_filename = 'DensityDM.z2_0.sim2.n256.rspace.dat' 
flux_filename = 'fluxz.z2_0.sim2.n256.zspace.dat'
vz_filename = 'Velocity_z.z2_0.sim2.n256.dat'

#out_filename = '...' 
inpars_filename = 'bestfit_pars.npy'
outpars_filename = 'bestfit_pars_global.npy'

verbose_parameters = False

npars_fit = 7

bounds_range = 0.2 

# Power spectrum multipoles computation
mumin = 0.
mumax = 1.
axis = 2
weight_l0 = 2.
weight_l2 = 1.
kkth_l0 = 1.0
kkth_l2 = 0.3

fit = True

prec = np.float64


# ***********************************
# ***********************************
# ***********************************
def fftr2c(arr):
    arr = np.fft.rfftn(arr, norm='ortho')

    return arr

def fftc2r(arr):
    arr = np.fft.irfftn(arr, norm='ortho')

    return arr

# ************************************
@njit(cache=True)
def k_squared(lbox,ngrid,ii,jj,kk):
    
      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)
      
      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)
      
      #if kk <= nc/2:
      kz = kfac*kk
      #else:
      #  kz = -kfac*np.float64(nc-k)
      
      k2 = kx**2+ky**2+kz**2

      return k2

# ***********************************
# ***********************************

@njit(cache=True)
def k_squared_nohermite(lbox,ngrid,ii,jj,kk):

      kfac = 2.0*np.pi/lbox

      if ii <= ngrid/2:
        kx = kfac*ii
      else:
        kx = -kfac*(ngrid-ii)

      if jj <= ngrid/2:
        ky = kfac*jj
      else:
        ky = -kfac*(ngrid-jj)

      if kk <= ngrid/2:
          kz = kfac*kk
      else:
          kz = -kfac*(ngrid-kk)                                                                                                           

      k2 = kx**2+ky**2+kz**2

      return k2

# ********************************
@njit(parallel=False, cache=True, fastmath=True)
def trilininterp(xx, yy, zz, arrin, lbox, ngrid):

    lcell = lbox/ngrid

    indxc = int(xx/lcell)
    indyc = int(yy/lcell)
    indzc = int(zz/lcell)

    wxc = xx/lcell - indxc
    wyc = yy/lcell - indyc
    wzc = zz/lcell - indzc

    if wxc <=0.5:
        indxl = indxc - 1
        if indxl<0:
            indxl += ngrid
        wxc += 0.5
        wxl = 1 - wxc
    elif wxc >0.5:
        indxl = indxc + 1
        if indxl>=ngrid:
            indxl -= ngrid
        wxl = wxc - 0.5
        wxc = 1 - wxl

    if wyc <=0.5:
        indyl = indyc - 1
        if indyl<0:
            indyl += ngrid
        wyc += 0.5
        wyl = 1 - wyc
    elif wyc >0.5:
        indyl = indyc + 1
        if indyl>=ngrid:
            indyl -= ngrid
        wyl = wyc - 0.5
        wyc = 1 - wyl

    if wzc <=0.5:
        indzl = indzc - 1
        if indzl<0:
            indzl += ngrid
        wzc += 0.5
        wzl = 1 - wzc
    elif wzc >0.5:
        indzl = indzc + 1
        if indzl>=0:
            indzl -= ngrid
        wzl = wzc - 0.5
        wzc = 1 - wzl

    wtot = wxc*wyc*wzc + wxl*wyc*wzc + wxc*wyl*wzc + wxc*wyc*wzl + wxl*wyl*wzc + wxl*wyc*wzl + wxc*wyl*wzl + wxl*wyl*wzl

    out = 0.

    out += arrin[indxc,indyc,indzc] * wxc*wyc*wzc
    out += arrin[indxl,indyc,indzc] * wxl*wyc*wzc
    out += arrin[indxc,indyl,indzc] * wxc*wyl*wzc
    out += arrin[indxc,indyc,indzl] * wxc*wyc*wzl
    out += arrin[indxl,indyl,indzc] * wxl*wyl*wzc
    out += arrin[indxc,indyl,indzl] * wxc*wyl*wzl
    out += arrin[indxl,indyc,indzl] * wxl*wyc*wzl
    out += arrin[indxl,indyl,indzl] * wxl*wyl*wzl

    return out        

# ********************************
@njit(parallel=True, fastmath=True, cache=True)
def real_to_redshift_space(delta, vz, ngrid, lbox, biaspars):

    lcell = lbox/ ngrid

    posx = np.zeros(ngrid**3)
    posy = np.zeros(ngrid**3)
    posz = np.zeros(ngrid**3)

    # Parallelize the outer loop                                                                                                                                                                            
    for ii in prange(ngrid):
        for jj in prange(ngrid):
            for kk in range(ngrid):
                
                # Initialize positions at the centre of the cell                                                                                                                                                    
                xtmp = lcell*(ii+0.5)
                ytmp = lcell*(jj+0.5)
                ztmp = lcell*(kk+0.5)
                
                indx = int(xtmp/lcell)
                indy = int(ytmp/lcell)
                indz = int(ztmp/lcell)
                
                ind3d = indz+ngrid*(indy+ngrid*indx)

                tind = int(tweb[ii,jj,kk]-1)
                dind = int(twebdelta[ii,jj,kk]-1)

                bv = biaspars[tind,dind,4]
                bb = biaspars[tind,dind,5]
                betarsd = biaspars[tind,dind,6]
                gamma = 1
                
                sigma = bb*(1. + delta[indx,indy,indz])**betarsd
                vzrand = np.random.normal(0,sigma)
                vzrand = np.sign(vzrand) * abs(vzrand) ** gamma
                vztmp = trilininterp(xtmp, ytmp, ztmp, vz, lbox, ngrid)
        
                vztmp += vzrand

                ztmp = ztmp + bv * vztmp #/ (ascale * HH)

                if ztmp<0:
                    ztmp += lbox
                elif ztmp>lbox:
                    ztmp -= lbox

                posx[ind3d] = xtmp
                posy[ind3d] = ytmp
                posz[ind3d] = ztmp

    return posx, posy, posz

# ***********************************
# ***********************************
@njit(parallel=False, cache=True)
def get_power(fsignal, Nbin, kmax, dk, mumin, mumax, axis):

    nmode = np.zeros(Nbin)
    kmode = np.zeros(Nbin)
    mono = np.zeros(Nbin)
    quadru = np.zeros(Nbin)
    hexa = np.zeros(Nbin)
    
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,ii,jj,kk))

                if ktot <= kmax:

                    k_par, k_per = get_k_par_per(lbox,ngrid,ii,jj,kk,axis)

                    # find the value of mu
                    if ii==0 and jj==0 and kk==0:  
                        mu = 0.0
                    else:    
                        mu = k_par/ktot
                    mu2 = mu*mu
                    #print(mu2)
                    #print('Done mu2')

                    # take the absolute value of k_par
                    if k_par<0.:
                        k_par = -k_par
                    
                    if mu>=mumin and mu<mumax:
                        nbin = int(ktot/dk-0.5)
                        akl = fsignal.real[ii,jj,kk]
                        bkl = fsignal.imag[ii,jj,kk]
                        kmode[nbin]+=ktot
                        delta2 = akl*akl+bkl*bkl
                        mono[nbin] += delta2
                        quadru[nbin] += delta2*(3.0*mu2-1.0)/2.0
                        hexa[nbin]   += delta2*(35.0*mu2*mu2 - 30.0*mu2 + 3.0)/8.0
                        nmode[nbin]+=1.

    for m in range(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            mono[m]/=nmode[m]
            quadru[m]/=nmode[m]*5.  # we need to multiply the multipoles by (2*ell + 1)
            hexa[m]/=nmode[m]*9.    # we need to multiply the multipoles by (2*ell + 1)

    mono = mono / (ngrid/2)**3
    quadru = quadru / (ngrid/2)**3
    hexa = hexa / (ngrid/2)**3

    return kmode, mono, quadru, hexa, nmode

# **********************************************
@njit(cache=True)
def get_k_par_per(lbox,ngrid,ii,jj,ll,axis):

    kfac = 2.0*np.pi/lbox

    if ii <= ngrid/2:
        kx = kfac*ii
    else:
        kx = -kfac*(ngrid-ii)

    if jj <= ngrid/2:
        ky = kfac*jj
    else:
        ky = -kfac*(ngrid-jj)

    if ll <= ngrid/2:
        kz = kfac*ll
    else:
        kz = -kfac*(ngrid-ll)

    # compute the value of k_par and k_perp
    if axis==0:   
        k_par = kx 
        k_per = np.sqrt(ky*ky + kz*kz)
    elif axis==1: 
        k_par = ky
        k_per = np.sqrt(kx*kx + kz*kz)
    else: 
        k_par = kz
        k_per = np.sqrt(kx*kx + ky*ky)
                                                                                                               
    return k_par, k_per

# ***********************************
# ***********************************
def measure_spectrum_old(signal):

    nbin = round(ngrid/2)
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))
    dk = kmax/nbin  # Bin width

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    print(fsignal.shape)

    kmode, power, nmode = get_power(fsignal, nbin, kmax, dk, kmode, power, nmode)
    
    return kmode[1:], power[1:]

def measure_spectrum(signal):

    nbin = int(ngrid//2)
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)

    kmax = np.pi * ngrid / lbox * np.sqrt(3.) #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))
    dk = kmax/nbin  # Bin width

    kmode, power_l0, power_l2, power_l4, nmode = get_power(fsignal, nbin, kmax, dk, mumin, mumax, axis)

    return kmode[1:], power_l0[1:], power_l2[1:], power_l4[1:] 

# **********************************************
@njit(parallel=True, cache=True)
def divide_by_k2(delta,ngrid, lbox):
    for ii in prange(ngrid):
        for jj in prange(ngrid):
            for kk in prange(round(ngrid/2.)): 
                k2 = k_squared(lbox,ngrid,ii,jj,kk) 
                if k2>0.:
                    delta[ii,jj,kk] /= -k_squared(lbox,ngrid,ii,jj,kk) 

    return delta

# ***********************************
# ***********************************
#@njit(parallel=True, cache=True)
def poisson_solver(delta, ngrid, lbox):

    delta = fftr2c(delta)

    delta = divide_by_k2(delta, ngrid, lbox)

    delta[0,0,0] = 0.

    delta = fftc2r(delta)

    return delta

# ***********************************
# ***********************************
@njit(parallel=True, cache=True, fastmath=True)
def biasmodel_withnorm(ngrid, lbox, delta, tweb, twebdelta, nmean, alpha, beta, dth, rhoeps, eps, rhoepsprime, epsprime, twebenv, twebdeltaenv):
     
    lcell = lbox/ ngrid

    # Allocate tracer field (may be replaced with delta if too memory consuming)
    ncounts = np.zeros((ngrid,ngrid,ngrid))

    #print(twebenv, twebdeltaenv)

    # FIRST LOOP: deterministic bias
    # Parallelize the outer loop
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                
                # Sample number counts
                if tweb[ii,jj,kk]==twebenv and twebdelta[ii,jj,kk]==twebdeltaenv and delta[ii,jj,kk]>=dth:
                    ncounts[ii,jj,kk] = (1.+delta[ii,jj,kk])**alpha
                else:
                    ncounts[ii,jj,kk] = 0. #* np.exp(-((1 + delta[ii,jj,kk])/rhoeps)**eps)# * np.exp(-((1 + delta[ii,jj,kk])/rhoepsprime)**epsprime)

    nc = 0.
    for ii in range(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):
                
                # Sample number counts
                if tweb[ii,jj,kk]==twebenv and twebdelta[ii,jj,kk]==twebdeltaenv:
                    nc+= ncounts[ii,jj,kk]
    
    denstot = nc / lbox**3


    normal = nmean / denstot
    
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                ncounts[ii,jj,kk] = nmean / denstot *  ncounts[ii,jj,kk]
                pnegbin = 1 - ncounts[ii,jj,kk]/(ncounts[ii,jj,kk] + beta)

                ncounts[ii,jj,kk] = negative_binomial(beta, pnegbin)

    return ncounts, normal

# ************************************************

@njit(parallel=True, cache=True, fastmath=True)
def biasmodel(ngrid, lbox, delta, tweb, twebdelta, biaspars):

    # Allocate tracer field (may be replaced with delta if too memory consuming)
    flux = np.zeros((ngrid,ngrid,ngrid))

    # FIRST LOOP: deterministic bias
    # Parallelize the outer loop
    for ii in prange(ngrid):
        for jj in range(ngrid):
            for kk in range(ngrid):

                tind = int(tweb[ii,jj,kk]-1)
                dind = int(twebdelta[ii,jj,kk]-1)

                aa = biaspars[tind,dind,0]
                alpha = biaspars[tind,dind,1]
                rho = biaspars[tind,dind,2]
                eps = biaspars[tind,dind,3]
   
                tau = aa * (1+delta[ii,jj,kk])**alpha * np.exp(-((1.+delta[ii,jj,kk])/rho)**eps)
                flux[ii,jj,kk] = np.exp(-tau)

    return flux

# ********************************
# ********************************
@njit(fastmath=True, cache=True)
def negative_binomial(n, p):
    if n>0:
        if p > 0. and p < 1.:
            gfunc = np.random.gamma(n, (1. - p) / p)
            Y = np.random.poisson(gfunc)

    else:
        Y = 0

    return Y

# ********************************
# ********************************
@njit(parallel=True, cache=True)
def get_cic(posx, posy, posz, lbox, ngrid):

    delta = np.zeros((ngrid,ngrid,ngrid))

    lcell = lbox / ngrid

    for ii in prange(ngrid**3):
        xx = posx[ii]
        yy = posy[ii]
        zz = posz[ii]
        indxc = int(xx/lcell)
        indyc = int(yy/lcell)
        indzc = int(zz/lcell)

        wxc = xx/lcell - indxc
        wyc = yy/lcell - indyc
        wzc = zz/lcell - indzc

        if wxc <=0.5:
            indxl = indxc - 1
            if indxl<0:
                indxl += ngrid
            wxc += 0.5
            wxl = 1 - wxc
        elif wxc >0.5:
            indxl = indxc + 1
            if indxl>=ngrid:
                indxl -= ngrid
            wxl = wxc - 0.5
            wxc = 1 - wxl

        if wyc <=0.5:
            indyl = indyc - 1
            if indyl<0:
                indyl += ngrid
            wyc += 0.5
            wyl = 1 - wyc
        elif wyc >0.5:
            indyl = indyc + 1
            if indyl>=ngrid:
                indyl -= ngrid
            wyl = wyc - 0.5
            wyc = 1 - wyl
            

        if wzc <=0.5:
            indzl = indzc - 1
            if indzl<0:
                indzl += ngrid
            wzc += 0.5
            wzl = 1 - wzc
        elif wzc >0.5:
            indzl = indzc + 1
            if indzl>=ngrid:
                indzl -= ngrid
            wzl = wzc - 0.5
            wzc = 1 - wzl

        delta[indxc,indyc,indzc] += wxc*wyc*wzc
        delta[indxl,indyc,indzc] += wxl*wyc*wzc
        delta[indxc,indyl,indzc] += wxc*wyl*wzc
        delta[indxc,indyc,indzl] += wxc*wyc*wzl
        delta[indxl,indyl,indzc] += wxl*wyl*wzc
        delta[indxc,indyl,indzl] += wxc*wyl*wzl
        delta[indxl,indyc,indzl] += wxl*wyc*wzl
        delta[indxl,indyl,indzl] += wxl*wyl*wzl

    #delta = delta/np.mean(delta) - 1.

    return delta

# ********************************
# ********************************
def chisquare(yy):

    xx = np.reshape(yy, (4,4,int(len(yy)/16)))

    # Define the parameters to fit
    aa11, alpha11, rho11, eps11, bv11, bb11, betarsd11 = xx[0,0,0], xx[0,0,1], xx[0,0,2], xx[0,0,3], xx[0,0,4], xx[0,0,5], xx[0,0,6]
    aa12, alpha12, rho12, eps12, bv12, bb12, betarsd12 = xx[0,1,0], xx[0,1,1], xx[0,1,2], xx[0,1,3], xx[0,1,4], xx[0,1,5], xx[0,1,6]
    aa13, alpha13, rho13, eps13, bv13, bb13, betarsd13 = xx[0,2,0], xx[0,2,1], xx[0,2,2], xx[0,2,3], xx[0,2,4], xx[0,2,5], xx[0,2,6]
    aa14, alpha14, rho14, eps14, bv14, bb14, betarsd14 = xx[0,3,0], xx[0,3,1], xx[0,3,2], xx[0,3,3], xx[0,3,4], xx[0,3,5], xx[0,3,6]

    aa21, alpha21, rho21, eps21, bv21, bb21, betarsd21 = xx[1,0,0], xx[1,0,1], xx[1,0,2], xx[1,0,3], xx[1,0,4], xx[1,0,5], xx[1,0,6]
    aa22, alpha22, rho22, eps22, bv22, bb22, betarsd22 = xx[1,1,0], xx[1,1,1], xx[1,1,2], xx[1,1,3], xx[1,1,4], xx[1,1,5], xx[1,1,6]
    aa23, alpha23, rho23, eps23, bv23, bb23, betarsd23 = xx[1,2,0], xx[1,2,1], xx[1,2,2], xx[1,2,3], xx[1,2,4], xx[1,2,5], xx[1,2,6]
    aa24, alpha24, rho24, eps24, bv24, bb24, betarsd24 = xx[1,3,0], xx[1,3,1], xx[1,3,2], xx[1,3,3], xx[1,3,4], xx[1,3,5], xx[1,3,6]

    aa31, alpha31, rho31, eps31, bv31, bb31, betarsd31 = xx[2,0,0], xx[2,0,1], xx[2,0,2], xx[2,0,3], xx[2,0,4], xx[2,0,5], xx[2,0,6]
    aa32, alpha32, rho32, eps32, bv32, bb32, betarsd32 = xx[2,1,0], xx[2,1,1], xx[2,1,2], xx[2,1,3], xx[2,1,4], xx[2,1,5], xx[2,1,6]
    aa33, alpha33, rho33, eps33, bv33, bb33, betarsd33 = xx[2,2,0], xx[2,2,1], xx[2,2,2], xx[2,2,3], xx[2,2,4], xx[2,2,5], xx[2,2,6]
    aa34, alpha34, rho34, eps34, bv34, bb34, betarsd34 = xx[2,3,0], xx[2,3,1], xx[2,3,2], xx[2,3,3], xx[2,3,4], xx[2,3,5], xx[2,3,6]

    aa41, alpha41, rho41, eps41, bv41, bb41, betarsd41 = xx[3,0,0], xx[3,0,1], xx[3,0,2], xx[3,0,3], xx[3,0,4], xx[3,0,5], xx[3,0,6]
    aa42, alpha42, rho42, eps42, bv42, bb42, betarsd42 = xx[3,1,0], xx[3,1,1], xx[3,1,2], xx[3,1,3], xx[3,1,4], xx[3,1,5], xx[3,1,6]
    aa43, alpha43, rho43, eps43, bv43, bb43, betarsd43 = xx[3,2,0], xx[3,2,1], xx[3,2,2], xx[3,2,3], xx[3,2,4], xx[3,2,5], xx[3,2,6]
    aa44, alpha44, rho44, eps44, bv44, bb44, betarsd44 = xx[3,3,0], xx[3,3,1], xx[3,3,2], xx[3,3,3], xx[3,3,4], xx[3,3,5], xx[3,3,6]

    posxnew, posynew, posznew = real_to_redshift_space(delta, vz, ngrid, lbox, xx)

    # Periodic BCs
    posxnew[posxnew<0.] += lbox
    posxnew[posxnew>=lbox] -= lbox
    posynew[posynew<0.] += lbox
    posynew[posynew>=lbox] -= lbox
    posznew[posznew<0.] += lbox
    posznew[posznew>=lbox] -= lbox

    delta_new = get_cic(posxnew, posynew, posznew, lbox, ngrid)
    delta_new = delta_new/np.mean(delta_new) - 1.

    flux_new = biasmodel(ngrid, lbox, delta_new, tweb, twebdelta, xx)

    kk, pk_l0, pk_l2, pk_l4 = measure_spectrum(flux_new)

    chisq_l0 = np.sum((pk_l0[np.where(kk<kkth_l0)]/pkref_l0[np.where(kk<kkth_l0)] - 1.)**2)/len(pk_l0[np.where(kk<kkth_l0)])
    chisq_l2 = np.sum((pk_l2[np.where(kk<kkth_l2)]/pkref_l2[np.where(kk<kkth_l2)] - 1.)**2)/len(pk_l2[np.where(kk<kkth_l2)])

    chisq = weight_l0*chisq_l0 + weight_l2*chisq_l2

    if verbose_parameters == True:
        print('PARS: ', xx)
        print('Monopole ratios (%): ', (pk_l0[np.where(kk<kkth_l0)]/pkref_l0[np.where(kk<kkth_l0)] - 1.)*100.)
        #print('Quadrupole ratios (%): ', (pk_l2[np.where(kk<kkth_l2)]/pkref_l2[np.where(kk<kkth_l2)] - 1.)*100.)
        print('----------------------------------------------')


    if plot_pk == True:
        plt.plot(kk, pkref_l0/pkref_l0, color='red', label='Ref')
        plt.plot(kk, pk_l0/pkref_l0, color='blue', linestyle='dashed', label='Mock l=0')
        plt.plot(kk, pk_l0/pkref_l0, color='green', linestyle='dashdot', label='Mock l=2')
        plt.fill_between(kk, 0.99*np.ones(len(kk)), 1.01*np.ones(len(kk)), color='gray', alpha=0.7)
        plt.fill_between(kk, 0.98*np.ones(len(kk)), 1.02*np.ones(len(kk)), color='gray', alpha=0.5)
        plt.fill_between(kk, 0.95*np.ones(len(kk)), 1.05*np.ones(len(kk)), color='gray', alpha=0.3)
        plt.fill_between(kk, 0.9*np.ones(len(kk)), 1.1*np.ones(len(kk)), color='gray', alpha=0.2)
        plt.xlabel('k')
        plt.ylabel('P(k) ratios')
        plt.xscale('log')
        #plt.yscale('log')
        plt.ylim([0.5, 1.5])
        plt.legend()
        plt.savefig('pk_ratios_flux_zspace_z%s.pdf' %(zstring), bbox_inches='tight')
        plt.show()

    return chisq

# ********************************
# ********************************
@njit(parallel=True, cache=True)
def gradfindiff(lbox,ngrid,arr,dim):

    fac = ngrid/(2*lbox)

    outarr = arr.copy()

    for xx in prange(ngrid):
        for yy in prange(ngrid):
            for zz in prange(ngrid):

                xdummy = np.array([xx,xx,xx,xx])
                ydummy = np.array([yy,yy,yy,yy])
                zdummy = np.array([zz,zz,zz,zz])
                xxr = xdummy[0]
                xxrr = xdummy[1]
                xxl = xdummy[2]
                xxll = xdummy[3]
                yyr = ydummy[0]
                yyrr = ydummy[1]
                yyl = ydummy[2]
                yyll = ydummy[3]
                zzr = zdummy[0]
                zzrr = zdummy[1]
                zzl = zdummy[2]
                zzll = zdummy[3]

                # Periodic BCs
                if dim == 1:
                    xxl = xx - 1
                    xxll = xx - 2
                    xxr = xx + 1
                    xxrr = xx + 2
                    
                    if xxl<0:
                        xxl += ngrid
                    if xxl>=ngrid:
                        xxl -= ngrid
                    
                    if xxll<0:
                        xxll += ngrid
                    if xxll>=ngrid:
                        xxll -= ngrid
                    
                    if xxr<0:
                        xxr += ngrid
                    if xxr>=ngrid:
                        xxr -= ngrid

                    if xxrr<0:
                        xxrr += ngrid
                    if xxrr>=ngrid:
                        xxrr -= ngrid


                elif dim == 2:
                    
                    yyl = yy - 1
                    yyll = yy - 2
                    yyr = yy + 1
                    yyrr = yy + 2
                    
                    if yyl<0:
                        yyl += ngrid
                    if yyl>=ngrid:
                        yyl -= ngrid
                    
                    if yyll<0:
                        yyll += ngrid
                    if yyll>=ngrid:
                        yyll -= ngrid
                    
                    if yyr<0:
                        yyr += ngrid
                    if yyr>=ngrid:
                        yyr -= ngrid

                    if yyrr<0:
                        yyrr += ngrid
                    if yyrr>=ngrid:
                        yyrr -= ngrid


                elif dim == 3:
                    
                    zzl = zz - 1
                    zzll = zz - 2
                    zzr = zz + 1
                    zzrr = zz + 2
                    
                    if zzl<0:
                        zzl += ngrid
                    if zzl>=ngrid:
                        zzl -= ngrid
                    
                    if zzll<0:
                        zzll += ngrid
                    if zzll>=ngrid:
                        zzll -= ngrid
                    
                    if zzr<0:
                        zzr += ngrid
                    if zzr>=ngrid:
                        zzr -= ngrid

                    if zzrr<0:
                        zzrr += ngrid
                    if zzrr>=ngrid:
                        zzrr -= ngrid

                outarr[xx,yy,zz] = -fac*((4.0/3.0)*(arr[xxl,yyl,zzl]-arr[xxr,yyr,zzr])-(1.0/6.0)*(arr[xxll,yyll,zzll]-arr[xxrr,yyrr,zzrr]))

    return outarr

# ********************************
# **********************************************
@njit(parallel=True, cache=True)
def get_tidal_invariants(arr, ngrid, lbox):

    # Get gradients exploiting simmetry of the tensor, i.e. gradxy=gradyx

    # X DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,1)
    #2nd derivs
    gradxx = gradfindiff(lbox,ngrid,grad,1)
    gradxy = gradfindiff(lbox,ngrid,grad,2)
    gradxz = gradfindiff(lbox,ngrid,grad,3)

    # Y DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,2)
    #2nd derivs
    gradyy = gradfindiff(lbox,ngrid,grad,2)
    gradyz = gradfindiff(lbox,ngrid,grad,3)

    # Y DIRECTION
    # 1st deriv
    grad = gradfindiff(lbox,ngrid,arr,3)
    #2nd derivs
    gradzz = gradfindiff(lbox,ngrid,grad,3)

    #del arr, grad

    lambda1 = np.zeros_like((gradxx))
    lambda2 = np.zeros_like((gradxx))
    lambda3 = np.zeros_like((gradxx))
    tweb = np.zeros_like((gradxx))

    # Compute eigenvalues    
    for ii in prange(ngrid):
        for jj in prange(ngrid):
            for kk in prange(ngrid):
                mat = np.array([[gradxx[ii,jj,kk],gradxy[ii,jj,kk],gradxz[ii,jj,kk]],[gradxy[ii,jj,kk],gradyy[ii,jj,kk],gradyz[ii,jj,kk]],[gradxz[ii,jj,kk],gradyz[ii,jj,kk],gradzz[ii,jj,kk]]])
                eigs = np.linalg.eigvals(mat)
                eigs = np.flip(np.sort(eigs))
                lambda1[ii,jj,kk] = eigs[0]
                lambda2[ii,jj,kk] = eigs[1]
                lambda3[ii,jj,kk] = eigs[2]
                if eigs[0]>=lambdath and eigs[1]>=lambdath and eigs[2]>=lambdath:
                    tweb[ii,jj,kk] = 1
                elif eigs[0]>=lambdath and eigs[1]>=lambdath and eigs[2]<lambdath:
                    tweb[ii,jj,kk] = 2
                elif eigs[0]>=lambdath and eigs[1]<lambdath and eigs[2]<lambdath:
                    tweb[ii,jj,kk] = 3
                elif eigs[0]<lambdath and eigs[1]<lambdath and eigs[2]<lambdath:
                    tweb[ii,jj,kk] = 4

    # Now compute invariants
    #del gradxx, gradxy, gradxz, gradyy,gradyz,gradzz
    
    #I1 = lambda1 + lambda2 + lambda3
    #I2 = lambda1 * lambda2 + lambda1 * lambda3 + lambda2 * lambda3
    #I3 = lambda2 * lambda2 * lambda3

    #del lambda1, lambda2, lambda3

    return tweb

# ********************************
# ********************************

parslist = []

bestfitpars = np.load(inpars_filename)

npars = int(len(bestfitpars))

boundslist = []

for ii in range(len(bestfitpars)):

    bnd = ((1-bounds_range)*bestfitpars[ii], (1+bounds_range)*bestfitpars[ii])
    boundslist.append(bnd)

bounds = np.array(boundslist)#, dth_bounds, rhoeps_bounds, eps_bounds])
#bestfit = np.array([1., 1., 1., 1., 1., 1., 1. ]) # z=3.0

bestfitpars = np.reshape(bestfitpars, (4,4,int(len(bestfitpars)/16)))

if fit==True:
    plot_pk = False
else:
    plot_pk = True

delta = np.fromfile(dm_filename, dtype=prec)
delta = delta/np.mean(delta) - 1.
delta = np.reshape(delta, (ngrid,ngrid,ngrid))

vz = np.fromfile(vz_filename, dtype=prec)
vz = np.reshape(vz, (ngrid,ngrid,ngrid))

fluxref = np.fromfile(flux_filename, dtype=prec)
fluxref = np.reshape(fluxref, (ngrid,ngrid,ngrid))

# Do T-web and Tweb-delta computaiton just once
# Solve Poisson equation in real space
print('Solving Poisson equation ...')
phi = poisson_solver(delta,ngrid, lbox) 

# Compute T-web in real space
print('Computing invariants of tidal field ...')
tweb = get_tidal_invariants(phi, ngrid, lbox)
twebdelta = get_tidal_invariants(delta, ngrid, lbox) # Now also the T-web is in redshift space

# Map density field from real to redshift space
posx, posy, posz = real_to_redshift_space(delta, vz, ngrid, lbox, bestfitpars) # Now delta is in redshift space
# Periodic BCs
posz[posz<0.] += lbox
posz[posz>=lbox] -= lbox

delta = get_cic(posx, posy, posz, lbox, ngrid)
delta = delta/np.mean(delta) - 1.
print(delta.shape)

# Solve Poisson equation in redshift space
print('Solving Poisson equation ...')
phi = poisson_solver(delta,ngrid, lbox) 

# Compute T-web in redshift space
print('Computing invariants of tidal field ...')
tweb = get_tidal_invariants(phi, ngrid, lbox) # Now also the T-web is in redshift space
twebdelta = get_tidal_invariants(delta, ngrid, lbox) # Now also the T-web is in redshift space
        
        
meandens = np.sum(fluxref)/lbox**3
        
kkref, pkref_l0, pkref_l2, pkref_l4 = measure_spectrum(fluxref)

if fit == True:
    print('=========================')
    print('Fitting ...')
    # Fit
    algorithm_param = {'max_num_iteration': 100,
                   'population_size':100,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':10}
            
    model=ga(function=chisquare,dimension=npars,variable_type='real',variable_boundaries=bounds, algorithm_parameters=algorithm_param)
    model.run()
    convergence=model.report
    solution=model.output_dict
            
    x0new = solution['variable']

    #ncounts_new = biasmodel(ngrid, lbox, delta, tweb, twebdelta, x0new)
    parslist = x0new.flatten()

    np.save(outpars_filename, parslist)

else:

    x0new = np.load(outpars_filename)
    x0new = np.reshape(x0new,(4,4,int(len(x0new)/16)))

    chisquare(x0new)
