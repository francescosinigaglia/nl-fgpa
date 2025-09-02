import numpy as np
from numba import njit, prange, int64
import numba as nb
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt

np.random.seed(123456)

redshift = 1.43
zstring = '1.430'

ngrid = 360
lbox = 2000.

dm_filename = '../../UCHUU/ALPT_DM_fields/deltaBOXOM0.314OL0.686G360V2000.0_ALPTrs6.000z%s.dat' %zstring
tweb_filename = '../../UCHUU/ALPT_DM_fields/Tweb_OM0.314OL0.686G360V2000.0_ALPTrs6.000z%s.dat' %zstring
twebdelta_filename = '../../UCHUU/ALPT_DM_fields/TwebDelta_OM0.314OL0.686G360V2000.0_ALPTrs6.000z%s.dat' %zstring 
gal_filename = 'Uchuu_NGP_real_space_z%s.dat' %zstring

out_filename = 'UCHUU_mock_BGS_NGP_real_space_z%s.DAT' %zstring
outpars_filename = 'bias_parameters_z%s.npy' %zstring 

twebenvs = [1,2,3,4]
twebdeltaenvs = [1,2,3,4]

verbose_parameters = False
kkth = 0.5
npars = 2

alpha_bounds = (0.01, 3.)
beta_bounds = (0.1, 100.)
dth_bounds = (-1, 0.5)
rhoeps_bounds = (0.3, 20.)
eps_bounds = (0.1, 3.)

bounds = np.array([alpha_bounds, beta_bounds])#, dth_bounds, rhoeps_bounds, eps_bounds])
bestfit = np.array([1.64833716, 1.64806699]) # z=3.0

fit = True

parslist = []

# ***********************************
# ***********************************
# ***********************************
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

# ***********************************
# ***********************************

@njit(parallel=False, cache=True)
def get_power(fsignal, Nbin, kmax, dk, kmode, power, nmode):
    
    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl = fsignal.real[i,j,k]
                    bkl = fsignal.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl*akl+bkl*bkl)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode

# ***********************************
# ***********************************
def measure_spectrum(signal):

    nbin = round(ngrid/2)
    
    fsignal = np.fft.fftn(signal) #np.fft.fftn(signal)

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))
    dk = kmax/nbin  # Bin width

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_power(fsignal, nbin, kmax, dk, kmode, power, nmode)
    
    return kmode[1:], power[1:]

# ***********************************
# ***********************************

@njit(parallel=True, cache=True, fastmath=True)
def biasmodel_local(ngrid, lbox, delta, tweb, twebdelta, nmean, alpha, beta, dth, rhoeps, eps, rhoepsprime, epsprime, twebenv, twebdeltaenv):
     
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
def chisquare(xx):

    # Define the parameters to fit
    alpha = xx[0]
    beta = xx[1]
    dth = -1#xx[2]
    rhoeps = 1e6#xx[3]
    eps = 1.#xx[4]
    rhoepsprime = 1.
    epsprime = 1.
    
    ncounts_new, normalization = biasmodel_local(ngrid, lbox, delta, tweb, twebdelta, meandens, alpha, beta, dth, rhoeps, eps, rhoepsprime, epsprime, twebenv, twebdeltaenv)
    if plot_pk == True:
          print('Maximum number of galaxies in cells: ', np.amax(ncounts_new))
          print('Total number of galaxies: ', np.sum(ncounts_new))

    #ncounts_new = ncounts_new / np.mean(ncounts_new) - 1
    #print(len(pk_l0[np.where(kk<kkth_l0)),pk_l0[np.where(kk<kkth_l0))
    
    ncounts_new_mask = np.zeros((ngrid,ngrid,ngrid))
    ncounts_new_mask[np.logical_and(tweb==twebenv,twebdelta==twebdeltaenv)] = ncounts_new[np.logical_and(tweb==twebenv,twebdelta==twebdeltaenv)]

    kk, pk = measure_spectrum(ncounts_new_mask)

    chisq = np.sum((pk[np.where(kk<kkth)]/pkgal[np.where(kk<kkth)] - 1.)**2)/len(pk[np.where(kk<kkth)])

    #chisq = weight_l0*chisq_l0 + weight_l2*chisq_l2

    if verbose_parameters == True:
        print('PARS: ', xx)
        print('Monopole ratios (%): ', (pk[np.where(kk<kkth)]/pkgal[np.where(kk<kkth)] - 1.)*100.)
        #print('Quadrupole ratios (%): ', (pk_l2[np.where(kk<kkth_l2)]/pkref_l2[np.where(kk<kkth_l2)] - 1.)*100.)
        print('----------------------------------------------')


    if plot_pk == True:
        plt.plot(kk, pkgal/pkgal, color='red', label='Ref')
        plt.plot(kk, pk/pkgal, color='green', linestyle='dashed', label='Mock')
        plt.fill_between(kk, 0.99*np.ones(len(kk)), 1.01*np.ones(len(kk)), color='gray', alpha=0.7)
        plt.fill_between(kk, 0.98*np.ones(len(kk)), 1.02*np.ones(len(kk)), color='gray', alpha=0.5)
        plt.fill_between(kk, 0.95*np.ones(len(kk)), 1.05*np.ones(len(kk)), color='gray', alpha=0.3)
        plt.fill_between(kk, 0.9*np.ones(len(kk)), 1.1*np.ones(len(kk)), color='gray', alpha=0.2)
        plt.xlabel('k')
        plt.ylabel('P(k) ratios')
        plt.xscale('log')
        #plt.yscale('log')
        #plt.ylim([0.85, 1.15])
        plt.legend()
        plt.savefig('pk_ratios_ngp_rspace_z%s.pdf' %zstring, bbox_inches='tight')
        plt.show()

    return chisq


# ********************************
# ********************************


if fit==True:
    plot_pk = False
else:
    plot_pk = True

delta = np.fromfile(dm_filename, dtype=np.float32)
delta = np.reshape(delta, (ngrid,ngrid,ngrid))

tweb = np.fromfile(tweb_filename, dtype=np.float32)
tweb = np.reshape(tweb, (ngrid,ngrid,ngrid))

twebdelta = np.fromfile(twebdelta_filename, dtype=np.float32)
twebdelta = np.reshape(twebdelta, (ngrid,ngrid,ngrid))

print(np.amin(delta), np.amax(delta))

ncounts = np.fromfile(gal_filename, dtype=np.float32)
ncounts = np.reshape(ncounts, (ngrid,ngrid,ngrid))

for twebenv in twebenvs:
    for twebdeltaenv in twebdeltaenvs:          
        
        meandens = np.sum(ncounts[np.logical_and(tweb==twebenv,twebdelta==twebdeltaenv)])/lbox**3

        print('=========================')
        print('Fitting %d %d ...' %(twebenv, twebdeltaenv))
        #print(np.amin(ncounts), np.amax(ncounts))
        #ncounts = ncounts / np.mean(ncounts) - 1
        
        ncounts_mask = np.zeros((ngrid,ngrid,ngrid))
        ncounts_mask[np.logical_and(tweb==twebenv,twebdelta==twebdeltaenv)] = ncounts[np.logical_and(tweb==twebenv,twebdelta==twebdeltaenv)]
        
        kkgal, pkgal = measure_spectrum(ncounts_mask)
        
        # Fit
        print('Fitting ...')
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

        alpha = x0new[0]
        beta = x0new[1]
        dth = -1#x0new[2]
        rhoeps = 1e6 #x0new[3]
        eps = 1.#x0new[4]
        rhoepsprime = 1e6 #x0new[3]
        epsprime = 1.#x0new[4]

        ncounts_new, normalization = biasmodel_local(ngrid, lbox, delta, tweb, twebdelta, meandens, alpha, beta, dth, rhoeps, eps, rhoepsprime, epsprime, twebenv, twebdeltaenv)

        parslist.append(normalization)
        parslist.append(alpha)
        parslist.append(beta)

      
np.save(outpars_filename, parslist)
