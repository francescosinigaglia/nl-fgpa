import numpy as np
from numba import njit, prange

# **********************************************
# **********************************************
# **********************************************
# INPUT PARAMETERS

# General parameters
flux_filename = '...' # Redshift space

dm_filename =  '...'  # Real space
vel_filename = '...'  # Real space

out_filename = '...'  # Output filename

bias_pars_filename = '...' # bias model parameters file
rsd_pars_filename = '...' # RSD model parameters file

kk_out_filename = '...' # k wavenumbers output filename for P(k)
pk_out_filename = '...' # amplitude output filename for P(k)

lbox = 500
ngrid = 256 

redshift = 2.

# Cosmological parameters
h = 0.6774
H0 = 100
Om = 0.3089
Orad = 0.
Ok = 0.
N_eff = 3.046
w_eos = -1
Ol = 1-Om-Ok-Orad
aa = 1./(1.+redshift)
num_part_per_cell = 8

# NL FGPA parameters                                                                                                                                                                                                                          
aa1 = 0.23                                                                                                                           
aa2 = 0.1                                                                                                                
aa3 = 0.2                                                                                                                      
aa4 = 0.15
alpha1 = 4.                                                                                                                          
alpha2 = 4.                                                                                                                          
alpha3 = 2.2                                                                                                                         
alpha4 = 2.03                                                                                                                        
                                                             
delta11 = 1.36                                                                                                                       
delta12 = 0.41                                                                                                                       
delta21 = 0.9                                                                                                                        
delta22 = 0.13                                                                                                           
delta31 = 0.85                                                                                                                       
delta32 = 0.6                                                                                                                   
delta41 = 1.0                                                                                                                        
delta42 = 0.25                                                                                                                   

norm = 1.4
nn = 0.1
pp = 0.6

lambdath = 0.1 

# Random seed for stochasticity reproducibility
np.random.seed(123456)

# **********************************************
# **********************************************
# **********************************************
def fftr2c(arr):
    arr = np.fft.rfftn(arr, norm='ortho')

    return arr

def fftc2r(arr):
    arr = np.fft.irfftn(arr, norm='ortho')

    return arr

# **********************************************
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

# **********************************************                                                                                    
def cross_spectrum(signal1, signal2):

    nbin = round(ngrid/2)

    fsignal1 = np.fft.fftn(signal1) #np.fft.fftn(signal)                                             
    fsignal2 = np.fft.fftn(signal2) #np.fft.fftn(signal)                                             

    kmax = np.pi * ngrid / lbox #np.sqrt(k_squared(L,nc,nc/2,nc/2,nc/2))            
    dk = kmax/nbin  # Bin width                                                                                                    

    nmode = np.zeros((nbin))
    kmode = np.zeros((nbin))
    power = np.zeros((nbin))

    kmode, power, nmode = get_cross_power(fsignal1, fsignal2, nbin, kmax, dk, kmode, power, nmode)

    return kmode[1:], power[1:]

# **********************************************                                                                                        
def compute_cross_correlation_coefficient(cross, power1,power2):
    ck = cross/(np.sqrt(power1*power2))
    return ck

# **********************************************                                                                                         
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

# **********************************************                                                                            
@njit(parallel=False, cache=True)
def get_cross_power(fsignal1, fsignal2, Nbin, kmax, dk, kmode, power, nmode):

    for i in prange(ngrid):
        for j in prange(ngrid):
            for k in prange(ngrid):
                ktot = np.sqrt(k_squared_nohermite(lbox,ngrid,i,j,k))
                if ktot <= kmax:
                    nbin = int(ktot/dk-0.5)
                    akl1 = fsignal1.real[i,j,k]
                    bkl1 = fsignal1.imag[i,j,k]
                    akl2 = fsignal2.real[i,j,k]
                    bkl2 = fsignal2.imag[i,j,k]
                    kmode[nbin]+=ktot
                    power[nbin]+=(akl1*akl2+bkl1*bkl2)
                    nmode[nbin]+=1

    for m in prange(Nbin):
        if(nmode[m]>0):
            kmode[m]/=nmode[m]
            power[m]/=nmode[m]

    power = power / (ngrid/2)**3

    return kmode, power, nmode
  
# **********************************************
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

    delta = delta/np.mean(delta) - 1.

    return delta

# **********************************************
# Real to redshift space mapping
#@njit(parallel=False, cache=True)
def real_to_redshift_space(delta, vel, tweb, ngrid, lbox, rsd_pars) :

    H = H0*h*np.sqrt(Om*(1+redshift)**3 + Ol)

    xx = np.repeat(np.arange(ngrid), ngrid**2*num_part_per_cell)*lcell
    yy = np.tile(np.repeat(np.arange(ngrid), ngrid *num_part_per_cell), ngrid)*lcell

    B1 = rsd_pars[0,0]
    b1 = rsd_pars[0,1]
    beta1 = rsd_pars[0,2]

    B2 = rsd_pars[1,0]
    b2 = rsd_pars[1,1]
    beta2 = rsd_pars[1,2]

    B3 = rsd_pars[2,0]
    b3 = rsd_pars[2,1]
    beta3 = rsd_pars[2,2]

    B4 = rsd_pars[3,0]
    b4 = rsd_pars[3,1]
    beta4 = rsd_pars[3,2]

    #RSD model
    velarr = np.repeat(vel,num_part_per_cell)

    # Coherent flows
    velbias = 0.*delta.copy()
    velbias[np.where(tweb==1)] = B1
    velbias[np.where(tweb==2)] = B2
    velbias[np.where(tweb==3)] = B3
    velbias[np.where(tweb==4)] = B4
    velbias = np.repeat(velbias, num_part_per_cell) 

    # Quasi-virialized motions
    sigma = 0.*delta.copy()
    sigma[np.where(tweb==1)] = b1*(1. + delta[np.where(tweb==1)])**beta1
    sigma[np.where(tweb==2)] = b2*(1. + delta[np.where(tweb==2)])**beta2
    sigma[np.where(tweb==3)] = b3*(1. + delta[np.where(tweb==3)])**beta3
    sigma[np.where(tweb==4)] = b4*(1. + delta[np.where(tweb==4)])**beta4
    sigma = np.repeat(sigma, num_part_per_cell)

    rand = np.random.normal(0,sigma)
    velarr = velarr + rand

    zz = np.tile(np.repeat(np.arange(ngrid), num_part_per_cell), ngrid**2)*lcell + velbias*velarr/(aa*H)

    # Periodic boundaries:
    zz[np.where(zz>lbox)] = zz[np.where(zz>lbox)] - lbox
    zz[np.where(zz<0)] = zz[np.where(zz<0)] + lbox

    delta = get_cic(xx, yy, zz, lbox, ngrid)

    return delta


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
    

# **********************************************
#@njit(parallel=True, cache=True)
def poisson_solver(delta, ngrid, lbox):

    delta = fftr2c(delta)

    delta = divide_by_k2(delta, ngrid, lbox)
    
    delta[0,0,0] = 0.

    delta = fftc2r(delta)

    return delta

# **********************************************
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

# **********************************************
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

# **********************************************
# **********************************************
# **********************************************

bias_pars = np.load(bias_pars_filename)
rsd_pars = np.load(rsd_pars_filename)

lcell = lbox/ngrid

aa = 1./(1.+redshift)
HH = 100.

# Read input arrays
flux = np.fromfile(open(flux_filename, 'r'), dtype=np.float64) # In redshift space
delta = np.fromfile(open(dm_filename, 'r'), dtype=np.float64)  # In real space
vel = np.fromfile(open(vel_filename, 'r'), dtype=np.float64)   # In real space

# Convert DM field to overdensity
delta = delta/np.mean(delta) - 1.
delta = np.reshape(delta, (ngrid,ngrid,ngrid))

# Solve Poisson equation in real space
print('Solving Poisson equation ...')
phi = poisson_solver(delta,ngrid, lbox) 

# Compute T-web in real space
print('Computing invariants of tidal field ...')
tweb = get_tidal_invariants(delta, ngrid, lbox)

# Map density field from real to redshift space
delta = real_to_redshift_space(delta, vel, tweb, ngrid, lbox) # Now delta is in redshift space

# Solve Poisson equation in redshift space
print('Solving Poisson equation ...')
phi = poisson_solver(delta,ngrid, lbox) 

# Compute T-web in redshift space
print('Computing invariants of tidal field ...')
tweb = get_tidal_invariants(phi, ngrid, lbox) # Now also the T-web is in redshift space
dweb = get_tidal_invariants(delta, ngrid, lbox) # Now also the T-web is in redshift space


print('===================')
print('T-web diagnostics, volume filling factors:')

print('')
print('REDSHIFT SPACE:')
print('Knots: ', 100*len(tweb[np.where(tweb==1)])/len(tweb))
print('Filaments: ', 100*len(tweb[np.where(tweb==2)])/len(tweb))
print('Sheets: ', 100*len(tweb[np.where(tweb==3)])/len(tweb))
print('Voids: ', 100*len(tweb[np.where(tweb==4)])/len(tweb))
print('===================')

# Flatten delta array
delta = delta.flatten()

# Compute NL FGPA
tau = 0. * delta.copy()
tau = tau.flatten()

delta[np.where(tweb==4)] += norm*np.random.negative_binomial(nn, pp, size=len(delta[np.where(tweb==4)]))

tau[np.where(tweb==1)] = aa1 * (1+delta[np.where(tweb==1)])**alpha1 * np.exp(-delta[np.where(tweb==1)]/delta11) * np.exp(delta[np.where(tweb==1)]/delta12)
tau[np.where(tweb==2)] = aa2 * (1+delta[np.where(tweb==2)])**alpha2 * np.exp(-delta[np.where(tweb==2)]/delta21) * np.exp(delta[np.where(tweb==2)]/delta22)
tau[np.where(tweb==3)] = aa3 * (1+delta[np.where(tweb==3)])**alpha3 * np.exp(-delta[np.where(tweb==3)]/delta31) * np.exp(delta[np.where(tweb==3)]/delta32)
tau[np.where(tweb==4)] = aa4 * (1+delta[np.where(tweb==4)])**alpha4 * np.exp(-delta[np.where(tweb==4)]/delta41) * np.exp(delta[np.where(tweb==4)]/delta42) #+ norm*np.random.negative_binomial(nn, pp, size=len(delta[np.where(tweb==4)]))
tau[np.where(tau<0)] = 0.
flux_new = np.exp(-tau)
flux_new = np.reshape(flux_new, (ngrid,ngrid,ngrid))

(flux_new.flatten()).astype('float32').tofile(out_filename)

# Measure power spectrum
kk, pk = measure_spectrum(flux_new)

np.save(kk_out_filename, kk)
np.save(pk_out_filename, pk)