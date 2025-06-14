#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("./src/")
import numpy as np
import matplotlib.pyplot as plt
from diag_rb import rb_open, rb_get_tri_filelist
from diag_geom import geom_set

### Read NetCDF data phi.*.nc or Zarr store gkvp.phi.*.zarr/ by xarray ###
FILETYPE="diag_nc"
# FILETYPE="GKV_nc"
# FILETYPE="GKV_zarr"
if FILETYPE=="diag_nc":
    xr_phi = rb_open('../post/data/phi.*.nc')
    xr_Al  = rb_open('../post/data/Al.*.nc')
    xr_mom = rb_open('../post/data/mom.*.nc')
    xr_fxv = rb_open('../post/data/fxv.*.nc')
    xr_cnt = rb_open('../post/data/cnt.*.nc')
    xr_trn = rb_open('../post/data/trn.*.nc')
    tri_filelist = rb_get_tri_filelist('../post/data/tri.*.nc')
    xr_tri_list=[]
    for file in tri_filelist:
        xr_tri=rb_open(file + '.*.nc')
        xr_tri_list.append(xr_tri)
elif FILETYPE=="GKV_nc":
    xr_phi = rb_open('../phi/gkvp.phi.*.nc')
    xr_Al  = rb_open('../phi/gkvp.Al.*.nc')
    xr_mom = rb_open('../phi/gkvp.mom.*.nc')
    xr_fxv = rb_open('../fxv/gkvp.fxv.*.nc')
    xr_cnt = rb_open('../cnt/gkvp.cnt.*.nc')
    xr_trn = rb_open('../phi/gkvp.trn.*.nc')
    tri_filelist = rb_get_tri_filelist('../phi/gkvp.tri.*.nc')
    xr_tri_list=[]
    for file in tri_filelist:
        xr_tri=rb_open(file + '.*.nc')
        xr_tri_list.append(xr_tri)
elif FILETYPE=="GKV_zarr":
    xr_phi = rb_open('../phi/gkvp.phi.*.zarr/')
    xr_Al  = rb_open('../phi/gkvp.Al.*.zarr/')
    xr_mom = rb_open('../phi/gkvp.mom.*.zarr/')
    xr_fxv = rb_open('../fxv/gkvp.fxv.*.zarr/')
    xr_cnt = rb_open('../cnt/gkvp.cnt.*.zarr/')
    xr_trn = rb_open('../phi/gkvp.trn.*.zarr/')
    tri_filelist = rb_get_tri_filelist('../phi/gkvp.tri.*.zarr/')
    xr_tri_list=[]
    for file in tri_filelist:
        xr_tri=rb_open(file + '.*.zarr/')
        xr_tri_list.append(xr_tri)

print("xr_phi:", xr_phi)
print("tri_filelist:", tri_filelist)

### Set geometric constants ###
geom_set(headpath='../src/gkvp_header.f90', nmlpath="../gkvp_namelist.001", mtrpath='../hst/gkvp.mtr.001')


# In[ ]:


from out_mominxy import phiinxy, Alinxy, mominxy
# Plot phi[y,x] at t[it], zz[iz]
it = 3
iz = 8
phiinxy(it, iz, xr_phi, flag="display")
Alinxy(it, iz, xr_Al, flag="display")
it = 3
iss = 0
imom = 2
iz = 8
mominxy(it, iss, imom, iz, xr_mom, flag="display")


# In[ ]:


from out_mominkxky import phiinkxky, Alinkxky, mominkxky
# Plot 0.5*<|phi|^2>[ky,kx] at t[it]
it = 3
phiinkxky(it, xr_phi, flag="display")
Alinkxky(it, xr_Al, flag="display")
it = 3
iss = 0
imom = 2
mominkxky(it, iss, imom, xr_mom, flag="display")


# In[ ]:


from out_mominrz import phiinrz
# Plot phi in cylindrical (R,Z) at t[it] and zeta
it = len(xr_phi['t'])-1
phiinrz(it, xr_phi, flag="display")


# In[ ]:


from out_mominvtk import phiinvtk
# Plot 3D phi in VTK file format
it = len(xr_phi['t'])-1
phiinvtk(it, xr_phi, flag="flux_tube", n_alp=4)
phiinvtk(it, xr_phi, flag="full_torus", n_alp=4)
phiinvtk(it, xr_phi, flag="field_aligned", n_alp=4)


# In[ ]:


from out_mominxmf import phiinxmf
# Plot 3D phi in XMF file format
it = len(xr_phi['t'])-1
phiinxmf(it, xr_phi, flag="flux_tube_coord", n_alp=4)
phiinxmf(it, xr_phi, flag="flux_tube_var",   n_alp=4)
phiinxmf(it, xr_phi, flag="full_torus_coord", n_alp=4)
phiinxmf(it, xr_phi, flag="full_torus_var",   n_alp=4)


# In[ ]:


from out_mominz import phiinz, phiinz_connect
# Plot phi along field line z at t[it], ky[my], kx[mx]
it = len(xr_phi['t'])-1
my = 1
mx = int((len(xr_phi.kx)-1)/2)
phiinz_connect(it, my, mx, xr_phi, xr_Al, normalize='phi0', flag='display')


# In[ ]:


from out_trninkxky import trninkxky
# Plot trn[ky,kx] at t[it], s[iss].
it = len(xr_trn['t'])-1
iss = 0 # Index of species
itrn = 10 # Index of outputs in trn.*.nc, see help(trninkxky)
trninkxky(it, iss, itrn, xr_trn, flag='display')


# In[ ]:


from out_ffinvm import fluxinvm_fxv, fluxinvm_cnt
# Plot flux_es[mu,vl] at at t[it], s[iss], zz[rankz] using fxv.*.nc or cnt.*.nc
### fluxinv_cnt ###
it = len(xr_fxv.t)-1
iss = 0 # Index of species
rankz = int(len(xr_fxv.zz) / 2)
fluxinvm_fxv(it, iss, rankz, xr_phi, xr_fxv, flag="display")

### fluxinvm_cnt ###
it = len(xr_cnt.t)-1
iss = 0 # Index of species
iz = int(len(xr_cnt.zz) / 2)
fluxinvm_cnt(it, iss, iz, xr_phi, xr_cnt, flag="display")


# In[ ]:


from out_dmd import phidmd
# DMD(Dynamic Mode Decomposition) analysis
my = int((len(xr_phi.ky)-1)/2)
mx = int((len(xr_phi.kx)-1)/2)
phidmd(xr_phi, my=my, mx=mx, flag='display')


# In[ ]:


# Examples of advanced use

### (1) Time step loop ###
nt=len(xr_phi['t'])
skip=10
for it in range(0,nt,skip):
    phiinxy(it, iz, xr_phi, flag="savefig")


### (2) Find nearest index zz[iz]=zz_target ###
zz_target = np.pi/3
zz_nearest=float(xr_phi['zz'].sel(zz=zz_target,method="nearest")) # Find nearest value
iz=np.where(xr_phi['zz']==zz_nearest)[0][0] # Pick up index
print("zz_target=",zz_target)
print("zz_nearest=", zz_nearest)
print("iz-1=", iz-1, ", zz[iz-1]=", float(xr_phi['zz'][iz-1]))
print("iz  =", iz  , ", zz[iz  ]=", float(xr_phi['zz'][iz]))
print("iz+1=", iz+1, ", zz[iz+1]=", float(xr_phi['zz'][iz+1]))


### (3) Time average ###
it_sta=10
it_end=30
ave=[]
for it in range(it_sta,it_end+1):
    ave.append(phiinkxky(it, xr_phi))
ave=np.array(ave)          # All time steps are stacked on axis=0.
ave=np.average(ave,axis=0) # Take average over time (axis=0, equidistant)
fig=plt.figure()
ax=fig.add_subplot(111)
quad=ax.pcolormesh(ave[:,:,0], ave[:,:,1], ave[:,:,2],
                   cmap='jet',shading="auto")
ax.set_title("Time average {0:f}<t<{1:f}".format(float(xr_phi['t'][it_sta]),float(xr_phi['t'][it_end])))
ax.set_xlabel(r"Radial wavenumber $kx$")
ax.set_ylabel(r"Poloidal wavenumber $ky$")
fig.colorbar(quad)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




