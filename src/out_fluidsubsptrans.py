#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:46:32 2020

@author: p-user
"""



def fluidsubsptrans_loop(it, iss, nfil, xr_phi, xr_Al, xr_mom, flag=None, outdir="../data/"):
    """
    Output subspace triad transfer function J_K^{P,Q} where K,P,Q are subspaces defined by filters.
    
    This function is valid for nmom=6.
    (Namely, 6 moments are particle-position n, u_para, p_para, p_perp, ql_para, ql_perp.)
    
    Parameters
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        xr_Al : xarray Dataset
            xarray Dataset of Al.*.nc, read by diag_rb
        xr_mom : xarray Dataset
            xarray Dataset of mom.*.nc, read by diag_rb
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        data[nfil,nfil,nfil,5]: Numpy array, dtype=np.float64
            # K = data[:,:,0]  # index of K
            # P = data[:,:,1]  # index of P
            # Q = data[:,:,2]  # index of Q
            # J_K^{P,Q}_es = data[:,:,3]
            # J_K^{P,Q}_em = data[:,:,4]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_fft import fft_backward_xyz, fft_forward_xyz
    from diag_geom import omg, ksq, g0, g1, bb ,Anum, Znum, tau, fcs, sgn # 計算に必要なglobal変数を呼び込む
    from diag_geom import nxw, nyw, ns, rootg, kx, ky, nx, global_ny, global_nz  # 格子点情報、時期座標情報、座標情報を呼び込む

    nmom = 6
   
    # 時刻t[it],粒子種issにおける４次元複素mom[mom,zz,ky,kx]を切り出す
    rephi = xr_phi['rephi'][it,:,:,:]  # dim: t, is, imom, zz, ky, kx
    imphi = xr_phi['imphi'][it,:,:,:]  # dim: t, is, imom, zz, ky, kx
    phi = rephi + 1.0j*imphi
    reAl = xr_Al['reAl'][it,:,:,:]     # dim: t, is, imom, zz, ky, kx
    imAl = xr_Al['imAl'][it,:,:,:]     # dim: t, is, imom, zz, ky, kx
    Al = reAl + 1.0j*imAl
    remom = xr_mom['remom'][it,iss,:,:,:,:]  # dim: t, is, imom, zz, ky, kx
    immom = xr_mom['immom'][it,iss,:,:,:,:]  # dim: t, is, imom, zz, ky, kx    
    mom = remom + 1.0j*immom
    
    phi = np.array(phi)
    Al = np.array(Al)
    mom = np.array(mom)

    # !- moments transform: gyrokinetic distribution -> non-adiabatic part
    mom[0] = mom[0] +sgn[iss]*fcs[iss]* g0[iss] *phi/tau[iss]
    mom[2] = mom[2] + 0.5* sgn[iss] * fcs[iss] * g0[iss] *phi
    mom[3] = mom[3] + sgn[iss] * fcs[iss] * phi * ((1.0 - bb[iss]) * g0[iss] + bb[iss] * g1[iss])  

    # !--- moments transform: non-adiabatic part -> Hermite-Laguerre coefficients
    mom[0] = Znum[iss] * mom[0] / fcs[iss]
    mom[1] = np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[1] / fcs[iss]
    mom[2] = 2.0* Znum[iss] * mom[2] / (fcs[iss] * tau[iss]) - mom[0]
    mom[3] = - Znum[iss] * mom[3] / (fcs[iss] * tau[iss]) + mom[0]
    mom[4] = 2.0 * np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[4] / (fcs[iss] * tau[iss]) - 3.0 * mom[1]
    mom[5] = - np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[5] / (fcs[iss] * tau[iss]) + mom[1]

    #! set filter
    subfilter = np.zeros((nfil, global_ny+1, 2*nx+1))    # "fliter"はPythonでは予約変数のため使用不可
    iz = global_nz # iz=0 for Fortran, but iz=global_nz for Python.
    for my in range(global_ny+1):
        for mx in range(2*nx+1):
            if ksq[iz,my,mx] < 0.5**2:
                if (my==0):
                    subfilter[0,my,mx] = 1.0
                else:
                    subfilter[1,my,mx] = 1.0   
                    
            elif ksq[iz,my,mx] < 1.0**2:
                if (my==0):
                    subfilter[2,my,mx] = 1.0
                else:
                    subfilter[3,my,mx] = 1.0           
            
            elif ksq[iz,my,mx] < 2.0**2:
                if (my==0):
                    subfilter[4,my,mx] = 1.0
                else:
                    subfilter[5,my,mx] = 1.0   
                    
            elif ksq[iz,my,mx] < 4.0**2:
                if (my==0):
                    subfilter[6,my,mx] = 1.0
                else:
                    subfilter[7,my,mx] = 1.0                   
                    
            elif ksq[iz,my,mx] < 8.0**2:
                if (my==0):
                    subfilter[8,my,mx] = 1.0
                else:
                    subfilter[9,my,mx] = 1.0  
    
            elif ksq[iz,my,mx] < 16.0**2:
                if (my==0):
                    subfilter[10,my,mx] = 1.0
                else:
                    subfilter[11,my,mx] = 1.0  
    
            else:
                if (my==0):
                    subfilter[12,my,mx] = 1.0
                else:
                    subfilter[13,my,mx] = 1.0      

    
    #!--- calc. subspace transfer ---
    wsubfilter = subfilter.reshape(1,nfil,1,global_ny+1,2*nx+1)
    wkx = kx.reshape(1,1,1,1,2*nx+1)
    wky = ky.reshape(1,1,1,global_ny+1,1)
    wmom = np.array(mom, dtype=np.complex128).reshape((nmom,1,2*global_nz,global_ny+1,2*nx+1))
    ikxf = wsubfilter[:,:,:,:,:] * 1.0j * wkx[:,:,:,:,:] * wmom[:,:,:,:,:]
    ikyf = wsubfilter[:,:,:,:,:] * 1.0j * wky[:,:,:,:,:] * wmom[:,:,:,:,:]
    dfdx = fft_backward_xyz(ikxf.reshape(np.prod(ikxf.shape[0:3]),ikxf.shape[3],ikxf.shape[4]))
    dfdx = dfdx.reshape(nmom,nfil,2*global_nz,dfdx.shape[-2],dfdx.shape[-1])
    dfdy = fft_backward_xyz(ikyf.reshape(np.prod(ikyf.shape[0:3]),ikyf.shape[3],ikyf.shape[4]))
    dfdy = dfdy.reshape(nmom,nfil,2*global_nz,dfdy.shape[-2],dfdy.shape[-1])

    # *************** subtrans_es の計算 *******************    
    wkx = kx.reshape(1,1,1,2*nx+1)
    wky = ky.reshape(1,1,global_ny+1,1)
    wsubfilter = subfilter.reshape(nfil,1,global_ny+1,2*nx+1)
    wphi = np.array(phi, dtype=np.complex128).reshape(1,2*global_nz,global_ny+1,2*nx+1) # 元のphiのdtype：complex128
    ikxp = wsubfilter[:,:,:,:] * 1.0j * wkx[:,:,:,:] * wphi[:,:,:,:] # Dim (nfil, zz, ky, kx)
    ikyp = wsubfilter[:,:,:,:] * 1.0j * wky[:,:,:,:] * wphi[:,:,:,:] 
    dpdx = fft_backward_xyz(ikxp.reshape(np.prod(ikxp.shape[0:2]),ikxp.shape[2],ikxp.shape[3]))
    dpdx = dpdx.reshape(1,nfil,2*global_nz,dpdx.shape[-2],dpdx.shape[-1])
    dpdy = fft_backward_xyz(ikyp.reshape(np.prod(ikyp.shape[0:2]),ikyp.shape[2],ikyp.shape[3]))
    dpdy = dpdy.reshape(1,nfil,2*global_nz,dpdy.shape[-2],dpdy.shape[-1])
    
    subtrans_es = np.zeros((nfil, nfil, nfil))    
    coeffiient = np.array([1,1,0.5,1,0.166666666666666,1]).reshape(nmom,1,1,1) * (fcs[iss] * tau[iss] / Znum[iss])
    cfsrf = np.sum(rootg[:])
    wfct = (rootg[:] / cfsrf).reshape(1,len(rootg),1,1)
    for jfil in range(nfil): # nfil
        for ifil in range(nfil): # nfil
            wkxy_es = 0.5*(- dpdx[:,ifil,:,:,:] * dfdy[:,jfil,:,:,:]                            + dpdy[:,ifil,:,:,:] * dfdx[:,jfil,:,:,:]                            - dpdx[:,jfil,:,:,:] * dfdy[:,ifil,:,:,:]                            + dpdy[:,jfil,:,:,:] * dfdx[:,ifil,:,:,:]) # nmom,z,y,x
            nf_es = fft_forward_xyz(wkxy_es.reshape(wkxy_es.shape[0]*wkxy_es.shape[1],wkxy_es.shape[2],wkxy_es.shape[3]))
            nf_es = nf_es.reshape(wkxy_es.shape[0],wkxy_es.shape[1],nf_es.shape[-2],nf_es.shape[-1])
            for kfil in range(nfil): # nfil
                temp_filter = np.array(subfilter[kfil,:,:]).reshape(1,1,subfilter.shape[-2],subfilter.shape[-1])
                temp_filter[:,:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現
                tk_es = 2.0 * temp_filter * coeffiient * (np.conj(mom) * nf_es).real
                # Flux surface average and summation over imom, summation over kx,ky
                subtrans_es[kfil,jfil,ifil] = np.sum(tk_es * wfct)
       
    # *************** subtrans_em の計算 *******************
    wkx = kx.reshape(1,1,1,2*nx+1)
    wky = ky.reshape(1,1,global_ny+1,1)
    wsubfilter = subfilter.reshape(nfil,1,global_ny+1,2*nx+1)
    wAl = np.array(Al, dtype=np.complex128).reshape(1,2*global_nz,global_ny+1,2*nx+1) # 元のAlのdtype：complex128
    ikxA = wsubfilter[:,:,:,:] * 1.0j * wkx[:,:,:,:] * wAl[:,:,:,:] # Dim (nfil, zz, ky, kx)
    ikyA = wsubfilter[:,:,:,:] * 1.0j * wky[:,:,:,:] * wAl[:,:,:,:] 
    dAdx = fft_backward_xyz(ikxA.reshape(np.prod(ikxA.shape[0:2]),ikxA.shape[2],ikxA.shape[3]))
    dAdx = dAdx.reshape(nfil,2*global_nz,dAdx.shape[-2],dAdx.shape[-1])
    dAdy = fft_backward_xyz(ikyA.reshape(np.prod(ikyA.shape[0:2]),ikyA.shape[2],ikyA.shape[3]))
    dAdy = dAdy.reshape(nfil,2*global_nz,dAdy.shape[-2],dAdy.shape[-1])

    # subtrans_em用に再定義する
    subtrans_em = np.zeros((nfil, nfil, nfil))    
    cfsrf = np.sum(rootg[:])
    wfct = (rootg[:] / cfsrf).reshape(len(rootg),1,1)
    for jfil in range(nfil):
        for ifil in range(nfil):
            #--- subtrans_em[imom=0] ---
            wkxy_em =                 0.5*(- dAdx[ifil,:,:,:] * dfdy[0,jfil,:,:,:]                      + dAdy[ifil,:,:,:] * dfdx[0,jfil,:,:,:]                      - dAdx[jfil,:,:,:] * dfdy[0,ifil,:,:,:]                      + dAdy[jfil,:,:,:] * dfdx[0,ifil,:,:,:] )
            nf_em = fft_forward_xyz(wkxy_em)
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                         np.sum(wfct[:,:,:]*(2.0*temp_filter*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                               *(np.conj(mom[1,:,:,:])*nf_em[:,:,:]).real))                

            wkxy_em =                 0.5*(- dAdx[ifil,:,:,:] * dfdy[1,jfil,:,:,:]                      + dAdy[ifil,:,:,:] * dfdx[1,jfil,:,:,:]                      - dAdx[jfil,:,:,:] * dfdy[1,ifil,:,:,:]                      + dAdy[jfil,:,:,:] * dfdx[1,ifil,:,:,:] )
            nf_em = fft_forward_xyz(wkxy_em)
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                          np.sum(wfct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                               *(np.conj(mom[0,:,:,:])*nf_em[:,:,:]).real))
            
            #--- subtrans_em[imom=1] ---
            #wkxy_em = \
            #    0.5*(- dAdx[ifil,:,:,:] * dfdy[1,jfil,:,:,:] \
            #         + dAdy[ifil,:,:,:] * dfdx[1,jfil,:,:,:] \
            #         - dAdx[jfil,:,:,:] * dfdy[1,ifil,:,:,:] \
            #         + dAdy[jfil,:,:,:] * dfdx[1,ifil,:,:,:] )
            #nf_em = fft_forward_xyz(wkxy_em) 
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                         np.sum(wfct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                               *(np.conj(mom[2,:,:,:])*nf_em[:,:,:]).real))

            wkxy_em =                 0.5*(- dAdx[ifil,:,:,:] * dfdy[2,jfil,:,:,:]                      + dAdy[ifil,:,:,:] * dfdx[2,jfil,:,:,:]                      - dAdx[jfil,:,:,:] * dfdy[2,ifil,:,:,:]                      + dAdy[jfil,:,:,:] * dfdx[2,ifil,:,:,:] ) 
            nf_em = fft_forward_xyz(wkxy_em)
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                         np.sum(wfct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                               *(np.conj(mom[1,:,:,:])*nf_em[:,:,:]).real))
            
            #--- subtrans_em[imom=2] ---
            #wkxy_em = \
            #    0.5*(- dAdx[ifil,:,:,:] * dfdy[2,jfil,:,:,:] \
            #         + dAdy[ifil,:,:,:] * dfdx[2,jfil,:,:,:] \
            #         - dAdx[jfil,:,:,:] * dfdy[2,ifil,:,:,:] \
            #         + dAdy[jfil,:,:,:] * dfdx[2,ifil,:,:,:] ) 
            #nf_em = fft_forward_xyz(wkxy_em)
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                         np.sum(wfct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                                *0.5*(np.conj(mom[4,:,:,:])*nf_em[:,:,:]).real))
            
            wkxy_em =                 0.5*(- dAdx[ifil,:,:,:] * dfdy[4,jfil,:,:,:]                      + dAdy[ifil,:,:,:] * dfdx[4,jfil,:,:,:]                      - dAdx[jfil,:,:,:] * dfdy[4,ifil,:,:,:]                      + dAdy[jfil,:,:,:] * dfdx[4,ifil,:,:,:] )  
            nf_em = fft_forward_xyz(wkxy_em)
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                         np.sum(wfct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                                *0.5*(np.conj(mom[2,:,:,:])*nf_em[:,:,:]).real))
            
            #--- subtrans_em[imom=3] ---
            wkxy_em =                 0.5*(- dAdx[ifil,:,:,:] * dfdy[3,jfil,:,:,:]                      + dAdy[ifil,:,:,:] * dfdx[3,jfil,:,:,:]                      - dAdx[jfil,:,:,:] * dfdy[3,ifil,:,:,:]                      + dAdy[jfil,:,:,:] * dfdx[3,ifil,:,:,:] )
            nf_em = fft_forward_xyz(wkxy_em)
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                         np.sum(wfct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                                *(np.conj(mom[5,:,:,:])*nf_em[:,:,:]).real))
            
            wkxy_em =                 0.5*(- dAdx[ifil,:,:,:] * dfdy[5,jfil,:,:,:]                      + dAdy[ifil,:,:,:] * dfdx[5,jfil,:,:,:]                      - dAdx[jfil,:,:,:] * dfdy[5,ifil,:,:,:]                      + dAdy[jfil,:,:,:] * dfdx[5,ifil,:,:,:]  ) 
            nf_em = fft_forward_xyz(wkxy_em)
            for kfil in range(nfil):
                temp_filter = np.array(subfilter[kfil:kfil+1,:,:]) # zz,ky,kx
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[kfil,jfil,ifil] = subtrans_em[kfil,jfil,ifil] +                         np.sum(wfct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss])                                            *(- np.sqrt(tau[iss]/Anum[iss]))                                                *(np.conj(mom[3,:,:,:])*nf_em[:,:,:]).real))

    subtrans = subtrans_es + subtrans_em
    
    fig=plt.figure(figsize=(16,16))
    for kfil in range(4*4):
        if kfil >= nfil:
            break
        ax = fig.add_subplot(4,4,kfil+1)
        ax.set_title("$J_K^{P,Q}$ "+"(K={:d})".format(kfil))
        ax.imshow(subtrans[kfil,:,:])
    plt.show()

    # 3次元配列の subtrans(kfil, jfil, ifil) を2次元化してテキストデータで保存する。 -------------------------------------------
    subtrans_2D = subtrans.reshape(nfil*nfil, nfil)
    filename = os.path.join(outdir,'fluidsubtrans_basedata_is{:02d}_t{:08d}.txt'.format(iss, it))
    with open(filename, 'w') as outfile:
        outfile.write('# subtrans_is = {:d} \n'.format(iss))
        outfile.write('# loop = {:d}, t = {:f} \n'.format(it, float(xr_mom['t'][it])))
        outfile.write('# 注記：    列方向： ifil個；　行方向： kfil x jfil個に配列した2次元データ \n')
        for data_slice in subtrans_2D:
            np.savetxt(outfile, data_slice.reshape(1, nfil), fmt='%.7e', delimiter='\t') 
            # data_sliceが1次元配列の場合、reshapeで2次元配列にしないと1個のデータ毎に改行され、縦方向にifil個のデータが表示される。
            # data_sliceが2次元配列の場合、reshapeは不要。
            outfile.write('\n')                              
        #np.savetxt(outfile, subtrans_2D, fmt='%.7e', delimiter='\t')
        #outfile.write('\n') 
    ### Program end ###    
    
    
    
if (__name__ == '__main__'):
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    
    s_time = time.time()
    xr_phi = rb_open('../../post/data/phi.*.nc')  
    xr_Al  = rb_open('../../post/data/Al.*.nc')                  
    xr_mom = rb_open('../../post/data/mom.*.nc')  

    it = 1; iss = 0; imom = 6; nfil = 14
    fluidsubsptrans_loop(it, iss, nfil, xr_phi, xr_Al, xr_mom, flag="display", outdir="../data/")

    e_time = time.time()
    pass_time = e_time - s_time
    print ('pass_time ={:12.5f}sec'.format(pass_time))


# In[ ]:




