#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:46:32 2020

@author: p-user
"""



def fluidsubsptrans_loop(it, iss, imom, nfil, xr_phi, xr_Al, xr_mom, flag=None, outdir="../data_fluid/"):

    """
    Output transfer diagnostics in (kx,ky)
    
    Parameters        # fluidtotaltransとして詳細部分は未修整
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis            
        imom : int
            index of tk_es
                imom= 0: 
                imom= 1: 
                imom= 2: 
                imom= 3: 
                imom= 4: 
                imom= 5: 
                imom= 6: 
            index of tk_em 
                imom= 0: 
                imom= 1: 
                imom= 2: 
                imom= 3: 
        xr_mom : xarray Dataset
            xarray Dataset of mom.*.nc, read by diag_rb
        outdir : str, optional
            Output directory path
            # Default: ./data_fluid/

    Returns
    -------
        for tk_es
            data_tk_es_sum[global_ny+1, 2*nx+1, 3]: Numpy array, dtype=np.float64
                # kx = data[:,:,0]
                # ky = data[:,:,1]
                #  = data_tk_es_sum[:,:,2]        
        for tk_em
            data_tk_em_sum[global_ny+1, 2*nx+1, 3]: Numpy array, dtype=np.float64
                # kx = data[:,:,0]
                # ky = data[:,:,1]
                #  = data_tk_em_sum[:,:,2]    
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    #from scipy import fft
    from diag_fft import fft_backward_xyz, fft_forward_xyz
    #from diag_intgrl import intgrl_thet
    from diag_geom import omg, ksq, g0, g1, bb ,Anum, Znum, tau, fcs, sgn # 計算に必要なglobal変数を呼び込む
    from diag_geom import nxw, nyw, ns, rootg, kx, ky, nx, global_ny, global_nz  # 格子点情報、時期座標情報、座標情報を呼び込む

    
    np.set_printoptions(precision=12, suppress=False) # pritt()で出力する場合、小数点以下の桁数の設定。　12桁に設定。 Falseで指数表示。
    #global iss
    
    #nx = int((len(xr_phi['kx'])-1)/2)
    #global_ny = int(len(xr_phi['ky'])-1)
    #global_nz = int(len(xr_phi['zz'])/2)
    
    
    # 時刻t[it]粒子種iss速度モーメントimomにおける三次元複素mom[zz,ky,kx]を切り出す
    rephi = xr_phi['rephi'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    imphi = xr_phi['imphi'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    phi = rephi + 1.0j*imphi
    
    reAl = xr_Al['reAl'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    imAl = xr_Al['imAl'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    Al = reAl + 1.0j*imAl

    remom = xr_mom['remom'][it,iss, :, :, :, :]  # dim: t, is, imom, zz, ky, kx
    immom = xr_mom['immom'][it,iss, :, :, :, :]  # dim: t, is, imom, zz, ky, kx    
    mom = remom + 1.0j*immom
       
    phi = phi.load() # <class 'xarray.core.dataarray.DataArray'>
    Al = Al.load()   # <class 'xarray.core.dataarray.DataArray'> 
    mom = mom.load() # <class 'xarray.core.dataarray.DataArray'> 

    print('確認：iss = ', iss )
    print('入力データの確認： fcs_type=', type(fcs), 'fcs=', fcs)
    print('入力データの確認： tau_type=', type(tau), 'tau=', tau)
    print('入力データの確認： Znum_type=', type(Znum),'Znum=', Znum)
    print('入力データの確認： Anim_type=',type(Anum), 'Azum=', Anum)
 
 
    # !- moments transform: gyrokinetic distribution -> non-adiabatic part
    # imom = 0
    mom[0,:,:,:] = mom[0,:,:,:] +sgn[iss]*fcs[iss]* g0[iss,:,:,:] *phi[:,:,:]/tau[iss]  

    # imom = 1
    #  
    
    # imom = 2
    mom[2,:,:,:] = mom[2,:,:,:] + 0.5* sgn[iss] * fcs[iss] * g0[iss,:,:,:] *phi[:,:,:]  

    # imom = 3

    mom[3,:,:,:] = mom[3,:,:,:] + sgn[iss] * fcs[iss] * phi[:,:,:] \
                    * ((1.0 - bb[iss,:,:,:]) * g0[iss,:,:,:] + bb[iss,:,:,:] * g1[iss,:,:,:])  

    
    # !--- moments transform: non-adiabatic part -> Hermite-Laguerre coefficients
    
    # imom = 0:
    mom[0,:,:,:] = Znum[iss] * mom[0,:,:,:] / fcs[iss]  
    
    # imom = 1:
    mom[1,:,:,:] = np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[1,:,:,:] / fcs[iss]  
    
    # imom = 2:
    mom[2,:,:,:] = 2.0* Znum[iss] * mom[2,:,:,:] / (fcs[iss] * tau[iss]) - mom[0,:,:,:]  
    
    # imom = 3:
    mom[3,:,:,:] = - Znum[iss] * mom[3,:,:,:] / (fcs[iss] * tau[iss]) + mom[0,:,:,:] 
    
    # imom = 4:
    mom[4,:,:,:] = 2.0 * np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[4,:,:,:] \
                    / (fcs[iss] * tau[iss]) - 3.0 * mom[1,:,:,:] 
    
    # imom = 5:
    mom[5,:,:,:] = - np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[5,:,:,:] \
                   / (fcs[iss] * tau[iss]) + mom[1,:,:,:] 

    
    #! set filter
    py_filter = np.zeros((nfil, global_ny+1, 2*nx+1))    # "fliter"はPythonでは予約変数のため使用不可
    
    print('global_nz=', global_nz)
    print('global_ny=', global_ny)
    print('nx=', nx)
    
    iz = global_nz # iz=0 for Fortran, but iz=global_nz for Python.
    for my in range(global_ny+1):
        for mx in range(2*nx+1):
            if ksq[iz,my,mx] < 0.5**2:
                if (my==0):
                    py_filter[0,my,mx] = 1.0
                else:
                    py_filter[1,my,mx] = 1.0   
                    
            elif ksq[iz,my,mx] < 1.0**2:
                if (my==0):
                    py_filter[2,my,mx] = 1.0
                else:
                    py_filter[3,my,mx] = 1.0           
            
            elif ksq[iz,my,mx] < 2.0**2:
                if (my==0):
                    py_filter[4,my,mx] = 1.0
                else:
                    py_filter[5,my,mx] = 1.0   
                    
            elif ksq[iz,my,mx] < 4.0**2:
                if (my==0):
                    py_filter[6,my,mx] = 1.0
                else:
                    py_filter[7,my,mx] = 1.0                   
                    
            elif ksq[iz,my,mx] < 8.0**2:
                if (my==0):
                    py_filter[8,my,mx] = 1.0
                else:
                    py_filter[9,my,mx] = 1.0  
    
            elif ksq[iz,my,mx] < 16.0**2:
                if (my==0):
                    py_filter[10,my,mx] = 1.0
                else:
                    py_filter[11,my,mx] = 1.0  
    
            else:
                if (my==0):
                    py_filter[12,my,mx] = 1.0
                else:
                    py_filter[13,my,mx] = 1.0      
                   
               
    #print('py_filter.shape=', py_filter.shape, '\n') # shape=(nfil, gloal_ny+1, 2*nx+1)

    
    #!--- calc. subspace transfer ---
    nmom = imom # imom = 6 : 引数として与えられる。

    wkx = kx.reshape(1,1,1,2*nx+1)        # ブロードヒャストを考えて4次元化
    wky = ky.reshape(1,1,global_ny+1,1)   # ブロードヒャストを考えて4次元化  
    
    py_filter4D = py_filter.reshape((nfil,1,global_ny+1,2*nx+1))  # ブロードヒャストを考えて4次元化
    print('py_filter4D.shape=', py_filter4D.shape)
    phi4D = np.array(phi,  dtype=np.complex128).reshape((1,2*global_nz,global_ny+1,2*nx+1)) # 元のphiのdtype：complex128
    print('phi4D.shape=', phi4D.shape, '; Class of phi4D >>>', type(phi4D))
   
    ikxp = py_filter4D[:,:,:,:] * 1.0j * wkx[:,:,:,:] * phi4D[:,:,:,:] # Dim (nfil,zz, ky, kx)
    ikyp = py_filter4D[:,:,:,:] * 1.0j * wky[:,:,:,:] * phi4D[:,:,:,:] 
    print('ikxp.shape=', ikxp.shape, '\n') # Ex. ikxp.shape= (14, 16, 7, 13) 
 
    
    ikxf5D = np.zeros((nmom,nfil,2*global_nz,global_ny+1,2*nx+1),  dtype=np.complex128) # 元のmomのdtype：complex128
    ikyf5D = np.zeros((nmom,nfil,2*global_nz,global_ny+1,2*nx+1),  dtype=np.complex128) # 元のmomのdtype：complex128
    py_filter5D = py_filter4D.reshape((1,nfil,1,global_ny+1,2*nx+1))  # ブロードヒャストを考えて5次元化
    w2kx = wkx.reshape((1,1,1,1,2*nx+1))         # ブロードヒャストを考えて5次元化
    w2ky = wky.reshape((1,1,1,global_ny+1,1))    # ブロードヒャストを考えて5次元化
    mom5D = np.array(mom, dtype=np.complex128).reshape((nmom,1,2*global_nz,global_ny+1,2*nx+1))  # ブロードヒャストを考えて5次元化
    print('mom5D.shape=', mom5D.shape)
    ikxf5D[:,:,:,:,:] = py_filter5D[:,:,:,:,:] * 1.0j * w2kx[:,:,:,:,:] * mom5D[:,:,:,:,:] 
    ikyf5D[:,:,:,:,:] = py_filter5D[:,:,:,:,:] * 1.0j * w2ky[:,:,:,:,:] * mom5D[:,:,:,:,:] 


    # 逆3次元FFTの実行
    #   4次元配列にfft_backward_xyzは適用不可のため、dfdx = fft_backward_xyz(ikxf, nxw=nxw, nyw=nyw) などは不成立。

    dpdx_phi = np.zeros((nfil,2*global_nz,2*nyw,2*nxw))  # 4次元実数配列; Ex. (14, 16, 20, 20) 
    dpdy_phi = np.zeros((nfil,2*global_nz,2*nyw,2*nxw))  # 4次元実数配列; Ex. (14, 16, 20, 20) 
    for ifil in range(nfil): # Ex. nfil=14
        dpdx_phi[ifil] = fft_backward_xyz(ikxp[ifil], nxw=nxw, nyw=nyw)  # dpdx_phi[ifil], ikxp[ifil] ：3次元配列
        dpdy_phi[ifil] = fft_backward_xyz(ikyp[ifil], nxw=nxw, nyw=nyw)  # dpdx_phi[ifil], ikxp[ifil] ：3次元配列
        
    dfdx5D = np.zeros((nmom,nfil,2*global_nz,2*nyw,2*nxw))  # 5次元実数配列; Ex. (6, 14, 16, 20, 20) 
    dfdy5D = np.zeros((nmom,nfil,2*global_nz,2*nyw,2*nxw))  # 5次元実数配列; Ex. (6, 14, 16, 20, 20) 
    
    for imom in range(nmom):
        for ifil in range(nfil):
            dfdx5D[imom,ifil] = fft_backward_xyz(ikxf5D[imom,ifil], nxw=nxw, nyw=nyw)  # dfdx5D[imom,ifil], ikxf5D[imom,ifil] ：3次元配列
            dfdy5D[imom,ifil] = fft_backward_xyz(ikyf5D[imom,ifil], nxw=nxw, nyw=nyw)  # dfdy5D[imom,ifil], ikyf5D[imom,ifil] ：3次元配列

    
    # *************** subtrans_es の計算 *******************    
    cfsrf = np.sum(rootg[:])
    fct = rootg[:]/cfsrf
    fct = fct.reshape(2*global_nz, 1, 1)
    print('確認： fct_type=', type(fct), 'fct.shape=', fct.shape)
    print('確認： fct_sum >>>\n', np.sum(fct),'\n')
    print('入力データの確認： fct >>> \n', fct.tolist(), '\n')   

    subtrans_es = np.zeros((nmom, nfil, nfil, nfil))    
    nf_es = np.zeros((2*global_nz,global_ny+1,2*nx+1), dtype=np.complex128) # nf_esは、imom, nfilの軸を持つ必要なし。 3次元配列の定義で良い。
    wkxy_es = np.zeros((2*global_nz,2*nyw,2*nxw)) # wkxyは、imom, nfilの軸を持つ必要なし。 3次元配列の定義で良い。
    coeffiient = np.array([1,1,0.5,1,0.166666666666666,1]) # nmom列の1次元配列


    wkxy_es_accumulation = 0.0
    nf_es_accumulation = 0+0j
    for imom in range(nmom): # nmom
        for jfil in range(nfil): # nfil
            for ifil in range(nfil): # nfil
                wkxy_es[:,:,:] = 0.5*(-dpdx_phi[ifil,:,:,:] * dfdy5D[imom,jfil,:,:,:]\
                                 + dpdy_phi[ifil,:,:,:] * dfdx5D[imom,jfil,:,:,:] \
                                 - dpdx_phi[jfil,:,:,:] * dfdy5D[imom,ifil,:,:,:] \
                                 + dpdy_phi[jfil,:,:,:] * dfdx5D[imom,ifil,:,:,:]) # wkxyは、imom, nfilの軸を持つ必要なし。 3次元配列の定義で良い。

                wkxy_es_accumulation = wkxy_es_accumulation + abs(wkxy_es[:,:,:]).sum()  # confirmation only
                
                nf_es[:,:,:] = fft_forward_xyz(wkxy_es[:,:,:]) 
                
                nf_es_accumulation = nf_es_accumulation + nf_es[:,:,:].sum()             # for confirmarion only!

                 
                for kfil in range(nfil): # nfil
                    temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                    #temp_filter = py_filter[kfil,:,:,:] # そのまま代入すると、　temp_filterの内容を変更に応じて、元のpy_filterの内容も変更される。
                    temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                    subtrans_es[imom,kfil,jfil,ifil] =\
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           * coeffiient[imom]\
                                               *(np.conj(mom[imom,:,:,:])*nf_es[:,:,:]).real))
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。

    #print('確認： wkxy_es_abs_accumulation_all-filter=', wkxy_es_accumulation)
    #print('確認： nf_em_accumulation_all-filter=\n', nf_es_accumulation, '\n')                

    subtrans_es_sum = np.sum(subtrans_es, axis=0)
    print('subtrans_es_sum.shape=', subtrans_es_sum.shape, '\n') # for confirmation
    #print('確認： subtrans_es_sum[2,3:8,3:6] >>> \n', subtrans_es_sum[2,3:8,3:6])  # for confirmation
    #print('確認： subtrans_es_sum[4,3:8,3:6] >>> \n', subtrans_es_sum[4,3:8,3:6])  # for confirmation
    
    print('確認： all_sum of_subtrans_es >>> \n', np.sum(subtrans_es_sum), '\n' ) # for confirmation
    print('確認： all_sum_of_abs(subtrans_es) >>> \n', np.sum(abs(subtrans_es_sum)), '\n' ) # for confirmation
    
    
    # *************** subtrans_em の計算 *******************

    Al4D = np.array(Al,  dtype=np.complex128).reshape((1,2*global_nz,global_ny+1,2*nx+1)) # 元のphiのdtype：complex128
    print('Class of Al4D >>>', type(Al4D), 'AL4D.shape=', Al4D.shape )
   

    ikxp_Al = py_filter4D[:,:,:,:] * 1.0j * wkx[:,:,:,:] * Al4D[:,:,:,:]  # Dim (nfil,2*global_nz,global_ny+1,2*nx+1)
    ikyp_Al = py_filter4D[:,:,:,:] * 1.0j * wky[:,:,:,:] * Al4D[:,:,:,:]  # Dim (nfil,2*global_nz,global_ny+1,2*nx+1)


    # 逆3次元FFTの実行
    #   4次元配列にfft_backward_xyzは適用不可のため、dfdx = fft_backward_xyz(ikxf, nxw=nxw, nyw=nyw) などは不成立。

    dpdx_Al = np.zeros((nfil,2*global_nz,2*nyw,2*nxw))  # 4次元実数配列
    dpdy_Al = np.zeros((nfil,2*global_nz,2*nyw,2*nxw))  # 4次元実数配列
    for ifil in range(nfil): # Ex. nfil=14
        dpdx_Al[ifil] = fft_backward_xyz(ikxp_Al[ifil], nxw=nxw, nyw=nyw)  # dpdx_phi[ifil], kxp[ifil] ：3次元配列
        dpdy_Al[ifil] = fft_backward_xyz(ikyp_Al[ifil], nxw=nxw, nyw=nyw)  # dpdx_phi[ifil], kxp[ifil] ：3次元配列
        

    
    # subtrans_em用に再定義する
    subtrans_em = np.zeros((nmom, nfil, nfil, nfil))    
    nf_em = np.zeros((2*global_nz,global_ny+1,2*nx+1), dtype=np.complex128) # nf_esは、imom, nfilの軸を持つ必要なし。 3次元配列の定義で良い。
    wkxy_em = np.zeros((2*global_nz,2*nyw,2*nxw)) # wkxyは、imom, nfilの軸を持つ必要なし。 3次元配列の定義で良い。    
    nf_em_accumulation = 0+0j 

    # imom == 0 ------------------------------------------------------------------------
    wkxy_accumulation_m0 = 0.0     # confirmation only
    nf_accumulation_m0 = 0.0       # confirmation only

    for jfil in range(nfil):
        for ifil in range(nfil):
            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil,:,:,:] * dfdy5D[0,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[0,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[0,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[0,ifil,:,:,:] )


            wkxy_accumulation_m0 = wkxy_accumulation_m0 + abs(wkxy_em[:,:,:]).sum()  # confirmation only
            
            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:])
            
            nf_accumulation_m0 = nf_accumulation_m0 + nf_em[:,:,:].sum()   # confirmation only

            
            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[0,kfil,jfil,ifil] =\
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss]))\
                                               *(np.conj(mom[1,:,:,:])*nf_em[:,:,:]).real))                

            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil] * dfdy5D[1,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[1,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[1,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[1,ifil,:,:,:]  ) 
                    

            wkxy_accumulation_m0 = wkxy_accumulation_m0 + abs(wkxy_em[:,:,:]).sum()  # confirmation only
              
            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:])    
            
            nf_accumulation_m0 = nf_accumulation_m0 + nf_em[:,:,:].sum()  # confirmation only


            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[0, kfil,jfil,ifil] = subtrans_em[0, kfil,jfil,ifil]+ \
                         np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss]))\
                                               *(np.conj(mom[0,:,:,:])*nf_em[:,:,:]).real)) # 追加： subtrans_em[0, kfil,jfil,ifil] +
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。
    #print('_____________________ imom_0 ________________________ ')    
    #print('確認_em： wkxy_accumulation_imom_0_all_loop =', wkxy_accumulation_m0)    # confirmation only             
    #print('確認_em： nf_accumulation_imom_0_all_loop = ', nf_accumulation_m0)   # confirmation only

            
    # imom == 1 ----------------------------------------------------------------------------
    for jfil in range(nfil): # nfil
        for ifil in range(nfil): # nfil
            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil,:,:,:] * dfdy5D[1,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[1,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[1,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[1,ifil,:,:,:] )

            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:]) 

            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[1, kfil,jfil,ifil] =\
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss]))\
                                               *(np.conj(mom[2,:,:,:])*nf_em[:,:,:]).real))
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。

            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil,:,:,:] * dfdy5D[2,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[2,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[2,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[2,ifil,:,:,:] ) 
     
            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:])
            
            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[1, kfil,jfil,ifil] = subtrans_em[1, kfil,jfil,ifil] + \
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss]))\
                                               *(np.conj(mom[1,:,:,:])*nf_em[:,:,:]).real)) # 追加： subtrans_em[1, kfil,jfil,ifil] +
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。


    
    # imom == 2 ---------------------------------------------------------------------------
    for jfil in range(nfil):
        for ifil in range(nfil):  # imom=1で作成した計算式を流用。
            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil,:,:,:] * dfdy5D[2,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[2,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[2,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[2,ifil,:,:,:] ) 
            
            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:])
  
            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[2, kfil,jfil,ifil] =\
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss])) \
                                               *0.5*(np.conj(mom[4,:,:,:])*nf_em[:,:,:]).real))
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。  

            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil,:,:,:] * dfdy5D[4,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[4,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[4,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[4,ifil,:,:,:] )  
                         
            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:])  
            
            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[2, kfil,jfil,ifil] = subtrans_em[2, kfil,jfil,ifil] + \
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss])) \
                                               *0.5*(np.conj(mom[2,:,:,:])*nf_em[:,:,:]).real)) # 追加： subtrans_em[2, kfil,jfil,ifil] +
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。  

    
    # imom == 3 ------------------------------------------------------------------------
    wkxy_accumulation_m3 = 0.0    # confirmation only
    nf_accumulation_m3 = 0.0      # confirmation only

    for jfil in range(nfil):
        for ifil in range(nfil):
            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil,:,:,:] * dfdy5D[3,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[3,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[3,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[3,ifil,:,:,:] )
                    
            wkxy_accumulation_m3 = wkxy_accumulation_m3 + abs(wkxy_em[:,:,:]).sum() # for confirmation only         
            
            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:])
            
            nf_accumulation_m3 = nf_accumulation_m3 + nf_em[:,:,:].sum()   # confirmation only

            
            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[3, kfil,jfil,ifil] =\
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss])) \
                                               *(np.conj(mom[5,:,:,:])*nf_em[:,:,:]).real))
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。  

            wkxy_em[:,:,:] = \
                0.5*( -dpdx_Al[ifil,:,:,:] * dfdy5D[5,jfil,:,:,:] \
                     + dpdy_Al[ifil,:,:,:] * dfdx5D[5,jfil,:,:,:] \
                         - dpdx_Al[jfil,:,:,:] * dfdy5D[5,ifil,:,:,:] \
                             + dpdy_Al[jfil,:,:,:] * dfdx5D[5,ifil,:,:,:]  ) 
                    
            wkxy_accumulation_m3 = wkxy_accumulation_m3 + abs(wkxy_em[:,:,:]).sum() # for confirmation only         
              
            nf_em[:,:,:] = fft_forward_xyz(wkxy_em[:,:,:])
            
            nf_accumulation_m3 = nf_accumulation_m3 + nf_em[:,:,:].sum()   # confirmation only

            
            for kfil in range(nfil):
                temp_filter = np.array(py_filter4D[kfil,:,:,:])  # temp_filter: 3次元配列
                temp_filter[:,0,0:nx+1] = 0.0 # Fortran: do my=0 do mx=1,nx の部分を表現 
                subtrans_em[3, kfil,jfil,ifil] = subtrans_em[3, kfil,jfil,ifil] + \
                        np.sum(fct[:,:,:]*(2.0*temp_filter[:,:,:]*(fcs[iss]*tau[iss]/Znum[iss]) \
                                           *(- np.sqrt(tau[iss]/Anum[iss])) \
                                               *(np.conj(mom[3,:,:,:])*nf_em[:,:,:]).real)) # 追加： subtrans_em[3, kfil,jfil,ifil] + 
                    # axisを指定しないので、iz、my、mx軸の全要素の合計となる。  




    
    subtrans_em_sum = np.sum(subtrans_em, axis=0) # sum_of_each_mom
    print('subtrans_em_sum.shape=', subtrans_em_sum.shape)
    #print('確認： subtrans_em_sum[2,3:8,3:6] >>> \n', subtrans_em_sum[2,3:8,3:6])
    #print('確認： subtrans_em_sum[4,3:8,3:6] >>> \n', subtrans_em_sum[4,3:8,3:6],'\n')
    
    print('確認： all_sum of_subtrans_es >>> \n', np.sum(subtrans_em_sum), '\n' ) # for confirmation
    print('確認： all_sum_of_abs(subtrans_es) >>> \n', np.sum(abs(subtrans_em_sum)), '\n' ) # for confirmation
    
    # es＋em の合計値
    subtrans = subtrans_es_sum + subtrans_em_sum
    print('subtrans.shape=', subtrans.shape)
    print('確認： subtrans[2,3:8,3:6]>>> \n', subtrans[2,3:8,3:6],'\n')
    print('確認： subtrans[4,3:8,3:6]>>> \n', subtrans[4,3:8,3:6],'\n')
    print('subtrans[10,7,13]>>> \n', subtrans[10,7,13],'\n')
    print('subtrans[7,13,10]>>> \n', subtrans[7,13,10],'\n')
    print('subtrans[13,10,7]>>> \n', subtrans[13,10,7],'\n') 
    print('\n  ***** Fortranとの整合性は確認済 *****')  
 

    ### Program end ###    
    
    
    
if (__name__ == '__main__'):
    
    #from diag_geom import geom_set
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time

    s_time = time.time()
    xr_phi = rb_open('../../post/data/phi.*.nc')  
    xr_Al  = rb_open('../../post/data/Al.*.nc')                  
    xr_mom = rb_open('../../post/data/mom.*.nc')  
    #print("\n***** 確認 if (__name__ == '__main__'):･･･ xr_momの属性 >>>\n", xr_mom, '\n')
    
    it = 1; iss = 0; imom = 6; nfil = 14
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')

    fluidsubsptrans_loop(it, iss, imom, nfil, xr_phi, xr_Al, xr_mom, flag="display", outdir="../data_fluid/")

    e_time = time.time()
    pass_time = e_time - s_time
    print ('pass_time ={:12.5f}sec'.format(pass_time))
       
    
    
    
    










    
    