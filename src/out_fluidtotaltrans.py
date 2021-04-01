#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:46:32 2020

@author: p-user
"""



def fluidtotaltrans_loop(it, iss, imom, xr_phi, xr_Al, xr_mom, flag=None, outdir="../data_fluid/"):

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
    from diag_intgrl import intgrl_thet
    from diag_geom import omg, ksq, g0, g1, bb ,Anum, Znum, tau, fcs, sgn # 計算に必要なglobal変数を呼び込む
    from diag_geom import nxw, nyw, ns, rootg, kx, ky, nx, global_ny, global_nz  # 格子点情報、時期座標情報、座標情報を呼び込む

    
    #nx = int((len(xr_phi['kx'])-1)/2)       # diag_geom.pyから変数情報を読み込む。
    #global_ny = int(len(xr_phi['ky'])-1)    # diag_geom.pyから変数情報を読み込む。
    #global_nz = int(len(xr_phi['zz'])/2)    # diag_geom.pyから変数情報を読み込む。
    
    
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
       
    phi = phi.load()
    Al = Al.load()
    mom = mom.load()
    #print('確認： type(mon) =', type(mom), '\n')

    #print('確認：iss = ', iss )
    #print('確認：g0[iss].shape = ', g0[iss].shape )
    
    # !- moments transform: gyrokinetic distribution -> non-adiabatic part
    # imom = 0
    mom[0] = mom[0] +sgn[iss]*fcs[iss]* g0[iss] *phi/tau[iss]

    # imom = 1
    # None
    
    # imom = 2
    mom[2] = mom[2] + 0.5* sgn[iss] * fcs[iss] * g0[iss] *phi
    
    # imom = 3
    mom[3] = mom[3] + sgn[iss] * fcs[iss] * phi * ((1.0 - bb[iss]) * g0[iss] + bb[iss] * g1[iss])  

    #print("imom 出力の確認： moments transform: gyrokinetic distribution -> non-adiabatic part")
    #print('imom=0 iz=0, my=1 mx=0~2 >>>\n', mom[0,0,1,0:3].compute) # 代表例


    # !--- moments transform: non-adiabatic part -> Hermite-Laguerre coefficients
    
    # imom = 0:
    mom[0] = Znum[iss] * mom[0] / fcs[iss]
    
    # imom = 1:
    mom[1] = np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[1] / fcs[iss]
    
    # imom = 2:
    mom[2] = 2.0* Znum[iss] * mom[2] / (fcs[iss] * tau[iss]) - mom[0]
    
    # imom = 3:
    mom[3] = - Znum[iss] * mom[3] / (fcs[iss] * tau[iss]) + mom[0]
    
    # imom = 4:
    mom[4] = 2.0 * np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[4] / (fcs[iss] * tau[iss]) - 3.0 * mom[1]
    
    # imom = 5:
    mom[5] = - np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[5] / (fcs[iss] * tau[iss]) + mom[1]

    #print("imom 出力の確認： moments transform: non-adiabatic part -> Hermite-Laguerre coefficients")
    #print('imom=0 iz=0, my=1 mx=0~2 >>>\n', mom[0,0,1,0:3].compute) # 代表例



    #!--- calc. total transfer ---
    nmom = imom # imom = 6 : 引数として与えられる。

    kx = kx.reshape(1,1,2*nx+1) 
    ky = ky.reshape(1,global_ny+1,1)  
   
    ikxf = 1j * kx * mom  #mom[0:nmom, :, :, :] Dim (imom, zz, ky, kx)、[:]を廃止して表記方法を合理化。
    ikyf = 1j * ky * mom  #mom[0:nmom, :, :, :]
    ikxp = 1j * kx * phi # Dim (zz, ky, kx)、[:]を廃止して表記方法を合理化。
    ikyp = 1j * ky * phi
    #print('ikxf.shape=', ikxf.shape, 'ikxp.shape=', ikxp.shape, '\n')
    
    # 逆3次元FFTの実行
    #   4次元配列にfft_backward_xyzは適用不可のため、dfdx = fft_backward_xyz(ikxf, nxw=nxw, nyw=nyw) などは不成立。

    # imom = 0
    dfdx_0 = fft_backward_xyz(ikxf[0])
    dfdy_0 = fft_backward_xyz(ikyf[0])

    # imom = 1    
    dfdx_1 = fft_backward_xyz(ikxf[1])
    dfdy_1 = fft_backward_xyz(ikyf[1])    

    # imom = 2    
    dfdx_2 = fft_backward_xyz(ikxf[2])
    dfdy_2 = fft_backward_xyz(ikyf[2])

    # imom = 3
    dfdx_3 = fft_backward_xyz(ikxf[3])
    dfdy_3 = fft_backward_xyz(ikyf[3])

    # imom = 4
    dfdx_4 = fft_backward_xyz(ikxf[4])
    dfdy_4 = fft_backward_xyz(ikyf[4])

    # imom = 5
    dfdx_5 = fft_backward_xyz(ikxf[5])
    dfdy_5 = fft_backward_xyz(ikyf[5])    
    
    dpdx_phi = fft_backward_xyz(ikxp)
    dpdy_phi = fft_backward_xyz(ikyp)
    
    
    cfsrf = np.sum(rootg[:])
    fct = rootg[:]/cfsrf
    fct = fct.reshape(2*global_nz, 1, 1)
    #print('確認： fct.shape=', fct.shape)

    # *************** tk_es の計算 *******************
    
    wkxy_es = np.zeros((nmom, 2*global_nz,2*nyw,2*nxw)) # 4次元実数配列
    nf_es = np.zeros((nmom, 2*global_nz,global_ny+1,2*nx+1), dtype=np.complex128) # 4次元実数複素数配列
    tk_es = np.zeros((nmom, 2*global_nz,global_ny+1,2*nx+1)) # 4次元実数配列
    
    wkxy_es[0] = -dpdx_phi * dfdy_0 + dpdy_phi * dfdx_0  # 3次元実数配列
    wkxy_es[1] = -dpdx_phi * dfdy_1 + dpdy_phi * dfdx_1
    wkxy_es[2] = -dpdx_phi * dfdy_2 + dpdy_phi * dfdx_2
    wkxy_es[3] = -dpdx_phi * dfdy_3 + dpdy_phi * dfdx_3
    wkxy_es[4] = -dpdx_phi * dfdy_4 + dpdy_phi * dfdx_4
    wkxy_es[5] = -dpdx_phi * dfdy_5 + dpdy_phi * dfdx_5
    
    # 4次元配列に適用できないため、nf_es = fft_forward_xyz(wkxy_es, nxw=nxw, nyw=nyw) は不成立。
    nf_es[0] = fft_forward_xyz(wkxy_es[0]) 
    nf_es[1] = fft_forward_xyz(wkxy_es[1]) 
    nf_es[2] = fft_forward_xyz(wkxy_es[2])
    nf_es[3] = fft_forward_xyz(wkxy_es[3])
    nf_es[4] = fft_forward_xyz(wkxy_es[4]) 
    nf_es[5] = fft_forward_xyz(wkxy_es[5])
    #print('確認： nf[0].shape=', nf_es[0].shape)    
 
    coeffiient = np.array([1,1,0.5,1,0.166666666666666,1]).reshape(nmom,1,1,1)   
    # この係数配列を設定することで、tk_es[0]= ~, ... ,tk_es[5]=~ と6種類の計算式の記述は不要。
    
    tk_es = tk_es + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * coeffiient
                                 * ((np.conj(mom)) * nf_es ).real)
 

    #print('確認： tk_es[0].shape=', tk_es[0].shape)
    #print('確認： tk_es[0, 0:2, 0:2, 0:2] >>>\n', tk_es[0, 0:2, 0:2, 0:2]) # 代表例   

    if nmom != 6: # Fortran code write(*,*) "nmom is wrong." に対応。
        print('nmom =', nmom)
    

    wkxy_em = np.zeros((nmom, 2*global_nz,2*nyw,2*nxw)) # 4次元実数配列
    nf_em = np.zeros((nmom, 2*global_nz,global_ny+1,2*nx+1), dtype=np.complex128) # 4次元実数複素数配列
    tk_em = np.zeros((nmom, 2*global_nz,global_ny+1,2*nx+1)) # 4次元実数配列
    
    ikxp[:] = 1j * kx[:] * Al[:, :, :]
    ikyp[:] = 1j * ky[:] * Al[:, :, :]
    
    dpdx_Al = fft_backward_xyz(ikxp) # for Al
    dpdy_Al = fft_backward_xyz(ikyp) # for Al
    
    # *************** tk_em の計算 *******************
    # imom = 0
    wkxy_em[0] = -dpdx_Al * dfdy_0 + dpdy_Al * dfdx_0  # 3次元実数配列, for Al
    nf_em[0] = fft_forward_xyz(wkxy_em[0])  # for Al
    tk_em[0] = tk_em[0] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                                 * ((np.conj(mom[1])) * nf_em[0, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)
    
    wkxy_em[1] = -dpdx_Al * dfdy_1 + dpdy_Al * dfdx_1  # 3次元実数配列, for Al
    nf_em[1] = fft_forward_xyz(wkxy_em[1])  # for Al
    tk_em[0] = tk_em[0] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                                     * ((np.conj(mom[0])) * nf_em[1, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)
    
    # imom = 1
    tk_em[1] = tk_em[1] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                                 * ((np.conj(mom[2])) * nf_em[1, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)  
    
    wkxy_em[2] = -dpdx_Al * dfdy_2 + dpdy_Al * dfdx_2  # 3次元実数配列, for Al
    nf_em[2] = fft_forward_xyz(wkxy_em[2])  # for Al
    tk_em[1] = tk_em[1] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                                 * ((np.conj(mom[1])) * nf_em[2, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)      
    
    # imom = 2
    tk_em[2] = tk_em[2] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * 0.5 * (-np.sqrt(tau[iss]/Anum[iss]))
                             * ((np.conj(mom[4])) * nf_em[2, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)

    wkxy_em[4] = -dpdx_Al * dfdy_4 + dpdy_Al * dfdx_4  # 3次元実数配列, for Al
    nf_em[4] = fft_forward_xyz(wkxy_em[4])  # for Al
    tk_em[2] = tk_em[2] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * 0.5 * (-np.sqrt(tau[iss]/Anum[iss]))
                             * ((np.conj(mom[2])) * nf_em[4, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)

    # imom = 3
    wkxy_em[3] = -dpdx_Al * dfdy_3 + dpdy_Al * dfdx_3  # 3次元実数配列, for Al
    nf_em[3] = fft_forward_xyz(wkxy_em[3])  # for Al
    tk_em[3] = tk_em[3] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                             * ((np.conj(mom[5])) * nf_em[3, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)  
    
    wkxy_em[5] = -dpdx_Al * dfdy_5 + dpdy_Al * dfdx_5  # 3次元実数配列, for Al
    nf_em[5] = fft_forward_xyz(wkxy_em[5])  # for Al
    tk_em[3] = tk_em[3] + fct * ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                             * ((np.conj(mom[3])) * nf_em[5, 0:2*global_nz, 0:global_ny+1, 0:2*nx+1] ).real)  
    
    
    #print('確認： tk_em[0].shape=', tk_em[0].shape)
    #print('確認： tk_em[0, 0:2, 0:2, 0:4] >>>\n', tk_em[0, 0:2, 0:2, 0:4]) # 代表例


    # 出力用に配列を整理する
    m_kx, m_ky = np.meshgrid(xr_phi['kx'], xr_phi['ky'])  # 2D-Plot用メッシュグリッドの作成
    
    #      ************** tk_esの計算 **************
    # iz方向の加算とimom=0〜5の加算を同時に行う。 # 3/22ミーティングでの指摘事項
    
    tk_em_sum = np.sum(tk_em, axis=(0, 1)) 
    
    
    #      ************** tk_emの計算 **************
    # iz方向の加算とimom=0〜5の加算を同時に行う。
    
    tk_es_sum = np.sum(tk_es, axis=(0, 1))    
    
    """
    calc_time = time.time()
    calc_pass_time = calc_time - s_time
    print ('\n *** calc_pass_time ={:12.5f}sec'.format(calc_pass_time))
    """

    ### データ出力 （tk_es, tk_emにおいて各imomの合計値として出力）###
    
    #      ************** tk_esの計算 **************    
    #tk_es_sum = tk_es_0 + tk_es_1 + tk_es_2 + tk_es_3 + tk_es_4 + tk_es_5 # 各imomの合計値を求める。
    data_tk_es_sum = np.stack([m_kx, m_ky, tk_es_sum], axis=2)            
    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("tk_es_is= {:02d}_sum ; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_es_sum = ax.pcolormesh(data_tk_es_sum[:,:,0], data_tk_es_sum[:,:,1], data_tk_es_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_es_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'tk_es_fluidtotaltranskxky_is_{:02d}_sum_t{:08d}.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'tk_es_fluidtotaltranskxky_is{:02d}_sum_t{:08d}.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of tk_es_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_tk_es_sum.shape))
            outfile.write('#           kx                ky                 tk_es_sum \n')
            for data_slice in data_tk_es_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')               
       
    #      ************** tk_emの計算 **************
    #tk_em_sum = tk_em_0 + tk_em_1 + tk_em_2 + tk_em_3  # 各imomの合計値を求める。
    data_tk_em_sum = np.stack([m_kx, m_ky, tk_em_sum], axis=2)      
              
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("tk_em_is= {:02d}_sum ; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_em_sum = ax.pcolormesh(data_tk_em_sum[:,:,0], data_tk_em_sum[:,:,1], data_tk_em_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_em_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'tk_em_fluidtotaltranskxky_is_{:02d}_sum_t{:08d}.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'tk_em_fluidtotaltranskxky_is{:02d}_sum_t{:08d}.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of tk_em_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_tk_em_sum.shape))
            outfile.write('#           kx                ky                 tk_em_sum \n')
            for data_slice in data_tk_em_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')                               
    
    else: # otherwise - return data array 
    
    
        return ( data_tk_es_sum, data_tk_em_sum ) # 複数の変数を返す時は、（ ）で」囲んでタプル形式にする。
           
                
   
    """
    # imom 個別の計算 ---> 別プログラムを参照。　個別計算のプログラムは削除＠2021.03.22


    """

    ### Program end ###    
    
    
    
if (__name__ == '__main__'):
    
    #from diag_geom import geom_set
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time
    global s_time

    s_time = time.time()
    xr_phi = rb_open('../../post/data/phi.*.nc')  
    xr_Al  = rb_open('../../post/data/Al.*.nc')                  
    xr_mom = rb_open('../../post/data/mom.*.nc')  
    #print("\n***** 確認 if (__name__ == '__main__'):･･･ xr_momの属性 >>>\n", xr_mom, '\n')
    
    it = 5; iss = 0; imom = 6
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')

    fluidtotaltrans_loop(it, iss, imom, xr_phi, xr_Al, xr_mom, flag="display", outdir="../data_fluid/")

    e_time = time.time()
    pass_time = e_time - s_time
    print ('pass_time ={:12.5f}sec'.format(pass_time))
       
    
    
    
    










    
    


# In[ ]:




