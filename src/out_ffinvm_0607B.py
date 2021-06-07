#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:41:16 2021

@author: plasma
"""


def fluxinvm_fxv(it, iss, xr_phi, xr_fxv, flag=None, outdir="../data_fluid/" ):
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    #from scipy import fft
    #from diag_fft import fft_backward_xyz, fft_forward_xyz
    #from diag_intgrl import intgrl_thet
    from diag_geom import omg, ksq, g0, g1, bb ,Anum, Znum, tau, fcs, sgn, dtout_ptn, dtout_fxv # 計算に必要なglobal変数を呼び込む
    from diag_geom import nxw, nyw, ns, rootg, kx, ky, nx, global_ny, \
                          global_nz, global_nm, global_nv, nz, ny, mu, vl, vp, rankz  # 格子点情報、時期座標情報、座標情報を呼び込む
    from scipy.special import  j0 # scipy.special.j0(x); scipy.special.j0(x); Parameters xarray_likeArgument (float).
    
    np.set_printoptions(precision=12, suppress=False) # pritt()で出力する場合、小数点以下の桁数の設定。　12桁に設定。 Falseで指数表示。
    #nx = int((len(xr_phi['kx'])-1)/2)       # diag_geom.pyから変数情報を読み込む。
    #global_ny = int(len(xr_phi['ky'])-1)    # diag_geom.pyから変数情報を読み込む。
    #global_nz = int(len(xr_phi['zz'])/2)    # diag_geom.pyから変数情報を読み込む。 
    
    iz = int(-global_nz + rankz * (2*nz) + global_nz ) # Ex. iz_value indicates xy-plane. iz=global_nz for Python.
    
    print(' *** OUTPUT fluxinym_fxv *** ')
    loop_fxv = it
    print('loop_fxv =', loop_fxv, '\n') 
    time_fxv = float(xr_fxv["t"][it] )
    print('--- real time_fxv ---',time_fxv, '\n')  
       

    loop_phi = int(time_fxv/ dtout_ptn + 0.5)
    print('loop_phi =', loop_phi, '\n')    
    #loop_time = float(loop_phi * dtout_ptn)
    xr_phi = xr_phi.sel(t=time_fxv, method="nearest") # time_cntに最も近い t を持つ配列を切り出す。
    nearest_phi_time = xr_phi["t"]
    print('nearest_phi_time=', nearest_phi_time, '\n')
    rephi = xr_phi['rephi'][iz, :, :]  # dim: t, zz, ky, kx ;Ex. dim(21,16,7,13) ; t=cnt_time, zz=0（xy平面）と指定済みなので2次元配列になる。
    imphi = xr_phi['imphi'][iz, :, :]  # dim: t, zz, ky, kx ;Ex. dim(21,16,7,13) ; t=cnt_time, zz=0（xy平面）と指定済みなので2次元配列になる。
    phi = rephi + 1.0j*imphi    
     # zz_index=8 for xr_phi indicates xy-plane.
     
    
    if nearest_phi_time - time_fxv > min(dtout_ptn, dtout_fxv):
        print('Error: wrong time in fluxinvm_fxv')
        print('time(fxv)=', time_fxv, '  time(phi)=', nearest_phi_time, '\n')    
    
    

    print('確認： zz_of phi =', np.array(xr_phi["zz"]), '\n')


    # 時刻t[it], 粒子種iss, xy平面（fxv_zz_index=1）における4次元複素数fxv[mu, vl, ky,kx]を切り出す
    time_fxv = float(xr_fxv["t"][it])
    print('--- real time_fxv ---',time_fxv, '\n')
    loop_phi = int(time_fxv / dtout_ptn)
    
    #xr_fxv = xr_fxv.sel(t=time_fxv, method="nearest")
    #nearest_time_fxv = float(xr_fxv["t"])
    #print('nearest_fxv_t=', nearest_time_fxv,'\n')
    
    # Index zz=1 indicates xy-plane.
    refxv = xr_fxv['refxv'][it, iss, :, :, 1, :, :]  # dim: t, is, mu, vl, zz, ky, kx ;Ex. dim(21,2,8,24,2,7,13)
    imfxv = xr_fxv['imfxv'][it, iss, :, :, 1, :, :]  # dim: t, is, mu, vl, zz, ky, kx ;Ex. dim(21,2,8,24,2,7,13)
    fxv = refxv + 1.0j*imfxv       
     # zz_index=1 for xr_fxv indicates xy-plane.
    
    phi = np.array(phi)
    fxv = np.array(fxv)
    print('phi.shape=', phi.shape)
    print('fxv.shape=', fxv.shape)



    #--- time for phi --- <xarray.DataArray 't' (t: 101)>
    #array([ 0.      ,  0.103847,  0.201586,  0.305433,  0.403171,  0.500909,
    #    0.604757,  0.702495,  0.800233,  0.904081,  1.001819,  1.105666,
    #    1.203405,  1.301143,  1.40499 ,  1.502728,  1.600467,  1.704314,
    #    1.802052,  1.9059  ,  2.003638,  2.101376,  2.205224,  2.302962,
    #    2.4007  ,  2.504547,  2.602286,  2.700024,  2.803871,  2.90161 ,
    #    3.005457,  3.103195,  3.200934,  3.304781,  3.402519,  3.500258,
    #    3.604105,  3.701843,  3.80569 ,  3.903429,  4.001167,  4.105014,
    #    4.202753,  4.300491,  4.404338,  4.502077,  4.605924,  4.703662,
    #    4.801401,  4.905248,  5.002986,  5.100725,  5.204572,  5.30231 ,
    #    5.400049,  5.503896,  5.601634,  5.705481,  5.80322 ,  5.900958,
    #    6.004805,  6.102544,  6.200282,  6.304129,  6.401868,  6.505715,
    #    6.603453,  6.701192,  6.805039,  6.902777,  7.000516,  7.104363,
    #    7.202101,  7.305948,  7.403687,  7.501425,  7.605272,  7.703011,
    #    7.800749,  7.904596,  8.002335,  8.100073,  8.20392 ,  8.301659,
    #    8.405506,  8.503244,  8.600983,  8.70483 ,  8.802568,  8.900307,
    #    9.004154,  9.101892,  9.205739,  9.303478,  9.401216,  9.505063,
    #    9.602802,  9.70054 ,  9.804387,  9.902126, 10.005973])
    
    #--- time for fxv --- <xarray.DataArray 't' (t: 101)>
    #array([ 0.      ,  0.103847,  0.201586,  0.305433,  0.403171,  0.500909,
    #    0.604757,  0.702495,  0.800233,  0.904081,  1.001819,  1.105666,
    #    1.203405,  1.301143,  1.40499 ,  1.502728,  1.600467,  1.704314,
    #    1.802052,  1.9059  ,  2.003638,  2.101376,  2.205224,  2.302962,
    #    2.4007  ,  2.504547,  2.602286,  2.700024,  2.803871,  2.90161 ,
    #    3.005457,  3.103195,  3.200934,  3.304781,  3.402519,  3.500258,
    #    3.604105,  3.701843,  3.80569 ,  3.903429,  4.001167,  4.105014,
    #    4.202753,  4.300491,  4.404338,  4.502077,  4.605924,  4.703662,
    #    4.801401,  4.905248,  5.002986,  5.100725,  5.204572,  5.30231 ,
    #    5.400049,  5.503896,  5.601634,  5.705481,  5.80322 ,  5.900958,
    #    6.004805,  6.102544,  6.200282,  6.304129,  6.401868,  6.505715,
    #    6.603453,  6.701192,  6.805039,  6.902777,  7.000516,  7.104363,
    #    7.202101,  7.305948,  7.403687,  7.501425,  7.605272,  7.703011,
    #    7.800749,  7.904596,  8.002335,  8.100073,  8.20392 ,  8.301659,
    #    8.405506,  8.503244,  8.600983,  8.70483 ,  8.802568,  8.900307,
    #    9.004154,  9.101892,  9.205739,  9.303478,  9.401216,  9.505063,
    #    9.602802,  9.70054 ,  9.804387,  9.902126, 10.005973])
    
    
    #zz_of phi =
    #  [-3.14159265359  -2.748893571891 -2.356194490192 -1.963495408494
    #   -1.570796326795 -1.178097245096 -0.785398163397 -0.392699081699
    #   0.              0.392699081699  0.785398163397  1.178097245096
    #   1.570796326795  1.963495408494  2.356194490192  2.748893571891]
    
        
    print('確認： iz=', iz, '\n')    # Ex. iz=0 for xr_phi
    zz_fxv = float(xr_fxv['zz'][1]) # Ex. zz_index=1 for xr_fxv indicates xy-plane.
    
    j0_py = np.zeros((global_nm+1, global_ny+1, 2*nx+1))

    
    for im in range(0, global_nm+1):
        for my in range(0, global_ny+1):
            for mx in range(0, 2*nx+1):
                kmo = np.sqrt(2.0*ksq[iz, my,mx]*mu[im]/omg[iz]) \
                    * np.sqrt(tau[iss]*Anum[iss]) / Znum[iss]
                j0_py[im,my,mx] = j0(kmo)
    
    print('確認： j0_py[0:2, 0:4, 0:4] >>>\n', j0_py[0:2, 0:4, 0:4], '\n')
    
    flux = np.zeros((global_nm+1, 2*global_nv))
    for im in range(0, global_nm+1): # index size of global_nm axis = 8 (0,1,...,7)
        for iv in range(0, 2*global_nv):  # index size of global_nv axis = 24 (0,1,...,23)
            #wr = 0.0
            #for my in range(0, global_ny+1): #４重for文 --> ２重のfor文
                #for mx in range(0, 2*nx+1):
                    #wr = wr + (-1.0j*ky[my]* j0_py[im,my,mx] * phi[my,mx] * np.conj(fxv[im,iv,my,mx]) ).real
                    # 関数fkの内容を確認する。
            #flux[im, iv] = wr
            flux[im, iv] = np.sum((-1.0j*ky[my]* j0_py[im,my,mx] * phi[my,mx] * np.conj(fxv[im,iv,my,mx]) ).real)

    print('確認： phi[0:3, 0:4] >>> \n', phi[0:3, 0:4], '\n')
    print('確認： np.conj(fxv)[0:3, 0:4] >>> \n', np.conj(fxv[global_nm, 2*global_nv-1, 0:3, 0:4]), '\n')
    print('確認： flux_fxv[0:3, 0:4] >>> \n', flux[0:3, 0:4], '\n')
    
    
    # 出力用に配列を整理する
    m_iv, m_im = np.meshgrid(xr_fxv['vl'], xr_fxv['mu'])  # 2D-Plot用メッシュグリッドの作成
    m_iv, m_ip = np.meshgrid(xr_fxv['vl'], vp[:,8]) # vpのxy平面値(iz=8)のメッシュグリッドを作成。　m_ivに対応。
    #data_flux = np.stack([m_iv, m_im, flux], axis=2)     
    data_flux = np.stack([m_iv, m_im, m_ip, flux], axis=2) # 出力する４種類の関数の各要素を第２軸に整列するように並べ替える。
    print('m_iv.shape=', m_iv.shape)
    print('m_im.shape=', m_im.shape)
    print('m_ip.shape=', m_ip.shape)
    print('data_flux.shape=', data_flux.shape, '\n')  

    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("flux_is= {:02d} ; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $vl$")
        ax.set_ylabel(r"Poloidal wavenumber $mu$")
        quad_flux = ax.pcolormesh(data_flux[:,:,0], data_flux[:,:,1], data_flux[:,:,3],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_flux)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluxinvm_rankz_is_{:02d}_t{:08d}.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'fluxinvm_rankz_is{:02d}_t{:08d}.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# flux_is = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t_phi = {:f} \n'.format(loop_fxv, float(nearest_phi_time)))
            outfile.write('### iz= {:d}, zz= {:f} \n'.format(iz, zz_fxv))
            outfile.write('#    vl                mu             vp               flux \n')
            for data_slice in data_flux:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')                               
    
    else: # otherwise - return data array 
        return data_flux


    print(' ------------------------ fluxinvm_fxv calculation end --------------------------\n')



def fluxinvm_cnt(it, iss, xr_phi, xr_cnt, flag=None, outdir="../data_fluid/" ):
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_geom import omg, ksq, g0, g1, bb ,Anum, Znum, tau, fcs, sgn, dtout_ptn, dtout_fxv # 計算に必要なglobal変数を呼び込む
    from diag_geom import nxw, nyw, ns, rootg, kx, ky, nx, global_ny, \
                          global_nz, global_nm, global_nv, nz, ny, rankz, mu, vl, vp  # 格子点情報、時期座標情報、座標情報を呼び込む
    from scipy.special import  j0 # scipy.special.j0(x); scipy.special.j0(x); Parameters xarray_likeArgument (float).
    
   
    iz = global_nz # xy平面の位置を示す。
    print(' *** OUTPUT fluxinym_fxv *** ')    
    loop_cnt = it
    print('loop_cnt =', loop_cnt, '\n')       
    time_cnt = float(xr_cnt["t"][it] )
    print('--- real time_cnt ---',time_cnt)  
    
    loop_phi = int(time_cnt/ dtout_ptn + 0.5)
    print('loop_phi =', loop_phi, '\n')    
    #loop_time = float(loop_phi * dtout_ptn)
    xr_phi = xr_phi.sel(t=time_cnt, method="nearest") # xr_Phiから、time_cntに最も近い t を持つ配列を切り出す。
    nearest_phi_time = xr_phi["t"] # time_cntに最も近い t を取り出す。
    print('nearest_phi_time=', nearest_phi_time, '\n')
    rephi = xr_phi['rephi'][iz, :, :]  # dim: t, zz, ky, kx ;Ex. dim(21,16,7,13) ; t=cnt_time, zz=0（xy平面）と指定済みなので2次元配列になる。
    imphi = xr_phi['imphi'][iz, :, :]  # dim: t, zz, ky, kx ;Ex. dim(21,16,7,13) ; t=cnt_time, zz=0（xy平面）と指定済みなので2次元配列になる。
    phi = rephi + 1.0j*imphi    

     
    
    if nearest_phi_time - time_cnt > min(dtout_ptn, dtout_fxv): # nearest_phi_timeとtime_cntを比較する。
        print('Error: wrong time in fluxinvm_fxv')
        print('time(fxv)=', time_cnt, '  time(phi)=', nearest_phi_time, '\n')    
    


    recnt = xr_cnt['recnt'][it, iss, :, :, iz, :, :]  # dim: t, is, mu, vl, zz, ky, kx ;Ex. dim(2,2,8,24,16,7,13)
    imcnt = xr_cnt['imcnt'][it, iss, :, :, iz, :, :]  # dim: t, is, mu, vl, zz, ky, kx ;Ex. dim(2,2,8,24,16,7,13)
    cnt = recnt + 1.0j*imcnt       

    
    phi = np.array(phi)
    cnt = np.array(cnt)
    print('phi.shape=', phi.shape)
    print('cnt.shape=', cnt.shape)

    print('\n 確認： zz_of phi =', np.array(xr_phi["zz"]), '\n')
    
   
    #--- time for cnt --- <xarray.DataArray 't' (t: 2)>
    #    array([1.069014, 2.095268])
    #    Coordinates:
    #       * t        (t) float64 1.069 2.095  <-- 1.069 at loop=0, 2.095 at loop=1
   
    
    zz_cnt = float(xr_cnt['zz'][iz]) # Ex. zz_index=8 for xr_cnt indicates xy-plane.
    print('確認： zz_for xy-plane=', zz_cnt, '\n')    # Ex. iz=0 for xr_phi
    
    j0_py = np.zeros((global_nm+1, global_ny+1, 2*nx+1))

    
    for im in range(0, global_nm+1):
        for my in range(0, global_ny+1):
            for mx in range(0, 2*nx+1):
                kmo = np.sqrt(2.0*ksq[iz, my,mx]*mu[im]/omg[iz]) \
                    * np.sqrt(tau[iss]*Anum[iss]) / Znum[iss]
                j0_py[im,my,mx] = j0(kmo) # j0 is Bessel function of the first kind of order 0.
    
    print('確認： j0_py[0:2, 0:4, 0:4] >>>\n', j0_py[0:2, 0:4, 0:4], '\n')
    
    flux = np.zeros((global_nm+1, 2*global_nv))
    for im in range(0, global_nm+1): # index size of global_nm axis = 8 (0,1,...,7)
        for iv in range(0, 2*global_nv):  # index size of global_nv axis = 24 (0,1,...,23)
            #wr = 0.0
            #for my in range(0, global_ny+1): #４重for文 --> ２重のfor文
                #for mx in range(0, 2*nx+1):
                    #wr = wr + (-1.0j*ky[my]* j0_py[im,my,mx] * phi[my,mx] * np.conj(cnt[im,iv,my,mx]) ).real
                    # 関数fkの内容を確認する。
            #flux[im, iv] = wr
            flux[im, iv] = np.sum((-1.0j*ky[my]* j0_py[im,my,mx] * phi[my,mx] * np.conj(cnt[im,iv,my,mx]) ).real)

    print('確認： phi[0:3, 0:4] >>> \n', phi[0:3, 0:4], '\n')
    print('確認： np.conj(cnt)[0:3, 0:4] >>> \n', np.conj(cnt[global_nm, 2*global_nv-1, 0:3, 0:4]), '\n')
    print('確認： flux_cnt[0:3, 0:4] >>> \n', flux[0:3, 0:4], '\n')
    
    
    # 出力用に配列を整理する
    m_iv, m_im = np.meshgrid(xr_cnt['vl'], xr_cnt['mu'])  # 2D-Plot用メッシュグリッドの作成
    m_iv, m_ip = np.meshgrid(xr_cnt['vl'], vp[:,iz]) # vpのxy平面値(iz=8)のメッシュグリッドを作成。　m_ivに対応。
   
    data_flux = np.stack([m_iv, m_im, m_ip, flux], axis=2) # 出力する４種類の関数の各要素を第２軸に整列するように並べ替える。
    print('確認： m_iv.shape=', m_iv.shape)
    print('確認： m_im.shape=', m_im.shape)
    print('確認： m_ip.shape=', m_ip.shape)
    print('確認： data_flux.shape=', data_flux.shape, '\n')  

    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("flux_is= {:02d} ; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $vl$")
        ax.set_ylabel(r"Poloidal wavenumber $mu$")
        quad_flux = ax.pcolormesh(data_flux[:,:,0], data_flux[:,:,1], data_flux[:,:,3],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_flux)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluxinvm_z_is_{:02d}_t{:08d}.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'fluxinvm_z_is{:02d}_t{:08d}.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# flux_is = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t_phi = {:f}, t_cnt = {:f} \n'.format(loop_cnt, float(nearest_phi_time), float(time_cnt)))
            outfile.write('### iz= {:d}, zz= {:f} \n'.format(iz, zz_cnt))
            outfile.write('#    vl                mu             vp               flux \n')
            for data_slice in data_flux:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')     
            #print('参考：第０軸最後のdata_slice=', data_slice)
    
    else: # otherwise - return data array 
        return data_flux





if (__name__ == '__main__'):
    
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time
    #from diag_geom import rankz

    s_time = time.time()
    xr_phi = rb_open('../../post/data/phi.*.nc')  
    #xr_Al  = rb_open('../../post/data/Al.*.nc')                  
    #xr_mom = rb_open('../../post/data/mom.*.nc')  
    xr_fxv = rb_open('../../post/data/fxv.*.nc')
    xr_cnt = rb_open('../../post/data/cnt.*.nc')
    print("\n***** 確認 if (__name__ == '__main__'):･･･ xr_phiの属性 >>>\n", xr_phi, '\n')
    print("\n***** 確認 if (__name__ == '__main__'):･･･ xr_fxvの属性 >>>\n", xr_fxv, '\n')
    print("\n***** 確認 if (__name__ == '__main__'):･･･ xr_cntの属性 >>>\n", xr_cnt, '\n')
    
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')    
    
    it = 70; iss = 0
    p_time_phi =xr_phi["t"]
    #print('--- time for phi ---',p_time_phi, '\n')      
    p_time_fxv =xr_fxv["t"]
    #print('--- time for fxv ---',p_time_fxv, '\n')    
    
    fluxinvm_fxv(it, iss, xr_phi, xr_fxv, flag="savetxt", outdir="../data_fluid/")
    
    it = 0; iss = 0
    #p_time_cnt =xr_cnt["t"]
    #print('--- time for cnt ---',p_time_cnt, '\n')     
    
    fluxinvm_cnt(it, iss, xr_phi, xr_cnt, flag="savetxt", outdir="../data_fluid/")
    
    e_time = time.time()
    pass_time = e_time - s_time
    print ('\n *** pass_time ={:12.5f}sec ***'.format(pass_time))    
    
    
    