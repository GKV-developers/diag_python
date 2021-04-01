#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:09:30 2020

@author: p-user
"""

def phiinz(it, mx, my, xr_phi, xr_Al, flag_normalize, out_flag ):
    """
    Output phi in (z) at mx,gmy,loop
    
    Parameters
    ----------
        it : int
            index of t-axis

        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb

        xr_Al : xarray Dataset
            xarray Dataset of Al.*.nc, read by diag_rb
            
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
        data[zz, re_phi, im_phi]: Numpy array, dtype=np.float64
 
    """

    import time
    start = time.time()
    import os
    import numpy as np
    #from diag_geom import geom_set
    from diag_geom import nml
    import matplotlib.pyplot as plt


    ### データ処理 ###
    # GKVパラメータを換算する
    #nx = int((len(xr_phi['kx'])-1)/2)
    global_ny= int(len(xr_phi['ky'])-1)

    
    ### パラメータ: gkvp_namelist から読み取る
    beta  = nml['physp']['beta']
    n_tht = nml['nperi']['n_tht']
    #m_j   = nml['nperi']['m_j']
    #del_c = nml['nperi']['del_c']
    
    ### 読み取った情報を元に座標、定数関数等を構築
    #ck = np.exp(2j*np.pi*del_c*n_tht*np.arange(global_ny+1))
    #dj = - m_j * n_tht * np.arange(global_ny+1)  # large_data(nx=85, global_ny=31)で確認するとき、m_j=4 に設定する。
    #print('\n確認：ck = ', ck )      # ck: numpy array
    #print('確認：dj = ', dj, '\n' )  # dy: numpy array

    # 時刻 t[it],位置 ky[my] & kx[mx]における一次元複素phi[zz]を切り出す
    rephi = xr_phi['rephi'][it,:, my, mx]  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][it,:, my, mx]  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi
    
    # 時刻 t[it],位置 ky[my] & kx[mx]における一次元複素Al[zz]を切り出す
    reAl = xr_Al['reAl'][it,:, my, mx]  # dim: t, zz, ky, kx
    imAl = xr_Al['imAl'][it,:, my, mx]  # dim: t, zz, ky, kx
    Al = reAl + 1.0j*imAl

    # z座標を作成
    global_nz= int(len(xr_phi['zz'])/2)
    lz = n_tht*np.pi
    zz = np.linspace(-lz,lz,2*global_nz,endpoint=False)
    

    # case study
    id = 'non_normalize' # id is involved in plot x_label and each file name.
    if (flag_normalize == 'phi0'):           # when phi is normalized by phi0
        phi = phi / phi[0]                   
        id = 'normalize'   # id changed when phi is normalized by phi0

    elif (flag_normalize == 'Al0'):          # when phi is normalized by (Al[0] / np.sqrt(beta))
        phi = phi / (Al[0] / np.sqrt(beta))  
        id = 'normalize'   # id changed when phi is normalized by Al0

    # 出力用の配列を整理する
    zz = zz.reshape(len(xr_phi['zz']), 1)  # numpy ２次元配列に変換
    phi = np.array(phi).reshape(len(xr_phi['zz']), 1)  # xarrayからnumpy ２次元配列に変換
    re_phi = phi.real
    im_phi = phi.imag
    data = np.concatenate([zz, re_phi, im_phi], axis=1)

    finish = time.time()
    pass_time = finish - start
    # process time
    #print('\n phiinz process time >>> {0:12.5f} sec'.format(pass_time), ' \n')

    
    # 出力の場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    outdir="./data_connect/"  # 専用の保管場所を設定
    if (out_flag == "display" or out_flag == "savefig"):
        # ２次元グラフの表示
        #plt.figure(figsize=(10.0, 8.0))
        #plt.rcParams['font.family'] = 'IPAexGothic'
        #plt.rcParams['font.size'] = 18
        #plt.rcParams['axes.labelsize'] = 18
        #plt.rcParams['legend.loc'] = 'best'
        #plt.rcParams['axes.linewidth'] = 2.0
        
        #plt.subplot(111)
        plt.plot(zz, re_phi, label='phi_real')
        plt.plot(zz, im_phi, label='phi_imag')
        plt.xlabel("zz     time_step="+str(it)+"__"+id, fontsize=18)
        plt.ylabel("phi_real & phi_imag", fontsize=18)
        plt.grid()
        leg = plt.legend(loc=1, fontsize=16) #
        leg.get_frame().set_alpha(1)

        if (out_flag == "display"):   # flag=="display" - show figure on display
            plt.show()
                    
        elif (out_flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}__{}.png'.format(mx, my, it, id))
            plt.savefig(filename)
            plt.close()

    elif (out_flag == "savetxt"):     # flag=="savetxt" - save data as txt
        # DATAの保存
        filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}__{}.dat'.format(mx, my, it, id))
        with open(filename, 'w') as outfile:
            outfile.write('# loop = {:d}, time = {:f}\n'.format(it, float(xr_phi['t'][it])))
            outfile.write('# mx = {:d}, kx = {:f}\n'.format(mx, float(xr_phi['kx'][mx])))
            outfile.write('# my = {:d}, ky = {:f} \n'.format(my, float(xr_phi['ky'][my])))
            outfile.write('#   zz             Re[phi]            Im[phi]\n')
            np.savetxt(outfile, data, fmt='%.7e', delimiter='   ')
            outfile.write('\n')        

    else: # otherwise - return data array
        return data


#from numba import jit
#@jit
def phiinz_connect(it, mx, my, xr_phi, xr_Al, flag_normalize, out_flag ):
    """
    Output phi in (z) at mx,gmy,loop
    
    Parameters
    ----------
        it : int
            index of t-axis

        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb

        xr_Al : xarray Dataset
            xarray Dataset of Al.*.nc, read by diag_rb
            
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
        data[zz, re_phi, im_phi]: Numpy array, dtype=np.float64
 
    """

    import time
    start = time.time()

    import os
    import numpy as np
    #from diag_geom import geom_set
    from diag_geom import nml, ck ,dj
    import matplotlib.pyplot as plt
    #from diag_fft import fft_backward_xy

    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_phi['kx'])-1)/2)
    global_ny= int(len(xr_phi['ky'])-1)
    
    ### パラメータ: gkvp_namelist から読み取る
    beta  = nml['physp']['beta']
    n_tht = nml['nperi']['n_tht']
    m_j   = nml['nperi']['m_j'] # diag_geomから関数として呼び出すことで良い。
    #del_c = nml['nperi']['del_c'] #未使用のため不要
    #print('m_j=', m_j)
    
    ### 読み取った情報を元に座標、定数関数等を構築
    #ck = np.exp(2j*np.pi*del_c*n_tht*np.arange(global_ny+1))
    #dj = - m_j * n_tht * np.arange(global_ny+1)     # large_data(nx=85, global_ny=31)で確認するとき、m_j=4 に設定する。
    # m_j = 4 に設定
    #print('\n確認：ck = ', ck )      # ck: numpy array
    #print('確認：dj = ', dj, '\n' )  # dy: numpy array

    # 時刻 t[it],位置 ky[my] & kx[mx]における一次元複素phi[zz]を切り出す
    rephi = xr_phi['rephi'][it,:, my, mx]  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][it,:, my, mx]  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi
    
    # 時刻 t[it],位置 ky[my] & kx[mx]における一次元複素Al[zz]を切り出す
    reAl = xr_Al['reAl'][it,:, my, mx]  # dim: t, zz, ky, kx
    imAl = xr_Al['imAl'][it,:, my, mx]  # dim: t, zz, ky, kx
    Al = reAl + 1.0j*imAl
    
    # z座標を作成
    global_nz= int(len(xr_phi['zz'])/2)
    lz = n_tht*np.pi
    zz = np.linspace(-lz,lz,2*global_nz,endpoint=False)
    #print('確認：zz >>>\n', zz)


    # case study
    id = 'non_normalize'  # id is involved in plot x_label and each file name.
    if (dj[my] == 0 ):
        #print('確認：dj[my] == 0 --> my=', my, '\n')
        if (flag_normalize == 'phi0'):           # when phi is normalized by phi0
            phi = phi / phi[0]   
            id = 'normalize'   # id changed when phi is normalized by phi0                
        elif (flag_normalize == 'Al0'):          # when phi is normalized by (Al[0] / np.sqrt(beta))
            phi = phi / (Al[0] / np.sqrt(beta))  
            id = 'normalize'   # id changed when phi is normalized by Al0

        # 出力用に配列を整理する
        zz = zz.reshape(len(xr_phi['zz']), 1)  # numpy ２次元配列に変換
        phi = np.array(phi).reshape(len(xr_phi['zz']), 1)  # xarrayからnumpy ２次元配列に変換
        #print('確認 phi >>>\n', phi)
        re_phi = phi.real
        im_phi = phi.imag

    else:
        # case of connect_min
        #print('確認：mx=', mx, 'nx=', nx)
        zz0 = np.linspace(-lz,lz,2*global_nz,endpoint=False)
        connect_min = int((nx + mx - int((len(xr_phi['kx'])-1)/2))/(abs(dj[my])))
        #print('確認：connect_min = ', connect_min, '\n')
        if (connect_min != 0 ):
            data1 = np.zeros((connect_min, len(zz), 3)) 
            #print('data1.shape >>>', data1.shape)
         
            for iconnect in range(connect_min, 0, -1):
                mxw = mx +iconnect*dj[my]
                #print('min_mxw=', mxw, 'iconnect=', iconnect)
                # mxwの定義により、時刻 t[it],位置 ky[my] & kx[mxw]における一次元複素phi[zz]を再計算
                rephi = xr_phi['rephi'][it, :, my, mxw]  # dim: t, zz, ky, kx
                imphi = xr_phi['imphi'][it, :, my, mxw]  # dim: t, zz, ky, kx
                phi = rephi + 1.0j*imphi
                
                if (flag_normalize == 'phi0'):           # when phi is normalized by phi0
                    phi = phi / phi[0]                      
                    id = 'normalize'   # id changed when phi is normalized by phi0
                    np_phi_min = np.array([phi])
                    print('確認：normalized_phi_min-part >>>\n', np_phi_min[0:2])
                elif (flag_normalize == 'Al0'):          # when phi is normalized by (Al[0] / np.sqrt(beta))
                    phi = phi / (Al[0] / np.sqrt(beta))     
                    id = 'normalize'   # id changed when phi is normalized by Al0

            # 出力用に配列を整理する    
                #zz = np.linspace(-lz,lz,2*global_nz,endpoint=False) # 初期設定値を読み込む
                zz = -2*np.pi * float(iconnect) + zz0                # 初期設定のzz0に加算して、新しい座標値を作る
                zz = zz.reshape(len(zz), 1)  # zz1を２次元配列にに変換
                phi1 = np.array(phi).reshape(len(xr_phi['zz']), 1)  # xarrayからnumpy ２次元配列に変換
                phi2 = ck[my]** iconnect * phi1
                re_phi = phi2.real
                im_phi = phi2.imag                
                data =np.concatenate([zz, re_phi, im_phi], axis=1)
                #print('data.shape >>>', data.shape)
                data1[connect_min-iconnect, :] = data # 各iconnectに対応するlen(zz)行３列の２次元配列を３次元配列に代入する
            data_connect1 = data1.reshape((connect_min*len(zz), 3))
              #connect_min回分の出力したデータ data1（3次元配列）を第1軸方向に結合して、connect_min行3列の2次元配列を作る。
            #print('data_connect1.shape >>>', data_connect1.shape)
            zz_min = data_connect1[:, 0:1]
            re_phi_min = data_connect1[:, 1:2]
            im_phi_min = data_connect1[:, 2:3]
            print('zz_min.shape >>>', zz_min.shape, '\n')


        # case of connect_max
        #print('確認：mx=', mx, 'nx=', nx)
        connect_max = int((nx - mx + int((len(xr_phi['kx'])-1)/2))/(abs(dj[my])))
        #print('確認：connect_max = ', connect_max)
        data2 = np.zeros((connect_max+1, len(zz), 3)) 
        #print('data2.shape >>>', data2.shape)
        
        for iconnect in range(0, connect_max+1):  # connect_max回分を出力
            mxw = mx - iconnect * dj[my]
            #print('max_mxw=', mxw, 'iconnect=', iconnect)
            # mxwの定義により、時刻 t[it],位置 ky[my] & kx[mxw]における一次元複素phi[zz]を再計算
            rephi = xr_phi['rephi'][it, :, my, mxw]  # dim: t, zz, ky, kx
            imphi = xr_phi['imphi'][it, :, my, mxw]  # dim: t, zz, ky, kx
            phi = rephi + 1.0j*imphi
            
            if (flag_normalize == 'phi0'):            # when phi is normalized by phi0
                phi = phi / phi[0]                      
                id = 'normalize'   # id changed when phi is normalized by phi0
                np_phi_max = np.array([phi])
                print('確認：normalized_phi_max-part >>>\n', np_phi_max[0:2])
            elif (flag_normalize == 'Al0'):           # when phi is normalized by (Al[0] / np.sqrt(beta))
                phi = phi / (Al[0] / np.sqrt(beta))     
                id = 'normalize'   # id changed when phi is normalized by phi0
                
        # 出力用に配列を整理する
            #zz = np.linspace(-lz,lz,2*global_nz,endpoint=False) # 初期設定値を読み込む
            zz = 2*np.pi * float(iconnect) + zz0                 # 初期設定のzz0に加算して、新しい座標値を作る
            zz = zz.reshape(len(zz), 1)  # zz1を２次元配列にに変換
            phi1 = np.array(phi).reshape(len(xr_phi['zz']), 1)  # xarrayからnumpy ２次元配列に変換
            phi2 = np.conj(ck[my]** iconnect) * phi1
            re_phi = phi2.real
            im_phi = phi2.imag
            #print('確認_max re_phi.shape>>>', re_phi.shape, iconnect)
            data =np.concatenate([zz, re_phi, im_phi], axis=1)
            #print('data.shape >>>', data.shape, iconnect)
            data2[iconnect, :] = data  # 各iconnectに対応するlen(zz)行３列の２次元配列を３次元配列に代入する
        data_connect2 = data2.reshape(((connect_max+1)*len(zz), 3))   
          #connect_min回分の出力したデータ data1（3次元配列）を第1軸方向に結合して、connect_max行3列の2次元配列を作る。
        #print('data_connect2.shape >>>', data_connect2.shape)
        zz_max = data_connect2[:, 0:1]      #第２軸の指定で、スライス表示させる理由は、connect_max行1列の2次元配列を作るため。
        re_phi_max = data_connect2[:, 1:2]
        im_phi_max = data_connect2[:, 2:3]
        print('zz_max.shape >>>', zz_max.shape)

        zz = np.concatenate([zz_min, zz_max], axis=0)  # connect_minとconnect_maxの3つの変数を行方向に結合する。
        re_phi = np.concatenate([re_phi_min, re_phi_max], axis=0)
        im_phi = np.concatenate([im_phi_min, im_phi_max], axis=0)
        print('\nzz.shape >>>', zz.shape)
        #print('re_phi.shape >>>', re_phi.shape)
        #print('im_phi.shape >>>', im_phi.shape)        



    # 保存用出力データフォーマット（savetxt）の作成（3つの変数を3列に並べる）
    data =np.concatenate([zz, re_phi, im_phi], axis=1)       
    print('data.shape >>>', data.shape)  
    
    
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    outdir="./data_connect/"
    if (out_flag == "display" or out_flag == "savefig"):
        # ２次元グラフの表示
        plt.figure(figsize=(10.0, 8.0))
        plt.plot(zz, re_phi, label='phi_real')
        plt.plot(zz, im_phi, label='phi_imag')
        plt.xlabel("zz     time_step="+str(it)+"__"+id)
        plt.ylabel("phi_real & phi_imag")
        plt.grid()
        plt.legend()
        

        if (out_flag == "display"):   # flag=="display" - show figure on display
                    plt.show()
                    
        elif (out_flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'phiinz_connect_mx{:04d}_my{:04d}_t{:08d}__{}.png'.format(mx, my, it, id))
            plt.savefig(filename)
            plt.close()

    elif (out_flag == "savetxt"):     # flag=="savetxt" - save data as txt
        # DATAの保存
 
        filename = os.path.join(outdir,'phiinz_connect_mx{:04d}_my{:04d}_t{:08d}__{}.dat'.format(mx, my, it, id))
        with open(filename, 'w') as outfile:
            outfile.write('# loop = {:d}, time = {:f}\n'.format(it, float(xr_phi['t'][it])))
            outfile.write('# mx = {:d}, kx = {:f}\n'.format(mx, float(xr_phi['kx'][mx])))
            outfile.write('# my = {:d}, ky = {:f} \n'.format(my, float(xr_phi['ky'][my])))
            outfile.write('#   zz             Re[phi]            Im[phi]\n')
            np.savetxt(outfile, data, fmt='%.7e', delimiter='   ')
            outfile.write('\n')        

    else: # otherwise - return data array
        return data



    finish = time.time()
    pass_time = finish - start
    # process time
    #print('\n phiinz_connect process time >>> {0:12.5f} sec'.format(pass_time), ' \n')



if (__name__ == '__main__'):
    
    from diag_geom import geom_set
    from diag_rb import rb_open

    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')

    xr_phi = rb_open('../../post/data/phi.*.nc')  # data_large_1118
    xr_Al  = rb_open('../../post/data/Al.*.nc')   # data_large_1118
    ### Examples of use ###
    it = 15; mx = int((len(xr_phi['kx'])-1)/2); my = 3
    
    #phiinz(it, mx, my, xr_phi, xr_Al, flag_normalize='phi0', out_flag='display' )
    
    phiinz_connect(it, mx, my, xr_phi, xr_Al, flag_normalize=None, out_flag='display' )





















