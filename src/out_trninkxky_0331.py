#!/usr/bin/env python
# coding: utf-8



#!/usr/bin/env python3
"""
Output 2D spectrum of electrostatic potential <|phi|^2>(kx,ky) 

Module dependency: diag_intgrl

Third-party libraries: numpy, matplotlib
"""

def trninkxky(it, iss, itrn, xr_trn, flag=None, outdir="./data/"):  # タイムステップ数itと表示・保存の選択番号numをmain programから引き受ける。
    """
    Output transfer diagnostics in (kx,ky)
    
    Parameters   # trninkxky用に未修整
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis            
        itrn : int
            index of moment-axis
            itrn= 0: Entropy S_s
            itrn= 1: Electrostatic field energy W_E
            itrn= 2: Magnetic field energy W_M
            itrn= 3: W_E to S_s interaction R_sE
            itrn= 4: W_M to S_s interaction R_sM
            itrn= 5: Entropy transfer via ExB nonlinearity I_sE
            itrn= 6: Entropy transfer via magnetic nonlinearity I_sM
            itrn= 7: Collisional dissipation D_s
            itrn= 8: Particle flux by ExB flows G_sE
            itrn= 9: Particle flux by magnetic flutters G_sM
            itrn=10: Energy flux by ExB flows Q_sE
            itrn=11: Energy flux by magnetic flutters Q_sM
        xr_trn : xarray Dataset
            xarray Dataset of trn.*.nc, read by diag_rb
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        data[global_ny+1,2*nx+1,3]: Numpy array, dtype=np.float64
            # kx = data[:,:,0]
            # ky = data[:,:,1]
            # trnkxky = data[:,:,2]    
    """

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    #from diag_intgrl import intgrl_thet

    ### データ処理 ###
    # 時刻t[it]粒子種iss、itrnにおける二次元実数trn[ky,kx]を切り出す
    trn = xr_trn['trn'][it,iss,itrn,:,:]  # dim: t, is, itrn, ky, kx
    #print('trn.shape >>>', trn)
    #np_trn =np.array(trn)
    #print('np_trn >>>\n', np_trn[0:3])

    # 出力用に配列を整理する
    m_kx, m_ky = np.meshgrid(xr_trn['kx'], xr_trn['ky'])  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky, trn],axis=2)
    print('data.dtype', data.dtype)

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if (flag == "display" or flag == "savefig"):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,2],
                             cmap='jet',shading="auto")
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-0.6, 0.6) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.5, 1.0) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("t = {:f}".format(float(xr_trn['t'][it])))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        fig.colorbar(quad)
        
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
            
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'trninkxky_itrn{:d}s{:d}_t{:08d}.png'.format(itrn, iss,it)) 
            plt.savefig(filename)
            plt.close()
            
    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'trninkxky_itrn{:d}s{:d}_t{:08d}.dat'.format(itrn, iss, it)) 
        with open(filename, 'w') as outfile:
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_trn['t'][it])))
            outfile.write('### Data shape: {} & Data type: {} ###\n'.format(data.shape, data.dtype))
            outfile.write('#     kx                 ky               <trn_'+str(itrn)+'>\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='     ')
                outfile.write('\n')
                
    else: # otherwise - return data array
        return data
    
    """
    ## 以下のコードは開発時のFortranデータとの整合性確認のために使用するものであり、開発完了時点で削除する。
    # confirmation for Fortran data  *ファイル名： main.pyで指定するitに合わせて変更すること。 10*it
    frtn_data = np.loadtxt('../post/data/trninkxky_s0_t00000130.dat', unpack=True)
    frtn_data.shape
    print('frtn_data.shape =', frtn_data.shape)
    
    itrn = frtn_data[itrn+2, :]  # frtn_dataの第０行＝kx、第１行＝kyのため　+2 とする。
    itrn = itrn.reshape(len(xr_trn['ky']), len(xr_trn['kx']))
    print('確認：itrn.shape >>>', itrn.shape) # shape... 2_dimension
    print('確認：itrn[0:5] >>>\n', itrn[0:5] )
    
    m_kx, m_ky = np.meshgrid(xr_trn['kx'], xr_trn['ky'])  # Fortran 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky, itrn],axis=2)
    print('data.dtype >>>', data.dtype, "data.shape >>>", data.shape)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,2],
                         cmap='jet',shading="auto")
    plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
    #ax.set_xlim(-0.6, 0.6) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
    #ax.set_ylim(-0.5, 1.0) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
    ax.set_title("Fortran_t = {:f}".format(float(xr_trn['t'][it])))
    ax.set_xlabel(r"Radial wavenumber $kx$")
    ax.set_ylabel(r"Poloidal wavenumber $ky$")
    fig.colorbar(quad)
    plt.show()
    
    # plot_savfor confirmatione
    filename = os.path.join(outdir,'Fortran_trninkxky_itrn{:d}s{:d}_t{:08d}.png'.format(itrn, iss,it)) 
    plt.savefig(filename)
    plt.close()
    """

# --------------------------------------------------------------


if (__name__ == '__main__'):
    import os
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time
    global s_time
    
    ### Read NetCDF data phi.*.nc by xarray ### 
    s_time = time.time()    
    xr_phi = rb_open('../../post/data/phi.*.nc')          
    xr_Al  = rb_open('../../post/data/Al.*.nc')    
    xr_mom = rb_open('../../post/data/mom.*.nc')   
    #xr_fxv = rb_open('../../post/data/fxv.*.nc')                    # no use @ Nov.11.2020 ~
    #xr_cnt = rb_open('../../post/data/cnt.*.nc')                    # no use @ Nov.11.2020 ~
    xr_trn = rb_open('../../post/data/trn.*.nc')       
    print("\n***** 確認 if (__name__ == '__main__'):･･･ xr_momの属性 >>>\n", xr_mom, '\n')

    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    
    it = 300   # time step No. 0 ~ 300
    iss = 0   # is値
    itrn = 5  # trn: 0 ～ 11の中から計算したい番号を選択
    trninkxky(it, iss, itrn, xr_trn, flag="display" )

    e_time = time.time()
    pass_time = e_time - s_time
    print ('\n *** total_pass_time ={:12.5f}sec'.format(pass_time))
    
    
    """
    ### Examples of use ###
    
    
    ### phiinkxky ###
    #help(phiinkxky)
    xr_phi = rb_open('../../post/data/phi.*.nc')
    #print(xr_phi)
    print("# Plot <|phi|^2> at t[it].")
    outdir='../data/phiinkxky/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_phi['t']),10):
        phiinkxky(it, xr_phi, flag="savefig", outdir=outdir)
    
    print("# Display <|phi|^2> at t[it].")
    phiinkxky(it, xr_phi, flag="display")
    print("# Save <|phi|^2> at t[it] as text files.")
    phiinkxky(it, xr_phi, flag="savetxt", outdir=outdir)
    
    
    ### Alinkxky ###
    #help(Alinkxky)
    xr_Al = rb_open('../../post/data/Al.*.nc')
    #print(xr_Al)
    print("# Plot <|Al|^2> at t[it].")
    outdir='../data/Alinkxky/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_Al['t']),10):
        Alinkxky(it, xr_Al, flag="savefig", outdir=outdir)
    
    print("# Display <|Al|^2> at t[it].")
    Alinkxky(it, xr_Al, flag="display")
    print("# Save <|Al|^2> at t[it] as text files.")
    Alinkxky(it, xr_Al, flag="savetxt", outdir=outdir)

    
    ### mominkxky ###
    #help(mominkxky)
    xr_mom = rb_open('../../post/data/mom.*.nc')
    #print(xr_mom)
    print("# Plot <|mom|^2> at t[it], iss, imom.")
    outdir='../data/mominkxky/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_mom['t']),10):
        mominkxky(it, iss, imom, xr_mom, flag="savefig", outdir=outdir)
    
    print("# Display <|mom|^2> at t[it], iss, imom.")
    mominkxky(it, iss, imom, xr_mom, flag="display")
    print("# Save <|mom|^2> at t[it], iss, imom as text files.")
    mominkxky(it, iss, imom, xr_mom, flag="savetxt", outdir=outdir)

    """





