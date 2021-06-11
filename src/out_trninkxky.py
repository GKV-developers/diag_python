#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output entropy balance diagnostics in (kx,ky) 

Module dependency: -

Third-party libraries: numpy, matplotlib
"""

def trninkxky(it, iss, itrn, xr_trn, flag=None, outdir="./data/"):
    """
    Output entropy balance diagnostics in (kx,ky) at given it, iss, itrn
    
    Parameters
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis            
        itrn : int
            index of entropy balance diagnostics
            # itrn= 0: Entropy S_s
            # itrn= 1: Electrostatic field energy W_E
            # itrn= 2: Magnetic field energy W_M
            # itrn= 3: W_E to S_s interaction R_sE
            # itrn= 4: W_M to S_s interaction R_sM
            # itrn= 5: Entropy transfer via ExB nonlinearity I_sE
            # itrn= 6: Entropy transfer via magnetic nonlinearity I_sM
            # itrn= 7: Collisional dissipation D_s
            # itrn= 8: Particle flux by ExB flows G_sE
            # itrn= 9: Particle flux by magnetic flutters G_sM
            # itrn=10: Energy flux by ExB flows Q_sE
            # itrn=11: Energy flux by magnetic flutters Q_sM
        xr_trn : xarray Dataset
            xarray Dataset of trn.*.nc, read by diag_rb
        flag : str, optional
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array
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

    ### データ処理 ###
    # 時刻t[it]粒子種iss、解析インデックスitrnにおける二次元実数trn[ky,kx]を切り出す
    trn = xr_trn['trn'][it,iss,itrn,:,:]

    # 出力用に配列を整理する
    m_kx, m_ky = np.meshgrid(xr_trn['kx'], xr_trn['ky'])  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky, trn],axis=2)

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
        ax.set_xlabel(r"Radial wavenumber $k_x$")
        ax.set_ylabel(r"Poloidal wavenumber $k_y$")
        fig.colorbar(quad)
        
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
            
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'trninkxky_itrn{:02d}s{:d}_t{:08d}.png'.format(itrn, iss,it)) 
            plt.savefig(filename)
            plt.close()
            
    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'trninkxky_itrn{:02d}s{:d}_t{:08d}.dat'.format(itrn, iss, it)) 
        with open(filename, 'w') as outfile:
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_trn['t'][it])))
            outfile.write('### Data shape: {} ##\n'.format(data.shape))
            outfile.write('#          kx            ky        <trn_'+str(itrn)+'>\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')
                
    else: # otherwise - return data array
        return data
    




if (__name__ == '__main__'):
    import os
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time
    
    ### Read NetCDF data phi.*.nc by xarray ### 
    s_time = time.time()    
    xr_trn = rb_open('../../post/data/trn.*.nc')       

    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    
    it = 300   # time step No. 0 ~ 300
    iss = 0   # is値
    itrn = 5  # trn: 0 ～ 11の中から計算したい番号を選択
    trninkxky(it, iss, itrn, xr_trn, flag="display", outdir="../data" )

    e_time = time.time()
    pass_time = e_time - s_time
    print ('\n *** total_pass_time ={:12.5f}sec'.format(pass_time))


# In[ ]:




