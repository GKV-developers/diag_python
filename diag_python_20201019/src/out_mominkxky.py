#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 2D spectrum of electrostatic potential <|phi|^2>(kx,ky) 

Module dependency: diag_intgrl

Third-party libraries: numpy, matplotlib
"""

def phiinkxky(it, xr_phi, flag=None):  # タイムステップ数itと表示・保存の選択番号numをmain programから引き受ける。
    """
    Output 2D spectrum of electrostatic potential <|phi|^2>[ky,kx] at t[it].
    <...> denotes flux-surface average in zz.
    
    Parameters
    ----------
        it : int
            index of t-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array

    Returns
    -------
        data[global_ny+1,2*nx+1,3]: Numpy array, dtype=np.float64
            # kx = data[:,:,0]
            # ky = data[:,:,1]
            # phikxky = data[:,:,2]    
    """ 
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_intgrl import intgrl_thet

    ### データ処理 ###
    # 時刻t[it]における三次元複素phi[z,ky,kx]を切り出す
    rephi = xr_phi['rephi'][it,:,:,:]  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][it,:,:,:]  # dim: t, zz, ky, kx
    phi_abs = 0.5 * (rephi*rephi + imphi*imphi) # xarray DataArray

    # diag_intgrl.pyから関数 intgrl_thet を呼び出し、z方向平均
    phi_intg = intgrl_thet(phi_abs)  # xarray DataArray

    # 出力用に配列を整理する
    m_kx, m_ky = np.meshgrid(xr_phi['kx'], xr_phi['ky'])  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky, phi_intg],axis=2)

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
        ax.set_title("t = {:f}".format(float(xr_phi['t'][it])))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        fig.colorbar(quad)
        
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
            
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = './data/phiinkxky_t{:08d}.png'.format(it) 
            plt.savefig(filename)
            plt.close()
            
    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = './data/phiinkxky_t{:08d}.dat'.format(it) 
        with open(filename, 'w') as outfile:
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_phi['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#           kx             ky      <|phi|^2>\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')
                
    else: # otherwise - return data array
        return data



# --------------------------------------------------------------

def Alinkxky(it, xr_Al, flag=None ):
    """
    Output 2D spectrum of magnetic potential <|Al|^2>[ky,kx] at t[it].
    <...> denotes flux-surface average in zz.
    
    Parameters
    ----------
        it : int
            index of t-axis
        xr_Al : xarray Dataset
            xarray Dataset of Al.*.nc, read by diag_rb
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array

    Returns
    -------
        data[global_ny+1,2*nx+1,3]: Numpy array, dtype=np.float64
            # kx = data[:,:,0]
            # ky = data[:,:,1]
            # Alkxky = data[:,:,2]    
    """ 
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_intgrl import intgrl_thet
    
    ### データ処理 ###
    # 時刻t[it]における三次元複素Al[z,ky,kx]を切り出す
    reAl = xr_Al['reAl'][it,:,:,:]  # dim: t, zz, ky, kx
    imAl = xr_Al['imAl'][it,:,:,:]  # dim: t, zz, ky, kx
    Al_abs = 0.5 * (reAl*reAl + imAl*imAl ) # xarray DataArray

    # diag_intgrl.pyから関数 intgrl_thet を呼び出し、z方向平均
    Al_intg = intgrl_thet(Al_abs)  # xarray DataArray

    # 出力用に配列を整理する
    m_kx, m_ky = np.meshgrid(xr_Al['kx'], xr_Al['ky'])  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky, Al_intg],axis=2)

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
        ax.set_title("t = {:f}".format(float(xr_Al['t'][it])))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        fig.colorbar(quad)
        
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
            
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = './data/Alinkxky_t{:08d}.png'.format(it) 
            plt.savefig(filename)
            plt.close()
            
    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = './data/Alinkxky_t{:08d}.dat'.format(it) 
        with open(filename, 'w') as outfile:
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_Al['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#           kx             ky      <|Al|^2>\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')
                
    else: # otherwise - return data array
        return data



# --------------------------------------------------------------

def mominkxky(it, iss, imom, xr_mom, flag=None ):
    """
    Output 2D spectrum of velocity moments <|mom|^2>[ky,kx] at t[it].
    <...> denotes flux-surface average in zz.
    
    Parameters
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis            
        imom : int
            index of moment-axis
            imom=0: dens
            imom=1: upara
            imom=2: ppara
            imom=3: pperp
            imom=4: qlpara
            imom=5: qlperp
        xr_mom : xarray Dataset
            xarray Dataset of mom.*.nc, read by diag_rb
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array

    Returns
    -------
        data[global_ny+1,2*nx+1,3]: Numpy array, dtype=np.float64
            # kx = data[:,:,0]
            # ky = data[:,:,1]
            # momkxky = data[:,:,2]
    """ 
    
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_intgrl import intgrl_thet
    
    ### データ処理 ###
    # 時刻t[it]粒子種iss速度モーメントimomにおける三次元複素mom[z,ky,kx]を切り出す
    remom = xr_mom['remom'][it,iss,imom,:,:,:]  # dim: t, iss, imom, zz, ky, kx
    immom = xr_mom['immom'][it,iss,imom,:,:,:]  # dim: t, iss, imom, zz, ky, kx
    mom_abs = 0.5 * (remom*remom + immom*immom) # xarray DataArray

    # diag_intgrl.pyから関数 intgrl_thet を呼び出し、z方向平均
    mom_intg = intgrl_thet(mom_abs)  # xarray DataArray

    # 出力用に配列を整理する
    m_kx, m_ky = np.meshgrid(xr_mom['kx'], xr_mom['ky'])  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky, mom_intg],axis=2)

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
        ax.set_title("t = {:f} (imom={:d},is={:d})".format(float(xr_mom['t'][it]), imom, iss))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        fig.colorbar(quad)
        
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
            
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = './data/mominkxky_mom{:d}s{:d}_t{:08d}.dat'.format(imom, iss, it) 
            plt.savefig(filename)
            plt.close()
            
    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = './data/mominkxky_mom{:d}s{:d}_t{:08d}.dat'.format(imom, iss, it)
        with open(filename, 'w') as outfile:
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#     kx             ky       <|mom-'+str(imom)+'|^2> \n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')
                
    else: # otherwise - return data array
        return data

    



if (__name__ == '__main__'):
    from diag_geom import geom_set
    from diag_rb import rb_open
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    xr_phi = rb_open('../../post/data/phi.*.nc')
    print(xr_phi)
    help(phiinkxky)
    # Plot 0.5*<|phi|^2>(kx,ky) at t[it]
    it = 2
    phiinkxky(it, xr_phi, flag="display")
    
    xr_Al = rb_open('../../post/data/Al.*.nc')
    print(xr_Al)
    # Plot 0.5*<|phi|^2>(kx,ky) at t[it]
    it = 2
    Alinkxky(it, xr_Al, flag="display")
    
    xr_mom = rb_open('../../post/data/mom.*.nc')
    print(xr_mom)
    # Plot 0.5*<|phi|^2>(kx,ky) at t[it]
    it = 2
    iss = 0
    imom = 3
    mominkxky(it, iss, imom, xr_mom, flag="display")


# In[ ]:





# In[ ]:




