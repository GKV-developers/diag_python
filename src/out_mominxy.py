#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 2D spectrum of electrostatic potential <|phi|^2>(kx,ky)

Module dependency: diag_fft, diag_geom

Third-party libraries: numpy, matplotlib
"""

def phiinxy(it, iz, xr_phi, flag=None, nxw=None, nyw=None, outdir="./data/"):
    """
    Output 2D electrostatic potential phi[y,x] at t[it], zz[iz].

    Parameters
    ----------
        it : int
            index of t-axis
        iz : int
            index of zz-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = nxw in gkvp_header.f90
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = nyw in gkvp_header.f90
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        data[2*nyw,2*nxw,3]: Numpy array, dtype=np.float64
            # xx = data[:,:,0]
            # yy = data[:,:,1]
            # phixy = data[:,:,2]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_fft import fft_backward_xy
    from diag_geom import nxw as nxw_geom
    from diag_geom import nyw as nyw_geom
    from diag_rb import safe_compute

    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_phi['kx'])-1)/2)
    gny = int(len(xr_phi['ky'])-1)
    if (nxw == None):
        nxw = nxw_geom
    if (nyw == None):
        nyw = nyw_geom

    # 時刻t[it]位置zz[iz]における二次元複素phi[ky,kx]を切り出す
    if 'rephi' in xr_phi and 'imphi' in xr_phi:
        rephi = xr_phi['rephi'][it,iz,:,:]  # dim: t, zz, ky, kx
        imphi = xr_phi['imphi'][it,iz,:,:]  # dim: t, zz, ky, kx
        phi = rephi + 1.0j * imphi
    elif 'phi' in xr_phi:
        phi = xr_phi['phi'][it,iz,:,:]  # dim: t, zz, ky, kx
    phi = safe_compute(phi)

    # diag_fft.pyから関数 fft_backward_xyを呼び出し、2次元逆フーリエ変換 phi[ky,kx]->phi[y,x]
    phixy = fft_backward_xy(phi,nxw=nxw,nyw=nyw) # Numpy array

    # x,y座標を作成
    kymin = float(xr_phi['ky'][1])
    ly = np.pi / kymin
    yy = np.linspace(-ly,ly,2*nyw,endpoint=False)
    if nx==0:
        lx = ly
    else:
        kxmin = float(xr_phi['kx'][nx+1])
        lx = np.pi / kxmin
    xx = np.linspace(-lx,lx,2*nxw,endpoint=False)

    # 出力用に配列を整理する
    m_xx, m_yy = np.meshgrid(xx, yy)  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_xx, m_yy, phixy],axis=2)

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if (flag == "display" or flag == "savefig"):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        vmax=np.max(abs(data[:,:,2]))
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,2],
                            cmap='jet',shading="auto",vmin=-vmax,vmax=vmax)
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-0.6, 0.6) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.5, 1.0) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("t = {:f}".format(float(xr_phi['t'][it])))
        ax.set_xlabel(r"Radial coordinate $x$")
        ax.set_ylabel(r"Poloidal coordinate $y$")
        fig.colorbar(quad)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'phiinxy_z{:04d}_t{:08d}.png'.format(iz,it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'phiinxy_z{:04d}_t{:08d}.dat'.format(iz,it))
        with open(filename, 'w') as outfile:
            outfile.write('# iz = {:d}, zz = {:f}\n'.format(iz, float(xr_phi['zz'][iz])))
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_phi['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#           xx             yy            phi\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')

    else: # otherwise - return data array
        return data



# -------------------------------------------------------------------------------

def Alinxy(it, iz, xr_Al, flag=None, nxw=None, nyw=None, outdir="./data/"):
    """
    Output 2D magnetic potential Al[y,x] at t[it], zz[iz].

    Parameters
    ----------
        it : int
            index of t-axis
        iz : int
            index of zz-axis
        xr_Al : xarray Dataset
            xarray Dataset of Al.*.nc, read by diag_rb
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = int(nx*1.5)+1
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = int(gny*1.5)+1
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        data[2*nyw,2*nxw,3]: Numpy array, dtype=np.float64
            # xx = data[:,:,0]
            # yy = data[:,:,1]
            # Alxy = data[:,:,2]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_fft import fft_backward_xy
    from diag_geom import nxw as nxw_geom
    from diag_geom import nyw as nyw_geom
    from diag_rb import safe_compute

    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_Al['kx'])-1)/2)
    gny = int(len(xr_Al['ky'])-1)
    if (nxw == None):
        nxw = nxw_geom
    if (nyw == None):
        nyw = nyw_geom

    # 時刻t[it]位置zz[iz]における二次元複素Al[ky,kx]を切り出す
    if 'reAl' in xr_Al and 'imAl' in xr_Al:
        reAl = xr_Al['reAl'][it,iz,:,:]  # dim: t, zz, ky, kx
        imAl = xr_Al['imAl'][it,iz,:,:]  # dim: t, zz, ky, kx
        Al = reAl + 1.0j * imAl
    elif 'Al' in xr_Al:
        Al = xr_Al['Al'][it,iz,:,:]  # dim: t, zz, ky, kx
    Al = safe_compute(Al)

    # diag_fft.pyから関数 fft_backward_xyを呼び出し、2次元逆フーリエ変換 Al[ky,kx]->Al(y,x)
    Alxy = fft_backward_xy(Al,nxw=nxw,nyw=nyw) # Numpy array

    # x,y座標を作成
    kymin = float(xr_Al['ky'][1])
    ly = np.pi / kymin
    yy = np.linspace(-ly,ly,2*nyw,endpoint=False)
    if nx==0:
        lx = ly
    else:
        kxmin = float(xr_Al['kx'][nx+1])
        lx = np.pi / kxmin
    xx = np.linspace(-lx,lx,2*nxw,endpoint=False)

    # 出力用に配列を整理する
    m_xx, m_yy = np.meshgrid(xx, yy)  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_xx, m_yy, Alxy],axis=2)

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if (flag == "display" or flag == "savefig"):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        vmax=np.max(abs(data[:,:,2]))
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,2],
                            cmap='jet',shading="auto",vmin=-vmax,vmax=vmax)
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-0.6, 0.6) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.5, 1.0) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("t = {:f}".format(float(xr_Al['t'][it])))
        ax.set_xlabel(r"Radial coordinate $x$")
        ax.set_ylabel(r"Poloidal coordinate $y$")
        fig.colorbar(quad)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'Alinxy_z{:04d}_t{:08d}.png'.format(iz,it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'Alinxy_z{:04d}_t{:08d}.dat'.format(iz,it))
        with open(filename, 'w') as outfile:
            outfile.write('# iz = {:d}, zz = {:f}\n'.format(iz, float(xr_Al['zz'][iz])))
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_Al['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#           xx             yy            Al\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')

    else: # otherwise - return data array
        return data



# ---------------------------------------------------------------------------------------

def mominxy(it, iss, imom, iz, xr_mom, flag=None, nxw=None, nyw=None, outdir='./data/'):
    """
    Output 2D velocity moments mom[y,x] at t[it], zz[iz].

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
        iz : int
            index of zz-axis
        xr_mom : xarray Dataset
            xarray Dataset of mom.*.nc, read by diag_rb
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # flag=="savetxt" - save data as txt
            # otherwise       - return data array
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = int(nx*1.5)+1
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = int(gny*1.5)+1
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        data[2*nyw,2*nxw,3]: Numpy array, dtype=np.float64
            # xx = data[:,:,0]
            # yy = data[:,:,1]
            # momxy = data[:,:,2]
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_fft import fft_backward_xy
    from diag_geom import nxw as nxw_geom
    from diag_geom import nyw as nyw_geom
    from diag_rb import safe_compute

    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_mom['kx'])-1)/2)
    gny = int(len(xr_mom['ky'])-1)
    if (nxw == None):
        nxw = nxw_geom
    if (nyw == None):
        nyw = nyw_geom

    # 時刻t[it]粒子種iss速度モーメントimom位置zz[iz]における二次元複素mom[ky,kx]を切り出す
    if 'remom' in xr_mom and 'immom' in xr_mom:
        remom = xr_mom['remom'][it,iss,imom,iz,:,:]  # dim: t, iss, imom, zz, ky, kx
        immom = xr_mom['immom'][it,iss,imom,iz,:,:]  # dim: t, iss, imom, zz, ky, kx
        mom = remom + 1.0j * immom
    elif 'mom' in xr_mom:
        mom = xr_mom['mom'][it,iss,imom,iz,:,:]  # dim: t, iss, imom, zz, ky, kx
    mom = safe_compute(mom)

    # diag_fft.pyから関数 fft_backward_xyを呼び出し、2次元逆フーリエ変換 mom[ky,kx]->mom[y,x]
    momxy = fft_backward_xy(mom, nxw=nxw, nyw=nyw) # Numpy array

    # x,y座標を作成
    kymin = float(xr_mom['ky'][1])
    ly = np.pi / kymin
    yy = np.linspace(-ly,ly,2*nyw,endpoint=False)
    if nx==0:
        lx = ly
    else:
        kxmin = float(xr_mom['kx'][nx+1])
        lx = np.pi / kxmin
    xx = np.linspace(-lx,lx,2*nxw,endpoint=False)

    # 出力用に配列を整理する
    m_xx, m_yy = np.meshgrid(xx, yy)  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_xx, m_yy, momxy],axis=2)

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if (flag == "display" or flag == "savefig"):
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        vmax=np.max(abs(data[:,:,2]))
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,2],
                            cmap='jet',shading="auto",vmin=-vmax,vmax=vmax)
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-0.6, 0.6) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.5, 1.0) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("t = {:f} (imom={:d},is={:d})".format(float(xr_mom['t'][it]), imom, iss))
        ax.set_xlabel(r"Radial coordinate $x$")
        ax.set_ylabel(r"Poloidal coordinate $y$")
        fig.colorbar(quad)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'mominxy_z{:04d}mom{:d}s{:d}_t{:08d}.png'.format(iz,imom,iss,it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'mominxy_z{:04d}mom{:d}s{:d}_t{:08d}.dat'.format(iz,imom,iss,it))
        with open(filename, 'w') as outfile:
            outfile.write('# iz = {:d}, zz = {:f}\n'.format(iz, float(xr_mom['zz'][iz])))
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#     xx             yy          <mom-'+str(imom)+'> \n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')

    else: # otherwise - return data array
        return data




if (__name__ == '__main__'):
    import os
    from diag_geom import geom_set
    from diag_rb import rb_open
    from time import time as timer
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')


    ### Examples of use ###


    ### phiinxy ###
    #help(phiinxy)
    xr_phi = rb_open('../../phi/gkvp.phi.*.zarr/')
    #print(xr_phi)
    from diag_geom import global_nz
    iz = global_nz # Index in z-grid
    zz=float(xr_phi['zz'][iz])
    print("# Plot phi[y,x] at t[it], zz[iz]. zz=",zz)
    outdir='../data/phiinxy/'
    os.makedirs(outdir, exist_ok=True)
    s_time = timer()
    for it in range(0,len(xr_phi['t']),len(xr_phi['t'])//10):
        phiinxy(it, iz, xr_phi, flag="savefig", outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))

    print("# Display phi[y,x] at t[it], zz[iz]. zz=",zz)
    phiinxy(it, iz, xr_phi, flag="display")
    print("# Save phi[y,x] as text files  at t[it], zz[iz]. zz=",zz)
    phiinxy(it, iz, xr_phi, flag="savetxt", outdir=outdir)


    ### Alinxy ###
    #help(Alinxy)
    xr_Al = rb_open('../../phi/gkvp.Al.*.zarr/')
    #print(xr_Al)
    from diag_geom import global_nz
    iz = global_nz # Index in z-grid
    zz=float(xr_Al['zz'][iz])
    print("# Plot Al[y,x] at t[it], zz[iz]. zz=",zz)
    outdir='../data/Alinxy/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_phi['t']),len(xr_phi['t'])//10):
        Alinxy(it, iz, xr_Al, flag="savefig", outdir=outdir)

    print("# Display Al[y,x] at t[it], zz[iz]. zz=",zz)
    Alinxy(it, iz, xr_Al, flag="display")
    print("# Save Al[y,x] as text files  at t[it], zz[iz]. zz=",zz)
    Alinxy(it, iz, xr_Al, flag="savetxt", outdir=outdir)


    ### mominxy ###
    #help(mominxy)
    xr_mom = rb_open('../../phi/gkvp.mom.*.zarr/')
    #print(xr_mom)
    from diag_geom import global_nz
    iss = 0 # Index of species
    imom = 3 # Index of velocity moment
    iz = global_nz # Index in z-grid
    zz=float(xr_mom['zz'][iz])
    print("# Plot mom[y,x] at t[it], iss, imom, zz[iz]. zz=",zz)
    outdir='../data/mominxy/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_phi['t']),len(xr_phi['t'])//10):
        mominxy(it, iss, imom, iz, xr_mom, flag="savefig", outdir=outdir)

    print("# Display mom[y,x] at t[it], zz[iz]. zz=",zz)
    mominxy(it, iss, imom, iz, xr_mom, flag="display")
    print("# Save mom[y,x] as text files  at t[it], zz[iz]. zz=",zz)
    mominxy(it, iss, imom, iz, xr_mom, flag="savetxt", outdir=outdir)


# In[ ]:





# In[ ]:




