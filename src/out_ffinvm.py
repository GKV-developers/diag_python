#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output velocity-dependent turbulent flux Gamma^v

Module dependency: diag_geom

Third-party libraries: numpy, scipy, matplotlib
"""


def fluxinvm_fxv(it, iss, rankz, xr_phi, xr_fxv, flag=None, outdir="../data/" ):
    """
    Output velocity-dependent turbulent flux Gamma^v[m,v] at t[it], zz[rankz].
    
    Parameters
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis
        rankz : int
            index of MPI rank in z
            # zz=zz[-nz] in GKV source of rankz
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        xr_fxv : xarray Dataset
            xarray Dataset of fxv.*.nc, read by diag_rb
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
        data[global_nm+1,2*global_nv,5]: Numpy array, dtype=np.float64
            # vl = data[:,:,0]          # Parallel velocity
            # mu = data[:,:,1]          # Magnetic moment
            # vp = data[:,:,2]          # Perpendicular velocity
            # Re[Gamma^v] = data[:,:,3] # Velocity dependent particle flux
            # Im[Gamma^v] = data[:,:,4] # * Imaginary part has no physical meaning, but information of the phase is avalable.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_geom import omg, ksq, Anum, Znum, tau, fcs, sgn, dtout_ptn, dtout_fxv # 計算に必要なglobal変数を呼び込む
    from diag_geom import kx, ky, vl, mu, vp, global_nz, global_nm, global_nv, nprocz  # 格子点情報、時期座標情報、座標情報を呼び込む
    from scipy.special import j0 # 0th-order Bessel function
    
    refxv = xr_fxv['refxv'][it,iss,:,:,rankz,:,:]  # dim: t, is, mu, vl, zz, ky, kx
    imfxv = xr_fxv['imfxv'][it,iss,:,:,rankz,:,:]  # dim: t, is, mu, vl, zz, ky, kx
    fxv = refxv + 1.0j*imfxv  # dim: mu, vl, ky, kx
    zz_fxv = float(fxv.zz)
    time_fxv = float(fxv.t)
    #print(zz_fxv, time_fxv)

    iz = int(-global_nz + rankz * (2*int(global_nz/nprocz)) + global_nz )
    rephi = xr_phi['rephi'][:,iz,:,:].sel(t=time_fxv, method="nearest")  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][:,iz,:,:].sel(t=time_fxv, method="nearest")  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi  # dim: ky, kx
    zz_phi = float(phi.zz)
    time_phi = float(phi.t)
    #print(zz_phi, time_phi)

    # Check nearest time of phi and fxv
    if time_phi - time_fxv > min(dtout_ptn, dtout_fxv):
        print('Error: wrong time in fluxinvm_fxv')
        print('time(fxv)=', time_fxv, '  time(phi)=', time_phi, '\n')

    # Set finite Larmor radius effect
    wksq = ksq[iz,:,:].reshape(1,ksq.shape[1],ksq.shape[2])
    wmu = mu[:].reshape(len(mu),1,1)
    kmo = np.sqrt(2.0*wksq*wmu/omg[iz]) * np.sqrt(tau[iss]*Anum[iss]) / Znum[iss]
    j0_myx = j0(kmo)
    
    # Calculate flux
    flux = np.zeros((global_nm+1, 2*global_nv), dtype=np.complex128)
    wky = ky.reshape(1,len(ky),1)
    wphi = np.array(phi).reshape(1,ksq.shape[1],ksq.shape[2])
    for iv in range(2*global_nv):
        wfxv = np.array(fxv[:,iv,:,:])
        flux[:,iv] = np.sum(-1.0j*wky*j0_myx*wphi*np.conj(wfxv), axis=(-2,-1))
    
    # 出力用に配列を整理する
    m_vl, m_mu = np.meshgrid(vl, mu)       # 2D-Plot用メッシュグリッドの作成
    m_vl, m_vp = np.meshgrid(vl, vp[:,iz]) # vpのxy平面値(iz=8)のメッシュグリッドを作成。
    data = np.stack([m_vl, m_mu, m_vp, flux.real, flux.imag], axis=2) # 出力する5種類の関数の各要素を第２軸に整列するように並べ替える。

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"Velocity-dependent flux $\Gamma^v$ "+"($s={:d},t={:f}$)".format(iss, time_fxv))
        ax.set_xlabel(r"Parallel velocity $v_\parallel$")
        ax.set_ylabel(r"Perpendicular velocity $v_\perp$")
        vmax=np.max([np.abs(data[:,:,3].min()),data[:,:,3].max()])
        quad = ax.pcolormesh(data[:,:,0], data[:,:,2], data[:,:,3],
                             cmap='RdBu_r',shading="auto",vmax=vmax,vmin=-vmax)
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluxinvm_rankz{:04d}s{:d}_t{:08d}.png'.format(rankz, iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'fluxinvm_rankz{:04d}s{:d}_t{:08d}.dat'.format(rankz, iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# rankz = {:d}, zz = {:f}\n'.format(rankz, zz_fxv))
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, time_fxv))
            outfile.write('### Infomation of phi ###')
            outfile.write('# iz = {:d}, zz = {:f}\n'.format(iz, zz_phi))
            outfile.write('# nearest time of phi = {:f}\n'.format(time_phi))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#          vl             mu             vp      Re[flux]      Im[flux]\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')                               
    
    else: # otherwise - return data array 
        return data

    
    

def fluxinvm_cnt(it, iss, iz, xr_phi, xr_cnt, flag=None, outdir="../data/" ):
    """
    Output velocity-dependent turbulent flux Gamma^v[m,v] at t[it], zz[rankz].
    
    Parameters
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis
        iz : int
            index of zz-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        xr_cnt : xarray Dataset
            xarray Dataset of cnt.*.nc, read by diag_rb
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
        data[global_nm+1,2*global_nv,5]: Numpy array, dtype=np.float64
            # vl = data[:,:,0]          # Parallel velocity
            # mu = data[:,:,1]          # Magnetic moment
            # vp = data[:,:,2]          # Perpendicular velocity
            # Re[Gamma^v] = data[:,:,3] # Velocity dependent particle flux
            # Im[Gamma^v] = data[:,:,4] # * Imaginary part has no physical meaning, but information of the phase is avalable.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_geom import omg, ksq, Anum, Znum, tau, fcs, sgn, dtout_ptn # 計算に必要なglobal変数を呼び込む
    from diag_geom import kx, ky, vl, mu, vp, global_nz, global_nm, global_nv, nprocz  # 格子点情報、時期座標情報、座標情報を呼び込む
    from scipy.special import j0 # 0th-order Bessel function
   

    recnt = xr_cnt['recnt'][it,iss,:,:,iz,:,:]  # dim: t, is, mu, vl, zz, ky, kx
    imcnt = xr_cnt['imcnt'][it,iss,:,:,iz,:,:]  # dim: t, is, mu, vl, zz, ky, kx
    cnt = recnt + 1.0j*imcnt  # dim: mu, vl, ky, kx
    zz_cnt = float(cnt.zz)
    time_cnt = float(cnt.t)
    #print(zz_cnt, time_cnt)

    rephi = xr_phi['rephi'][:,iz,:,:].sel(t=time_cnt, method="nearest")  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][:,iz,:,:].sel(t=time_cnt, method="nearest")  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi  # dim: ky, kx
    zz_phi = float(phi.zz)
    time_phi = float(phi.t)
    #print(zz_phi, time_phi)
        
    # Check nearest time of phi and cnt
    if time_phi - time_cnt > dtout_ptn:
        print('Error: wrong time in fluxinvm_cnt')
        print('time(cnt)=', time_cnt, '  time(phi)=', time_phi, '\n')

    # Set finite Larmor radius effect
    wksq = ksq[iz,:,:].reshape(1,ksq.shape[1],ksq.shape[2])
    wmu = mu[:].reshape(len(mu),1,1)
    kmo = np.sqrt(2.0*wksq*wmu/omg[iz]) * np.sqrt(tau[iss]*Anum[iss]) / Znum[iss]
    j0_myx = j0(kmo)

    # Calculate flux
    flux = np.zeros((global_nm+1, 2*global_nv), dtype=np.complex128)
    wky = ky.reshape(1,len(ky),1)
    wphi = np.array(phi).reshape(1,ksq.shape[1],ksq.shape[2])
    for iv in range(2*global_nv):
        wcnt = np.array(cnt[:,iv,:,:])
        flux[:,iv] = np.sum(-1.0j*wky*j0_myx*wphi*np.conj(wcnt), axis=(-2,-1))
    
    # 出力用に配列を整理する
    m_vl, m_mu = np.meshgrid(vl, mu)       # 2D-Plot用メッシュグリッドの作成
    m_vl, m_vp = np.meshgrid(vl, vp[:,iz]) # vpのxy平面値(iz=8)のメッシュグリッドを作成。
    data = np.stack([m_vl, m_mu, m_vp, flux.real, flux.imag], axis=2) # 出力する5種類の関数の各要素を第２軸に整列するように並べ替える。

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"Velocity-dependent flux $\Gamma^v$ "+"($s={:d},t={:f}$)".format(iss, time_cnt))
        ax.set_xlabel(r"Parallel velocity $v_\parallel$")
        ax.set_ylabel(r"Perpendicular velocity $v_\perp$")
        vmax=np.max([np.abs(data[:,:,3].min()),data[:,:,3].max()])
        quad = ax.pcolormesh(data[:,:,0], data[:,:,2], data[:,:,3],
                             cmap='RdBu_r',shading="auto",vmax=vmax,vmin=-vmax)
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluxinvm_z{:04d}s{:d}_t{:08d}.png'.format(iz, iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'fluxinvm_z{:04d}s{:d}_t{:08d}.dat'.format(iz, iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# iz = {:d}, zz = {:f}\n'.format(iz, zz_cnt))
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, time_cnt))
            outfile.write('### Infomation of phi ###')
            outfile.write('# iz = {:d}, zz = {:f}\n'.format(iz, zz_phi))
            outfile.write('# nearest time of phi = {:f}\n'.format(time_phi))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#          vl             mu             vp      Re[flux]      Im[flux]\n')
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
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    
    
    ### Examples of use ###
    
    
    ### fluxinvm_fxv ###
    #help(phiinz_connect)
    xr_phi = rb_open('../../post/data/phi.*.nc')  
    xr_fxv = rb_open('../../post/data/fxv.*.nc')
    #print(xr_phi)
    from diag_geom import nprocz
    iss = 0 # Index of species
    rankz = int(nprocz / 2)
    zz = float(xr_fxv.zz[rankz])
    print("# Plot flux_es[mu,vl] at t[it], s[iss], zz[rankz]. zz=",zz)
    outdir='../data/fluxinvm_fxv/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_fxv['t']),10):
        fluxinvm_fxv(it, iss, rankz, xr_phi, xr_fxv, flag="savefig", outdir=outdir)
    print("# Display flux_es[mu,vl] at t[it], s[iss], zz[rankz]. zz=",zz)
    it = len(xr_fxv.t)-1
    fluxinvm_fxv(it, iss, rankz, xr_phi, xr_fxv, flag="display")
    print("# Save flux_es[mu,vl] as text files at t[it], s[iss], zz[rankz]. zz=",zz)
    fluxinvm_fxv(it, iss, rankz, xr_phi, xr_fxv, flag="savetxt", outdir=outdir)
    
    
    
    ### fluxinvm_cnt ###
    #help(phiinz_connect)
    xr_phi = rb_open('../../post/data/phi.*.nc')  
    xr_cnt = rb_open('../../post/data/cnt.*.nc')
    #print(xr_phi)
    from diag_geom import global_nz
    iss = 0 # Index of species
    iz = global_nz
    zz = float(xr_cnt.zz[iz])
    print("# Plot flux_es[mu,vl] at t[it], s[iss], zz[iz]. zz=",zz)
    outdir='../data/fluxinvm_cnt/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_cnt['t']),1):
        fluxinvm_cnt(it, iss, iz, xr_phi, xr_cnt, flag="savefig", outdir=outdir)
    print("# Display flux_es[mu,vl] at t[it], s[iss], zz[iz]. zz=",zz)
    it = len(xr_cnt.t)-1
    fluxinvm_cnt(it, iss, iz, xr_phi, xr_cnt, flag="display")
    print("# Save flux_es[mu,vl] as text files at t[it], s[iss], zz[iz]. zz=",zz)
    fluxinvm_cnt(it, iss, iz, xr_phi, xr_cnt, flag="savetxt", outdir=outdir)


# In[ ]:




