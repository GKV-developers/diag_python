#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 2D spectrum of total triad transfer function in fluid approximation 

Module dependency: diag_geom, diag_fft, diag_intgrl

Third-party libraries: numpy, matplotlib
"""


def fluidtotaltrans_loop(it, iss, xr_phi, xr_Al, xr_mom, flag=None, outdir="../data/"):
    """
    Output total triad transfer function of electrostatic Tk_es[ky,kx] and 
    electromagnetic Tk_em[ky,kx] at t[it].
    
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
        data[global_ny+1, 2*nx+1, 4]: Numpy array, dtype=np.float64
            # kx = data[:,:,0]
            # ky = data[:,:,1]
            # Tk_es = data[:,:,2]
            # Tk_em = data[:,:,3]    
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_fft import fft_backward_xyz, fft_forward_xyz
    from diag_intgrl import intgrl_thet
    from diag_geom import kx, ky, rootg, g0, g1, bb, Anum, Znum, tau, fcs, sgn 
    from diag_rb import safe_compute
    
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

    #!--- calc. total transfer ---
    ikxf = 1j * kx.reshape(1,1,1,len(kx)) * mom[:,:,:,:]
    ikyf = 1j * ky.reshape(1,1,len(ky),1) * mom[:,:,:,:]
    dfdx = fft_backward_xyz(ikxf[:,:,:,:].reshape(ikxf.shape[0]*ikxf.shape[1],ikxf.shape[2],ikxf.shape[3]))
    dfdx = dfdx.reshape(ikxf.shape[0],ikxf.shape[1],dfdx.shape[-2],dfdx.shape[-1])
    dfdy = fft_backward_xyz(ikyf[:,:,:,:].reshape(ikyf.shape[0]*ikyf.shape[1],ikyf.shape[2],ikyf.shape[3]))
    dfdy = dfdy.reshape(ikyf.shape[0],ikyf.shape[1],dfdy.shape[-2],dfdy.shape[-1])
    
    #!--- calc electrostatic Tk_es ---    
    ikxp = 1j * kx.reshape(1,1,len(kx)) * phi[:,:,:]
    ikyp = 1j * ky.reshape(1,len(ky),1) * phi[:,:,:]
    dpdx = fft_backward_xyz(ikxp)
    dpdx = dpdx.reshape(1,dpdx.shape[0],dpdx.shape[1],dpdx.shape[2])
    dpdy = fft_backward_xyz(ikyp)
    dpdy = dpdy.reshape(1,dpdy.shape[0],dpdy.shape[1],dpdy.shape[2])
    # nonlinear term
    wkxy_es = - dpdx * dfdy + dpdy * dfdx
    nf_es = fft_forward_xyz(wkxy_es.reshape(wkxy_es.shape[0]*wkxy_es.shape[1],wkxy_es.shape[2],wkxy_es.shape[3]))
    nf_es = nf_es.reshape(wkxy_es.shape[0],wkxy_es.shape[1],nf_es.shape[-2],nf_es.shape[-1])
    # Tk_es
    coeffiient = np.array([1,1,0.5,1,0.166666666666666,1]).reshape(nmom,1,1,1)
    tk_es = ((fcs[iss] * tau[iss] / Znum[iss]) * coeffiient * ((np.conj(mom)) * nf_es[:,:,:,:]).real)
    # Flux surface average and summation over imom
    cfsrf = np.sum(rootg[:])
    wfct = (rootg[:] / cfsrf).reshape(1,len(rootg), 1, 1)
    tk_es_sum = np.sum(tk_es * wfct, axis=(0, 1))    

    #!--- calc electromagnetic Tk_em ---    
    ikxA = 1j * kx.reshape(1,1,len(kx)) * Al[:,:,:]
    ikyA = 1j * ky.reshape(1,len(ky),1) * Al[:,:,:]
    dAdx = fft_backward_xyz(ikxA)
    dAdx = dAdx.reshape(1,dAdx.shape[0],dAdx.shape[1],dAdx.shape[2])
    dAdy = fft_backward_xyz(ikyA)
    dAdy = dAdy.reshape(1,dAdy.shape[0],dAdy.shape[1],dAdy.shape[2])
    # nonlinear term
    wkxy_em = - dAdx * dfdy + dAdy * dfdx
    nf_em = fft_forward_xyz(wkxy_em.reshape(wkxy_em.shape[0]*wkxy_em.shape[1],wkxy_em.shape[2],wkxy_em.shape[3]))
    nf_em = nf_em.reshape(wkxy_em.shape[0],wkxy_em.shape[1],nf_em.shape[-2],nf_em.shape[-1])
    # Tk_em    
    tk_em = np.zeros_like(tk_es) # 4次元実数配列
    # imom = 0
    tk_em[0] = tk_em[0] + ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[1])) * nf_em[0,:,:,:] ).real)
    tk_em[0] = tk_em[0] + ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[0])) * nf_em[1,:,:,:] ).real)
    # imom = 1
    tk_em[1] = tk_em[1] + ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[2])) * nf_em[1,:,:,:] ).real)  
    tk_em[1] = tk_em[1] + ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[1])) * nf_em[2,:,:,:] ).real)      
    # imom = 2
    tk_em[2] = tk_em[2] + ((fcs[iss] * tau[iss] / Znum[iss]) * 0.5 * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[4])) * nf_em[2,:,:,:] ).real)
    tk_em[2] = tk_em[2] + ((fcs[iss] * tau[iss] / Znum[iss]) * 0.5 * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[2])) * nf_em[4,:,:,:] ).real)
    # imom = 3
    tk_em[3] = tk_em[3] + ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[5])) * nf_em[3,:,:,:] ).real)  
    tk_em[3] = tk_em[3] + ((fcs[iss] * tau[iss] / Znum[iss]) * (-np.sqrt(tau[iss]/Anum[iss]))
                           * ((np.conj(mom[3])) * nf_em[5,:,:,:] ).real)  
    # Flux surface average and summation over imom
    cfsrf = np.sum(rootg[:])
    wfct = (rootg[:] / cfsrf).reshape(1,len(rootg), 1, 1)
    tk_em_sum = np.sum(tk_em * wfct, axis=(0, 1))   

    # 出力用に配列を整理する
    m_kx, m_ky = np.meshgrid(xr_phi['kx'], xr_phi['ky'])  # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky, tk_es_sum, tk_em_sum], axis=2)

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    # 場合分け
    if flag == 'display' or flag == 'savefig':
        # --- plot electrostatic Tk_es
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"$T_{k,es}$ "+"($s={:01d},t={:f}$)".format(iss, float(xr_phi['t'][it])))
        ax.set_xlabel(r"Radial wavenumber $k_x$")
        ax.set_ylabel(r"Poloidal wavenumber $k_y$")
        quad_es_sum = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,2],
                                    cmap='jet',shading="auto")
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad_es_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluidtotaltranskxky_Tk_es_s{:01d}_t{:08d}.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()
            
        # --- plot electromagnetic Tk_em
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"$T_{k,em}$ "+"($s={:01d},t={:f}$)".format(iss, float(xr_phi['t'][it])))
        ax.set_xlabel(r"Radial wavenumber $k_x$")
        ax.set_ylabel(r"Poloidal wavenumber $k_y$")
        quad_es_sum = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,3],
                                    cmap='jet',shading="auto")
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.05, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad_es_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluidtotaltranskxky_Tk_em_s{:01d}_t{:08d}.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'fluidtotaltranskxky_s{:01d}_t{:08d}.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#           kx             ky         Tk_es         Tk_em\n')
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

    xr_phi = rb_open('../../post/data/phi.*.nc')  
    xr_Al  = rb_open('../../post/data/Al.*.nc')                  
    xr_mom = rb_open('../../post/data/mom.*.nc')  
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    
    it = 5; iss = 0
    outdir='../data/fluidtotaltrans/'
    os.makedirs(outdir, exist_ok=True)
    s_time = timer()
    for it in range(0, len(xr_phi['t']), len(xr_phi['t'])//10):
        fluidtotaltrans_loop(it, iss, xr_phi, xr_Al, xr_mom, flag="savefig", outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))
    it = len(xr_phi.t)-1
    fluidtotaltrans_loop(it, iss, xr_phi, xr_Al, xr_mom, flag="display", outdir=outdir)
    fluidtotaltrans_loop(it, iss, xr_phi, xr_Al, xr_mom, flag="savetxt", outdir=outdir)


# In[ ]:





# In[ ]:




