#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numba_compat import njit

@njit
def compute_es_loops(jkpq_es, jpqk_es, jqkp_es,
                    kx_ed, ky_ed, phi_ed, mom_ed, ceff,
                    mx, my, global_ny, nx):
    """
    jkpq_es, jpqk_es, jqkp_es の計算を行う関数です。
    """
    for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
        qy = -py-my
        for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):
            qx = -px-mx
            jkpq_es[:,:,py,px] = - ceff[:,:] * (-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx]) \
                                   * np.real((phi_ed[:,:,py,px]*mom_ed[:,:,qy,qx] \
                                            -phi_ed[:,:,qy,qx]*mom_ed[:,:,py,px])*mom_ed[:,:,my,mx])
            jpqk_es[:,:,py,px] = - ceff[:,:] * (-kx_ed[qx]*ky_ed[my]+ky_ed[qy]*kx_ed[mx]) \
                                   * np.real((phi_ed[:,:,qy,qx]*mom_ed[:,:,my,mx] \
                                            -phi_ed[:,:,my,mx]*mom_ed[:,:,qy,qx])*mom_ed[:,:,py,px])
            jqkp_es[:,:,py,px] = - ceff[:] * (-kx_ed[mx]*ky_ed[py]+ky_ed[my]*kx_ed[px]) \
                                   * np.real((phi_ed[:,:,my,mx]*mom_ed[:,:,py,px] \
                                            -phi_ed[:,:,py,px]*mom_ed[:,:,my,mx])*mom_ed[:,:,qy,qx])

@njit
def compute_em_loops(jkpq_em, jpqk_em, jqkp_em,
                    kx_ed, ky_ed, Al_ed, mom_ed, ceff,
                    mx, my, global_ny, nx, im_idx1, im_idx2, block_idx):
    """
    jkpq_em, jpqk_em, jqkp_em の計算を行う関数です。
    """
    for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
        qy = -py-my
        for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):
            qx = -px-mx
            jkpq_em[block_idx,:,py,px] = \
                (-ceff*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*\
                np.real( Al_ed[:,py,px]*mom_ed[im_idx1,:,qy,qx]*mom_ed[im_idx2,:,my,mx]\
                        -Al_ed[:,qy,qx]*mom_ed[im_idx1,:,py,px]*mom_ed[im_idx2,:,my,mx]\
                        +Al_ed[:,py,px]*mom_ed[im_idx2,:,qy,qx]*mom_ed[im_idx1,:,my,mx]\
                        -Al_ed[:,qy,qx]*mom_ed[im_idx2,:,py,px]*mom_ed[im_idx1,:,my,mx]) )
            jpqk_em[block_idx,:,py,px] = \
                (-ceff*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*\
                np.real( Al_ed[:,qy,qx]*mom_ed[im_idx1,:,my,mx]*mom_ed[im_idx2,:,py,px]\
                        -Al_ed[:,my,mx]*mom_ed[im_idx1,:,qy,qx]*mom_ed[im_idx2,:,py,px]\
                        +Al_ed[:,qy,qx]*mom_ed[im_idx2,:,my,mx]*mom_ed[im_idx1,:,py,px]\
                        -Al_ed[:,my,mx]*mom_ed[im_idx2,:,qy,qx]*mom_ed[im_idx1,:,py,px]) )
            jqkp_em[block_idx,:,py,px] = \
                (-ceff*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*\
                np.real( Al_ed[:,my,mx]*mom_ed[im_idx1,:,py,px]*mom_ed[im_idx2,:,qy,qx]\
                        -Al_ed[:,py,px]*mom_ed[im_idx1,:,my,mx]*mom_ed[im_idx2,:,qy,qx]\
                        +Al_ed[:,my,mx]*mom_ed[im_idx2,:,py,px]*mom_ed[im_idx1,:,qy,qx]\
                        -Al_ed[:,py,px]*mom_ed[im_idx2,:,my,mx]*mom_ed[im_idx1,:,qy,qx]) )


def fluiddetaltrans_loop_calc(diag_mx, diag_my, it, iss, phi_ed, Al_ed, mom_ed,
                                flag, outdir, tt):
    import os
    import numpy as np
    from diag_geom import ky_ed, kx_ed, global_ny, nx, global_nz, fcs, tau, Znum, rootg
    import matplotlib.pyplot as plt

    nmom = mom_ed.shape[0]
    mx = diag_mx
    my = diag_my

    # ----- jkpq_es, jpqk_es, jqkp_es の計算 ----------------------------------
    jkpq_es = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1), dtype=np.float64)
    jpqk_es = np.zeros_like(jkpq_es)
    jqkp_es = np.zeros_like(jkpq_es)

    ceff_es = np.array([1, 1, 0.5, 1, 0.166666666666666, 1], dtype=np.float64).reshape(nmom,1)
    ceff_es = ceff_es * 0.5 * fcs[iss] * tau[iss] * Znum[iss]
    phi_ed = phi_ed.reshape(1, 2*global_nz, 2*global_ny+1, 2*nx+1) # momと同一の次元に合わせる。

    # es 計算ループを呼び出します。
    compute_es_loops(jkpq_es, jpqk_es, jqkp_es,
                    kx_ed, ky_ed, phi_ed, mom_ed, ceff_es,
                    mx, my, global_ny, nx)

    # Flux surface average and summation over imom
    cfsrf = np.sum(rootg[:])
    wfct = (rootg[:] / cfsrf).reshape(1, len(rootg), 1, 1)
    jkpq_es_sum = np.sum(jkpq_es * wfct, axis=(0, 1))
    jpqk_es_sum = np.sum(jpqk_es * wfct, axis=(0, 1))
    jqkp_es_sum = np.sum(jqkp_es * wfct, axis=(0, 1))

    # ----- jkpq_em, jpqk_em, jqkp_em の計算 ----------------------------------
    jkpq_em = np.zeros((4, 2*global_nz, 2*global_ny+1, 2*nx+1), dtype=np.float64)
    jpqk_em = np.zeros_like(jkpq_em)
    jqkp_em = np.zeros_like(jkpq_em)

    # imom = 0
    ceff0 = 0.5 * fcs[iss] * tau[iss] * Znum[iss] * (-np.sqrt(tau[iss] / Znum[iss]))  # ※元コードの sqrt 部は Anum 等に依存
    compute_em_loops(jkpq_em, jpqk_em, jqkp_em,
                    kx_ed, ky_ed, Al_ed, mom_ed, ceff0,
                    mx, my, global_ny, nx, 0, 1, 0)

    # imom = 1
    ceff1 = 0.5 * fcs[iss] * tau[iss] * Znum[iss] * (-np.sqrt(tau[iss] / Znum[iss]))
    compute_em_loops(jkpq_em, jpqk_em, jqkp_em,
                    kx_ed, ky_ed, Al_ed, mom_ed, ceff1,
                    mx, my, global_ny, nx, 1, 2, 1)

    # imom = 2
    ceff2 = 0.5 * fcs[iss] * tau[iss] * Znum[iss] * (-np.sqrt(tau[iss] / Znum[iss])) * 0.5
    compute_em_loops(jkpq_em, jpqk_em, jqkp_em,
                    kx_ed, ky_ed, Al_ed, mom_ed, ceff2,
                    mx, my, global_ny, nx, 2, 4, 2)

    # imom = 3
    ceff3 = 0.5 * fcs[iss] * tau[iss] * Znum[iss] * (-np.sqrt(tau[iss] / Znum[iss]))
    compute_em_loops(jkpq_em, jpqk_em, jqkp_em,
                    kx_ed, ky_ed, Al_ed, mom_ed, ceff3,
                    mx, my, global_ny, nx, 3, 5, 3)

    # Flux surface average and summation over imom
    jkpq_em_sum = np.sum(jkpq_em * wfct, axis=(0, 1))
    jpqk_em_sum = np.sum(jpqk_em * wfct, axis=(0, 1))
    jqkp_em_sum = np.sum(jqkp_em * wfct, axis=(0, 1))

    # 出力用に配列を整理する
    kx_sht = np.fft.fftshift(kx_ed)  # この関数で配列データを昇順で並び替え。
    ky_sht = np.fft.fftshift(ky_ed)  # この関数で配列データを昇順で並び替え。
    jkpq_es_sum_sht = np.fft.fftshift(jkpq_es_sum) # データの並び替え
    jpqk_es_sum_sht = np.fft.fftshift(jpqk_es_sum)
    jqkp_es_sum_sht = np.fft.fftshift(jqkp_es_sum)
    jkpq_em_sum_sht = np.fft.fftshift(jkpq_em_sum)
    jpqk_em_sum_sht = np.fft.fftshift(jpqk_em_sum)
    jqkp_em_sum_sht = np.fft.fftshift(jqkp_em_sum)
    m_kx, m_ky = np.meshgrid(kx_sht, ky_sht) # 2D-Plot用メッシュグリッドの作成
    data = np.stack([m_kx, m_ky,
                    jkpq_es_sum_sht, jpqk_es_sum_sht, jqkp_es_sum_sht,
                    jkpq_em_sum_sht, jpqk_em_sum_sht, jqkp_em_sum_sht],
                    axis=2)

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if flag == 'display' or flag == 'savefig' :
        # plot jkpq_es as a function of (px,py) for a given (kx,ky)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"$J_{k(es)}^{p,q}$ "+"($s={:01d},t={:f},k_x={:f},k_y={:f}$)".format(iss, tt, kx_ed[diag_mx], ky_ed[diag_my]))
        ax.set_xlabel(r"Radial wavenumber $p_x$")
        ax.set_ylabel(r"Poloidal wavenumber $p_y$")
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,2],
                            cmap='jet',shading="auto")
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad)
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluiddetailtransinkxky_jkpq_es_x{:04d}y{:04d}s{:01d}_t{:08d}.png'.format(diag_mx,diag_my,iss,it))
            plt.savefig(filename)
            plt.close()

        # plot jpqk_es as a function of (px,py) for a given (kx,ky)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"$J_{p(es)}^{q,k}$ "+"($s={:01d},t={:f},k_x={:f},k_y={:f}$)".format(iss, tt, kx_ed[diag_mx], ky_ed[diag_my]))
        ax.set_xlabel(r"Radial wavenumber $p_x$")
        ax.set_ylabel(r"Poloidal wavenumber $p_y$")
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,3],
                            cmap='jet',shading="auto")
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad)
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluiddetailtransinkxky_jpqk_es_x{:04d}y{:04d}s{:01d}_t{:08d}.png'.format(diag_mx,diag_my,iss,it))
            plt.savefig(filename)
            plt.close()

        # plot jkpq_em as a function of (px,py) for a given (kx,ky)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"$J_{k(em)}^{p,q}$ "+"($s={:01d},t={:f},k_x={:f},k_y={:f}$)".format(iss, tt, kx_ed[diag_mx], ky_ed[diag_my]))
        ax.set_xlabel(r"Radial wavenumber $p_x$")
        ax.set_ylabel(r"Poloidal wavenumber $p_y$")
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,5],
                            cmap='jet',shading="auto")
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad)
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluiddetailtransinkxky_jkpq_em_x{:04d}y{:04d}s{:01d}_t{:08d}.png'.format(diag_mx,diag_my,iss,it))
            plt.savefig(filename)
            plt.close()

        # plot jpqk_em as a function of (px,py) for a given (kx,ky)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        ax.set_title(r"$J_{p(em)}^{q,k}$ "+"($s={:01d},t={:f},k_x={:f},k_y={:f}$)".format(iss, tt, kx_ed[diag_mx], ky_ed[diag_my]))
        ax.set_xlabel(r"Radial wavenumber $p_x$")
        ax.set_ylabel(r"Poloidal wavenumber $p_y$")
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:,:,6],
                            cmap='jet',shading="auto")
        plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        #ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        #ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        fig.colorbar(quad)
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'fluiddetailtransinkxky_jpqk_em_x{:04d}y{:04d}s{:01d}_t{:08d}.png'.format(diag_mx,diag_my,iss,it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'fluiddetailtransinkxky_x{:04d}y{:04d}s{:01d}_t{:08d}.png'.format(diag_mx,diag_my,iss,it))
        with open(filename, 'w') as outfile:
            outfile.write('# loop    = {:d},  t = {:f}\n'.format(it, tt))
            outfile.write('# diag_mx = {:d}, kx = {:f}\n'.format(diag_mx, kx_ed[diag_mx]))
            outfile.write('# diag_my = {:d}, ky = {:f}\n'.format(diag_my, ky_ed[diag_my]))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#           px             py    J_k^pq(es)    J_p^qk(es)    J_q^kp(es)    J_k^pq(em)    J_p^qk(em)    J_q^kp(em)\n')
            for data_slice in data:
                np.savetxt(outfile, data_slice, fmt='%.7e')
                outfile.write('\n')

    else: # otherwise - return data array
        return data


def fluiddetailtrans_loop(diag_mx, diag_my, it, iss, xr_phi, xr_Al, xr_mom, flag, outdir="../data_fluid/"):
    """
    Output detailed symmetric triad transfer function J_k^{p,q} for a given kx[diag_mx],ky[diag_my] at t[it].
    Electrostatic J_k^{p,q}_es and electromagnetic J_k^{p,q}_em are separate.

    Parameters   # fluiddetailtrans用に未修整
    ----------
        diag_mx : int
            index of kx-axis
        diag_my : int
            index of ky-axis
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
        data[2*global_ny+1,2*nx+1,8]: Numpy array, dtype=np.float64
            # px = data[:,:,0]           # -kxmax < px < kxmax
            # py = data[:,:,1]           # -kymax < py < kymax
            # J_k^{p,q}_es = data[:,:,2] # as a function of (px,py), for a given (kx,ky)
            # J_p^{q,k}_es = data[:,:,3] # as a function of (px,py), for a given (kx,ky)
            # J_q^{k,p}_es = data[:,:,4] # as a function of (px,py), for a given (kx,ky)
            # J_k^{p,q}_em = data[:,:,5] # as a function of (px,py), for a given (kx,ky)
            # J_p^{q,k}_em = data[:,:,6] # as a function of (px,py), for a given (kx,ky)
            # J_q^{k,p}_em = data[:,:,7] # as a function of (px,py), for a given (kx,ky)
    """
    import os
    import numpy as np
    from diag_geom import rootg, g0, g1, bb, Anum, Znum, tau, fcs, sgn # 計算に必要なglobal変数を呼び込む
    from diag_rb import safe_compute

    nx = int((len(xr_phi['kx'])-1)/2)
    global_ny = int(len(xr_phi['ky'])-1)
    global_nz = int(len(xr_phi['zz'])/2)
    nmom = len(xr_mom["imom"])

    # 時刻t[it],粒子種issにおける４次元複素mom[mom,zz,ky,kx]を切り出す
    if 'rephi' in xr_phi and 'imphi' in xr_phi:
        rephi = xr_phi['rephi'][it,:,:,:]  # dim: t, is, imom, zz, ky, kx
        imphi = xr_phi['imphi'][it,:,:,:]  # dim: t, is, imom, zz, ky, kx
        phi = rephi + 1.0j*imphi
        reAl = xr_Al['reAl'][it,:,:,:]     # dim: t, is, imom, zz, ky, kx
        imAl = xr_Al['imAl'][it,:,:,:]     # dim: t, is, imom, zz, ky, kx
        Al = reAl + 1.0j*imAl
        remom = xr_mom['remom'][it,iss,:,:,:,:]  # dim: t, is, imom, zz, ky, kx
        immom = xr_mom['immom'][it,iss,:,:,:,:]  # dim: t, is, imom, zz, ky, kx
        mom = remom + 1.0j*immom
    elif 'phi' in xr_phi:
        phi = xr_phi['phi'][it,:,:,:]  # dim: t, is, imom, zz, ky, kx
        Al = xr_Al['Al'][it,:,:,:]     # dim: t, is, imom, zz, ky, kx
        mom = xr_mom['mom'][it,iss,:,:,:,:]  # dim: t, is, imom, zz, ky, kx

    phi = safe_compute(phi)
    Al = safe_compute(Al)
    mom = safe_compute(mom)

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

    # 共役複素数を含む拡張した配列(*_ed)の作成
    phi_ed = np.zeros((2*global_nz, 2*global_ny+1, 2*nx+1),dtype=np.complex128)
    phi_ed[:, 0:global_ny+1, 0:nx+1] = phi[:, 0:global_ny+1, nx:2*nx+1]
    phi_ed[:, 0:global_ny+1, nx+1:2*nx+1] = phi[:, 0:global_ny+1, 0:nx]
    phi_ed[:, global_ny+1:2*global_ny+1, nx+1:2*nx+1] = np.conj(phi[:, global_ny:0:-1, 2*nx:nx:-1])
    phi_ed[:, global_ny+1:2*global_ny+1, 0:nx+1] = np.conj(phi[:, global_ny:0:-1, nx::-1])
    Al_ed = np.zeros((2*global_nz, 2*global_ny+1, 2*nx+1), dtype=np.complex128)
    Al_ed[:, 0:global_ny+1, 0:nx+1] = Al[:, 0:global_ny+1, nx:2*nx+1]
    Al_ed[:, 0:global_ny+1, nx+1:2*nx+1] = Al[:, 0:global_ny+1, 0:nx]
    Al_ed[:, global_ny+1:2*global_ny+1, nx+1:2*nx+1] = np.conj(Al[:, global_ny:0:-1, 2*nx:nx:-1])
    Al_ed[:, global_ny+1:2*global_ny+1, 0:nx+1] = np.conj(Al[:, global_ny:0:-1, nx::-1])
    mom_ed = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1), dtype=np.complex128)
    mom_ed[:, :, 0:global_ny+1, 0:nx+1] = np.array(mom[:,:, 0:global_ny+1, nx:2*nx+1])
    mom_ed[:, :, 0:global_ny+1, nx+1:2*nx+1] = mom[:, :, 0:global_ny+1, 0:nx]
    mom_ed[:, :, global_ny+1:2*global_ny+1, nx+1:2*nx+1] = np.conj(mom[:, :, global_ny:0:-1, 2*nx:nx:-1])
    mom_ed[:, :, global_ny+1:2*global_ny+1, 0:nx+1] = np.conj(mom[:, :, global_ny:0:-1, nx::-1])

    data = fluiddetaltrans_loop_calc(diag_mx, diag_my, it, iss, phi_ed, Al_ed, mom_ed, flag, outdir, float(xr_mom['t'][it]))

    return data


if (__name__ == '__main__'):
    import os
    from diag_geom import geom_set
    from diag_rb import rb_open
    from time import time as timer
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')

    xr_phi = rb_open('../../phi/gkvp.phi.*.zarr/')
    xr_Al  = rb_open('../../phi/gkvp.Al.*.zarr/')
    xr_mom = rb_open('../../phi/gkvp.mom.*.zarr/')

    diag_mx = 0; diag_my = 3
    iss = 0
    outdir='../data/fluiddetailtrans/'
    os.makedirs(outdir, exist_ok=True)
    s_time = timer()
    for it in range(0, len(xr_phi['t']), len(xr_phi['t'])//10):
        fluiddetailtrans_loop(diag_mx, diag_my, it, iss, xr_phi, xr_Al, xr_mom, flag="savefig", outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))
    it = len(xr_phi.t)-1
    fluiddetailtrans_loop(diag_mx, diag_my, it, iss, xr_phi, xr_Al, xr_mom, flag="display")
    fluiddetailtrans_loop(diag_mx, diag_my, it, iss, xr_phi, xr_Al, xr_mom, flag="savetxt", outdir=outdir)


# In[ ]:




