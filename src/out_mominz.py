#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output field-aligned profile of a Fourier mode phi(z)

Module dependency: diag_geom

Third-party libraries: numpy, matplotlib
"""

def phiinz(it, my, mx, xr_phi, xr_Al, normalize=None, flag=None, outdir="./data/" ):
    """
    Output phi(z) at t[it], ky[my], kx[mx]

    Parameters
    ----------
        it : int
            index of t-axis
        my : int
            index of ky-axis
        mx : int
            index of kx-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        xr_Al : xarray Dataset
            xarray Dataset of Al.*.nc, read by diag_rb
        normalize : str, optional
            # normalize="phi0" - output phi(z)/phi(z=0)
            # normalize="Al0"  - output phi(z)/(Al(z=0)/np.sqrt(beta))
            # otherwise        - No normalization
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
        data[zz, re_phi, im_phi]: Numpy array, dtype=np.float64

    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_geom import nml
    from diag_rb import safe_compute

    ### パラメータ: gkvp_namelist から読み取る
    beta  = nml['physp']['beta']

    # 時刻 t[it],位置 ky[my] & kx[mx]における一次元複素phi[zz]を切り出す
    rephi = xr_phi['rephi'][it,:, my, mx]  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][it,:, my, mx]  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi
    phi = safe_compute(phi)

    # 規格化
    if (normalize == 'phi0'):
        rephi0 = xr_phi['rephi'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        imphi0 = xr_phi['imphi'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        phi0 = complex(rephi0 + 1.0j*imphi0)
        phi = phi / phi0
    elif (normalize == 'Al0'):
        reAl0 = xr_Al['reAl'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        imAl0 = xr_Al['imAl'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        Al0 = complex(reAl0 + 1.0j*imAl0)
        phi = phi / (Al0 / np.sqrt(beta))

    # 出力用の配列を整理する
    data = np.stack([xr_phi['zz'], phi.real, phi.imag], axis=1)

    # 出力の場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if (flag == "display" or flag == "savefig"):
        fig = plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.plot(data[:,0], data[:,1], label=r'Re[$\phi_k$]')
        ax.plot(data[:,0], data[:,2], label=r'Im[$\phi_k$]')
        ax.set_title("$t=${:f}, $k_x=${:f}, $k_y=${:f}".format(float(xr_phi['t'][it]), float(xr_phi['kx'][mx]), float(xr_phi['ky'][my])))
        ax.set_xlabel("Field-aligned coordinate z")
        if (normalize == 'phi0'):
            ax.set_ylabel(r"Electrostatic potential $\phi_k(z)/[\phi_k(z=0)]$")
        elif (normalize == 'Al0'):
            ax.set_ylabel(r"Electrostatic potential $\phi_k(z)/[v_A A_{\parallel k}(z=0)]$")
        else:
            ax.set_ylabel(r"Electrostatic potential $\phi_k$")
        ax.legend()
        ax.grid()

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            if (normalize == 'phi0'):
                filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}_phi0norm.png'.format(mx, my, it))
            elif (normalize == 'Al0'):
                filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}_Al0norm.png'.format(mx, my, it))
            else:
                filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}.png'.format(mx, my, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        if (normalize == 'phi0'):
            filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}_phi0norm.dat'.format(mx, my, it))
        elif (normalize == 'Al0'):
            filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}_Al0norm.dat'.format(mx, my, it))
        else:
            filename = os.path.join(outdir,'phiinz_mx{:04d}my{:04d}_t{:08d}.dat'.format(mx, my, it))
        with open(filename, 'w') as outfile:
            outfile.write('# loop = {:d}, time = {:f}\n'.format(it, float(xr_phi['t'][it])))
            outfile.write('# mx = {:d}, kx = {:f}\n'.format(mx, float(xr_phi['kx'][mx])))
            outfile.write('# my = {:d}, ky = {:f} \n'.format(my, float(xr_phi['ky'][my])))
            outfile.write('#           zz        Re[phi]       Im[phi]\n')
            np.savetxt(outfile, data, fmt='%.7e', delimiter='   ')
            outfile.write('\n')

    else: # otherwise - return data array
        return data




def phiinz_connect(it, my, mx, xr_phi, xr_Al, normalize=None, flag=None, outdir="./data/" ):
    """
    Output phi(z) at t[it], ky[my], kx[mx]

    Parameters
    ----------
        it : int
            index of t-axis
        my : int
            index of ky-axis
        mx : int
            index of kx-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        xr_Al : xarray Dataset
            xarray Dataset of Al.*.nc, read by diag_rb
        normalize : str, optional
            # normalize="phi0" - output phi(z)/phi(z=0)
            # normalize="Al0"  - output phi(z)/(Al(z=0)/np.sqrt(beta))
            # otherwise        - No normalization
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
        data[zz, re_phi, im_phi]: Numpy array, dtype=np.float64

    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_geom import nml, ck ,dj
    from diag_rb import safe_compute

    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_phi['kx'])-1)/2)

    ### パラメータ: gkvp_namelist から読み取る
    beta  = nml['physp']['beta']
    n_tht = nml['nperi']['n_tht']

    # 時刻 t[it],位置 ky[my]における2次元複素phi[zz,kx]を切り出す
    if 'rephi' in xr_phi and 'imphi' in xr_phi:
        rephi = xr_phi['rephi'][it,:,my,:]  # dim: t, zz, ky, kx
        imphi = xr_phi['imphi'][it,:,my,:]  # dim: t, zz, ky, kx
        phi_zx = rephi + 1.0j*imphi
    elif 'phi' in xr_phi:
        phi_zx = xr_phi['phi'][it,:,my,:]  # dim: t, zz, ky, kx
    phi_zx = safe_compute(phi_zx)

    # 規格化
    if (normalize == 'phi0'):
        if 'rephi' in xr_phi and 'imphi' in xr_phi:
            rephi0 = xr_phi['rephi'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
            imphi0 = xr_phi['imphi'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
            phi0 = rephi0 + 1.0j*imphi0
        elif 'phi' in xr_phi:
            phi0 = xr_phi['phi'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        phi0 = safe_compute(phi0)
        phi0 = complex(phi0)
        phi_zx = phi_zx / phi0
    elif (normalize == 'Al0'):
        if 'reAl' in xr_Al and 'imAl' in xr_Al:
            reAl0 = xr_Al['reAl'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
            imAl0 = xr_Al['imAl'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
            Al0 = reAl0 + 1.0j*imAl0
        elif 'Al' in xr_Al:
            Al0 = xr_Al['Al'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        reAl0 = xr_Al['reAl'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        imAl0 = xr_Al['imAl'][it,:,my,mx].sel(zz=0,method="nearest")  # dim: t, zz, ky, kx
        Al0 = reAl0 + 1.0j*imAl0
        Al0 = safe_compute(Al0)
        Al0 = complex(Al0)
        phi_zx = phi_zx / (Al0 / np.sqrt(beta))

    if (dj[my] == 0 ):
        # 出力用に配列を整理する
        zz = np.array(xr_phi['zz'])
        phi = np.array(phi_zx[:,mx])

    else:
        zz0 = np.array(xr_phi['zz'])
        zz = []
        phi = []
        # case of connect_min
        connect_min = int((nx + mx - int((len(xr_phi['kx'])-1)/2))/(abs(dj[my])))
        if (connect_min != 0 ):
            for iconnect in range(connect_min, 0, -1):
                mxw = mx +iconnect*dj[my]
                wphi = phi_zx[:,mxw]
                wzz = -2*np.pi*n_tht * float(iconnect) + zz0 # 初期設定のzz0に加算して、新しい座標値を作る
                wphi = ck[my]**iconnect * wphi
                zz.append(wzz)
                phi.append(wphi)
        # case of connect_max
        connect_max = int((nx - mx + int((len(xr_phi['kx'])-1)/2))/(abs(dj[my])))
        for iconnect in range(0, connect_max+1):  # connect_max回分を出力
            mxw = mx - iconnect * dj[my]
            wphi = phi_zx[:,mxw]
            wzz = 2*np.pi*n_tht * float(iconnect) + zz0 # 初期設定のzz0に加算して、新しい座標値を作る
            wphi = np.conj(ck[my]**iconnect) * wphi
            zz.append(wzz)
            phi.append(wphi)
        zz = np.array(zz).ravel()
        phi = np.array(phi).ravel()

    # 保存用出力データフォーマット（savetxt）の作成（3つの変数を3列に並べる）
    data =np.stack([zz, phi.real, phi.imag], axis=1)

    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if (flag == "display" or flag == "savefig"):
        fig = plt.figure(figsize=(12,4))
        ax = fig.add_subplot(111)
        ax.plot(data[:,0], data[:,1], label=r'Re[$\phi_k$]')
        ax.plot(data[:,0], data[:,2], label=r'Im[$\phi_k$]')
        ax.set_title("$t=${:f}, $k_x=${:f}, $k_y=${:f}".format(float(xr_phi['t'][it]), float(xr_phi['kx'][mx]), float(xr_phi['ky'][my])))
        ax.set_xlabel("Field-aligned coordinate z")
        if (normalize == 'phi0'):
            ax.set_ylabel(r"Electrostatic potential $\phi_k(z)/[\phi_k(z=0)]$")
        elif (normalize == 'Al0'):
            ax.set_ylabel(r"Electrostatic potential $\phi_k(z)/[v_A A_{\parallel k}(z=0)]$")
        else:
            ax.set_ylabel(r"Electrostatic potential $\phi_k$")
        ax.legend()
        ax.grid()

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            if (normalize == 'phi0'):
                filename = os.path.join(outdir,'phiinz_connect_mx{:04d}my{:04d}_t{:08d}_phi0norm.png'.format(mx, my, it))
            elif (normalize == 'Al0'):
                filename = os.path.join(outdir,'phiinz_connect_mx{:04d}my{:04d}_t{:08d}_Al0norm.png'.format(mx, my, it))
            else:
                filename = os.path.join(outdir,'phiinz_connect_mx{:04d}my{:04d}_t{:08d}.png'.format(mx, my, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        if (normalize == 'phi0'):
            filename = os.path.join(outdir,'phiinz_connect_mx{:04d}my{:04d}_t{:08d}_phi0norm.dat'.format(mx, my, it))
        elif (normalize == 'Al0'):
            filename = os.path.join(outdir,'phiinz_connect_mx{:04d}my{:04d}_t{:08d}_Al0norm.dat'.format(mx, my, it))
        else:
            filename = os.path.join(outdir,'phiinz_connect_mx{:04d}my{:04d}_t{:08d}.dat'.format(mx, my, it))
        with open(filename, 'w') as outfile:
            outfile.write('# loop = {:d}, time = {:f}\n'.format(it, float(xr_phi['t'][it])))
            outfile.write('# mx = {:d}, kx = {:f}\n'.format(mx, float(xr_phi['kx'][mx])))
            outfile.write('# my = {:d}, ky = {:f} \n'.format(my, float(xr_phi['ky'][my])))
            outfile.write('#           zz        Re[phi]       Im[phi]\n')
            np.savetxt(outfile, data, fmt='%.7e', delimiter='   ')
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


    ### phiinz ###
    #help(phiinz_connect)
    xr_phi = rb_open('../../phi/gkvp.phi.*.zarr/')
    xr_Al = rb_open('../../phi/gkvp.Al.*.zarr/')
    #print(xr_phi)
    from diag_geom import nx
    mx = nx # Index in kx
    my = 1  # Index in ky
    kx = float(xr_phi.kx[mx])
    ky = float(xr_phi.ky[my])
    print("# Plot phi[z] at t[it], ky[my], kx[mx]. kx=",kx, ", ky=",ky)
    outdir='../data/phiinz/'
    os.makedirs(outdir, exist_ok=True)
    s_time = timer()
    for it in range(0,len(xr_phi['t']),len(xr_phi['t'])//10):
        phiinz_connect(it, my, mx, xr_phi, xr_Al, normalize=None, flag='savefig', outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))

    print("# Display phi[z] at t[it], ky[my], kx[mx]. kx=",kx, ", ky=",ky)
    phiinz_connect(it, my, mx, xr_phi, xr_Al, normalize='phi0', flag='display')
    print("# Save phi[z] as text files at t[it], ky[my], kx[mx]. kx=",kx, ", ky=",ky)
    phiinz_connect(it, my, mx, xr_phi, xr_Al, flag='savetxt', outdir=outdir)


# In[ ]:




