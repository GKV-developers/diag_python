#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 2D electrostatic potential phi in cylindrical (R,Z) of the poloidal cross-section

Module dependency: diag_geom, diag_fft

Third-party libraries: numpy, scipy, matplotlib
"""

def field_aligned_coordinates_salpha(wxx,zz,rho,q_0,s_hat,eps_r):
    """
    Calculate major radius R, height Z, safety factor profile q(r) from GKV field-aligned coordinates
    
    Parameters
    ----------
        wxx : float
            Radial x in GKV coordinates
        zz : Numpy array
            Field-aligned z in GKV coordinates
        rho, q_0, s_hat, eps_r, ... : float
            Coefficients for magnetic geometry
    Returns
    -------
        mr : Numpy array
            Major radius R
        z_car : Numpy array
            Height Z
        q_r : float
            Safety factor profile q(r)
    """
    import numpy as np
    wtheta=zz
    wsr=eps_r+rho*wxx

    wmr=1+wsr*np.cos(wtheta)
    wz_car=wsr*np.sin(wtheta)
    
    q_r = q_0 * (1.0 + s_hat * wxx * rho / eps_r)
    return wmr, wz_car, q_r



def field_aligned_coordinates_miller(wxx,zz,rho,q_0,s_hat,eps_r,dRmildr,dZmildr,kappa,s_kappa,delta,s_delta,zetasq,s_zetasq):
    """
    Calculate major radius R, height Z, safety factor profile q(r) from GKV field-aligned coordinates
    
    Parameters
    ----------
        wxx : float
            Radial x in GKV coordinates
        zz : Numpy array
            Field-aligned z in GKV coordinates
        rho, q_0, s_hat, eps_r, ... : float
            Coefficients for magnetic geometry
    Returns
    -------
        mr : Numpy array
            Major radius R
        z_car : Numpy array
            Height Z
        q_r : float
            Safety factor profile q(r)
    """
    import numpy as np
    wtheta=zz
    wsr=eps_r+rho*wxx
    
    kappa_r = kappa * (1.0 + s_kappa * wxx * rho / eps_r)
    delta_r = delta + np.sqrt(1.0 - delta**2) * s_delta * wxx * rho / eps_r
    zetasq_r = zetasq + s_zetasq * wxx * rho / eps_r
    Rmil_r = 1.0 + dRmildr * wxx * rho
    Zmil_r = 0.0 + dZmildr * wxx * rho
    wmr = Rmil_r + wsr * np.cos(wtheta + np.arcsin(delta_r) * np.sin(wtheta))
    wz_car = Zmil_r + wsr * kappa_r * np.sin(wtheta + zetasq_r * np.sin(2*wtheta))
    
    q_r = q_0 * (1.0 + s_hat * wxx * rho / eps_r)
    return wmr, wz_car, q_r



def phiinrz(it, xr_phi, flag=None, n_alp=4, zeta=0.0, nxw=None, nyw=None, nzw=None, outdir="./data/"):
    """
    Output 2D electrostatic potential phirz in cylindrical (R,Z) of the poloidal cross-section at t[it], zeta.
    
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
        n_alp : int, optional
            1/n_alp partition of torus
        zeta : float, optional
            Toroidal angle of the poloidal cross-section
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = nxw in gkvp_header.f90 
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = nxw in gkvp_header.f90
        nzw : int, optional
            (grid number in poloidal direction) = 2*nzw+1
            # Default: nzw = int(nyw*n_alp*q_0)
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        data[2*nzw+1,2*nxw+1,3]: Numpy array, dtype=np.float64
            #  Major radius R = data[:,:,0]
            #        Height Z = data[:,:,1]
            # Potential phirz = data[:,:,2]    
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import fft, interpolate
    from diag_geom import nml, dj, ck
    from diag_geom import nxw as nxw_geom
    from diag_geom import nyw as nyw_geom
    import time
    
    t1=time.time()
    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_phi['kx'])-1)/2)
    global_ny = int(len(xr_phi['ky'])-1)
    global_nz = int(len(xr_phi['zz'])/2)
    if (nxw == None):
        nxw = nxw_geom
    if (nyw == None):
        nyw = nyw_geom
     
    # GKV座標(x,y,z)を作成
    kymin = float(xr_phi['ky'][1])
    ky = kymin * np.arange(global_ny+1)
    ly = np.pi / kymin
    if nx==0:
        s_hat = nml['confp']['s_hat']
        m_j = nml['nperi']['m_j']
        if (abs(s_hat) < 1e-10):
            kxmin = kymin
        elif (m_j == 0):
            kxmin = kymin
        else:
            kxmin = abs(2*np.pi*s_hat*kymin / m_j)
    else:
        kxmin = float(xr_phi['kx'][nx+1])
    lx = np.pi / kxmin
    xx = np.linspace(-lx,lx,2*nxw,endpoint=False)
    lz = - float(xr_phi['zz'][0])
    zz = np.linspace(-lz,lz,2*global_nz+1)
        
    # トロイダル座標系のパラメータ設定。トロイダル方向分割数n_alpが大きい程、L_ref/rho_refが大きい描画
    eps_r = nml['confp']['eps_r']
    q_0 = nml['confp']['q_0']
    s_hat = nml['confp']['s_hat']
    n_tht = nml['nperi']['n_tht'] # Modify for n_tht
    rho = np.pi*eps_r/(q_0*ly*n_alp) # = Larmor radius rho_ref/L_ref
    # Parameters for Miller geometry
    dRmildr=-0.1;dZmildr=0;kappa=1.5;s_kappa=0.7;delta=0.4;s_delta=1.3;zetasq=0;s_zetasq=0
    #print("# Plotted as Larmor radius rho/L_ref = ", rho)
    if lx*rho > eps_r:
        print("# WARNING in out_mominvtk. lx*rho < eps_r is recommended. Set larger n_alp.")
        print("# lx=",lx,", rho=",rho,", eps_r=",eps_r,", n_alp=",n_alp )
    
    # 時刻t[it]における三次元複素phi[z,ky,kx]を切り出す
    rephi = xr_phi['rephi'][it,:,:,:]  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][it,:,:,:]  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi
    #t2=time.time();print("#time(init)=",t2-t1)
    
    t1=time.time()
    # 磁力線z方向の準周期境界条件
    phi_zkykx = np.zeros([2*global_nz+1,global_ny+1,2*nx+1],dtype=np.complex128)
    phi_zkykx[0:2*global_nz,:,:] = phi[:,:,:]
    iz = 2*global_nz
#     for my in range(global_ny+1):
#         for mx in range(2*nx+1):
#             mwp = mx - dj[my]
#             if (mwp < 0 or mwp > 2*nx):
#                 phi_zkykx[iz,my,mx] = 0.0
#             else:
#                 phi_zkykx[iz,my,mx] = np.conjugate(ck[my]) * phi[0,my,mwp]
    for my in range(global_ny+1):
        #phi_zkykx[iz,my,0:2*nx+1] = np.conjugate(ck[my]) * phi[0,my,0-dj[my]:2*nx+1-dj[my]]
        if dj[my]<0:
            if 0<2*nx+1+dj[my] and 0-dj[my]<2*nx+1: 
                phi_zkykx[iz,my,0:2*nx+1+dj[my]] = np.conjugate(ck[my]) * phi[0,my,0-dj[my]:2*nx+1]
        else:
            if 0+dj[my]<2*nx+1 and 0<2*nx+1-dj[my]:
                phi_zkykx[iz,my,0+dj[my]:2*nx+1] = np.conjugate(ck[my]) * phi[0,my,0:2*nx+1-dj[my]]
    #t2=time.time();print("#time(bound)=",t2-t1)
    
    t1=time.time()
    # x方向のみ逆FFT。入力 phi[z,ky,kx] -> 出力 phi[z,ky,x]
    phi_zkyx = np.zeros([2*global_nz+1,global_ny+1,2*nxw],dtype=np.complex128) # fft.ifft用Numpy配列
    phi_zkyx[:,:, 0:nx+1] = phi_zkykx[:,:, nx:2*nx+1] # 波数空間配列の並び替え
    phi_zkyx[:,:, 2*nxw-nx:2*nxw] = phi_zkykx[:,:, 0:nx]
    phi_zkyx = fft.ifft(phi_zkyx,axis=2) * (2*nxw) # phi[x] = Sum_kx phi[kx]*exp[i(kx*x)]
    phi_zkyx = np.concatenate((phi_zkyx,phi_zkyx[:,:,0:1]),axis=2)
    #t2=time.time();print("#time(fft_x)=",t2-t1)
    
    t1=time.time()
    # z方向を2*global_nz+1点から2*nzw+1点に補完する
    if (nzw == None):
        nzw=int(nyw*n_alp*q_0)
    poly_interp = interpolate.CubicSpline(zz,phi_zkyx,axis=0)
    zz_interp = np.linspace(-lz/n_tht,lz/n_tht,2*nzw+1) # Modify for n_tht>1
    phi_interp = poly_interp(zz_interp)
    #t2=time.time();print("#time(interp)=",t2-t1)
    
    t1=time.time()
    ### Prepare structured grid
    npol=2*nzw+1
    nrad=2*nxw+1
    xx=np.linspace(-lx,lx,nrad)
    wmr=np.zeros([npol,nrad],dtype=np.float64)
    wz_car=np.zeros([npol,nrad],dtype=np.float64)
    for ix in range(nrad):
        wxx = xx[ix]
        # Circular s-alpha geometry
        wmr[:,ix],wz_car[:,ix],_ = field_aligned_coordinates_salpha(wxx,zz_interp,rho,q_0,s_hat,eps_r)
#         # Non-circular Miller geometry
#         wmr[:,ix],wz_car[:,ix],_ = field_aligned_coordinates_miller(wxx,zz_interp,rho,q_0,s_hat,eps_r,  
#                                                                     dRmildr,dZmildr,kappa,s_kappa,delta,s_delta,zetasq,s_zetasq)
    #t2=time.time();print("#time(grid)=",t2-t1)

    # y方向にも逆FFT。ただし、位置zetaの点のみ評価。
    t1=time.time()
    q_r = (q_0 * (1.0 + s_hat * xx * rho / eps_r)).reshape(1,1,nrad)
    wtheta = zz_interp[:].reshape(npol,1,1)
    wyy = eps_r*(q_r*wtheta -zeta)/(q_0*rho)
    wyy = wyy + ly # since -ly<=yy<ly in GKV, rather than 0<=yy<2*ly
    phi_pol = 2*np.sum(np.exp(1j*ky.reshape(1,global_ny+1,1)*wyy) * phi_interp[:,:,:], axis=1).real
    phi_pol[:,:] = phi_pol[:,:] - phi_interp[:,0,:].real
    #t2=time.time();print("#time(fft_y)=",t2-t1)

    # 出力用に配列を整理する
    data = np.stack([wmr,wz_car,phi_pol],axis=2)

    ### データ出力 ###
    # 場合分け：flag = "display", "savefig", "savetxt", それ以外なら配列dataを返す
    if (flag == "display" or flag == "savefig"):
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(111)
        vmax=np.max(abs(data[:,:,2]))
        quad = ax.pcolormesh(data[:,:,0], data[:,:,1], data[:-1,:-1,2],
                             cmap='jet',shading="flat",vmin=-vmax,vmax=vmax)
        ax.set_title("t = {:f}".format(float(xr_phi['t'][it])))
        ax.set_aspect('equal')
        ax.set_xlabel("R")
        ax.set_ylabel("Z")
        fig.colorbar(quad)
        
        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()
            
        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'phiinrz_t{:08d}.png'.format(it)) 
            plt.savefig(filename)
            plt.close()
            
    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'phiinrz_t{:08d}.dat'.format(it))
        with open(filename, 'w') as outfile:
            outfile.write('# n_alp = {:d}\n'.format(n_alp))
            outfile.write('# zeta = {:f}\n'.format(zeta))
            outfile.write('# it = {:d}, t = {:f}\n'.format(it, float(xr_phi['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data.shape))
            outfile.write('#            R              Z            phi\n')
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
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    
    
    ### Examples of use ###
    
    
    ### phiinrz ###
    #help(phiinrz)
    xr_phi = rb_open('../../post/data/phi.*.nc')
    #print(xr_phi)
    print("# Plot phi in poloidal cross-section (R,Z) at t[it].")
    outdir='../data/phiinrz/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_phi['t']),10):
        t1=time.time()
        phiinrz(it, xr_phi, flag="savefig", outdir=outdir)
        #t2=time.time();print("time=",t2-t1)
    
    print("# Display phi in poloidal cross-section (R,Z) at t[it].")
    phiinrz(it, xr_phi, flag="display")
    print("# Save phi in poloidal cross-section (R,Z) at t[it] as text files.")
    phiinrz(it, xr_phi, flag="savetxt", outdir=outdir)


# In[ ]:




