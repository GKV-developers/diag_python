#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 2D electrostatic potential phi in cylindrical (R,Z) of the poloidal cross-section

Module dependency: diag_geom, diag_fft

Third-party libraries: numpy, matplotlib
"""


def phi_zkykx2zyx_fluxtube(phi, nxw, nyw):
    """
    Extend boundary values, from phi[2*global_nz,global_ny+1,2*nx+1] in (z,ky,kx) 
    to phi[2*global_nz+1,2*nyw+1,2*nxw+1] in (z,y,x)
    
    Parameters
    ----------
        phi[2*global_nz,global_ny+1,2*nx+1] : Numpy array, dtype=np.complex128
            xarray Dataset of phi.*.nc, read by diag_rb
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = int(nx*1.5)+1 
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = int(global_ny*1.5)+1 

    Returns
    -------
        data[2*nyw,2*nxw,3] : Numpy array, dtype=np.float64
            # xx = data[:,:,0]
            # yy = data[:,:,1]
            # phixy = data[:,:,2]    
    """
    import numpy as np
    from diag_geom import dj, ck
    from diag_fft import fft_backward_xyz

    # GKVパラメータを換算する
    nx = int((phi.shape[2]-1)/2)
    global_ny = int(phi.shape[1]-1)
    global_nz = int(phi.shape[0]/2)
    
    # Pseudo-periodic boundary condition along a field line z
    phi_zkykx = np.zeros([2*global_nz+1,global_ny+1,2*nx+1],dtype=np.complex128)
    phi_zkykx[0:2*global_nz,:,:] = phi[:,:,:]
    iz = 2*global_nz
    for my in range(global_ny+1):
        for mx in range(2*nx+1):
            mwp = mx - dj[my]
            if (mwp < 0 or mwp > 2*nx):
                phi_zkykx[iz,my,mx] = 0.0
            else:
                phi_zkykx[iz,my,mx] = np.conjugate(ck[my]) * phi[0,my,mwp]

    phi_zyx = np.zeros([2*global_nz+1,2*nyw+1,2*nxw+1],dtype=np.float64)

    # diag_fft.pyから関数 fft_backward_xyzを呼び出し、2次元逆フーリエ変換 phi[z,ky,kx]->phi[z,y,x]
    phi_zyx[:,0:2*nyw,0:2*nxw] = fft_backward_xyz(phi_zkykx, nxw=nxw, nyw=nyw)
    
    # Periodic boundary in (x,y)
    phi_zyx[:,2*nyw,:] = phi_zyx[:,0,:]
    phi_zyx[:,:,2*nxw] = phi_zyx[:,:,0]
    return phi_zyx


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




def phiinrz(it, xr_phi, flag=None, n_alp=4, zeta=0.0, nxw=None, nyw=None):
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
            # Default: nxw = int(nx*1.5)+1 
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = int(global_ny*1.5)+1 

    Returns
    -------
        data[:,:,3]: Numpy array, dtype=np.float64
            #  Major radius R = data[:,:,0]
            #        Height Z = data[:,:,1]
            # Potential phirz = data[:,:,2]    
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_geom import nml

    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_phi['kx'])-1)/2)
    global_ny = int(len(xr_phi['ky'])-1)
    global_nz = int(len(xr_phi['zz'])/2)
    if (nxw == None):
        nxw = int(nx*1.5)+1
    if (nyw == None):
        nyw = int(global_ny*1.5)+1

    # 時刻t[it]における三次元複素phi[z,ky,kx]を切り出す
    rephi = xr_phi['rephi'][it,:,:,:]  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][it,:,:,:]  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi

    # 境界上の点を拡張した phi[2*global_nz+1,2*nyw+1,2*nxw+1] in (z,y,x)
    phi_zyx = phi_zkykx2zyx_fluxtube(phi,nxw=nxw,nyw=nyw) # Numpy array
    #print(phi_zyx.shape, phi_zyx.dtype)
    
    # GKV座標(x,y,z)を作成
    kxmin = float(xr_phi['kx'][nx+1])
    lx = np.pi / kxmin
    xx = np.linspace(-lx,lx,2*nxw+1)
    kymin = float(xr_phi['ky'][1])
    ly = np.pi / kymin
    yy = np.linspace(-ly,ly,2*nyw+1)
    lz = - float(xr_phi['zz'][0])
    zz = np.linspace(-lz,lz,2*global_nz+1)
    
    # トロイダル座標系のパラメータ設定。トロイダル方向分割数n_alpが大きい程、L_ref/rho_refが大きい描画
    eps_r = nml['confp']['eps_r']
    q_0 = nml['confp']['q_0']
    s_hat = nml['confp']['s_hat']
    rho = np.pi*eps_r/(q_0*ly*n_alp) # = Larmor radius rho_ref/L_ref
    # Parameters for Miller geometry
    dRmildr=-0.1;dZmildr=0;kappa=1.5;s_kappa=0.7;delta=0.4;s_delta=1.3;zetasq=0;s_zetasq=0
    #print("# Plotted as Larmor radius rho/L_ref = ", rho)
    
    # 磁力線(ix,i_alp,iy)毎に、磁力線z方向に関数補間し、zeta=0.0でのR,Z,phiの値を求める。
    wzz = zz
    wtheta = wzz
    scatt=[]
    import scipy.interpolate as interpolate
    for ix in range(2*nxw+1): # Define flux surface
        wxx=xx[ix]
        # Circular s-alpha geometry
        wmr, wz_car, q_r = field_aligned_coordinates_salpha(wxx,wzz,rho,q_0,s_hat,eps_r)
#         # Non-circular Miller geometry
#         wmr, wz_car, q_r = field_aligned_coordinates_miller(wxx,wzz,rho,q_0,s_hat,eps_r,  
#                                                             dRmildr=-0.1,dZmildr=0,kappa=1.5,s_kappa=0.7,delta=0.4,s_delta=1.3,zetasq=0,s_zetasq=0)
        for i_alp in range(n_alp): # Define flux tube
            for iy in range(2*nyw+1): # Define field line
                wyy=yy[iy]
                wzeta=q_r*wtheta - q_0*wyy*rho/eps_r - i_alp*2*np.pi/n_alp
                wphi=phi_zyx[:,iy,ix]

                ### Interpolation: zeta -> theta, phi
                poly_phi = interpolate.Akima1DInterpolator(wzeta,wphi,axis=0)
                poly_theta = interpolate.Akima1DInterpolator(wzeta,wtheta,axis=0)
                poly_mr = interpolate.Akima1DInterpolator(wzeta,wmr,axis=0)
                poly_z_car = interpolate.Akima1DInterpolator(wzeta,wz_car,axis=0)

                ### Find points at mod(zeta,2*pi)==0
                iz0=int(wzeta.min()/(2*np.pi)) 
                for zeta0 in np.arange(2*np.pi*iz0,wzeta.max(),2*np.pi): 
                    theta0=poly_theta(zeta0)
                    mr0=poly_mr(zeta0)
                    z_car0=poly_z_car(zeta0)
                    phi0=poly_phi(zeta0)
                    #print(ix,iy,i_alp,zeta0,wxx,theta0,mr0,z_car0,phi0)
                    scatt.append([wxx,theta0,mr0,z_car0,phi0])

    scatt=np.array(scatt)

#     ### Plot scattered points for check
#     print(scatt.shape)
#     fig = plt.figure(figsize=[8,8])
#     ax = fig.add_subplot(111)
#     ax.scatter(scatt[:,2],scatt[:,3])
#     ax.set_aspect('equal')
#     ax.set_xlabel("R")
#     ax.set_ylabel("Z")
#     plt.show()
    
    ### Prepare structured grid
    npol=3*int(2*nyw*n_alp*q_0+1)
    nrad=(2*nxw+1)
    zz_pol=np.linspace(-np.pi,np.pi,npol)
    xx=np.linspace(-lx,lx,nrad)
    wmr=np.zeros([nrad,npol],dtype=np.float64)
    wz_car=np.zeros([nrad,npol],dtype=np.float64)
    for ix in range(nrad):
        wxx = xx[ix]
        # Circular s-alpha geometry
        wmr[ix,:],wz_car[ix,:],_ = field_aligned_coordinates_salpha(wxx,zz_pol,rho,q_0,s_hat,eps_r)
#         # Non-circular Miller geometry
#         wmr[ix,:],wz_car[ix,:],_ = field_aligned_coordinates_miller(wxx,zz_pol,rho,q_0,s_hat,eps_r,  
#                                                                     dRmildr,dZmildr,kappa,s_kappa,delta,s_delta,zetasq,s_zetasq)

    ### Interpolation from 2D scattered data to 2D structured grid
    points=(scatt[:,2],scatt[:,3])  # Coordinates of scattered data
    values=(scatt[:,4])             # Values of scattered data
    outgrid=(wmr,wz_car)            # Structured grid on which values are interpolated
    phi_pol=interpolate.griddata(points,values,outgrid,method="cubic",fill_value=0)
    #print(phi_pol.shape)

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
            filename = './data/phiinrz_t{:08d}.png'.format(it) 
            plt.savefig(filename)
            plt.close()
            
    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = './data/phiinrz_t{:08d}.dat'.format(it) 
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
    from diag_geom import geom_set
    from diag_rb import rb_open
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    xr_phi = rb_open('../../post/data/phi.*.nc')
    print(xr_phi)
    help(phiinrz)
    # Plot phi[y,x] at t[it], zz[iz]
    it = 60
    phiinrz(it, xr_phi, flag="display")
    #phiinrz(it, xr_phi, flag="savefig")
    #phiinrz(it, xr_phi, flag="savetxt")
    data=phiinrz(it, xr_phi, flag=None)
    print(data.shape)


# In[ ]:





# In[ ]:




