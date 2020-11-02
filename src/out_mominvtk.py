#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 3D electrostatic potential phi in VTK file format

Module dependency: diag_geom, diag_fft

Third-party libraries: numpy, pyevtk
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


def cartesian_coordinates_salpha(i_alp,n_alp,xx,yy,zz):
    """
    Calculate Cartesian coordinates (x_car,y_car,z_car) from GKV coordinates (xx,yy,zz)
    Depending on number of partitioned torus i_alp, n_alp
    
    Parameters
    ----------
        i_alp : int
            i_alp-th fluxtube representing one of partitions
        n_alp : int
            1/n_alp partition of torus
        xx : Numpy array
            Radial x in GKV coordinates
        yy : Numpy array
            Field-line-label y in GKV coordinates
        zz : Numpy array
            Field-aligned z in GKV coordinates
    Returns
    -------
        x_car, y_car, z_car : Numpy array
            x,y,z in Cartesian coordinates
    """
    import numpy as np
    from diag_geom import nml
    
    eps_r = nml['confp']['eps_r']
    q_0 = nml['confp']['q_0']
    s_hat = nml['confp']['s_hat']
    kymin = nml['nperi']['kymin']
    ly = np.pi / kymin
    rho = np.pi*eps_r/(q_0*ly*n_alp) # = Larmor radius rho_ref/L_ref
    
    wzz=zz.reshape(len(zz),1,1)
    wyy=yy.reshape(1,len(yy),1)
    wxx=xx.reshape(1,1,len(xx))
    wtheta=wzz
    wsr=eps_r+rho*wxx
    wmr=1+wsr*np.cos(wzz)
    wzeta=q_0*(wzz+(s_hat*wxx*wzz-wyy)*rho/eps_r)
    wx_car=wmr*np.cos(wzeta)
    wy_car=wmr*np.sin(wzeta)
    wz_car=wsr*np.sin(wtheta)
    #print(wz_car.shape)
    wz_car=wz_car*np.ones((len(zz),len(yy),len(xx))) # to adjust the array shape
    #wz_car=np.tile(wz_car,(1,2*nyw+1,1)) # Another way to adjust the array shape                   
    #print(wz_car.shape)
    return wx_car, wy_car, wz_car



def phiinvtk(it, xr_phi, flag="flux_tube", n_alp=4, nxw=None, nyw=None):
    """
    Output 3D electrostatic potential phi in VTK file format at t[it].
    
    Parameters
    ----------
        it : int
            index of t-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        flag : str
            # flag=="flux_tube" - show figure on display
            # flag=="full_torus" - save figure as png
            # flag=="field_aligned" - save data as txt
        n_alp : int, optional
            1/n_alp partition of torus
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = int(nx*1.5)+1 
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = int(global_ny*1.5)+1 

    Returns
    -------
        None
            VTK files are dumped in data/ directory.
    """
    import numpy as np
    from pyevtk.hl import gridToVTK

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

    ### Plot flux-tube ###
    # GKV座標(x,y,z)を作成
    kxmin = float(xr_phi['kx'][nx+1])
    lx = np.pi / kxmin
    xx = np.linspace(-lx,lx,2*nxw+1)
    kymin = float(xr_phi['ky'][1])
    ly = np.pi / kymin
    yy = np.linspace(-ly,ly,2*nyw+1)
    lz = - float(xr_phi['zz'][0])
    zz = np.linspace(-lz,lz,2*global_nz+1)

    # GKV座標系からCartesian座標系の値を計算
    i_alp=0
    wx_car, wy_car, wz_car = cartesian_coordinates_salpha(i_alp, n_alp, xx, yy, zz)
    
    # diag_fft.pyから関数 fft_backward_xyzを呼び出し、3次元逆フーリエ変換 phi[zz,ky,kx]-> phi[z,y,x]
    # 境界上の点を拡張済み phi[2*global_nz+1,2*nyw+1,2*nxw+1] in (z,y,x)
    phi_zyx = phi_zkykx2zyx_fluxtube(phi,nxw=nxw,nyw=nyw) # Numpy array

    ### Output a VTK-structured-grid file *.vts ###
    gridToVTK("./data/phiinvtk_tube_t{:08d}".format(it), 
              wx_car.astype(np.float32), 
              wy_car.astype(np.float32), 
              wz_car.astype(np.float32),
              pointData = {"phi": phi_zyx.astype(np.float32)})
    
#     # 3D surface plot 1 for check
#     import matplotlib.pyplot as plt
#     fig = plt.figure(figsize=[8,8])
#     ax = fig.add_subplot(111,projection="3d")
#     ax.set_title("Flux tube-1")
#     ax.plot_surface(wx_car[:,:,0], wy_car[:,:,0], wz_car[:,:,0] )
#     ax.plot_surface(wx_car[:,:,-1],wy_car[:,:,-1],wz_car[:,:,-1])
#     ax.plot_surface(wx_car[:,0,:], wy_car[:,0,:], wz_car[:,0,:] )
#     ax.plot_surface(wx_car[:,-1,:],wy_car[:,-1,:],wz_car[:,-1,:])
#     ax.set_xlim(-1.5,1.5)
#     ax.set_ylim(-1.5,1.5)
#     ax.set_zlim(-1,1)
#     plt.show()

#     # 3D surface plot 2 for check
#     fig = plt.figure(figsize=[8,8])
#     ax = fig.add_subplot(111,projection="3d")
#     ax.set_title("Flux tube-2")
#     for iz in range(0,2*global_nz+1,4):
#         ax.plot_surface(wx_car[iz,:,:], wy_car[iz,:,:], wz_car[iz,:,:])
#     ax.set_xlim(-1.5,1.5)
#     ax.set_ylim(-1.5,1.5)
#     ax.set_zlim(-1,1)
#     plt.show()


if (__name__ == '__main__'):
    from diag_geom import geom_set
    from diag_rb import rb_open
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    xr_phi = rb_open('../../post/data/phi.*.nc')
    print(xr_phi)
    help(phiinvtk)
    # Plot phi[y,x] at t[it], zz[iz]
    it = 60
    phiinvtk(it, xr_phi, flag="flux_tube")


# In[ ]:




