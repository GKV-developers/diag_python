#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 3D electrostatic potential phi in XMF file format

Module dependency: diag_geom, diag_fft

Third-party libraries: numpy, scipy
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
    #    for mx in range(2*nx+1):
    #        mwp = mx - dj[my]
    #        if (mwp < 0 or mwp > 2*nx):
    #            phi_zkykx[iz,my,mx] = 0.0
    #        else:
    #            phi_zkykx[iz,my,mx] = np.conjugate(ck[my]) * phi[0,my,mwp]
        if dj[my]<0:
            if 0<2*nx+1+dj[my] and 0-dj[my]<2*nx+1: 
                phi_zkykx[iz,my,0:2*nx+1+dj[my]] = np.conjugate(ck[my]) * phi[0,my,0-dj[my]:2*nx+1]
        else:
            if 0+dj[my]<2*nx+1 and 0<2*nx+1-dj[my]:
                phi_zkykx[iz,my,0+dj[my]:2*nx+1] = np.conjugate(ck[my]) * phi[0,my,0:2*nx+1-dj[my]]

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
    wzeta=q_0*(wzz+(s_hat*wxx*wzz-wyy)*rho/eps_r) - i_alp*2*np.pi/n_alp  
      # - i_alp*2*np.pi/n_alp を追加することで、full_torusの計算でも使用可能になる。
    wx_car=wmr*np.cos(wzeta)
    wy_car=wmr*np.sin(wzeta)
    wz_car=wsr*np.sin(wtheta)
    #print(wz_car.shape)
    wz_car=wz_car*np.ones((len(zz),len(yy),len(xx))) # to adjust the array shape
    #wz_car=np.tile(wz_car,(1,2*nyw+1,1)) # Another way to adjust the array shape                   
    #print(wz_car.shape)
    return wx_car, wy_car, wz_car



def phiinxmf(it, xr_phi, flag=None, n_alp=4, nxw=None, nyw=None, nzw=None, outdir='./data/'):
    """
    Output 3D electrostatic potential phi in VTK file format at t[it].

    Parameters
    ----------
        it : int
            index of t-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        flag : str
            # flag=="flux_tube_coord" - coordinates of single flux-tube
            # flag=="flux_tube_var" - header&variables of single flux-tube
            # flag=="full_torus_coord" - coordinates of full torus with n_alp copies
            # flag=="full_torus_var" - header&variables of full torus with n_alp copies
        n_alp : int, optional
            1/n_alp partition of torus
            # Default: n_alp = 4
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = nxw in gkvp_header.f90 
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = nxw in gkvp_header.f90
        nzw : int, optional
            (grid number in zz) = 2*nzw+1
            # Default: nzw = global_nz
            # When nzw /= global_nz, data is interpolated along a field line z
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        None
            VTK files are dumped in outdir/.
    """
    import os
    import numpy as np
    import scipy.interpolate as interpolate
    from diag_geom import nml
    from diag_geom import nxw as nxw_geom
    from diag_geom import nyw as nyw_geom
    from diag_rb import safe_compute

    ### データ処理 ###
    # GKVパラメータを換算する
    nx = int((len(xr_phi['kx'])-1)/2)
    global_ny = int(len(xr_phi['ky'])-1)
    global_nz = int(len(xr_phi['zz'])/2)
    if (nxw == None):
        nxw = nxw_geom
    if (nyw == None):
        nyw = nyw_geom

    # 時刻t[it]における三次元複素phi[z,ky,kx]を切り出す
    if 'rephi' in xr_phi and 'imphi' in xr_phi:
        rephi = xr_phi['rephi'][it,:,:,:]  # dim: t, zz, ky, kx
        imphi = xr_phi['imphi'][it,:,:,:]  # dim: t, zz, ky, kx
        phi = rephi + 1.0j*imphi
    elif 'phi' in xr_phi:
        phi = xr_phi['phi'][it,:,:,:]  # dim: t, zz, ky, kx
    phi = safe_compute(phi)

    # GKV座標(x,y,z)を作成
    kymin = float(xr_phi['ky'][1])
    ly = np.pi / kymin
    yy = np.linspace(-ly,ly,2*nyw+1)
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

    # diag_fft.pyから関数 fft_backward_xyzを呼び出し、3次元逆フーリエ変換 phi[zz,ky,kx]-> phi[z,y,x]
    # 境界上の点を拡張済み phi[2*global_nz+1,2*nyw+1,2*nxw+1] in (z,y,x)
    phi_zyx = phi_zkykx2zyx_fluxtube(phi,nxw=nxw,nyw=nyw) # Numpy array；関数の呼び出し

    # z方向を2*global_nz+1点から2*nzw+1点に補完する
    if (nzw is not None) and (flag != "field_aligned"):
        poly_as = interpolate.Akima1DInterpolator(zz,phi_zyx,axis=0)
        #poly_as = interpolate.CubicSpline(zz,phi_zyx,axis=0)
        zz = np.linspace(-lz,lz,2*nzw+1)
        phi_zyx = poly_as(zz)

    ### データ出力 ###
    if (flag == "flux_tube_coord"):

        i_alp=0
        # GKV座標系からCartesian座標系の値を計算　（関数の呼び出し）
        wx_car, wy_car, wz_car = cartesian_coordinates_salpha(i_alp, n_alp, xx, yy, zz)

        # Output binary grid data file *.bin
        wx_car.astype(np.float32).tofile(os.path.join(outdir,'phiinxmf_tube_xcoord.bin'))
        wy_car.astype(np.float32).tofile(os.path.join(outdir,'phiinxmf_tube_ycoord.bin'))
        wz_car.astype(np.float32).tofile(os.path.join(outdir,'phiinxmf_tube_zcoord.bin'))

    elif (flag == "flux_tube_var"):

        # Output binary phi data file *.bin # 精度の変換（float64 to float32）
        phi_zyx.astype(np.float32).tofile(os.path.join(outdir,f'phiinxmf_var_t{it:08d}.bin'))

        ### Output tube header.xmf ###
        time=float(xr_phi['t'][it])
        with open(os.path.join(outdir,f'phiinxmf_tube_header_t{it:08d}.xmf'), mode='w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
            f.write('\n')
            f.write('<Xdmf>\n')
            f.write('<Domain>\n')
            f.write('\n')
            f.write('<Grid Name="phi" GridType="Collection" CollectionType="Temporal">\n')
            f.write('<Time TimeType="HyperSlab">\n')
            f.write('<DataItem Format="XML" NumberType="Float" Dimensions="3">\n')
            f.write(f'{time:17.6f}{0.0:17.6f}{1:17d}\n' )
            f.write('</DataItem>\n')
            f.write('</Time>\n')
            f.write('\n')
            #f.write(f'<Grid Name="phi_var{it} " GridType="Collection" CollectionType="Spatial">\n')
            f.write(f'<Grid Name="phi_var{it}"> \n')
            f.write(f'<Topology Type="3DSMesh" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
            f.write('</Topology>\n')
            f.write('<Geometry Type="X_Y_Z">\n')
            f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
            f.write('  phiinxmf_tube_xcoord.bin\n')
            f.write('</DataStructure>\n')
            f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
            f.write('  phiinxmf_tube_ycoord.bin\n')
            f.write('</DataStructure>\n')
            f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
            f.write('  phiinxmf_tube_zcoord.bin\n')
            f.write('</DataStructure>\n')
            f.write('</Geometry>\n')
            f.write('<Attribute Active="1" Type="Scalar" Center="Node" Name="phi">\n')
            f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
            f.write(f'  phiinxmf_var_t{it:08d}.bin\n')
            f.write('</DataStructure>\n')
            f.write('</Attribute>\n')
            f.write('</Grid>\n')
            f.write('\n')
            f.write('</Grid><!-- End GridType="Collection" CollectionType="Temporal" -->\n')
            f.write('</Domain>\n')
            f.write('</Xdmf>\n')

    elif (flag == "full_torus_coord"):

        for i_alp in range(n_alp):
            # GKV座標系からCartesian座標系の値を計算　（関数の呼び出し）
            wx_car, wy_car, wz_car = cartesian_coordinates_salpha(i_alp, n_alp, xx, yy, zz)

            ### Output binary grid data file *.bin ###
            wx_car.astype(np.float32).tofile(os.path.join(outdir, f'phiinxmf_full_alp{i_alp:03d}_xcoord.bin'))
            wy_car.astype(np.float32).tofile(os.path.join(outdir, f'phiinxmf_full_alp{i_alp:03d}_ycoord.bin'))
            wz_car.astype(np.float32).tofile(os.path.join(outdir, f'phiinxmf_full_alp{i_alp:03d}_zcoord.bin'))

    elif (flag == "full_torus_var"): # 0: var & header (flux_torus)

        # Output binary phi data file *.bin # 精度の変換（float64 to float32）
        phi_zyx.astype(np.float32).tofile(os.path.join(outdir, f'phiinxmf_var_t{it:08d}.bin')) 

        ### Output full torus header.xmf ###
        time=float(xr_phi['t'][it])
        with open(os.path.join(outdir,f'phiinxmf_full_header_t{it:08d}.xmf'), mode='w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd">\n')
            f.write('\n')
            f.write('<Xdmf>\n')
            f.write('<Domain>\n')
            f.write('\n')
            f.write('<Grid Name="phi" GridType="Collection" CollectionType="Temporal">\n')
            f.write('<Time TimeType="HyperSlab">\n')
            f.write('<DataItem Format="XML" NumberType="Float" Dimensions="3">\n')
            f.write(f'{time:17.6f}{0.0:17.6f}{1:17d}\n' )
            f.write('</DataItem>\n')
            f.write('</Time>\n')
            f.write('\n')
            f.write(f'<Grid Name="phi_var{it} " GridType="Collection" CollectionType="Spatial">\n')
            for i_alp in range(n_alp):
                f.write('<Grid Name="phi_var'+str(it)+'_alp'+str(i_alp)+'">\n')
                f.write(f'<Topology Type="3DSMesh" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
                f.write('</Topology>\n')
                f.write('<Geometry Type="X_Y_Z">\n')
                f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
                f.write(f'  phiinxmf_full_alp{i_alp:03d}_xcoord.bin\n')
                f.write('</DataStructure>\n')
                f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
                f.write(f'  phiinxmf_full_alp{i_alp:03d}_ycoord.bin\n')
                f.write('</DataStructure>\n')
                f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
                f.write(f'  phiinxmf_full_alp{i_alp:03d}_zcoord.bin\n')
                f.write('</DataStructure>\n')
                f.write('</Geometry>\n')
                f.write('<Attribute Active="1" Type="Scalar" Center="Node" Name="phi">\n')
                f.write(f'<DataStructure DataType="Float" Precision="4" Dimensions="{len(zz):17d}{2*nyw+1:17d}{2*nxw+1:17d} "  Format="Binary" Endian="Little"> \n')
                f.write(f'  phiinxmf_var_t{it:08d}.bin\n')
                f.write('</DataStructure>\n')
                f.write('</Attribute>\n')
                f.write('</Grid>\n')
                f.write('\n')
            f.write('</Grid><!-- End GridType="Collection" CollectionType="Spatial" -->\n')
            f.write('\n')
            f.write('</Grid><!-- End GridType="Collection" CollectionType="Temporal" -->\n')
            f.write('</Domain>\n')
            f.write('</Xdmf>\n')

    else:
        print('#Current flag=',flag)
        print('#flag should be "flux_tube_coord","flux_tube_var","full_torus_coord","full_torus_var"')


if (__name__ == '__main__'):
    import os
    from diag_geom import geom_set
    from diag_rb import rb_open
    from time import time as timer
    ### Initialization ###
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')



    ### Examples of use ###


    ### phiinxmf ###
    xr_phi = rb_open('../../phi/gkvp.phi.*.zarr/')
    #print(xr_phi)
    #help(phiinxmf)
    from diag_geom import global_nz
    print("# Output phi[z,y,x] at t[it] in flux_tube XMF format *.xmf")
    outdir='../data/xmf_tube/'
    os.makedirs(outdir, exist_ok=True)
    s_time = timer()
    it = 0
    phiinxmf(it, xr_phi, flag="flux_tube_coord",nzw=5*global_nz,outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))
    s_time = timer()
    for it in range(0,len(xr_phi['t']),len(xr_phi['t'])//10):
        phiinxmf(it, xr_phi, flag="flux_tube_var",nzw=5*global_nz,outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))

    print("# Output phi[z,y,x] at t[it] in full_torus XMF format *.xmf")
    outdir='../data/xmf_full/'
    os.makedirs(outdir, exist_ok=True)
    s_time = timer()
    it = 0
    phiinxmf(it, xr_phi, flag="full_torus_coord",nzw=3*global_nz,outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))
    s_time = timer()
    for it in range(0,len(xr_phi['t']),len(xr_phi['t'])//10):
        phiinxmf(it, xr_phi, flag="full_torus_var",nzw=3*global_nz,outdir=outdir)
    e_time = timer(); print('\n *** total_pass_time ={:12.5f}sec'.format(e_time-s_time))


# In[ ]:




