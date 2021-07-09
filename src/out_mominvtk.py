#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output 3D electrostatic potential phi in VTK file format

Module dependency: diag_geom, diag_fft

Third-party libraries: numpy, scipy, pyevtk
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


def gridToVTK_with_start(path, x, y, z, cellData=None, pointData=None, fieldData=None, start=None):
    """
    Write data values as a rectilinear or rectangular grid.
    Parameters
    ----------
    path : str
        name of the file without extension where data should be saved.
    x : array-like
        x coordinate axis.
    y : array-like
        y coordinate axis.
    z : array-like
        z coordinate axis.
    cellData : dict, optional
        dictionary containing arrays with cell centered data.
        Keys should be the names of the data arrays.
        Arrays must have the same dimensions in all directions and must contain
        only scalar data.
    pointData : dict, optional
        dictionary containing arrays with node centered data.
        Keys should be the names of the data arrays.
        Arrays must have same dimension in each direction and
        they should be equal to the dimensions of the cell data plus one and
        must contain only scalar data.
    fieldData : dict, optional
        dictionary with variables associated with the field.
        Keys should be the names of the variable stored in each array.
    start : tuple, optional
        start position of data extent.
    Returns
    -------
    str
        Full path to saved file.
    Notes
    -----
    coordinates of the nodes of the grid. They can be 1D or 3D depending if
    the grid should be saved as a rectilinear or logically structured grid,
    respectively.
    Arrays should contain coordinates of the nodes of the grid.
    If arrays are 1D, then the grid should be Cartesian,
    i.e. faces in all cells are orthogonal.
    If arrays are 3D, then the grid should be logically structured
    with hexahedral cells.
    In both cases the arrays dimensions should be
    equal to the number of nodes of the grid.
    """

    import numpy as np
    from pyevtk.vtk import VtkFile, VtkStructuredGrid
    
    # Extract dimensions
    if start is None:
        start = (0, 0, 0)
    s = x.shape
    end = (start[0]+s[0]-1, start[1]+s[1]-1, start[2]+s[2]-1)

    w = VtkFile(path, VtkStructuredGrid)
    w.openGrid(start=start, end=end)
    w.openPiece(start=start, end=end)
    w.openElement("Points")
    w.addData("coordinates", (x, y, z))
    w.closeElement("Points")
    # Point data
    if pointData:
        keys = list(pointData.keys())
        w.openData("Point", scalars=keys[0])
        for key in keys:
            data = pointData[key]
            w.addData(key, data)
        w.closeData("Point")
    # Cell data
    if cellData:
        keys = list(cellData.keys())
        w.openData("Cell", scalars=keys[0])
        for key in keys:
            data = cellData[key]
            w.addData(key, data)
        w.closeData("Cell")
    # Field data
    # https://www.visitusers.org/index.php?title=Time_and_Cycle_in_VTK_files#XML_VTK_files
    if fieldData:
        keys = list(fieldData.keys())
        w.openData("Field")  # no attributes in FieldData
        for key in keys:
            data = fieldData[key]
            w.addData(key, data)
        w.closeData("Field")
    w.closePiece()
    w.closeGrid()
    # Write coordinates
    w.appendData((x, y, z))
    # Write data
    if pointData is not None:
        keys = list(pointData.keys())
        for key in keys:
            data = pointData[key]
            w.appendData(data)
    if cellData is not None:
        keys = list(cellData.keys())
        for key in keys:
            data = cellData[key]
            w.appendData(data)
    if fieldData is not None:
        keys = list(fieldData.keys())
        for key in keys:
            data = fieldData[key]
            w.appendData(data)
    w.save()
    return w.getFileName()


def phiinvtk(it, xr_phi, flag=None, n_alp=4, nxw=None, nyw=None, nzw=None, outdir="./data/"):
    """
    Output 3D electrostatic potential phi in VTK file format at t[it].
    
    Parameters
    ----------
        it : int
            index of t-axis
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        flag : str
            # flag=="flux_tube" - single flux-tube
            # flag=="full_torus" - full torus with n_alp copies
            # flag=="field_aligned" - (x,y,z) in GKV coordinates
        n_alp : int, optional
            1/n_alp partition of torus
            # Default: n_alp = 4
        nxw : int, optional
            (grid number in xx) = 2*nxw+1
            # Default: nxw = nxw in gkvp_header.f90
        nyw : int, optional
            (grid number in yy) = 2*nyw+1
            # Default: nyw = nyw in gkvp_header.f90
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
            VTK files are dumped in data/ directory.
    """
    import os
    import numpy as np
    import scipy.interpolate as interpolate
    from pyevtk.hl import gridToVTK
    from diag_geom import nml
    from diag_geom import nxw as nxw_geom
    from diag_geom import nyw as nyw_geom

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
    rephi = xr_phi['rephi'][it,:,:,:]  # dim: t, zz, ky, kx
    imphi = xr_phi['imphi'][it,:,:,:]  # dim: t, zz, ky, kx
    phi = rephi + 1.0j*imphi

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
    xx = np.linspace(-lx,lx,2*nxw+1)
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
    if (flag == "flux_tube"):
        
        i_alp=0
        # GKV座標系からCartesian座標系の値を計算　（関数の呼び出し）        
        wx_car, wy_car, wz_car = cartesian_coordinates_salpha(i_alp, n_alp, xx, yy, zz)

        ### Output a VTK-structured-grid file *.vts ###
        gridToVTK(os.path.join(outdir,'phiinvtk_tube_t{:08d}'.format(it)), 
                  wx_car.astype(np.float32), 
                  wy_car.astype(np.float32), 
                  wz_car.astype(np.float32),
                  pointData = {"phi": phi_zyx.astype(np.float32)})
        
    elif (flag == "full_torus"):

        ### Output full torus by bundling multiple flux tubes ###
        ### % a partitioned-VTK-structured-grid file *.pvts % ### 
        ### %  and multiple VTK-structured-grid files *.vts % ###
        for i_alp in range(n_alp):
            # GKV座標系からCartesian座標系の値を計算　（関数の呼び出し）
            wx_car, wy_car, wz_car = cartesian_coordinates_salpha(i_alp, n_alp, xx, yy, zz)
            
            # VTKファイル出力用関数の呼び出し
            gridToVTK_with_start(path=os.path.join(outdir,'phiinvtk_full_t{:08d}_alp{:03d}'.format(it,i_alp)),
                                 x=wx_car.astype(np.float32), 
                                 y=wy_car.astype(np.float32), 
                                 z=wz_car.astype(np.float32),
                                 pointData = {"phi": phi_zyx.astype(np.float32)},
                                 start=(0,2*nyw*i_alp,0))
        
        with open(os.path.join(outdir,'phiinvtk_full_t{:08d}.pvts'.format(it)), mode="w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="PStructuredGrid" version="1.0" byte_order="LittleEndian" header_type="UInt64">\n')
            f.write('<PStructuredGrid WholeExtent="{} {} {} {} {} {}" GhostLevel="#">\n'.format(0,len(zz)-1,0,2*nyw*n_alp,0,2*nxw))
            f.write('<PPoints>\n')
            f.write('<PDataArray Name="points" NumberOfComponents="3" type="Float32"/>\n')
            f.write('</PPoints>\n')
            f.write('<PPointData scalars="phi">\n')
            f.write('<PDataArray Name="phi" NumberOfComponents="1" type="Float32"/>\n')
            f.write('</PPointData>\n')
            for i_alp in range(n_alp):
                f.write('<Piece Extent="{} {} {} {} {} {}" Source="./phiinvtk_full_t{:08d}_alp{:03d}.vts"/>\n'.format(0,len(zz)-1,2*nyw*i_alp,2*nyw*(i_alp+1),0,2*nxw,it,i_alp))
            f.write('</PStructuredGrid>\n')
            f.write('</VTKFile>\n')

    elif (flag == "field_aligned"):

        ### Output a VTK-structured-grid file *.vti
        phi_xyz = phi_zyx.transpose()    # 変数の並びをFortranの phi_xyz に合わせる。
        from pyevtk.hl import imageToVTK
        imageToVTK(os.path.join(outdir,'phiinvtk_align_t{:08d}'.format(it)),
                   pointData = {"phi": phi_xyz.astype(np.float32)})

    else:  # otherwise - return data array
        return phi_zyx
        
        

if (__name__ == '__main__'):
    import os
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')

    
    ### Examples of use ###
    
    
    ### phiinvtk ###
    #help(phiinvtk)
    xr_phi = rb_open('../../post/data/phi.*.nc')
    #print(xr_phi)
    from diag_geom import global_nz
    print("# Output phi[z,y,x] at t[it] in flux_tube VTK format *.vts")
    outdir='../data/vts_tube/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_phi['t']),10):
        t1=time.time()
        phiinvtk(it, xr_phi, flag="flux_tube",nzw=3*global_nz,outdir=outdir)
        #t2=time.time();print("time=",t2-t1)
    
    
    print("# Output phi[z,y,x] at t[it] in full_torus VTK format *.pvts")
    outdir='../data/pvts_full/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_phi['t']),10):
        t1=time.time()
        phiinvtk(it, xr_phi, flag="full_torus",nzw=3*global_nz,outdir=outdir)
        #t2=time.time();print("time=",t2-t1)
    
    
    print("# Output phi[z,y,x] at t[it] in field_aligned VTK format *.vti")
    outdir='../data/vti_aligned/'
    os.makedirs(outdir, exist_ok=True)
    for it in range(0,len(xr_phi['t']),10):
        t1=time.time()
        phiinvtk(it, xr_phi, flag="field_aligned",outdir=outdir)
        #t2=time.time();print("time=",t2-t1)


# In[ ]:




