#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
"""
Module for geometric constants

Module dependency: -

Third-party libraries: numpy, re, f90nml
"""

### GKV parameters from gkvp_namelist ###
nml = None

### 磁気座標情報: hst/gkvp.mtr.001 から読み取る
theta  = None
omg    = None
domgdx = None
domgdy = None
domgdz = None
ggxx   = None
ggxy   = None
ggxz   = None
ggyy   = None
ggyz   = None
ggzz   = None
rootg  = None

### 格子点数: src/gkvp_header.f90 から読み取る
nxw       = None
nyw       = None
nx        = None
global_ny = None
global_nz = None
global_nv = None
global_nm = None

### 読み取った情報を元に座標、定数関数等を構築
xx     = None
yy     = None
kx     = None
ky     = None
zz     = None
vl     = None
mu     = None

vp     = None
ksq    = None
fmx    = None
ck     = None
dj     = None


def read_f90_parameters(filename, argname, argtype=float):
    """
    Read the value of argname from filename, in whose form ' argname = value'.
    Note that there is two spaces, one just before 'argname' and the other between 'argname' and '='.

    Parameters
    ----------
    filename : str
    argname : str
    argtype : Numeric types {int, float}

    Returns
    -------
    arg : argtype
        The value of 'argname' in numeric type 'argtype'.
    """
    import re
    with open(filename) as f:                             # ASCIIファイルを開く。
        for line in f.readlines():                        # ファイルの各行を読み込み、
            if not (line.strip()).startswith("!"):        # !で始まる行はFortranコメント行なので除外。
                if line.find(" " + argname + " =") != -1: # " argname ="という文字列を含むかどうか判定。
                    arg = line
    arg=re.sub(r'.+' + argname + ' =','',arg) # "argname ="以前の文字を削除。(正規表現： . 改行以外の任意の文字列, + 直前の文字のくり返し)
    arg=re.sub(r'[,!].+\n','',arg)            # コロン","または感嘆符"!"以降の文字と改行コード\nを削除。（正規表現： [abc] "a"または"b"または"c"）
    arg=re.sub(r'd','e',arg)                  # Fortran の倍精度実数を表す d の文字を Pythonの実数でも使える e に置き換える。
    arg=re.sub(r'_DP','e0',arg)               # Fortran の倍精度実数を表す _DP の文字を Pythonの実数でも使える e0 に置き換える。
    if (argtype==str):
        arg=re.sub(r'["\']','',arg).strip()   # Fortran文字列の""や''を削除。
    else:
        arg=argtype(arg)                      # 文字列型を argtype型に変換。
    return arg


def geom_set(headpath='../src/gkvp_header.f90', nmlpath='../gkvp_namelist.001', mtrpath='../hst/gkvp.mtr.001'):
    """
    Read NetCDF files by xarray

    Parameters
    ----------
        ncpath : str
            directory path of phi.*.nc

    Returns
    -------
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc
    """
    global nml
    global theta, omg, domgdx, domgdy, domgdz, ggxx, ggxy, ggxz, ggyy, ggyz, ggzz, rootg
    global nxw, nyw, nx, global_ny, global_nz, global_nv, global_nm
    global xx, yy, kx, ky, zz, vl, mu
    global vp, ksq, fmx, ck, dj
    
    import numpy as np
    import f90nml
    
    ### 磁気座標情報: hst/gkvp.mtr.001 から読み取る
    mtr = np.loadtxt(mtrpath, comments='#')
    zz     = mtr[:,0]
    theta  = mtr[:,1]
    omg    = mtr[:,2]
    domgdx = mtr[:,3]
    domgdy = mtr[:,4]
    domgdz = mtr[:,5]
    ggxx   = mtr[:,6]
    ggxy   = mtr[:,7]
    ggxz   = mtr[:,8]
    ggyy   = mtr[:,9]
    ggyz   = mtr[:,10]
    ggzz   = mtr[:,11]
    rootg  = mtr[:,12]
    
    ### 格子点数: src/gkvp_header.f90 から読み取る
    nxw = read_f90_parameters(headpath, "nxw", int)
    nyw = read_f90_parameters(headpath, "nyw", int)
    nx = read_f90_parameters(headpath, "nx", int)
    global_ny = read_f90_parameters(headpath, "global_ny", int)
    global_nz = read_f90_parameters(headpath, "global_nz", int)
    global_nv = read_f90_parameters(headpath, "global_nv", int)
    global_nm = read_f90_parameters(headpath, "global_nm", int)
    #print(nxw, nyw, nx, global_ny, global_nz, global_nv, global_nm)
    
    ### パラメータ: gkvp_namelist から読み取る
    nml=f90nml.read(nmlpath)
    vmax  = nml['physp']['vmax']
    n_tht = nml['nperi']['n_tht']
    kymin = nml['nperi']['kymin']
    m_j   = nml['nperi']['m_j']
    del_c = nml['nperi']['del_c']
    s_hat = nml['confp']['s_hat']
    #print(vmax,n_tht,kymin,m_j,del_c,s_hat)
    
    ### 読み取った情報を元に座標、定数関数等を構築
    if (abs(s_hat) < 1e-10):
        m_j = 0
        kxmin = kymin
    elif (m_j == 0):
        kxmin = kymin
    else:
        kxmin = abs(2*np.pi*s_hat*kymin / m_j)
    lx = np.pi / kxmin
    ly = np.pi / kymin
    lz = n_tht*np.pi
    dz = lz / global_nz
    dv = 2*vmax / (2*global_nv-1)
    mmax = vmax
    dm = mmax / (global_nm)
    
    xx = np.linspace(-lx,lx,2*nxw,endpoint=False)
    yy = np.linspace(-ly,ly,2*nyw,endpoint=False)
    kx = kxmin * np.arange(-nx,nx+1)
    ky = kymin * np.arange(global_ny+1)
    zz = np.linspace(-lz,lz,2*global_nz,endpoint=False)
    vl = np.linspace(-vmax,vmax,2*global_nv,endpoint=True)
    mu = 0.5 * (dm * np.arange(global_nm+1))**2
    #print("kx=",kx); print("ky=",ky); print("zz=",zz); print("vl=",vl); print("mu=",mu)
    
    vp = np.sqrt(2*mu.reshape(global_nm+1,1)*omg.reshape(1,2*global_nz))
    dvp = np.sqrt(2*(0.5*dm**2)*omg) # = vp[1,:]
    #print(vp.shape); print(vp[1,:]); print(dvp)
    
    wkx = kx.reshape(1,1,2*nx+1)
    wky = ky.reshape(1,global_ny+1,1)
    wggxx = ggxx.reshape(2*global_nz,1,1)
    wggxy = ggxx.reshape(2*global_nz,1,1)
    wggyy = ggxx.reshape(2*global_nz,1,1)
    ksq = wkx**2*wggxx + 2*wkx*wky*wggxy + wky**2*wggyy
    #print(ksq.shape)
    
    womg = omg.reshape(1,1,2*global_nz)
    wvl = vl.reshape(1,2*global_nv,1)
    wmu = mu.reshape(global_nm+1,1,1)
    fmx = np.exp(-0.5*wvl**2 -womg*wmu) / np.sqrt((2*np.pi)**3)
    #print(fmx.shape)
    
    ck = np.exp(2j*np.pi*del_c*n_tht*np.arange(global_ny+1))
    dj = - m_j * n_tht * np.arange(global_ny+1)
    #print(ck.shape, dj.shape)

    return



if (__name__ == '__main__'):
    import matplotlib.pyplot as plt
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    print(xx.shape,"xx=",xx)
    print(yy.shape,"yy=",yy)
    print(kx.shape,"kx=",kx)
    print(ky.shape,"ky=",ky)
    print(zz.shape,"zz=",zz)
    print(vl.shape,"vl=",vl)
    print(mu.shape,"mu=",mu)
    print(vp.shape,"vp=",vp)
    print(ksq.shape,"ksq=",ksq)
    print(fmx.shape,"fmx=",fmx)
    print(ck.shape,"ck=",ck)
    print(dj.shape,"dj=",dj)

    fig = plt.figure(figsize=[8,12])
    ax = fig.add_subplot(6,2,1)
    ax.plot(zz,omg,label="B")
    ax.legend()
    ax = fig.add_subplot(6,2,2)
    ax.plot(zz,domgdx,label="dB/dx")
    ax.legend()
    ax = fig.add_subplot(6,2,3)
    ax.plot(zz,domgdy,label="dB/dy")
    ax.legend()
    ax = fig.add_subplot(6,2,4)
    ax.plot(zz,domgdz,label="dB/dz")
    ax.legend()
    ax = fig.add_subplot(6,2,5)
    ax.plot(zz,ggxx,label=r"$g^{xx}$")
    ax.legend()
    ax = fig.add_subplot(6,2,6)
    ax.plot(zz,ggxy,label=r"$g^{xy}$")
    ax.legend()
    ax = fig.add_subplot(6,2,7)
    ax.plot(zz,ggxz,label=r"$g^{xz}$")
    ax.legend()
    ax = fig.add_subplot(6,2,8)
    ax.plot(zz,ggyy,label=r"$g^{yy}$")
    ax.legend()
    ax = fig.add_subplot(6,2,9)
    ax.plot(zz,ggyz,label=r"$g^{yz}$")
    ax.legend()
    ax = fig.add_subplot(6,2,10)
    ax.plot(zz,ggzz,label=r"$g^{zz}$")
    ax.legend()
    ax = fig.add_subplot(6,2,11)
    ax.plot(zz,rootg,label=r"$\sqrt{g}$")
    ax.legend()
    plt.show()


# In[ ]:





# In[ ]:




