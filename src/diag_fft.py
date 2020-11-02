#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
"""
Module for FFT

Module dependency: -

Third-party libraries: numpy, scipy
"""

def fft_backward_xy(phikxky, nxw=None, nyw=None):
    """
    Backward Fourier transform phi[ky,kx]->phi[y,x]
    
    Parameters
    ----------
        phikxky[global_ny+1,2*nx+1] : Numpy array, dtype=np.complex128
            phi[ky,kx] in wavenumber space
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = int(nx*1.5)+1 
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = int(gny*1.5)+1 

    Returns
    -------
        phixy[2*nyw,2*nxw] : Numpy array, dtype=np.float64
            phi[y,x] in real space
    """
    import numpy as np
    from scipy import fft

    # GKVパラメータを換算する
    nx = int((phikxky.shape[1]-1)/2)
    gny = int(phikxky.shape[0]-1)
    if (nxw == None):
        nxw = int(nx*1.5)+1
    if (nyw == None):
        nyw = int(gny*1.5)+1

    # 2次元逆フーリエ変換 phi[ky,kx] -> phi[y,x]
    phixy = np.zeros([2*nyw,2*nxw],dtype=np.complex128) # fft.ifft2用Numpy配列
    phixy[0:gny+1, 0:nx+1] = phikxky[0:gny+1, nx:2*nx+1] # 波数空間配列の並び替え
    phixy[0:gny+1, 2*nxw-nx:2*nxw] = phikxky[0:gny+1, 0:nx]
    phixy[2*nyw-gny:2*nyw, 2*nxw-nx:2*nxw] = np.conj(phikxky[gny:0:-1, 2*nx:nx:-1])
    phixy[2*nyw-gny: 2*nyw, 0:nx+1] = np.conj(phikxky[gny+1:0:-1, nx::-1])
    
    phixy = fft.ifft2(phixy) * (2*nxw)*(2*nyw) # phi[y,x] = Sum_kx Sum_ky phi[ky,kx]*exp[i(kx*x+ky*y)]
    phixy = phixy.real # phi[y,x]は実数配列
    return phixy


def fft_backward_xyz(phikxkyz, nxw=None, nyw=None): # 3次元配列用逆FFT（最後の2軸に対して計算）
    """
    Backward Fourier transform phi[z,ky,kx]->phi[z,y,x]
    Arbitrary length of z is applicable.
    
    Parameters
    ----------
        phikxkyz[:,global_ny+1,2*nx+1] : Numpy array, dtype=np.complex128
            phi[z,ky,kx] in wavenumber space [ky,kx] and z
        nxw : int, optional
            (grid number in xx) = 2*nxw
            # Default: nxw = int(nx*1.5)+1 
        nyw : int, optional
            (grid number in yy) = 2*nyw
            # Default: nyw = int(gny*1.5)+1 

    Returns
    -------
        phixy[:,2*nyw,2*nxw] : Numpy array, dtype=np.float64
            phi[z,y,x] in real space
    """
    import numpy as np
    from scipy import fft
    # GKVパラメータを換算する
    nx = int((phikxkyz.shape[2]-1)/2)
    gny = int(phikxkyz.shape[1]-1)
    len_z = phikxkyz.shape[0]
    if (nxw == None):
        nxw = int(nx*1.5)+1
    if (nyw == None):
        nyw = int(gny*1.5)+1

    # 2次元逆フーリエ変換 phi[z,ky,kx] -> phi[z,y,x]
    phixyz = np.zeros([len_z,2*nyw,2*nxw],dtype=np.complex128) # fft.ifft2用Numpy配列
    phixyz[:, 0:gny+1, 0:nx+1] = phikxkyz[:, 0:gny+1, nx:2*nx+1] # 波数空間配列の並び替え
    phixyz[:, 0:gny+1, 2*nxw-nx:2*nxw] = phikxkyz[:, 0:gny+1, 0:nx]
    phixyz[:, 2*nyw-gny:2*nyw, 2*nxw-nx:2*nxw] = np.conj(phikxkyz[:, gny:0:-1, 2*nx:nx:-1])
    phixyz[:, 2*nyw-gny: 2*nyw, 0:nx+1] = np.conj(phikxkyz[:, gny+1:0:-1, nx::-1])
    
    phixyz = fft.ifft2(phixyz, axes=(-2,-1)) * (2*nxw)*(2*nyw) # phi[y,x] = Sum_kx Sum_ky phi[ky,kx]*exp[i(kx*x+ky*y)]  
    phixyz = phixyz.real # phi[z,y,x]は実数配列
    return phixyz


if (__name__ == '__main__'):
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_geom import geom_set
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    from diag_geom import nxw, nyw, nx, global_ny, xx, yy, kx, ky

    temp=np.zeros([global_ny+1,2*nx+1],dtype=np.complex128)
    temp[1,nx+2]=1.0
    tempxy=fft_backward_xy(temp,nxw=nxw,nyw=nyw)
    print(tempxy.shape)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    quad=ax.pcolormesh(kx,ky,np.abs(temp),shading="auto")
    fig.colorbar(quad)
    plt.show()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    quad=ax.pcolormesh(xx,yy,tempxy,shading="auto")
    fig.colorbar(quad)
    plt.show()
    
    temp=np.zeros([4,global_ny+1,2*nx+1],dtype=np.complex128)
    temp[0,1,nx+2]=1.0
    temp[1,2,nx+2]=1.0
    temp[2,3,nx+2]=1.0
    temp[3,4,nx+2]=1.0
    tempxy=fft_backward_xyz(temp,nxw=nxw,nyw=nyw)
    print(tempxy.shape)
    iz=3
    fig=plt.figure()
    ax=fig.add_subplot(111)
    quad=ax.pcolormesh(kx,ky,np.abs(temp[iz,:,:]),shading="auto")
    fig.colorbar(quad)
    plt.show()
    fig=plt.figure()
    ax=fig.add_subplot(111)
    quad=ax.pcolormesh(xx,yy,tempxy[iz,:,:],shading="auto")
    fig.colorbar(quad)
    plt.show()


# In[ ]:





# In[ ]:




