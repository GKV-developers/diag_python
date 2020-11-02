#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
"""
Module for integrals on flux tube

Module dependency: diag_geom.py

Third-party libraries: numpy
"""


def intgrl_thet(w3):
    """
    Calculate flux-surface average in zz
    
    Parameters
    ----------
        w3[zz,:,:] : Numpy array

    Returns
    -------
        w2[]:,:] : Numpy array
            w2 = <w3> = int w3 sqrt(g)*dz / (int sqrt(g)*dz)
    """
    import numpy as np
    
#     # 実空間 Jacobian をメトリックデータから読み込む
#     mtr = np.loadtxt('../../hst/gkvp.mtr.001', comments='#')
#     rootg = mtr[:,12]
#     cfsrf = np.sum(rootg)
#     fct = rootg / cfsrf
    from diag_geom import rootg
    cfsrf = np.sum(rootg)
    fct = rootg / cfsrf
    
    # zz方向平均を計算する
    ww = w3 * fct.reshape(len(fct),1,1) # Numpy broadcast: fctはky,kxに依らない
    w2 = np.sum(ww, axis=0) # データの軸順序が(zz,ky,kx)なので、axis=0について足し上げる
    return w2




if (__name__ == '__main__'):
    import numpy as np
    from diag_geom import geom_set
    geom_set(headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')
    from diag_geom import kx, ky, zz
    
    wkx=kx.reshape(1,1,len(kx))
    wky=ky.reshape(1,len(ky),1)
    wzz=zz.reshape(len(zz),1,1)
    temp=np.exp(-wzz**2/2)*np.exp(wkx**2+wky**2)
    temp_ave_z=intgrl_thet(temp)
    print(temp_ave_z)


# In[ ]:





# In[ ]:




