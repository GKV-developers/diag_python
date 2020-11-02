#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
"""
Module for reading NetCDF files

Module dependency: -

Third-party libraries: xarray, dask, glob
"""

def rb_open(ncpath='../post/data/phi.*.nc'):
    """
    Read NetCDF files by xarray

    Parameters
    ----------
        ncpath : str
            directory path of phi.*.nc

    Returns
    -------
        xr_dataset : xarray Dataset
            xarray Dataset of phi.*.nc
    """
    import xarray as xr
    xr_dataset = xr.open_mfdataset(ncpath, combine='by_coords')
    # 相対パスは、このプログラム（diag_rb.py）を基準にした位置ではなく、これを呼び出したmain_programから見た相対位置を指定すること！
    return xr_dataset


def rb_get_tri_filelist(ncpath='../post/data/tri.*.nc'):
    """
    Get a list of NetCDF files tri.mx****my****.***.nc

    Parameters
    ----------
        ncpath : str
            directory path of tri.mx****my****.***.nc

    Returns
    -------
        tri_filelist : list of str
            list of tri.mx****my****
    """
    import glob
    tri_filelist=sorted(glob.glob(ncpath))             # tri.*ncに該当するファイル名を取得
    tri_filelist=[file[:-7] for file in tri_filelist]  # ファイル名末尾の .***.nc を削除
    tri_filelist=sorted(set(tri_filelist))             # 重複したファイル名 tri.mx****my*** はリストから削除
    num_triad_diag = len(tri_filelist)
    #print("num_triad_diag:", num_triad_diag)
    #print("tri_filelist:", tri_filelist)
    return tri_filelist


if (__name__ == '__main__'):
    xr_phi = rb_open('../../post/data/phi.*.nc')
    print("xr_phi", xr_phi, "\n")
    xr_Al = rb_open('../../post/data/Al.*.nc')
    print("xr_Al", xr_Al, "\n")
    xr_mom = rb_open('../../post/data/mom.*.nc')
    print("xr_mom", xr_mom, "\n")
    xr_fxv = rb_open('../../post/data/fxv.*.nc')
    print("xr_fxv", xr_fxv, "\n")
    xr_cnt = rb_open('../../post/data/cnt.*.nc')
    print("xr_cnt", xr_cnt, "\n")
    xr_trn = rb_open('../../post/data/trn.*.nc')
    print("xr_trn", xr_trn, "\n")
    
    tri_filelist = rb_get_tri_filelist('../../post/data/tri.*.nc')
    print(tri_filelist)
    xr_tri_list=[]
    for file in tri_filelist:
        xr_tri=rb_open(file + '.*.nc')
        xr_tri_list.append(xr_tri)
        print("xr_tri", xr_tri, "\n")


# In[ ]:





# In[ ]:




