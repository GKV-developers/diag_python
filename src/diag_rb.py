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

def safe_compute(tensor, safety_factor=1.5, enable_logging=False):
    """
    tensor.compute() を呼び出す前に、テンソルのサイズとシステムの使用可能メモリを比較します。

    Parameters
    ----------
    tensor : dask.array.Array または類似のオブジェクト
        compute() を呼び出す対象のテンソルです。
    safety_factor : float, optional
        計算時に発生する一時的なメモリ使用量を見越して、必要メモリに掛ける係数です (デフォルト: 1.5)。
    enable_logging : bool, optional
        True の場合、ログを標準出力します。
    Returns
    -------
    dask.array.Array または類似のオブジェクト
        十分なメモリ領域がある場合は tensor.compute() の結果を返します。
        その他の場合は tensor をそのまま返します。
    """
    import psutil
    # テンソルのメモリ量をバイト単位で推定します。
    # tensor.size と tensor.dtype.itemsize を用いる。
    try:
        required = tensor.size * tensor.dtype.itemsize * safety_factor
    except AttributeError:
        raise TypeError("The input tensor must have 'size' and 'dtype.itemsize' attributes.")

    # システムの利用可能メモリをバイト単位で取得します。
    avail = psutil.virtual_memory().available

    # メモリ量を標準出力します。
    if enable_logging:
        print(f"safe_compute(): Required memory (with safety factor {safety_factor}): {required / (1024**3):.2f} GB")
        print(f"safe_compute(): Available memory: {avail / (1024**3):.2f} GB")

    if required < avail:
        # 十分なメモリ領域がある場合、compute() を実行します。
        if enable_logging:
            print("safe_compute(): Sufficient memory available. Proceeding with compute().")
        return tensor.compute()
    else:
        # 十分なメモリ領域がない場合はそのまま返します。
        if enable_logging:
            print(f"safe_compute(): Not enough memory to compute() tensor.")
        return tensor

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

    import dask.array as da
    x = da.random.random((1000, 100, 10))
    result = safe_compute(x, safety_factor=1e6, enable_logging=True)
    result = safe_compute(x, enable_logging=True)


# In[ ]:





# In[ ]:




