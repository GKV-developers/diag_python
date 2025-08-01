# Diagnostics tool: diag_python

Post-processing tool for GKV binary output
**diag_python** is the python version of diag, which read Zarr format files of GKV binary output.



### How to use diag_python
1. Copy whole diag_python/ into the output directory of GKV. For example,
- YOUR_GKV_EXECUTED_DIR/
  - diag_python/
  - cnt/ (Zarr format files \*.zarr will be read by diag_python)
  - fxv/ (Zarr format files \*.zarr will be read by diag_python)
  - phi/ (Zarr format files \*.zarr will be read by diag_python)
  - hst/ (gkvp.mtr.001 will be read by diag_python)
  - src/ (gkvp_header.f90 will be read by diag_python)
  - log/
  - gkvp_namelist.001 (gkvp_namelist.001 will be read by diag_python)
  - sub.q.001


2. Initial settings in main.py
    ```
    ### Read NetCDF data phi.*.nc or Zarr store gkvp.phi.*.zarr/ by xarray ###
    FILETYPE="GKV_zarr"
    # FILETYPE="GKV_nc"
    # FILETYPE="diag_nc"
    if FILETYPE=="GKV_zarr":
        xr_phi = rb_open('../phi/gkvp.phi.*.zarr/')
        xr_Al  = rb_open('../phi/gkvp.Al.*.zarr/')
        xr_mom = rb_open('../phi/gkvp.mom.*.zarr/')
        xr_fxv = rb_open('../fxv/gkvp.fxv.*.zarr/')
        xr_cnt = rb_open('../cnt/gkvp.cnt.*.zarr/')
        xr_trn = rb_open('../phi/gkvp.trn.*.zarr/')
        tri_filelist = rb_get_tri_filelist('../phi/gkvp.tri.*.zarr/')
        xr_tri_list=[]
        for file in tri_filelist:
            xr_tri=rb_open(file + '.*.zarr/')
            xr_tri_list.append(xr_tri)
    elif FILETYPE=="GKV_nc":
        xr_phi = rb_open('../phi/gkvp.phi.*.nc')
        xr_Al  = rb_open('../phi/gkvp.Al.*.nc')
        xr_mom = rb_open('../phi/gkvp.mom.*.nc')
        xr_fxv = rb_open('../fxv/gkvp.fxv.*.nc')
        xr_cnt = rb_open('../cnt/gkvp.cnt.*.nc')
        xr_trn = rb_open('../phi/gkvp.trn.*.nc')
        tri_filelist = rb_get_tri_filelist('../phi/gkvp.tri.*.nc')
        xr_tri_list=[]
        for file in tri_filelist:
            xr_tri=rb_open(file + '.*.nc')
            xr_tri_list.append(xr_tri)
    elif FILETYPE=="diag_nc":
        xr_phi = rb_open('../post/data/phi.*.nc')
        xr_Al  = rb_open('../post/data/Al.*.nc')
        xr_mom = rb_open('../post/data/mom.*.nc')
        xr_fxv = rb_open('../post/data/fxv.*.nc')
        xr_cnt = rb_open('../post/data/cnt.*.nc')
        xr_trn = rb_open('../post/data/trn.*.nc')
        tri_filelist = rb_get_tri_filelist('../post/data/tri.*.nc')
        xr_tri_list=[]
        for file in tri_filelist:
            xr_tri=rb_open(file + '.*.nc')
            xr_tri_list.append(xr_tri)
    
    print("xr_phi:", xr_phi)
    print("tri_filelist:", tri_filelist)
    
    ### Set geometric constants ###
    geom_set(headpath='../src/gkvp_header.f90', nmlpath="../gkvp_namelist.001", mtrpath='../hst/gkvp.mtr.001')
    ```


3. Call functions, e.g.:
    ```
    # Plot phi[y,x] at t[it], zz[iz]
    it = 3
    iz = 8
    phiinxy(it, iz, xr_phi, flag="display")
    ```
    see help(phiinxy) for details.
