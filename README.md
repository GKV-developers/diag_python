# Diagnostics tool: diag_python

Post-processing tool for GKV binary output
**diag_python** is the python version of diag, which read NetCDF files converted from GKV binary output.



### How to use diag_python
1. Copy whole diag_python/ into the output directory of GKV. For example,
- YOUR_GKV_EXECUTED_DIR/
  - diag_python/
  - cnt/
  - fxv/
  - phi/
  - hst/ (gkvp.mtr.001 will be read by diag_python)
  - src/
  - log/
  - gkvp_namelist.001 (gkvp_namelist.001 will be read by diag_python)
  - sub.q.001
  - post/ (NetCDF files \*.nc created by diag will be read by diag_python)


2. Initial settings in main.py
    ```
    ### Read NetCDF data phi.*.nc by xarray ### 
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
    # print("xr_phi:", xr_phi)
    # print("tri_filelist:", tri_filelist)
    
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
