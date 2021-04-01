#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 11:46:32 2020

@author: p-user
"""

def fluiddetaltrans_loop_calc(diag_mx, diag_my, it, phi_ed, Al_ed, mom_ed, flag, outdir="../data_fluid/"):
    """
    docstrings;.....
    """
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from diag_fft import fft_backward_xy, fft_backward_xyz, fft_forward_xyz
    from diag_intgrl import intgrl_thet
    from diag_geom import omg, ksq, g0, g1, bb ,Anum, Znum, tau, fcs, sgn # 計算に必要なglobal変数を呼び込む
    from diag_geom import nxw, nyw, ns, rootg, kx, ky, ky_ed, kx_ed, nx, global_ny, global_nz  # 格子点情報、磁気座標情報、座標情報を呼び込む
    
    #nx = int((len(xr_phi['kx'])-1)/2)       # diag_geom.pyから変数情報を読み込む。
    #global_ny = int(len(xr_phi['ky'])-1)    # diag_geom.pyから変数情報を読み込む。
    #global_nz = int(len(xr_phi['zz'])/2)    # diag_geom.pyから変数情報を読み込む。
    
    cfsrf = np.sum(rootg[:])
    fct = rootg[:]/cfsrf
    #fct = fct.reshape(2*global_nz, 1, 1)
    #print('確認： fct.shape=', fct.shape)

    nmom = 6
    
    # ----- jkpq_es, jpqk_es, jqkp_es の計算 ----------------------------------
    jkpq_es = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1 ))    
    jpqk_es = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1 ))
    jqkp_es = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1 )) 
    
    mx = diag_mx; my = diag_my
    
    coeff = 0.5*fcs[iss]*tau[iss]*Znum[iss] # ceffの計算式記述の合理化
    ceff = np.zeros((nmom, 1))
    #imom = 0
    ceff[0] = coeff 
    #imom = 1
    ceff[1] = coeff 
    #imom = 2
    ceff[2] = coeff* 0.5 
    #imom = 3
    ceff[3] = coeff
    #imom = 4
    ceff[4] = coeff* 0.166666666666666
    #imom = 5
    ceff[5] = coeff
    
    phi_ed = phi_ed.reshape(1, 2*global_nz, 2*global_ny+1, 2*nx+1) # momと同一の次元に合わせる。
    for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
        qy = -py-my
        #print('py=', py, '  qy=', qy)
        for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):
            qx = -px-mx
            #print('py=,', py, ',   qy=,', qy,'px=,', px, ',  qx=,', qx)                       
            jkpq_es[:,:,py,px] = jkpq_es[:,:,py,px] + fct*                 (-ceff[:]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real((phi_ed[:,:,py,px]*mom_ed[:,:,qy,qx]                        -phi_ed[:,:,qy,qx]*mom_ed[:,:,py,px])*mom_ed[:,:,my,mx]))   
            #print('確認 jkpq_es[0,:,py,px]=', jkpq_es[0,0,py,px])
    
            
            jpqk_es[:,:,py,px] = jpqk_es[:,:,py,px] + fct*                 (-ceff[:]*(-kx_ed[qx]*ky_ed[my]+ky_ed[qy]*kx_ed[mx])*                np.real((phi_ed[:,:,qy,qx]*mom_ed[:,:,my,mx]                        -phi_ed[:,:,my,mx]*mom_ed[:,:,qy,qx])*mom_ed[:,:,py,px]))  
            #print('確認 jpqk_es[0,:,py,px].shape=', jpqk_es[0,:,py,px].shape)
            

            jqkp_es[:,:,py,px] = jqkp_es[:,:,py,px] + fct*                (-ceff[:]*(-kx_ed[mx]*ky_ed[py]+ky_ed[my]*kx_ed[px])*                np.real((phi_ed[:,:,my,mx]*mom_ed[:,:,py,px]                        -phi_ed[:,:,py,px]*mom_ed[:,:,my,mx])*mom_ed[:,:,qy,qx]))
            #print('確認 jqkp_es[0,:,py,px].shape=', jqkp_es[0,:,py,px].shape)
                    

    # ----- jkpq_em, jpqk_em, jqkp_em の計算 ----------------------------------
    jkpq_em = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1 ))    
    jpqk_em = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1 ))
    jqkp_em = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1 ))     

    mx = diag_mx; my = diag_my
    
    ceff = np.zeros((nmom, 1))  # ~_emの計算のため、イニシャライズする。  
    coeff = 0.5*fcs[iss]*tau[iss]*Znum[iss]
    
    imom = 0
    ceff[imom] = coeff * (- np.sqrt(tau[iss]/Anum[iss]))
    
    for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
        qy = -py-my
        #print('qy=', qy)
        for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):
            qx = -px-mx
            #print('   qx=', qx)           
            jkpq_em[0,:,py,px] = jkpq_em[0,:,py,px] + fct*                 (-ceff[0]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,py,px]*mom_ed[0,:,qy,qx]*mom_ed[1,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[0,:,py,px]*mom_ed[1,:,my,mx]                        +Al_ed[:,py,px]*mom_ed[1,:,qy,qx]*mom_ed[0,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[1,:,py,px]*mom_ed[0,:,my,mx]) )   
            #print('確認 jkpq_em[0,:,py,px]=', jkpq_em[0,0,py,px])
    
            
            jpqk_em[0,:,py,px] = jpqk_em[0,:,py,px] + fct*                 (-ceff[0]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,qy,qx]*mom_ed[0,:,my,mx]*mom_ed[1,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[0,:,qy,qx]*mom_ed[1,:,py,px]                        +Al_ed[:,qy,qx]*mom_ed[1,:,my,mx]*mom_ed[0,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[1,:,qy,qx]*mom_ed[0,:,py,px]) )  
            #print('確認 jpqk_em[0,:,py,px].shape=', jpqk_em[0,:,py,px].shape)
            

            jqkp_em[0,:,py,px] = jqkp_em[0,:,py,px] + fct*                 (-ceff[0]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,my,mx]*mom_ed[0,:,py,px]*mom_ed[1,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[0,:,my,mx]*mom_ed[1,:,qy,qx]                        +Al_ed[:,my,mx]*mom_ed[1,:,py,px]*mom_ed[0,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[1,:,my,mx]*mom_ed[0,:,qy,qx]) ) 
            #print('確認 jqkp_em[0,:,py,px].shape=', jqkp_em[0,:,py,px].shape)    
    
    imom = 1
    ceff[imom] = coeff * (- np.sqrt(tau[iss]/Anum[iss]))             
    
    for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
        qy = -py-my
        for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):
            qx = -px-mx
                       
            jkpq_em[1,:,py,px] = jkpq_em[1,:,py,px] + fct*                 (-ceff[1]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,py,px]*mom_ed[1,:,qy,qx]*mom_ed[2,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[1,:,py,px]*mom_ed[2,:,my,mx]                        +Al_ed[:,py,px]*mom_ed[2,:,qy,qx]*mom_ed[1,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[2,:,py,px]*mom_ed[1,:,my,mx]) )   
            #print('確認 jkpq_em[1,:,py,px]=', jkpq_em[1,0,py,px])
    
            
            jpqk_em[1,:,py,px] = jpqk_em[1,:,py,px] + fct*                 (-ceff[1]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,qy,qx]*mom_ed[1,:,my,mx]*mom_ed[2,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[1,:,qy,qx]*mom_ed[2,:,py,px]                        +Al_ed[:,qy,qx]*mom_ed[2,:,my,mx]*mom_ed[1,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[2,:,qy,qx]*mom_ed[1,:,py,px]) )  
            #print('確認 jpqk_em[1,:,py,px].shape=', jpqk_em[1,:,py,px].shape)
            

            jqkp_em[1,:,py,px] = jqkp_em[1,:,py,px] + fct*                 (-ceff[1]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,my,mx]*mom_ed[1,:,py,px]*mom_ed[2,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[1,:,my,mx]*mom_ed[2,:,qy,qx]                        +Al_ed[:,my,mx]*mom_ed[2,:,py,px]*mom_ed[1,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[2,:,my,mx]*mom_ed[1,:,qy,qx]) ) 
            #print('確認 jqkp_em[1,:,py,px].shape=', jqkp_em[1,:,py,px].shape)         


    
    imom = 2
    ceff[imom] = coeff * (- np.sqrt(tau[iss]/Anum[iss])) * 0.5           
    
    for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
        qy = -py-my
        for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):
            qx = -px-mx
                       
            jkpq_em[2,:,py,px] = jkpq_em[2,:,py,px] + fct*                 (-ceff[2]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,py,px]*mom_ed[2,:,qy,qx]*mom_ed[4,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[2,:,py,px]*mom_ed[4,:,my,mx]                        +Al_ed[:,py,px]*mom_ed[4,:,qy,qx]*mom_ed[2,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[4,:,py,px]*mom_ed[2,:,my,mx]) )   
            #print('確認 jkpq_em[1,:,py,px]=', jkpq_em[2,0,py,px])
    
            
            jpqk_em[2,:,py,px] = jpqk_em[2,:,py,px] + fct*                 (-ceff[2]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,qy,qx]*mom_ed[2,:,my,mx]*mom_ed[4,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[2,:,qy,qx]*mom_ed[4,:,py,px]                        +Al_ed[:,qy,qx]*mom_ed[4,:,my,mx]*mom_ed[2,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[4,:,qy,qx]*mom_ed[2,:,py,px]) )  
            #print('確認 jpqk_em[2,:,py,px].shape=', jpqk_em[2,:,py,px].shape)
            

            jqkp_em[2,:,py,px] = jqkp_em[2,:,py,px] + fct*                 (-ceff[2]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,my,mx]*mom_ed[2,:,py,px]*mom_ed[4,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[2,:,my,mx]*mom_ed[4,:,qy,qx]                        +Al_ed[:,my,mx]*mom_ed[4,:,py,px]*mom_ed[2,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[4,:,my,mx]*mom_ed[2,:,qy,qx]) ) 
            #print('確認 jqkp_em[2,:,py,px].shape=', jqkp_em[2,:,py,px].shape)         


    
    imom = 3
    ceff[imom] = coeff * (- np.sqrt(tau[iss]/Anum[iss]))             
    
    for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
        qy = -py-my
        for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):
            qx = -px-mx
                       
            jkpq_em[3,:,py,px] = jkpq_em[3,:,py,px] + fct*                 (-ceff[0]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,py,px]*mom_ed[3,:,qy,qx]*mom_ed[5,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[3,:,py,px]*mom_ed[5,:,my,mx]                        +Al_ed[:,py,px]*mom_ed[5,:,qy,qx]*mom_ed[3,:,my,mx]                        -Al_ed[:,qy,qx]*mom_ed[5,:,py,px]*mom_ed[3,:,my,mx]) )   
            #print('確認 jkpq_em[3,:,py,px]=', jkpq_em[3,0,py,px])
    
            
            jpqk_em[3,:,py,px] = jpqk_em[3,:,py,px] + fct*                 (-ceff[0]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,qy,qx]*mom_ed[3,:,my,mx]*mom_ed[5,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[3,:,qy,qx]*mom_ed[5,:,py,px]                        +Al_ed[:,qy,qx]*mom_ed[5,:,my,mx]*mom_ed[3,:,py,px]                        -Al_ed[:,my,mx]*mom_ed[5,:,qy,qx]*mom_ed[3,:,py,px]) )  
            #print('確認 jpqk_em[3,:,py,px].shape=', jpqk_em[3,:,py,px].shape)
            

            jqkp_em[3,:,py,px] = jqkp_em[3,:,py,px] + fct*                 (-ceff[0]*(-kx_ed[px]*ky_ed[qy]+ky_ed[py]*kx_ed[qx])*                np.real(Al_ed[:,my,mx]*mom_ed[3,:,py,px]*mom_ed[5,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[3,:,my,mx]*mom_ed[5,:,qy,qx]                        +Al_ed[:,my,mx]*mom_ed[5,:,py,px]*mom_ed[3,:,qy,qx]                        -Al_ed[:,py,px]*mom_ed[5,:,my,mx]*mom_ed[3,:,qy,qx]) ) 
            #print('確認 jqkp_em[3,:,py,px].shape=', jqkp_em[3,:,py,px].shape)  

    
    # iz方向の加算とimomごとの加算を同時に実行。結果はFortranのテキスト出力に相当
    jkpq_es_sum = np.sum(jkpq_es, axis=(0, 1))
    jpqk_es_sum = np.sum(jpqk_es, axis=(0, 1))
    jqkp_es_sum = np.sum(jqkp_es, axis=(0, 1))
    #print('確認： jkpq_es_sum.shape=', jkpq_es_sum.shape)  # Ex. jkpq_es_sum.shape= (13, 13)    
    jkpq_em_sum = np.sum(jkpq_em, axis=(0, 1)) 
    jpqk_em_sum = np.sum(jpqk_em, axis=(0, 1))
    jqkp_em_sum = np.sum(jqkp_em, axis=(0, 1)) 
    

    # ---以下、Python化を保留 前山先生と調整＠2021.3.15 ---------------------------------------------------- 
    """
    tk_es = np.zeros((nmom, 2*global_ny+1, 2*nx+1 ))
    tk_es_pos = np.zeros((nmom, 2*global_ny+1, 2*nx+1 ))
    tk_es_neg = np.zeros((nmom, 2*global_ny+1, 2*nx+1 ))
    tk_em = np.zeros((nmom, 2*global_ny+1, 2*nx+1 ))
    tk_em_pos = np.zeros((nmom, 2*global_ny+1, 2*nx+1 ))
    tk_em_neg = np.zeros((nmom, 2*global_ny+1, 2*nx+1 ))

    
    for pn in range(nmom): # nmom = 6

        for py in range(max(-global_ny-my, -global_ny), min(global_ny, global_ny-my)+1):
            for px in range(max(-nx-mx, -nx), min(nx, nx-mx)+1):

                tk_es[pn, py, px] = tk_es[pn, py, px] + jkpq_es_iz[pn, py, px]
    
                if (jkpq_es_iz[pn, py, px]) < 0.0:
                    tk_es_neg[pn, py, px] = tk_es_neg[pn, py, px] + jkpq_es_iz[pn, py, px]
                else:
                    tk_es_pos[pn, py, px] = tk_es_pos[pn, py, px] + jkpq_es_iz[pn, py, px]
    
                tk_em[pn, py, px] = tk_em[pn, py, px] + jkpq_em_iz[pn, py, px]
                
                if (jkpq_em_iz[pn, py, px]) < 0.0:
                    tk_em_neg[pn, py, px] = tk_em_neg[pn, py, px] + jkpq_em_iz[pn, py, px]
                else:
                    tk_em_pos[pn, py, px] = tk_em_pos[pn, py, px] + jkpq_em_iz[pn, py, px] 
    
    print('確認：tk_es.shape=', tk_es.shape)
    print('確認：tk_es_neg.shape=', tk_es_neg.shape)
    print('確認：tk_es_pos.shape=', tk_es_pos.shape)
    
    # imom毎の数値を合計
    tk_es_sum = np.sum(tk_es, axis=0)
    tk_es_neg_sum = np.sum(tk_es_neg, axis=0)
    tk_es_pos_sum = np.sum(tk_es_pos, axis=0)
    
    tk_em_sum = np.sum(tk_em, axis=0)
    tk_em_neg_sum = np.sum(tk_em_neg, axis=0)
    tk_em_pos_sum = np.sum(tk_em_pos, axis=0)    
    
    """

    # 出力用に配列を整理する
    
    kx_sht = np.fft.fftshift(kx_ed)    # この関数で配列データを昇順で並び替え。
    wky_sht = np.fft.fftshift(ky_ed)  # この関数で配列データを昇順で並び替え。
    #print('確認：kx_sht >>> \n', kx_sht) 
    #print('確認：wky_sht >>> \n', wky_sht) 
    
    m_kx, m_ky = np.meshgrid(kx_sht, wky_sht)  # 2D-Plot用メッシュグリッドの作成  
    
     
    jkpq_es_sum_sht = np.fft.fftshift(jkpq_es_sum) # データの並び替え
    jpqk_es_sum_sht = np.fft.fftshift(jpqk_es_sum)
    jqkp_es_sum_sht = np.fft.fftshift(jqkp_es_sum)
    jkpq_em_sum_sht = np.fft.fftshift(jkpq_em_sum)
    jpqk_em_sum_sht = np.fft.fftshift(jpqk_em_sum)
    jqkp_em_sum_sht = np.fft.fftshift(jqkp_em_sum)
    
    calc_time = time.time()
    calc_pass_time = calc_time - s_time
    print ('\n *** calc_pass_time ={:12.5f}sec'.format(calc_pass_time))
    
    # 各関数毎に出力（flag:diag_main.py or 'if (__name__ == '__main__')' で設定）
    
    # [1] jkpq_es_sum
    data_jkpq_es_sum = np.stack([m_kx, m_ky, jkpq_es_sum_sht], axis=2) # Ex. data_jkpq_es_sum.shape = (13, 13, 3)
    #print('\n 確認： data_jkpq_es_sum.shape =', data_jkpq_es_sum.shape )
    #print('data_jkpq_es_sum >>> \n', data_jkpq_es_sum)
       
    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("jkpq_es_sum_is= {:02d}_; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_es_sum = ax.pcolormesh(data_jkpq_es_sum[:,:,0], data_jkpq_es_sum[:,:,1], data_jkpq_es_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_es_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'jkpq_es_sum_fluiddetailtransinkxky_is_{:02d}_sum_t{:08d}_sht.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'jkpq_es_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of jkpq_es_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_jkpq_es_sum.shape))
            outfile.write('#           kx(px)            ky(py)             jkpq_es_sum \n')
            for data_slice in data_jkpq_es_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')               
           
    
    # [2] jpqk_es_sum
    data_jpqk_es_sum = np.stack([m_kx, m_ky, jpqk_es_sum_sht], axis=2)            
    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("jpqk_es_sum_is= {:02d}_; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_es_sum = ax.pcolormesh(data_jpqk_es_sum[:,:,0], data_jpqk_es_sum[:,:,1], data_jpqk_es_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_es_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'jpqk_es_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'jpqk_es_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of jpqk_es_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_jpqk_es_sum.shape))
            outfile.write('#           kx(px)            ky(py)             jpqk_es_sum \n')
            for data_slice in data_jpqk_es_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')    
    
    
    # [3] jqkp_es_sum
    data_jqkp_es_sum = np.stack([m_kx, m_ky, jqkp_es_sum_sht], axis=2)            
    #print('data_jqkp_es_sum >>> \n', data_jqkp_es_sum)
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("jqkp_es_sum_is= {:02d}_; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_es_sum = ax.pcolormesh(data_jqkp_es_sum[:,:,0], data_jqkp_es_sum[:,:,1], data_jqkp_es_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_es_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'jqkp_es_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'jqkp_es_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of jqkp_es_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_jqkp_es_sum.shape))
            outfile.write('#           kx(px)            ky(py)             jqkp_es_sum \n')
            for data_slice in data_jqkp_es_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')    


    # [4] jkpq_em_sum
    data_jkpq_em_sum = np.stack([m_kx, m_ky, jkpq_em_sum_sht], axis=2)            
    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("jkpq_em_sum_is= {:02d}_; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_em_sum = ax.pcolormesh(data_jkpq_em_sum[:,:,0], data_jkpq_em_sum[:,:,1], data_jkpq_em_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_em_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'jkpq_em_sum_fluiddetailtransinkxky_is_{:02d}_sum_t{:08d}_sht.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'jkpq_em_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of jkpq_em_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_jkpq_em_sum.shape))
            outfile.write('#           kx(px)            ky(py)             jkpq_em_sum \n')
            for data_slice in data_jkpq_em_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')    


    # [5] jpqk_em_sum
    data_jpqk_em_sum = np.stack([m_kx, m_ky, jpqk_em_sum_sht], axis=2)            
    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("jpqk_em_sum_is= {:02d}_; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_em_sum = ax.pcolormesh(data_jpqk_em_sum[:,:,0], data_jpqk_em_sum[:,:,1], data_jpqk_em_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_em_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'jpqk_em_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'jpqk_em_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of jpqk_em_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_jpqk_em_sum.shape))
            outfile.write('#           kx(px)            ky(py)             jpqk_es_sum \n')
            for data_slice in data_jpqk_em_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')  


    # [6] jqkp_em_sum
    data_jqkp_em_sum = np.stack([m_kx, m_ky, jqkp_em_sum_sht], axis=2)            
    
    # 場合分け
    if flag == 'display' or flag == 'savefig' :
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        #plt.axis('tight') # 見やすさを優先するときは、このコマンドを有効にする
        ax.set_xlim(-1.55, 1.55) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_ylim(-0.65, 0.65) # 軸範囲を指定するときは、plt.axis('tight') を無効にする
        ax.set_title("jqkp_em_sum_is= {:02d}_; t= {:08d}".format(iss, it))
        ax.set_xlabel(r"Radial wavenumber $kx$")
        ax.set_ylabel(r"Poloidal wavenumber $ky$")
        quad_em_sum = ax.pcolormesh(data_jqkp_em_sum[:,:,0], data_jqkp_em_sum[:,:,1], data_jqkp_em_sum[:,:,2],
                             cmap='jet',shading="auto")        
        fig.colorbar(quad_em_sum)

        if (flag == "display"):   # flag=="display" - show figure on display
            plt.show()

        elif (flag == "savefig"): # flag=="savefig" - save figure as png
            filename = os.path.join(outdir,'jqkp_em_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.png'.format(iss, it))
            plt.savefig(filename)
            plt.close()

    elif (flag == "savetxt"):     # flag=="savetxt" - save data as txt
        filename = os.path.join(outdir,'jqkp_em_sum_fluiddetailtransinkxky_is{:02d}_sum_t{:08d}_sht.txt'.format(iss, it))
        with open(filename, 'w') as outfile:
            outfile.write('# sum of jqkp_em_iss = {:d} \n'.format(iss))
            outfile.write('# loop = {:d}, t = {:f}\n'.format(it, float(xr_mom['t'][it])))
            outfile.write('### Data shape: {} ###\n'.format(data_jqkp_em_sum.shape))
            outfile.write('#           kx(px)            ky(py)             jqkp_em_sum \n')
            for data_slice in data_jqkp_es_sum:
                np.savetxt(outfile, data_slice, fmt='%.7e', delimiter='\t')
                outfile.write('\n')

    return (data_jkpq_es_sum, data_jpqk_es_sum, data_jqkp_es_sum, data_jkpq_em_sum, data_jpqk_em_sum, data_jqkp_em_sum)


    # ---------------------------------------------------------------------------------------                       


def fluiddetailtrans_loop(it, iss, xr_phi, xr_Al, xr_mom, flag, outdir="../data_fluid/"):

    """
    Output transfer diagnostics in (kx,ky)
    
    Parameters   # fluiddetailtrans用に未修整
    ----------
        it : int
            index of t-axis
        iss : int
            index of species-axis            
        imom : int
            index of moment-axis
            itrn= 0: Entropy S_s
            itrn= 1: Electrostatic field energy W_E
            itrn= 2: Magnetic field energy W_M
            itrn= 3: W_E to S_s interaction R_sE
            itrn= 4: W_M to S_s interaction R_sM
            itrn= 5: Entropy transfer via ExB nonlinearity I_sE
            itrn= 6: Entropy transfer via magnetic nonlinearity I_sM
            itrn= 7: Collisional dissipation D_s
            itrn= 8: Particle flux by ExB flows G_sE
            itrn= 9: Particle flux by magnetic flutters G_sM
            itrn=10: Energy flux by ExB flows Q_sE
            itrn=11: Energy flux by magnetic flutters Q_sM
        xr_mom : xarray Dataset
            xarray Dataset of mom.*.nc, read by diag_rb
        outdir : str, optional
            Output directory path:
            # Default: ./data/

    Returns
    -------
        data[global_ny+1,2*nx+1,3]: Numpy array, dtype=np.float64
            # kx = data[:,:,0]
            # ky = data[:,:,1]
            # trnkxky = data[:,:,2]    
    """


    import os
    import numpy as np
    import matplotlib.pyplot as plt
    #from scipy import fft
    from diag_fft import fft_backward_xy, fft_backward_xyz, fft_forward_xyz
    from diag_intgrl import intgrl_thet
    from diag_geom import omg, ksq, g0, g1, bb ,Anum, Znum, tau, fcs, sgn # 計算に必要なglobal変数を呼び込む
    from diag_geom import nxw, nyw, ns, rootg, kx, ky  # 格子点情報、時期座標情報、座標情報を呼び込む
    
    nx = int((len(xr_phi['kx'])-1)/2)
    global_ny = int(len(xr_phi['ky'])-1)
    global_nz = int(len(xr_phi['zz'])/2)
    
    # 時刻t[it]粒子種iss速度モーメントimomにおける三次元複素mom[zz,ky,kx]を切り出す
    rephi = xr_phi['rephi'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    imphi = xr_phi['imphi'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    phi = rephi + 1.0j*imphi
    
    reAl = xr_Al['reAl'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    imAl = xr_Al['imAl'][it, :, :, :]  # dim: t, is, imom, zz, ky, kx
    Al = reAl + 1.0j*imAl

    remom = xr_mom['remom'][it,iss, :, :, :, :]  # dim: t, is, imom, zz, ky, kx
    immom = xr_mom['immom'][it,iss, :, :, :, :]  # dim: t, is, imom, zz, ky, kx    
    mom = remom + 1.0j*immom
       
    phi = phi.load()
    Al = Al.load()
    mom = mom.load()
    #print('確認： type(mon) =', type(mom), '\n') # <class 'xarray.core.dataarray.DataArray'> 
    #print('確認： mom.shape =', mom.shape, '\n')

     
    # !- moments transform: gyrokinetic distribution -> non-adiabatic part
    # imom = 0
    mom[0] = mom[0] +sgn[iss]*fcs[iss]* g0[0] *phi/tau[iss]

    # imom = 1
    # none
    
    # imom = 2
    mom[2] = mom[2] + 0.5* sgn[iss] * fcs[iss] * g0[0] *phi

    # imom = 3
    mom[3] = mom[3] + sgn[iss] * fcs[iss] * phi * ((1.0 - bb[0]) * g0[0] + bb[0] * g1[0])

    #print("imom 出力の確認： moments transform: gyrokinetic distribution -> non-adiabatic part")
    #print('imom=0 iz=0, my=1 mx=  : >>>\n', mom[0][0,1,:].compute) # imom=0, iz=0, ky=1, kx= :

    
    # !--- moments transform: non-adiabatic part -> Hermite-Laguerre coefficients
    
    # imom = 0:
    mom[0] = Znum[iss] * mom[0] / fcs[iss]
    
    # imom = 1:
    mom[1] = np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[1] / fcs[iss]
    
    # imom = 2:
    mom[2] = 2.0* Znum[iss] * mom[2] / (fcs[iss] * tau[iss]) - mom[0]
    
    # imom = 3:
    mom[3] = - Znum[iss] * mom[3] / (fcs[iss] * tau[iss]) + mom[0]
    
    # imom = 4:
    mom[4] = 2.0 * np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[4] / (fcs[iss] * tau[iss]) - 3.0 * mom[1]
    
    # imom = 5:
    mom[5] = - np.sqrt(Anum[iss] / tau[iss]) * Znum[iss] * mom[5] / (fcs[iss] * tau[iss]) + mom[1]

        
    # ----------------- complex conjugateの算出 ------------------
    
    # 共役複素数を含む拡張した配列(*_ed)の作成
    # phiの拡張配列
    phi_ed = np.zeros((2*global_nz, 2*global_ny+1, 2*nx+1),dtype=np.complex128) # Ex. (16, 13, 13)
    phi_ed[:, 0:global_ny+1, 0:nx+1] = phi[:, 0:global_ny+1, nx:2*nx+1]
    # 第1象限    0:7 -> 7個   0:7 -> 7個        0:7 -> 7個     6:13 -> 7個      
    phi_ed[:, 0:global_ny+1, nx+1:2*nx+1] = phi[:, 0:global_ny+1, 0:nx]
    # 第2象限    0:7 -> 7個   7:13 -> 6個       0:7 -> 7個     0:6 -> 6個 
    phi_ed[:, global_ny+1:2*global_ny+1, nx+1:2*nx+1] = np.conj(phi[:, global_ny:0:-1, 2*nx:nx:-1])
    # 第3象限    7:13 -> 6個              7:13 -> 6個                6:0 -> 6個   12:6 -> 6個                                                        
    phi_ed[:, global_ny+1:2*global_ny+1, 0:nx+1] = np.conj(phi[:, global_ny:0:-1, nx::-1])                                           
    # 第4象限    7:13 -> 6個             0:7 -> 7個              6:0 -> 6個   7:0 -> 7個
    print('\n確認 phi_ed.shape=', phi_ed.shape)
    
    
    # Alの拡張配列
    Al_ed = np.zeros((2*global_nz, 2*global_ny+1, 2*nx+1), dtype=np.complex128) # Ex. (16, 13, 13)
    Al_ed[:, 0:global_ny+1, 0:nx+1] = Al[:, 0:global_ny+1, nx:2*nx+1]
    #           0:7 -> 7個   0:7 -> 7個        0:7 -> 7個     6:13 -> 7個      
    Al_ed[:, 0:global_ny+1, nx+1:2*nx+1] = Al[:, 0:global_ny+1, 0:nx]
    #           0:7 -> 7個   6:12 -> 6個       0:7 -> 7個     0:6 -> 6個 
    Al_ed[:, global_ny+1:2*global_ny+1, nx+1:2*nx+1] = np.conj(Al[:, global_ny:0:-1, 2*nx:nx:-1]) # 修正@0312
    #           7:13 -> 6個              6:12 -> 6個                6:0 -> 6個   12:6 -> 6個                                                        
    Al_ed[:, global_ny+1:2*global_ny+1, 0:nx+1] = np.conj(Al[:, global_ny:0:-1, nx::-1])                                           
    #           7:13 -> 6個             0:7 -> 7個              6:0 -> 6個   7:0 -> 7個    

    nmom = 6
    mom_ed = np.zeros((nmom, 2*global_nz, 2*global_ny+1, 2*nx+1), dtype=np.complex128) # Ex. (6, 16, 13, 13)
    mom_ed[:, :, 0:global_ny+1, 0:nx+1] = np.array(mom[:,:, 0:global_ny+1, nx:2*nx+1])
    #              0:7 -> 7個   0:7 -> 7個        0:7 -> 7個     6:13 -> 7個      
    mom_ed[:, :, 0:global_ny+1, nx+1:2*nx+1] = mom[:, :, 0:global_ny+1, 0:nx]
    #              0:7 -> 7個   6:12 -> 6個       0:7 -> 7個     0:6 -> 6個 
    mom_ed[:, :, global_ny+1:2*global_ny+1, nx+1:2*nx+1] = np.conj(mom[:, :, global_ny:0:-1, 2*nx:nx:-1])  # 修正@0312
    #              7:13 -> 6個              6:12 -> 6個                6:0 -> 6個   12:6 -> 6個                                                        
    mom_ed[:, :, global_ny+1:2*global_ny+1, 0:nx+1] = np.conj(mom[:, :, global_ny:0:-1, nx::-1])                                           
    #              7:13 -> 6個             0:7 -> 7個              6:0 -> 6個   7:0 -> 7個    
    """
    print('\n確認 mom_ed.shape=', mom_ed.shape, '\n')
    
    
    filename = os.path.join(outdir,'mom_expand_array_t={:02d}.txt'.format(it))
    with open(filename, 'w') as outfile:
        np.savetxt(outfile, mom_ed[0, 0, :, :], fmt='%.7e',delimiter='\t', newline='\n')
        outfile.write('\n')   
        outfile.write('\n') 
    
    filename = os.path.join(outdir,'mom_array_t={:02d}.txt'.format(it))
    with open(filename, 'w') as outfile:
        np.savetxt(outfile, mom[0, 0, :, :], fmt='%.7e',delimiter='\t', newline='\n')
        outfile.write('\n\n')        
    """
        
    diag_mx = 0; diag_my = 3
    fluiddetaltrans_loop_calc(diag_mx, diag_my, it, phi_ed, Al_ed, mom_ed, flag, outdir=outdir)
    
    return 
    
    ### Program end ###    
    
    
    
if (__name__ == '__main__'):
    
    from diag_geom import geom_set
    from diag_rb import rb_open
    import time
    global s_time
    
    s_time = time.time()
    xr_phi = rb_open('../../post/data/phi.*.nc')  
    xr_Al  = rb_open('../../post/data/Al.*.nc')                  
    xr_mom = rb_open('../../post/data/mom.*.nc')  
    #print("\n***** 確認 if (__name__ == '__main__'):･･･ xr_momの属性 >>>\n", xr_mom, '\n')
    
    it = 30; iss = 0  # 開発用ベンチマーク用データでは time step = 0,1,2, ..., 30
    geom_set( headpath='../../src/gkvp_header.f90', nmlpath="../../gkvp_namelist.001", mtrpath='../../hst/gkvp.mtr.001')

    fluiddetailtrans_loop(it, iss, xr_phi, xr_Al, xr_mom, flag="savetxt", outdir="../data/")

    e_time = time.time()
    pass_time = e_time - s_time
    print ('\n *** total_pass_time ={:12.5f}sec'.format(pass_time))
    
    


# In[ ]:




