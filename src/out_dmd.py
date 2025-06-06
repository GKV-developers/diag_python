#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
"""
Output DMD modes, eigenvalues and amplitudes (DMD: Dynamic Mode Decomposition)

Module dependency: -

Third-party libraries: numpy, matplotlib, pydmd
"""

def phidmd(xr_phi, mz=None, my=None, mx=None, tsta=0, tend=-1, dmd_method="DMD", DMD_param=None, flag=None, outdir="./data/"):
    """
    Output DMD modes, eigenvalues and amplitudes (DMD: Dynamic Mode Decomposition)

    Parameters
    ----------
        xr_phi : xarray Dataset
            xarray Dataset of phi.*.nc, read by diag_rb
        mz : int, optional
            If specified, slice only at this index along the zz-axis; if None, do not slice (use all indices).
            # Default: None
        my : int, optional
            If specified, slice only at this index along the ky-axis; if None, do not slice (use all indices).
            # Default: None
        mx : int, optional
            If specified, slice only at this index along the kx-axis; if None, do not slice (use all indices).
            # Default: None
        tsta : int, optional
            starting time index for slicing (inclusive)
            # Default: 0
        tend : int, optional
            ending time index for slicing (exclusive)
            # Default: -1 (till end)
        dmd_method : str, optional
            DMD method to apply: "DMD", "MrDMD", or "HankelDMD"
            # Default: "DMD"
        DMD_param : dict, optional
            dictionary of hyperparameters for the chosen DMD method
            # Default: None
        flag : str
            # flag=="display" - show figure on display
            # flag=="savefig" - save figure as png
            # otherwise       - return data array
            # Default: None
        outdir : str, optional
            Output directory path
            # Default: ./data/

    Returns
    -------
        eigs[r]: Numpy array, dtype=np.complex128
            r: number of modes

        amplitudes[r]: Numpy array, dtype=np.complex128
            r: number of modes

        modes[n, r]: Numpy array, dtype=np.complex128
            n: number of spatial points
            r: number of modes
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from pydmd import DMD, MrDMD, HankelDMD

    ### Data Processing ###
    # Create phi (complex) sliced by tsta:tend
    if 'rephi' in xr_phi and 'imphi' in xr_phi:
        rephi = xr_phi['rephi'][tsta:tend,:,:,:]  # dim: t, zz, ky, kx
        imphi = xr_phi['imphi'][tsta:tend,:,:,:]  # dim: t, zz, ky, kx
        phi = (rephi + 1.0j*imphi).values
    elif 'phi' in xr_phi:
        phi = xr_phi['phi'][tsta:tend,:,:,:]  # dim: t, zz, ky, kx
        phi = phi.values

    # Prepare a list of full slices for all axes
    slices = [slice(None)] * phi.ndim

    # If an index is specified for any axis, slice only that element
    if mz is not None:
        slices[1] = slice(mz, mz + 1)
    if my is not None:
        slices[2] = slice(my, my + 1)
    if mx is not None:
        slices[3] = slice(mx, mx + 1)

    # Obtain the sliced array
    sliced_phi = phi[tuple(slices)]

    # Keep the time axis and flatten the other axes to reshape into (time, space)
    reshaped_phi = sliced_phi.reshape(sliced_phi.shape[0], -1)
    transposed_phi = reshaped_phi.T

    # Apply hyperparameters
    if dmd_method == "DMD":
        if DMD_param == None:
            svd_rank = 500
        else:
            svd_rank = DMD_param["svd_rank"]
    elif dmd_method == "MrDMD":
        if DMD_param == None:
            svd_rank = 500
            max_level = 5
            max_cycles = 1
        else:
            svd_rank = DMD_param["svd_rank"]
            max_level = DMD_param["max_level"]
            max_cycles = DMD_param["max_cycles"]
    elif dmd_method == "HankelDMD":
        if DMD_param == None:
            svd_rank = 500
            d = 10
        else:
            svd_rank = DMD_param["svd_rank"]
            d = DMD_param["d"]

    # Create and fit the DMD model
    if dmd_method == "DMD":
        dmd = DMD(svd_rank=svd_rank)
    elif dmd_method == "MrDMD":
        sub_dmd = DMD(svd_rank=svd_rank)
        dmd = MrDMD(sub_dmd, max_level=max_level, max_cycles=max_cycles)
    elif dmd_method == "HankelDMD":
        dmd = HankelDMD(svd_rank=svd_rank, d=d)

    dmd.fit(transposed_phi)

    # Retrieve eigenvalues, modes, dynamics, and amplitudes
    eigs = dmd.eigs                 # DMD eigenvalues (complex)
    modes = dmd.modes               # DMD modes (complex eigenvectors)
    dynamics = dmd.dynamics         # Dynamics (complex)
    if dmd_method in ("DMD", "HankelDMD"):
        amplitudes = dmd.amplitudes # Mode amplitudes (complex)
    elif dmd_method == "MrDMD":
        amplitudes = np.linalg.pinv(modes) @ transposed_phi[:, 0] # Mode amplitudes (complex)

    # Reconstruct the original data matrix
    phi_reconstructed = modes @ dynamics
    phi_reconstructed = phi_reconstructed.T  # back to (time, space)

    # Post-processing for HankelDMD
    if dmd_method == "HankelDMD":
        # Number of original variables
        n = transposed_phi.shape[0]
        # Extract only the first n stacked components
        phi_reconstructed = phi_reconstructed[:, :n]

    ### Data Output ### 
    # Scatter plot on the complex plane
    plt.figure(figsize=(6,6))
    plt.scatter(eigs.real, eigs.imag, marker='o')
    # Draw unit circle for reference
    circle = plt.Circle((0, 0), 1.0, color='r', fill=False, linestyle='--')
    ax = plt.gca()
    ax.add_patch(circle)

    # Configure axes and grid
    ax.axhline(0, color='black', linewidth=0.5)  # real axis
    ax.axvline(0, color='black', linewidth=0.5)  # imaginary axis
    ax.set_xlabel("Real part")
    ax.set_ylabel("Imag part")
    ax.set_title("DMD Eigenvalues on Complex Plane")
    ax.set_aspect('equal', 'box')
    ax.grid(True)

    if (flag == "display"):
        plt.show()
    elif (flag == "savefig"):
        # Save plot to file without displaying
        filename = os.path.join(outdir,'complex_plane.png') 
        plt.savefig(filename)
        plt.close()

    # Plot original vs reconstructed data
    if (flag == "display" or flag == "savefig"):
        # --- Compute common colormap range (vmin, vmax) ---
        orig = reshaped_phi.real        # original data
        recon = phi_reconstructed.real  # reconstructed data

        vmin = np.minimum(orig.min(), recon.min())
        vmax = np.maximum(orig.max(), recon.max())

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(orig, 
                   aspect='auto', 
                   extent=[0, orig.shape[1], tsta, tsta + orig.shape[0] - 1],
                   origin='lower',
                   vmin=vmin, vmax=vmax)
        plt.title("original data")
        plt.xlabel("Spatial index")
        plt.ylabel("Time step")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(recon, 
                   aspect='auto', 
                   extent=[0, recon.shape[1], tsta, tsta + recon.shape[0] - 1],
                   origin='lower',
                   vmin=vmin, vmax=vmax)
        plt.title("reconstructed data")
        plt.xlabel("Spatial index")
        plt.ylabel("Time step")
        plt.colorbar()

        plt.tight_layout()
        if (flag == "display"):
            plt.show()
        elif (flag == "savefig"):
            # Save plot to file without displaying
            filename = os.path.join(outdir,'dmd_reconstructed.png') 
            plt.savefig(filename)
            plt.close()

    # Sort modes by descending real part of log(eigenvalues)
    sorted_indices = np.argsort(np.log(eigs).real)[::-1]

    # Reorder modes and dynamics according to sorted indices
    sorted_modes = dmd.modes[:, sorted_indices]
    sorted_dynamics = dmd.dynamics[sorted_indices, :]
    # Post-processing for HankelDMD
    if dmd_method == "HankelDMD":
        # Number of original variables
        n = transposed_phi.shape[0]
        # Extract only the first n stacked components
        sorted_modes = sorted_modes[:n, :]

    # Branch: display/savefig/savetxt or return data arrays
    if (flag == "display" or flag == "savefig"):
        # Case when exactly one of mx, my, mz is None
        if [mx, my, mz].count(None) == 1:
            plt.figure(figsize=(8, 10))
            # Plot real/imag parts of top 4 modes
            for i in range(4):
                plt.subplot(4, 2, 2*i+1)
                plt.plot(sorted_modes[:, i].real, label=f'real part')
                plt.plot(sorted_modes[:, i].imag, label=f'imaginary part')
                plt.title(f'Mode {i+1}')
                plt.xlabel('Spatial index')
                plt.ylabel('Mode amplitute')
                plt.legend()
            # Plot real/imag parts of top 4 dynamics
            for i in range(4):
                plt.subplot(4, 2, 2*i + 2)
                nt = sorted_dynamics.shape[1]
                time = np.arange(tsta, tsta + nt)
                plt.plot(time, sorted_dynamics[i, :].real, label=f'real part')
                plt.plot(time, sorted_dynamics[i, :].imag, label=f'imaginary part')
                plt.title(f'Dynamics {i+1}')
                plt.xlabel('Time step')
                plt.ylabel('Dynamics amplitude')
                plt.legend()
        else:
            plt.figure(figsize=(8, 10))
            # Plot real/imag parts of top 4 dynamics only
            for i in range(4):
                plt.subplot(4, 1, i + 1)
                nt = sorted_dynamics.shape[1]
                time = np.arange(tsta, tsta + nt)
                plt.plot(time, sorted_dynamics[i, :].real, label=f'real part')
                plt.plot(time, sorted_dynamics[i, :].imag, label=f'imaginary part')
                plt.title(f'Dynamics {i+1}')
                plt.xlabel('Time step')
                plt.ylabel('Dynamics amplitude')
                plt.legend()

        plt.tight_layout()
        if (flag == "display"):
            plt.show()
        elif (flag == "savefig"):
            # Save plot to file without displaying
            filename = os.path.join(outdir,'mode.png')
            plt.savefig(filename)
            plt.close()
        return

    elif (flag == "None"):
        return eigs, amplitudes, modes

    elif (flag == "savetxt"):
        print("flag='savetxt' is not supported.")
        return




if (__name__ == '__main__'):
    import os
    from diag_rb import rb_open


    ### Examples of use ###


    ### phidmd ###
    #help(phidmd)
    xr_phi = rb_open('../../phi/gkvp.phi.*.zarr/')
    #print(xr_phi)
    print("# Plot DMD eigenvalues, reconstructed data, modes, and dynamics.")
    outdir='../data/phidmd/'
    os.makedirs(outdir, exist_ok=True)
    my = int((len(xr_phi.ky)-1)/2)
    mx = int((len(xr_phi.kx)-1)/2)
    phidmd(xr_phi, my=my, mx=mx, flag='display')
    phidmd(xr_phi, my=my, mx=mx, flag='savefig', outdir=outdir)


# In[ ]:




