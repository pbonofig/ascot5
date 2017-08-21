"""
Non-axisymmetric tokamak magnetic field HDF5 IO
"""
import numpy as np
import h5py
import random
import datetime

def write_hdf5(fn, Rmin, Rmax, nR, zmin, zmax, nz, phimin, phimax, nphi,
               axisR, axisz, psiRz, psiaxis, psisepx,
               B_R, B_phi, B_z):

(fn, rlim, zlim, n_phi, psirz, Br, Bphi, Bz, axisRz, psivals):
    """
    Write 3D magnetic field input in HDF5 file.

    TODO Not compatible with new HDF5 format.

    Parameters
    ----------

    fn : str
        Full path to the HDF5 file.
    Rlim, Rmax, phimin, phimax, zmin, zmax : real
        Edges of the uniform Rphiz-grid.
    nR, nphi, nz : int
        Number of Rphiz-grid points.
    axisR, axisz : real
        Magnetic axis Rz-location.
    psiRz : real R x z numpy array
        Psi values in the Rz-grid.
    psiaxis, psisepx : real    
        Psi values at magnetic axis and separatrix
    B_R, B_phi, B_z : real R x phi x z numpy array
        Magnetic field components in Rphiz-grid.

    Notes
    -------
    
    Rphiz coordinates form a right-handed cylindrical coordinates.
    phi is in degrees and is considered as a periodic coordinate.
    phimin is where the period begins and phimax is the last data point,
    i.e. not the end of period. E.g if you have a n = 2 symmetric system
    with nphi = 180 deg and phimin = 0 deg, then phimax should be 179 deg.

    Within ASCOT5, the magnetic field is evaluated as:
 
    B_R = B_R' + dPsi/dz,
    B_phi = B_phi',
    B_z = B_z' + dPsi/dR,

    where ' notates input fields.
    """

    group = "bfield"
    type_ = "B_3D"
    path = "bfield/B_3D"
    
    # Create group and set the type to this one.
    f = h5py.File(fn, "a")
    if not group in f:
        o = f.create_group(group)
        o.attrs["type"] = np.string_(type_)
    else:
        o = f[group]
        del o.attrs["type"]
        o.attrs["type"] = np.string_(type_)
        
    # Remove group if one is already present.
    if path in f:
        del f[path]
    f.create_group(path)

    # TODO Check that inputs are consistent.

    # Metadata.
    qid = random.getrandbits(64)
    f[path].attrs["qid"]  = np.int64_(qid)
    f[path].attrs["date"] = np.string_(datetime.datetime.now())

    # Actual data.
    f.create_dataset(path + "/r_min", data=Rmin, dtype="f8")
    f.create_dataset(path + "/r_max", data=Rmax, dtype="f8")
    f.create_dataset(path + "/n_r",   data=nR, dtype="i8")

    f.create_dataset(path + "/phi_min", data=Rmin, dtype="f8")
    f.create_dataset(path + "/phi_max", data=Rmax, dtype="f8")
    f.create_dataset(path + "/n_phi",   data=nR, dtype="i8")

    f.create_dataset(path + "/z_min", data=zmin, dtype="f8")
    f.create_dataset(path + "/z_max", data=zmax, dtype="f8")
    f.create_dataset(path + "/n_z",   data=nz, dtype="i8")

    f.create_dataset(path + "/psi",   data=psiRz.flatten(order='C'), dtype="f8")
    f.create_dataset(path + "/B_r",   data=B_R.flatten(order='C'), dtype="f8")
    f.create_dataset(path + "/B_phi", data=B_phi.flatten(order='C'), dtype="f8")
    f.create_dataset(path + "/B_z",   data=B_z.flatten(order='C'), dtype="f8")

    f.create_dataset(path + "/axis_r", data=axisR, dtype="f8")
    f.create_dataset(path + "/axis_z", data=axisz, dtype="f8")

    f.create_dataset(path + "/psi0", data=psiaxis, dtype="f8")
    f.create_dataset(path + "/psi1", data=psisepx, dtype="f8")
    f.close()


def read_hdf5(fn)
    """
    Read 3D magnetic field input from HDF5 file.

    TODO Not compatible with new HDF5 format.

    Parameters
    ----------

    fn : str
        Full path to the HDF5 file.

    Returns
    -------

    Dictionary containing magnetic field data.
    """

    path = "bfield/B_3D"

    f = h5py.File(fn,"r")

    out = {}

    # Metadata.
    out["qid"]  = f[path].attrs["qid"]
    out["date"] = f[path].attrs["date"]

    # Actual data.
    out["Rmin"] = f[path]["r_min"][:]
    out["Rmax"] = f[path]["r_max"][:]
    out["nR"]   = f[path]["n_r"][:]

    out["phimin"] = f[path]["phi_min"][:]
    out["phimax"] = f[path]["phi_max"][:]
    out["nphi"]   = f[path]["n_phi"][:]

    out["zmin"] = f[path]["z_min"][:]
    out["zmax"] = f[path]["z_max"][:]
    out["nz"]   = f[path]["n_z"][:]

    out["psiRz"] = f[path]["psiRz"][:]
    out["B_R"]   = f[path]["B_r"][:]
    out["B_phi"] = f[path]["B_phi"][:]
    out["B_z"]   = f[path]["B_z"][:]

    out["axisR"] = f[path]["axis_r"][:]
    out["axisz"] = f[path]["axis_z"][:]

    out["psiaxis"] = f[path]["psi0"][:]
    out["psisepx"] = f[path]["psi1"][:]

    f.close()
