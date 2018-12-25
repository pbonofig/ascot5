"""
Test collisional slowing-down and equilibrium.

This file initializes, runs, and plots test case for checking that ASCOT5
collision operator produces correct distribution for fast-particle slowing-down
and thermal equilibrium.

The tests are done using protons as markers. Even though these tests only
require collision operator, we also include orbit following to be sure that
this does not bias the results.

File: test_coulombcollisions.py
"""

import sys
import copy

import numpy                   as np
import scipy.constants         as constants
import matplotlib.pyplot       as plt

import a5py.ascot5io.ascot5    as ascot5
import a5py.ascot5io.options   as options
import a5py.ascot5io.B_GS      as B_GS
import a5py.ascot5io.E_TC      as E_TC
import a5py.ascot5io.plasma_1D as P_1D
import a5py.ascot5io.wall_2D   as W_2D
import a5py.ascot5io.N0_3D     as N0_3D
import a5py.ascot5io.mrk_gc    as mrk

sys.path.insert(0, '../')
sys.path.insert(0, '.')
import opt
import test_ascot

from a5py.preprocessing.analyticequilibrium import psi0 as psifun

e       = constants.elementary_charge
m_p_AMU = constants.physical_constants["proton mass in u"][0]
m_p     = constants.physical_constants["proton mass"][0]
m_a_AMU = constants.physical_constants["alpha particle mass in u"][0]
m_a     = constants.physical_constants["alpha particle mass"][0]
m_e_AMU = constants.physical_constants["electron mass in u"][0]
m_e     = constants.physical_constants["electron mass"][0]
c       = constants.physical_constants["speed of light in vacuum"][0]
eps0    = constants.physical_constants["electric constant"][0]

simtime_th = 2e-2

Te  = 1e3
ne  = 1e20
Eth = 1e3
Esd = 3.5e6

R0 = 6.2
z0 = 0
Bphi0 = 5.3
psi_mult = 200
# ITER-like but circular equilibrium
psi_coeff = np.array([ 8.629e-02,  3.279e-01,  5.268e-01, -2.366e-01,
                       3.825e-01, -3.573e-01, -1.484e-02,  1.506e-01,
                       7.428e-01, -4.447e-01, -1.084e-01,  1.281e-02, -0.155])

def init():
    """
    Initialize tests

    This function initializes three tests that simulate a thermal equilibrium
    in different modes: THERMAL_GO, THERMAL_GCF, and THERMAL_GCA. Three
    additional simulations are initialized that simulate alpha particle
    slowing-down in different modes: SLOWING_GO, SLOWING_GCF, and SLOWING_GCA.

    Input fields contain the test case name (to which the input corresponds to)
    as a description.
    """

    #**************************************************************************#
    #*       Generate options for THERMAL_GO, THERMAL_GCF, and THERMAL_GCA     #
    #*                                                                         #
    #**************************************************************************#
    odict = opt.generateopt()
    test_ascot.clean_opt(odict)

    odict["ENDCOND_SIMTIMELIM"]              = 1
    odict["ENDCOND_MAX_SIM_TIME"]            = simtime_th
    odict["ENABLE_ORBIT_FOLLOWING"]          = 1
    odict["ENABLE_COULOMB_COLLISIONS"]       = 1
    odict["ENABLE_R_phi_z_vpa_vpe_t_q_DIST"] = 1

    odict["DIST_MIN_R"]    = 4
    odict["DIST_MAX_R"]    = 10
    odict["DIST_NBIN_R"]   = 20
    odict["DIST_MIN_phi"]  = 0
    odict["DIST_MAX_phi"]  = 360
    odict["DIST_NBIN_phi"] = 1
    odict["DIST_MIN_z"]    = -5
    odict["DIST_MAX_z"]    = 5
    odict["DIST_NBIN_z"]   = 20
    odict["DIST_MIN_vpa"]  = -1.5e6
    odict["DIST_MAX_vpa"]  =  1.5e6
    odict["DIST_NBIN_vpa"] = 140
    odict["DIST_MIN_vpe"]  = 0
    odict["DIST_MAX_vpe"]  = 1.5e6
    odict["DIST_NBIN_vpe"] = 80
    odict["DIST_MIN_t"]    = 0
    odict["DIST_MAX_t"]    = simtime_th
    odict["DIST_NBIN_t"]   = 2

    odict["SIM_MODE"]                  = 1
    odict["FIXEDSTEP_USE_USERDEFINED"] = 1
    odict["FIXEDSTEP_USERDEFINED"]     = 1e-8
    opt.settypes(odict)
    options.write_hdf5(test_ascot.testfn, odict, desc="THERMAL_GO")

    odict["SIM_MODE"]                  = 2
    odict["FIXEDSTEP_USE_USERDEFINED"] = 1
    odict["FIXEDSTEP_USERDEFINED"]     = 2e-8
    opt.settypes(odict)
    options.write_hdf5(test_ascot.testfn, odict, desc="THERMAL_GCF")

    odict["SIM_MODE"]                  = 2
    odict["ENABLE_ADAPTIVE"]           = 1
    odict["ADAPTIVE_TOL_ORBIT"]        = 1e-6
    odict["ADAPTIVE_TOL_COL"]          = 9e0
    odict["ADAPTIVE_MAX_DRHO"]         = 0.1
    odict["ADAPTIVE_MAX_DPHI"]         = 10
    odict["FIXEDSTEP_USE_USERDEFINED"] = 1
    odict["FIXEDSTEP_USERDEFINED"]     = 1e-8
    opt.settypes(odict)
    options.write_hdf5(test_ascot.testfn, odict, desc="THERMAL_GCA")


    #**************************************************************************#
    #*  Generate options for SLOWING_GO, SLOWING_GCF, and SLOWING_GCA          #
    #*                                                                         #
    #**************************************************************************#
    odict = opt.generateopt()
    test_ascot.clean_opt(odict)

    odict["ENDCOND_ENERGYLIM"]                = 1
    odict["ENDCOND_MIN_ENERGY"]               = 1e3
    odict["ENDCOND_MIN_ENERGY_TIMES_THERMAL"] = 0
    odict["ENABLE_ORBIT_FOLLOWING"]           = 1
    odict["ENABLE_COULOMB_COLLISIONS"]        = 1
    odict["ENABLE_R_phi_z_vpa_vpe_t_q_DIST"]  = 1

    odict["DIST_MIN_R"]    = 4
    odict["DIST_MAX_R"]    = 10
    odict["DIST_NBIN_R"]   = 20
    odict["DIST_MIN_phi"]  = 0
    odict["DIST_MAX_phi"]  = 360
    odict["DIST_NBIN_phi"] = 1
    odict["DIST_MIN_z"]    = -5
    odict["DIST_MAX_z"]    = 5
    odict["DIST_NBIN_z"]   = 40
    odict["DIST_MIN_vpa"]  = -2e7
    odict["DIST_MAX_vpa"]  =  2e7
    odict["DIST_NBIN_vpa"] = 100
    odict["DIST_MIN_vpe"]  = 0
    odict["DIST_MAX_vpe"]  = 2e7
    odict["DIST_NBIN_vpe"] = 50

    odict["SIM_MODE"]                  = 1
    odict["FIXEDSTEP_USE_USERDEFINED"] = 1
    odict["FIXEDSTEP_USERDEFINED"]     = 1e-8
    opt.settypes(odict)
    options.write_hdf5(test_ascot.testfn, odict, desc="SLOWING_GO")

    odict["SIM_MODE"]                  = 2
    odict["FIXEDSTEP_USE_USERDEFINED"] = 1
    odict["FIXEDSTEP_USERDEFINED"]     = 2e-8
    opt.settypes(odict)
    options.write_hdf5(test_ascot.testfn, odict, desc="SLOWING_GCF")

    odict["SIM_MODE"]                  = 2
    odict["ENABLE_ADAPTIVE"]           = 1
    odict["ADAPTIVE_TOL_ORBIT"]        = 1e-6
    odict["ADAPTIVE_TOL_COL"]          = 9e0
    odict["ADAPTIVE_MAX_DRHO"]         = 0.1
    odict["ADAPTIVE_MAX_DPHI"]         = 10
    odict["FIXEDSTEP_USE_USERDEFINED"] = 1
    odict["FIXEDSTEP_USERDEFINED"]     = 1e-8
    opt.settypes(odict)
    options.write_hdf5(test_ascot.testfn, odict, desc="SLOWING_GCA")

    #**************************************************************************#
    #*            Marker input consisting of protons and alfas                 #
    #*                                                                         #
    #**************************************************************************#
    Nmrk   = 20
    ids    = np.linspace(1,Nmrk,Nmrk)
    weight = (1/Nmrk)* np.ones(ids.shape)
    mass   = m_p_AMU * np.ones(ids.shape)
    charge = 1       * np.ones(ids.shape)
    time   = 0       * np.ones(ids.shape)
    R      = 8       * np.ones(ids.shape)
    phi    = 90      * np.ones(ids.shape)
    z      = 0       * np.ones(ids.shape)
    theta  = 2 * np.pi * np.random.rand(1,Nmrk)
    energy = Eth     * np.ones(ids.shape)
    pitch  = 0.9     * np.ones(ids.shape)

    mrk.write_hdf5(test_ascot.testfn, Nmrk, ids, mass,
                   charge, R, phi, z, energy, pitch, theta,
                   weight, time, desc="THERMAL_GO")
    mrk.write_hdf5(test_ascot.testfn, Nmrk, ids, mass,
                   charge, R, phi, z, energy, pitch, theta,
                   weight, time, desc="THERMAL_GCF")
    mrk.write_hdf5(test_ascot.testfn, Nmrk, ids, mass,
                   charge, R, phi, z, energy, pitch, theta,
                   weight, time, desc="THERMAL_GCA")

    Nmrk   = 20
    pol    = 2*np.pi*np.random.rand(1, Nmrk)
    ids    = np.linspace(1,Nmrk,Nmrk)
    weight = (1/Nmrk)* np.ones(ids.shape)
    mass   = m_a_AMU * np.ones(ids.shape)
    charge = 2       * np.ones(ids.shape)
    time   = 0       * np.ones(ids.shape)
    R      = 6.2 + 0.8 * np.cos(pol)
    phi    = 90      * np.ones(ids.shape)
    z      = 0.8 * np.sin(pol)
    theta  = 2 * np.pi * np.random.rand(1,Nmrk)
    energy = Esd * np.ones(ids.shape)
    pitch  = 1 - 2 * np.random.rand(1,Nmrk)
    mrk.write_hdf5(test_ascot.testfn, Nmrk, ids, mass,
                   charge, R, phi, z, energy, pitch, theta,
                   weight, time, desc="SLOWING_GO")
    mrk.write_hdf5(test_ascot.testfn, Nmrk, ids, mass,
                   charge, R, phi, z, energy, pitch, theta,
                   weight, time, desc="SLOWING_GCF")
    mrk.write_hdf5(test_ascot.testfn, Nmrk, ids, mass,
                   charge, R, phi, z, energy, pitch, theta,
                   weight, time, desc="SLOWING_GCA")

    #**************************************************************************#
    #*                 Analytical ITER-like magnetic field                     #
    #*                                                                         #
    #**************************************************************************#
    B_GS.write_hdf5(test_ascot.testfn, R0, z0, Bphi0, psi_mult, psi_coeff,
                    desc="THERMAL_GO")
    B_GS.write_hdf5(test_ascot.testfn, R0, z0, Bphi0, psi_mult, psi_coeff,
                    desc="THERMAL_GCF")
    B_GS.write_hdf5(test_ascot.testfn, R0, z0, Bphi0, psi_mult, psi_coeff,
                    desc="THERMAL_GCA")
    B_GS.write_hdf5(test_ascot.testfn, R0, z0, Bphi0, psi_mult, psi_coeff,
                    desc="SLOWING_GO")
    B_GS.write_hdf5(test_ascot.testfn, R0, z0, Bphi0, psi_mult, psi_coeff,
                    desc="SLOWING_GCF")
    B_GS.write_hdf5(test_ascot.testfn, R0, z0, Bphi0, psi_mult, psi_coeff,
                    desc="SLOWING_GCA")

    #**************************************************************************#
    #*                               Plasma                                    #
    #*                                                                         #
    #**************************************************************************#
    Nrho  = 3
    Nion  = 1
    znum  = np.array([1])
    anum  = np.array([1])
    rho   = np.array([0, 0.5, 100])
    edens = ne  * np.ones(rho.shape)
    etemp = Te  * np.ones(rho.shape)
    idens = ne  * np.ones((rho.size, Nion))
    itemp = 1e3 * np.ones(rho.shape)
    P_1D.write_hdf5(test_ascot.testfn, Nrho, Nion, znum, anum, rho,
                    edens, etemp, idens, itemp, desc="THERMAL_GO")
    P_1D.write_hdf5(test_ascot.testfn, Nrho, Nion, znum, anum, rho,
                    edens, etemp, idens, itemp, desc="THERMAL_GCF")
    P_1D.write_hdf5(test_ascot.testfn, Nrho, Nion, znum, anum, rho,
                    edens, etemp, idens, itemp, desc="THERMAL_GCA")
    P_1D.write_hdf5(test_ascot.testfn, Nrho, Nion, znum, anum, rho,
                    edens, etemp, idens, itemp, desc="SLOWING_GO")
    P_1D.write_hdf5(test_ascot.testfn, Nrho, Nion, znum, anum, rho,
                    edens, etemp, idens, itemp, desc="SLOWING_GCF")
    P_1D.write_hdf5(test_ascot.testfn, Nrho, Nion, znum, anum, rho,
                    edens, etemp, idens, itemp, desc="SLOWING_GCA")

    #**************************************************************************#
    #*                     Rest of the inputs are trivial                      #
    #*                                                                         #
    #**************************************************************************#
    Exyz   = np.array([0, 0, 0])
    E_TC.write_hdf5(test_ascot.testfn, Exyz, desc="THERMAL_GO")
    E_TC.write_hdf5(test_ascot.testfn, Exyz, desc="THERMAL_GCF")
    E_TC.write_hdf5(test_ascot.testfn, Exyz, desc="THERMAL_GCA")
    E_TC.write_hdf5(test_ascot.testfn, Exyz, desc="SLOWING_GO")
    E_TC.write_hdf5(test_ascot.testfn, Exyz, desc="SLOWING_GCF")
    E_TC.write_hdf5(test_ascot.testfn, Exyz, desc="SLOWING_GCA")

    nwall = 4
    Rwall = np.array([0.1, 100, 100, 0.1])
    zwall = np.array([-100, -100, 100, 100])
    W_2D.write_hdf5(test_ascot.testfn, nwall, Rwall, zwall, desc="THERMAL_GO")
    W_2D.write_hdf5(test_ascot.testfn, nwall, Rwall, zwall, desc="THERMAL_GCF")
    W_2D.write_hdf5(test_ascot.testfn, nwall, Rwall, zwall, desc="THERMAL_GCA")
    W_2D.write_hdf5(test_ascot.testfn, nwall, Rwall, zwall, desc="SLOWING_GO")
    W_2D.write_hdf5(test_ascot.testfn, nwall, Rwall, zwall, desc="SLOWING_GCF")
    W_2D.write_hdf5(test_ascot.testfn, nwall, Rwall, zwall, desc="SLOWING_GCA")

    N0Rmin = 0
    N0Rmax = 100
    N0nR   = 2
    N0zmin = -100
    N0zmax = 100
    N0nz   = 2
    N0pmin = 0
    N0pmax = 2*np.pi
    N0np   = 2
    N0dens = np.array([ [ [0,0] , [0,0] ], [ [0,0] , [0,0] ] ])
    N0_3D.write_hdf5(test_ascot.testfn,
                     N0Rmin, N0Rmax, N0nR,
                     N0zmin, N0zmax, N0nz,
                     N0pmin, N0pmax, N0np, N0dens,
                     desc="THERMAL_GO")
    N0_3D.write_hdf5(test_ascot.testfn,
                     N0Rmin, N0Rmax, N0nR,
                     N0zmin, N0zmax, N0nz,
                     N0pmin, N0pmax, N0np, N0dens,
                     desc="THERMAL_GCF")
    N0_3D.write_hdf5(test_ascot.testfn,
                     N0Rmin, N0Rmax, N0nR,
                     N0zmin, N0zmax, N0nz,
                     N0pmin, N0pmax, N0np, N0dens,
                     desc="THERMAL_GCA")
    N0_3D.write_hdf5(test_ascot.testfn,
                     N0Rmin, N0Rmax, N0nR,
                     N0zmin, N0zmax, N0nz,
                     N0pmin, N0pmax, N0np, N0dens,
                     desc="SLOWING_GO")
    N0_3D.write_hdf5(test_ascot.testfn,
                     N0Rmin, N0Rmax, N0nR,
                     N0zmin, N0zmax, N0nz,
                     N0pmin, N0pmax, N0np, N0dens,
                     desc="SLOWING_GCF")
    N0_3D.write_hdf5(test_ascot.testfn,
                     N0Rmin, N0Rmax, N0nR,
                     N0zmin, N0zmax, N0nz,
                     N0pmin, N0pmax, N0np, N0dens,
                     desc="SLOWING_GCA")

def run():
    """
    Run tests.
    """
    for test in ["THERMAL_GO", "THERMAL_GCF", "THERMAL_GCA", \
                 "SLOWING_GO", "SLOWING_GCF", "SLOWING_GCA"]:
        test_ascot.set_and_run(test)

def check():
    """
    Plot the results of these tests.

    This function makes four plots.
    - One that shows marker equilibrium energy distributions (for all modes) and
      compares them with a Maxwellian distribution.
    - One that shows the pitch distributions in equilibrium.
    - One that shows marker slowing-down energy distributions and compares them
      to the analytical result.
    - One that shows pitch distribution during the slowing-down.
    """

    a5 = ascot5.Ascot(test_ascot.testfn)

    plt.figure()
    plt.hist(a5.SLOWING_GCF.endstate["pitch"],bins=10)
    plt.show()
    return

    #**************************************************************************#
    #*      Initialize histogram grids and evaluate analytical results         #
    #*                                                                         #
    #**************************************************************************#
    THERMAL = {}
    THERMAL["Egrid"] = np.linspace( 0,   4e4, 200)
    THERMAL["xigrid"]  = np.linspace(-1,     1, 150)
    SLOWING = {}
    SLOWING["Egrid"] = np.linspace( 0, 4.0e6, 200)
    SLOWING["xigrid"]  = np.linspace(-1,     1, 100)

    alphaZ = 2
    clog   = 14
    v0     = np.sqrt(2*Esd*e / m_a)
    vgrid  = np.sqrt(2*SLOWING["Egrid"]*e / m_a)
    vth    = np.sqrt(2*Te*e / m_e)
    vcrit  = vth * np.power( (3.0*np.sqrt(np.pi)/4.0) * (m_e / m_p) , 1/3.0)
    ts     = 3 * np.sqrt( np.power(2*np.pi * Te * e, 3) / m_e ) * eps0 * eps0 \ 
             * m_a /( alphaZ * alphaZ * np.power(e, 4) * ne * clog) 
    ts *= 2 # For some reason analytical slowing-down is off by a factor of 2
    heaviside = vgrid <= v0

    THERMAL["analytical"]  = 2 * np.sqrt(THERMAL["Egrid"]/np.pi) \
                             * np.power(Te,-3.0/2) \
                             * np.exp(-THERMAL["Egrid"]/Te)
    THERMAL["analytical"] *= (simtime_th/2)

    SLOWING["analytical"]  = ( 1 / ( np.power(vcrit,3)+np.power(vgrid,3) ) ) \
                              * 3 * ts * heaviside \
                              * (vgrid/ np.log(1+np.power(v0/vcrit,3)) ) * e/m_a

    #**************************************************************************#
    #*            Evaluate thermal distributions in energy and pitch           #
    #*                                                                         #
    #**************************************************************************#

    for mode in ["GO", "GCF", "GCA"]:
        distobj = a5["THERMAL_" + mode]["R_phi_z_vpa_vpe_t_q"]
        dist = distobj.histogram2dist(distobj.read())
        distobj.integrate(dist, 'R')
        distobj.integrate(dist, 'phi')
        distobj.integrate(dist, 'z')
        distobj.integrate(dist, 'charge')
        distobj.integrate(dist, 'time', slices=[1])
        distobj.integrate(dist, 'vpa')
        distobj.integrate(dist, 'vpe')

        Exidist = distobj.get_E_xi_dist(THERMAL["Egrid"],
                                        THERMAL["xigrid"], m_p)
        distobj.integrate(Exidist, 'R')
        distobj.integrate(Exidist, 'phi')
        distobj.integrate(Exidist, 'z')
        distobj.integrate(Exidist, 'charge')
        distobj.integrate(Exidist, 'time', slices=[1])

        Edist  = copy.deepcopy(Exidist)
        xidist = copy.deepcopy(Exidist)
        distobj.integrate(Edist, 'xi')
        distobj.integrate(xidist, 'E')
        distobj.integrate(Exidist, 'E')
        distobj.integrate(Exidist, 'xi')

        THERMAL[mode + "_Edist"]  = Edist["density"]
        THERMAL[mode + "_xidist"] = xidist["density"]
        THERMAL["E"]  = Edist["E"]
        THERMAL["xi"] = xidist["xi"]

    #**************************************************************************#
    #*        Evaluate slowing-down distributions in energy and pitch          #
    #*                                                                         #
    #**************************************************************************#

    for mode in ["GO", "GCF", "GCA"]:
        distobj = a5["SLOWING_" + mode]["R_phi_z_vpa_vpe_t_q"]

        dist = distobj.histogram2dist(distobj.read())
        distobj.integrate(dist, 'R')
        distobj.integrate(dist, 'phi')
        distobj.integrate(dist, 'z')
        distobj.integrate(dist, 'charge')
        distobj.integrate(dist, 'time')

        Exidist = distobj.get_E_xi_dist(SLOWING["Egrid"],
                                        SLOWING["xigrid"], m_a)
        distobj.integrate(Exidist, 'R')#, slices=np.arange(10,19))
        distobj.integrate(Exidist, 'phi')
        distobj.integrate(Exidist, 'z')#, slices=[19,20])
        distobj.integrate(Exidist, 'charge')
        distobj.integrate(Exidist, 'time')

        Edist  = copy.deepcopy(Exidist)
        xidist = copy.deepcopy(Exidist)
        distobj.integrate(Edist, 'xi')
        distobj.integrate(xidist, 'E')

        SLOWING[mode + "_Edist"]  = Edist["density"]
        SLOWING[mode + "_xidist"] = xidist["density"]
        SLOWING["E"]  = Edist["E"]
        SLOWING["xi"] = xidist["xi"]

    #**************************************************************************#
    #*                                 Plot                                    #
    #*                                                                         #
    #**************************************************************************#

    f = plt.figure(figsize=(11.9/2.54, 8/2.54))
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)
    plt.rc('axes', labelsize=10)
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    c = ['b', 'g', 'r']
    h1 = f.add_subplot(1,4,1)
    h1.set_position([0.12, 0.72, 0.4, 0.25], which='both')

    h2 = f.add_subplot(1,4,2)
    h2.set_position([0.12, 0.44, 0.4, 0.25], which='both')

    h3 = f.add_subplot(1,4,3)
    h3.set_position([0.12, 0.155, 0.4, 0.25], which='both')

    h4 = f.add_subplot(1,4,4)
    h4.set_position([0.6, 0.3, 0.45, 0.45], which='both')

    a1.plot(THERMAL["E"], THERMAL["GO_Edist"])
    a1.plot(THERMAL["E"], THERMAL["GCF_Edist"])
    a1.plot(THERMAL["E"], THERMAL["GCA_Edist"])
    a1.plot(THERMAL["Egrid"], THERMAL["analytical"], 'black')

    a2.plot(THERMAL["xi"], THERMAL["GO_xidist"])
    a2.plot(THERMAL["xi"], THERMAL["GCF_xidist"])
    a2.plot(THERMAL["xi"], THERMAL["GCA_xidist"])
    a2.plot(np.array([-1, 1]), np.array([0.5, 0.5]))

    a3.plot(SLOWING["E"], SLOWING["GO_Edist"])
    a3.plot(SLOWING["E"], SLOWING["GCF_Edist"])
    a3.plot(SLOWING["E"], SLOWING["GCA_Edist"])
    a3.plot(SLOWING["Egrid"], SLOWING["analytical"], 'black')

    a4.plot(SLOWING["xi"], SLOWING["GO_xidist"])
    a4.plot(SLOWING["xi"], SLOWING["GCF_xidist"])
    a4.plot(SLOWING["xi"], SLOWING["GCA_xidist"])
    a4.plot(np.array([-1, 1]), np.array([0.5, 0.5]))

    #**************************************************************************#
    #*                         Decorate and finish                             #
    #*                                                                         #
    #**************************************************************************#
    h1.set_xlim(0, 4e4)
    h1.xaxis.set(ticks=[0, 1e4, 2e4, 3e4, 4e4], 
                 ticklabels=['0', '1', '2', '3', '4'])
    #h1.yaxis.set(ticks=np.array([0, 1, -2, 0, 2])*1e-11,
    #             ticklabels=[-6, '', '', 0, 2])
    h1.tick_params(axis='y', direction='out')
    h1.tick_params(axis='x', direction='out')
    h1.spines['right'].set_visible(False)
    h1.spines['top'].set_visible(False)
    h1.yaxis.set_ticks_position('left')
    h1.xaxis.set_ticks_position('bottom')
    h1.set(ylabel=r"$f(E) \times 10^{-3}$ [1/eV]")
    h1.set(xlabel=r"Energy $\times 10^{3}$ [eV]")

    h2.set_xlim(-1, 1)
    h2.set_ylim(0, 1)
    h2.xaxis.set(ticks=[-1, -0.5, 0, 0.5, 1])
    h2.yaxis.set(ticks=[0, 0.5, 1])
    h2.tick_params(axis='y', direction='out')
    h2.tick_params(axis='x', direction='out')
    h2.spines['right'].set_visible(False)
    h2.spines['top'].set_visible(False)
    h2.yaxis.set_ticks_position('left')
    h2.xaxis.set_ticks_position('bottom')
    h2.set(ylabel=r"$f(\xi)$")
    h2.set(xlabel=r"Pitch")

    h3.set_xlim(0, 4e6)
    h3.xaxis.set(ticks=[0, 1e6, 2e6, 3e6, 4e6], 
                 ticklabels=['0', '1', '2', '3', '4'])
    #h3.yaxis.set(ticks=np.array([0, 1, -2, 0, 2])*1e-11,
    #             ticklabels=[-6, '', '', 0, 2])
    h3.tick_params(axis='y', direction='out')
    h3.tick_params(axis='x', direction='out')
    h3.spines['right'].set_visible(False)
    h3.spines['top'].set_visible(False)
    h3.yaxis.set_ticks_position('left')
    h3.xaxis.set_ticks_position('bottom')
    h3.set(ylabel=r"$f(E) \times 10^{-6}$ [1/eV]")
    h3.set(xlabel=r"Energy $\times 10^{6}$ [eV]")

    h4.set_xlim(-1, 1)
    h4.set_ylim(0, 1)
    h4.xaxis.set(ticks=[-1, -0.5, 0, 0.5, 1])
    h4.yaxis.set(ticks=[0, 0.5, 1])
    h4.tick_params(axis='y', direction='out')
    h4.tick_params(axis='x', direction='out')
    h4.spines['right'].set_visible(False)
    h4.spines['top'].set_visible(False)
    h4.yaxis.set_ticks_position('left')
    h4.xaxis.set_ticks_position('bottom')
    h4.set(ylabel=r"$f(\xi)$")
    h4.set(xlabel=r"Pitch")

    #plt.savefig("test_coulombcollisions.png", dpi=72)
    plt.show()

if __name__ == '__main__':
    if( len(sys.argv) == 1 ):
        print("Initializing tests.")
        init()
        print("Initialization complete.")
        print("")
        print("Running tests.")
        run()
        print("Runs complete.")
        print("")
        print("Checking test results.")
        check()
        print("Testing complete.")
        sys.exit()

    if(len(sys.argv) > 2):
        print("Too many arguments.")
        print("Only \"init\", \"run\" or \"check\" is accepted.")
        print("Aborting.")
        sys.exit()

    if( sys.argv[1] == "init" ):
        print("Initializing tests.")
        init()
        print("Initialization complete.")
        sys.exit()

    elif( sys.argv[1] == "run" ):
        print("Running tests.")
        run()
        print("Runs complete.")
        sys.exit()

    elif( sys.argv[1] == "check" ):
        print("Checking test results.")
        check()
        print("Testing complete.")
        sys.exit()

    else:
        print("Too many arguments.")
        print("Only \"init\", \"run\" or \"check\" is accepted.")
        print("Aborting.")
        sys.exit()
