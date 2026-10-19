"""orbitkicks: Calculates orbit "kicks" of fast ions in axisymmetric 
fusion devices. This is done by examining changes in the constants
of motion: energy and canonical toroidal momentum. It is assumed that
the magnetic moment is held constant (true for low freq. modes)
"""
import numpy as np
import unyt as u

from datetime import datetime
from scipy.interpolate import RectBivariateSpline
from scipy.constants import elementary_charge as QE
from scipy.constants import proton_mass as MP
from scipy.constants import atomic_mass as AMU

from a5py.ascot5io.options import Opt

import matplotlib.pyplot as plt

import h5py
from a5py.ascot5io.marker import GC as mgc
from a5py.ascot5io.marker import Prt as mprt
#from a5py.ascot5io.options import Opt as myopt

class Orbitkicks():
    """
    Class for computing orbit-kicks based on the ORBIT-kick
    code by M. Podesta (see PPCF 2014 and 2017).

    The main function is simkick which performs a series of
    ASCOT runs and calculates the "kick matrices" which
    are 5D phase-space matrices to compute P(DE,DP|E,P,mu).
    That is, the average change in energy and canonical
    toroidal momentum given the location in phase-space 
    (E,Pz,Mu). Theses calculations are performed with 
    nprt number of markers for nloop iterations.

    All other functions with this class are in support of 
    this main function. 

    The main output is a Ufile calle pDEDP.AEP. This is of
    the correct form to pass directly to TRANSP to calculate
    the enhanced anomalyous fast ion transport. Note that it
    is species dependent. 

    Inputs require a sampling time, tsamp, which should be a
    little longer than the mode period of interest. Optional 
    inputs include the simulation run time, number of markers,
    number of itrerations, guiding-center mode or full-orbit, 
    and matrix bounds and dimensions.

    Attributes
    ----------
    _ascot : :class:`.Ascot`
        Ascot objecct used to run Orbitkicks
    """

    def __init__(self,ascot):
        self._ascot = ascot

    def simkick(self,
                gcmode=True,nprt=10000,nloop=5,
                anum=2,znum=1,mrk=None,dtav=2e-6,
                e_min_keV=1.0,e_max_keV=150.0,e_bins=15,
                pz_min=-1.2,pz_max=1.0,pz_bins=40,
                mu_min=0.0,mu_max=1.4,mu_bins=16,
                de_min_keV=-1.0e-3,de_max_keV=1.0e-3,de_bins=29,
                dpz_min=-1.0e-8,dpz_max=1.0e-8,dpz_bins=29):
        """
        Parameters
        ----------
        dtsamp : float [s]
            Time over which to sample orbit kicks. Should be a little bigger than
            the mode period of interest.
        tsim : float [s]
            Simulation run time. Default is 0.2 ms
        gcmode : boolean
            Option to run in guiding-center. Default is True
        nprt : int
            Number of markers per iteration. Default is 10000
        nloop : int
            Number of iterations to perform, i.e. the total marker count will
            be nprt*nloop. Defualt is 5. 
        e_min : float [eV]
            Minimum energy to calculate kicks. Default 1000 eV
        e_max : float [eV]
            Maximum energy to calculate kicks. Default 150e3 eV
        e_bins : int
            Number of bins in energy array. Default is 15
        pz_min : float [unitless]
            Minimum tor momentum to calculate kicks. Default -1.2
        pz_max : float [unitless]
            Minimum tor momentum to calculate kicks. Default 1.0
        pz_bins : int
            Number of bins in momentum array. Default is 40
        mu_min : float [unitless]
            Minimum mag moment to calculate kicks. Default 0.0
        mu_max : float [unitless]
            Minimum mag moment to calculate kicks. Default 1.0
        mu_bins : int
            Number of bins in mag moment array. Default is 16
        de_min : float [eV]
            Minimum change in energy. Default 1e-6 eV
        de_max : float [eV]
            Maximum change in energy. Default 2e-6 eV
        de_bins : int
            Number of bins in kick energy array. Default is 29; must be odd!
        dpz_min : float [unitless]
            Minimum change in momentum. Default 1e-8
        dpz_max : float [unitless]
            Minimum change in  momentum. Default 2e-8
        dpz_bins : int
            Number of bins in kick momentum array. Default is 29; must be odd!
        """
        from a5py import Ascot
        import subprocess
    
        #check de_bins is odd for 0 center
        if de_bins % 2 == 0:
            raise ValueError("de_bins must be odd so zero is a bin center")

        #check dpz_bins is odd for 0 center
        if dpz_bins % 2 == 0:
            raise ValueError("dpz_bins must be odd so zero is a bin center")
        
        #make storage for kicks
        e_arr = np.linspace(e_min_keV,e_max_keV,e_bins)
        mu_arr = np.linspace(mu_min,mu_max,mu_bins)
        pz_arr = np.linspace(pz_min,pz_max,pz_bins)
        de_arr = np.linspace(de_min_keV,de_max_keV,de_bins)
        dpz_arr = np.linspace(dpz_min,dpz_max,dpz_bins)
        pdedp = np.zeros([e_bins,pz_bins,mu_bins,de_bins,dpz_bins])
        pdedp_optimize = True
        maxDE_kick = 0.0
        maxDPz_kick = 0.0

        #print kick storage memory size, float=8 bytes
        mem1 = 8*(e_bins+pz_bins+mu_bins+de_bins+dpz_bins)/1e9
        mem2 = 8*e_bins*pz_bins*mu_bins*de_bins*dpz_bins/1e9
        print('Rough memory usage for kick storage: '+str(mem1+mem2)+' GB'+'\n')

        #get mode frequency
        mhd = self._ascot.data.mhd.active.read()
        omega = np.asarray(mhd["omega"],dtype=float).ravel() #rad/s
        nmodes = np.unique(np.asarray(mhd["nmodes"],dtype=int).ravel()) #tor mode num
        
        #check only one mode is active
        if len(nmodes) > 1:
            raise ValueError("More than 1 tor mode number is active - choose a single mode")
        else:
            omega = omega[0]

        #set sampling time and sim time
        mode_period = 2.0*np.pi/omega #sec
        dtsamp = 1.05*mode_period #sec
        tsim = dtsamp*50
        dtav = 5e-7

        #check for too short smoothing time
        if dtav > 0.3*dtsamp:
            raise ValueError('Smoothing time too long compared to dtsamp')

        #print run information
        print('Simulation time: '+str(tsim)+'\n')
        print('Sampling time: '+str(dtsamp)+'\n')
        print('Mode period: '+str(mode_period)+'\n')
        print('Total markers: '+str(nprt*nloop)+'\n')
        if gcmode:
            print('Guiding-center simulation'+'\n')
        else:
            print('Full orbit simulation'+'\n')

        #set orbitwrite interval to have sufficient averaging
        ow_int = min(dtav/10.0,dtsamp/20.0)

        #get number of orbitwrite npoints from orbitwrite interval
        ow_npoint = int(np.ceil(1.2*tsim/ow_int))+10
        
        #make run options
        opt = Opt.get_default()
        self.set_simkick_opts(opt,tsim=tsim,ow_int=ow_int,
                              ow_npoint=ow_npoint,gcmode=gcmode)
        #self._ascot.data.create_input("opt",**opt)
        Opt.write_hdf5("test_kicks.h5",**opt)
        #self._ascot.simulation_initoptions(**opt)

        #print rough storage for marker data (8 output variables below)
        mem = 8*nprt*ow_npoint*8/1e9
        print('Rough memory usage for marker data: '+str(mem)+' GB'+'\n')

        #loop over sub-simulations and calculate kicks
        for iloop in range(0,nloop):
            #print beginning of loop
            print('Starting iteration '+str(iloop+1)+'\n')

            #only optimize first loop to avoid over interpolation of kicks
            if iloop > 0:
                pdedp_optimize = False

            #get needed bfield quantities for marker init and unit norms
            bstr = self.get_kick_bfield()
        
            #initialize markers
            if mrk is None:
                mrk = self.uni_mrk_orbit_deposit(bstr,emin_keV=e_min_keV,emax_keV=e_max_keV,
                                                 anum=anum,znum=znum,nprt=nprt)
            #self._ascot.data.create_input("gc",**mrk)
            #self._ascot.write("test_kicks.h5")
            dum = mgc.write_hdf5("test_kicks.h5",**mrk)
            #self._ascot.simulation_initmarkers(**mrk)

            #do simulation
            print('Sub-simulation start...'+'\n')
            self._ascot.simulation_initinputs()

            #self._ascot.write("test_kicks.h5")
            
            #vrun = self._ascot.simulation_run()
            subprocess.run(["/u/pbonofig/ascot_dev/ascot5/build/ascot5_main","test_kicks.h5"])
            print('Sub-simulation end'+'\n')
            a5_out = Ascot("test_kicks.h5")
            id_arr = a5_out.getstate("ids", state="ini") #particle IDs
            t_fin = a5_out.getstate("time", state="end") #end time [s]
            #r_orb, z_orb, ekin_orb, psi_orb = a5_out.data.active.getorbit("r","z","ekin","psi",ids=1)
            print(id_arr)
            #with h5py.File("test_kicks.h5",'r') as fout:
            #    id_arr = fout["results/endstate/ids"][:]
            #    print(id_arr)
            #mil = self._ascot.data.active.getorbit("ekin")
            #print(mil)
            #get marker ids and lost times
            id_arr = vrun.getstate("ids",state="ini") #particle IDs
            t_fin = vrun.getstate("time",state="end") #end time [s]

            #check kick ranges based on inputs only on first loop
            if pdedp_optimize == True:
                pz_arr,mu_arr = self.check_bdry(e_max_keV,anum,znum,
                                                mu_min,mu_max,mu_bins,
                                                pz_min,pz_max,pz_bins,bstr)

            #needed for magnetic quantities
            self._ascot.input_init(bfield=True) 

            #print start
            print('Computing (DE,DP) kicks...'+'\n')
            
            all_kicks = [] #kick storage 
            for j in range(0,len(id_arr)):
                #get orbit info vs. time
                torb,eorb,muorb,wgtorb,pitorb,psiorb,bphiorb,borb,rorb,zorb,pzorb = vrun.getorbit("time",
                                                                                                  "ekin",
                                                                                                  "mu",
                                                                                                  "weight",
                                                                                                  "pitch",
                                                                                                  "psi",
                                                                                                  "bphi",
                                                                                                  "bnorm",
                                                                                                  "r",
                                                                                                  "z",
                                                                                                  "pphi",
                                                                                            ids=id_arr[j])
                
                #limit calculations to before marker is terminated
                tind = np.where(torb < t_fin[j])[0]
                #tind = tind[1:]
                torb = torb[tind] #[s]
                eorb = eorb[tind] #[eV]
                muorb = muorb[tind] #[eV/T]
                wgtorb = wgtorb[tind] #[#/s]
                pitorb = pitorb[tind] #[vpar/v]
                psiorb = psiorb[tind] #[Wb]
                bphiorb = bphiorb[tind] #[T]
                borb = borb[tind] #[T]
                rorb = rorb[tind] #[m]
                zorb = zorb[tind] #[m]
                pzorb = pzorb[tind] #[amu*m/s]

                order = np.argsort(torb)
                torb = torb[order]
                eorb = eorb[order] #[eV]
                muorb = muorb[order] #[eV/T]
                wgtorb = wgtorb[order] #[#/s]
                pitorb = pitorb[order] #[vpar/v]
                psiorb = psiorb[order] #[Wb]
                bphiorb = bphiorb[order] #[T]
                borb = borb[order] #[T]
                rorb = rorb[order] #[m]
                zorb = zorb[order] #[m]
                pzorb = pzorb[order] #[amu*m/s]

                psi_spl  = bstr["fpsi"].ev(rorb, zorb)
                bphi_spl = bstr["fphi"].ev(rorb, zorb)
                btot_spl = bstr["ftot"].ev(rorb, zorb)

                #print("psi ASCOT-spline max diff:", np.nanmax(np.abs(np.asarray(psiorb) - psi_spl)))
                #print("bphi ASCOT-spline max diff:", np.nanmax(np.abs(np.asarray(bphiorb) - bphi_spl)))
                #print("B ASCOT-spline max diff:", np.nanmax(np.abs(np.asarray(borb) - btot_spl)))

                #conver to Roscoe units
                eorb_keV = eorb/1.0e3 #[keV]
                eorb_prime = self.energy_prime(eorb_keV,bstr,anum=anum,znum=znum)

                muorb_norm = self.calc_mu_norm(muorb,eorb_keV,bstr)
                #print('mypit',pitorb)
                #ds = np.asarray(np.sign(pitorb))
                #dmu = np.asarray(muorb_norm)
                #db = np.asarray(borb)
                #dbc = float(bstr["bcenter"])
                #newpit = ds*np.sqrt(1-dmu*db/dbc)
                #print('newpit',newpit)
                #print(pitorb-newpit)
                #pitorb = newpit
                #pzorb_norm = self.calc_pz_orbit(eprime=eorb_prime,pitch=pitorb,
                #                                bnorm=borb,bphi=bphiorb,
                #                                psi=psiorb,bstr=bstr,rlocal=rorb)
                pzorb_norm = self.calc_pz_orbit(
                    eprime=eorb_prime,
                    pitch=pitorb,
                    bnorm=btot_spl,
                    bphi=bphi_spl,
                    psi=psi_spl,
                    bstr=bstr,
                    rlocal=rorb)

                #cases = {
                #    "orbit psi, orbit B": (psiorb, borb, bphiorb),
                #    "spline psi, orbit B": (psi_spl, borb, bphiorb),
                #    "orbit psi, spline B": (psiorb, btot_spl, bphi_spl),
                #    "spline psi, spline B": (psi_spl, btot_spl, bphi_spl),
                #}

                #for name, (psi_use, B_use, Bphi_use) in cases.items():
                #    pz = self.calc_pz_orbit(
                #        eprime=eorb_prime,
                #        pitch=pitorb,
                #        bnorm=B_use,
                #        bphi=Bphi_use,
                #        psi=psi_use,
                #        bstr=bstr,
                #        rlocal=rorb,
                #    )

                #    good = np.isfinite(pz)
                #    x = pz[good]
                    
                #    if len(x) > 1:
                #        print(
                #            name,
                #            "p2p=", np.nanmax(x) - np.nanmin(x),
                #            "std=", np.nanstd(x),
                #            "first-last=", x[-1] - x[0],
                #            "finite=", good.sum(), "/", len(good),
                #        )
                #    else:
                #        print(name, "not enough finite points")
                
                #print(j,'a',eorb_keV)
                #print(j,'aa',eorb_prime)
                #print(j,'b',muorb_norm)
                print(j,'cc',pzorb)
                print(j,'c',pzorb_norm)
                #print(j,np.nanmax(pzorb_norm)-np.nanmin(pzorb_norm),np.std(pzorb_norm))
                print(j,'w',torb)
                #print(j,'e',pzdum)
                print(j,"Pz range:", np.nanmax(pzorb_norm)-np.nanmin(pzorb_norm))
                print(j,"Pz std:", np.std(pzorb_norm))
                print(j,"Pz start/end:", pzorb_norm[0], pzorb_norm[-1],
                      "difference:", pzorb_norm[-1] - pzorb_norm[0])
                print(j, "mu_norm:", np.nanmean(muorb_norm))
                
                #average full-orbit over gyroperiod
                #can't record GC and GO positions simultaneously
                if not gcmode:
                    gc = self.find_gc_equiv(torb,eorb_keV,pzorb_norm,muorb_norm,
                                            borb,anum,znum)
                    e_for_kick = gc["eorb_gc"]
                    pz_for_kick = gc["pzorb_gc"]
                    mu_for_kick = gc["muorb_gc"]
                else:
                    e_for_kick = eorb_keV
                    pz_for_kick = pzorb_norm
                    mu_for_kick = muorb_norm

                #calculate kicks for given orbit
                kick_str,maxDE_kick,maxDPz_kick = self.pdedp_calc_kicks(dtsamp,dtav,torb,
                                                                        e_for_kick,mu_for_kick,pz_for_kick,
                                                                        maxDE_kick,maxDPz_kick)
                all_kicks.append(kick_str)
                #print(j,'d',maxDE_kick,maxDPz_kick)

            #loop through markers has ended
            print('Finished computing (DE,DP) kicks'+'\n')
            
            #free postproc bfield for next iteration loop
            self._ascot.input_free()

            #check (DE,DPz) ranges only on first loop after kick calcs
            if pdedp_optimize == True:
                de_arr,dpz_arr = self.check_DEDP(maxDE_kick,maxDPz_kick,
                                                 de_max_keV,de_min_keV,de_bins,
                                                 dpz_max,dpz_min,dpz_bins)

            #record kicks to kick matrix
            self.pdedp_record_kicks(pdedp,e_arr,pz_arr,mu_arr,de_arr,dpz_arr,
                                    all_kicks)

            #free markers and diagnostics for next iteration
            self._ascot.simulation_free(markers=True,diagnostics=True)

            #print end of loop
            print('Completed iteration '+str(iloop+1)+'/'+str(nloop)+'\n')

        #iteration loops have ended and we have finished all simulations
        
        #normalize matrix
        self.pdedp_finalize(pdedp,de_arr,dpz_arr)

        #write ufile
        self.write_pdedp_ufile(dtsamp,e_arr,mu_arr,pz_arr,de_arr,dpz_arr,
                               pdedp,ami=anum,zmi=znum)

        #output structure if using routine interactively
        mystr = {'e_arr':e_arr,'mu_arr':mu_arr,'pz_arr':pz_arr,
                 'de_arr':de_arr,'dpz_arr':dpz_arr,'pdedp':pdedp,
                 'dtsamp':dtsamp}
        
        return mystr

    def set_simkick_opts(self,opt,tsim=0.0002,
                         ow_int=1e-8,ow_npoint=30000,
                         gcmode=True):
        """
        Parameters
        ----------
        opt : options group
            Default options object
        tsim : float [s]
            Simulation run time. Default is 0.2ms
        gcmode : boolean
            Option to run in guiding-center. Default is True
        """
        #settings dependent on GC or F0
        if gcmode: 
            sim_mode = 2 #Guiding-center
            rec_mode = 1
        else: #FO
            sim_mode = 1 #full-orbit
            rec_mode = 0
            
        #change options from the default
        opt.update({
            "SIM_MODE":sim_mode,
            "ENABLE_ADAPTIVE":0,
            "RECORD_MODE":rec_mode,
            "FIXEDSTEP_USE_USERDEFINED":0,
            "FIXEDSTEP_USERDEFINED":1e-9,
            "FIXEDSTEP_GYRODEFINED":10,
            "ENDCOND_SIMTIMELIM":1, #end at max mileage
            "ENDCOND_RHOLIM":1, #end at rho>=1
            "ENDCOND_LIM_SIMTIME":10.0,
            "ENDCOND_MAX_MILEAGE":tsim, #mileage is actual sim time
            "ENDCOND_MAX_CPUTIME":10.0,
            "ENDCOND_MAX_RHO":1.0,
            "ENABLE_ORBIT_FOLLOWING":1, #orbit following ON
            "ENABLE_MHD":1, #include MHD
            "ENABLE_ORBITWRITE":1, #store marker orbits
            "ORBITWRITE_MODE":1, #collect orbits on dt intervals
            "ORBITWRITE_NPOINT":ow_npoint, #max orbit collection points
            "ORBITWRITE_INTERVAL":ow_int, #time interval to collect marker info
        })
            
        return

    '''def get_kick_bfield(self):
        """
        Parameters
        ----------
        self : Class
            Ascot object used to run Orbitkicks
        """
        #unpack initialized Bfield
        self._ascot.input_init(bfield=True)
        bout = self._ascot.data.bfield.active.read()
        raxis = bout['axisr'][0]
        zaxis = bout['axisz'][0]
        br = bout['br']
        bz = bout['bz']
        bphi = bout['bphi']
        btot = np.sqrt(br**2+bz**2+bphi**2)
        rmin = bout['rmin'][0]
        rmax = bout['rmax'][0]
        nr = bout['nr'][0]
        zmin = bout['zmin'][0]
        zmax = bout['zmax'][0]
        nz = bout['nz'][0]
        psiwall = bout['psi1'][0]
        psi = bout['psi']
        psiaxis = bout["psi0"][0]
        self._ascot.input_free()
        
        #1D R and Z arrays
        rarr = np.linspace(rmin,rmax,nr) #[m]
        zarr = np.linspace(zmin,zmax,nz) #[m]

        #interpolate to finer grid
        ftot = RectBivariateSpline(rarr,zarr,btot) #[T]
        fphi = RectBivariateSpline(rarr,zarr,bphi) #[T]
        fpsi = RectBivariateSpline(rarr,zarr,psi) #[Wb]
        newr = np.linspace(rmin,rmax,nr*5) #[m]
        newz = np.linspace(zmin,zmax,nz*5) #[m]
        meshR, meshZ = np.meshgrid(newr,newz,indexing="ij")
        btot = ftot.ev(meshR,meshZ)
        bphi = fphi.ev(meshR,meshZ)
        psi = fpsi.ev(meshR,meshZ)

        #construct normalized psi grid
        psi_norm_grid = (psi-psiaxis)/(psiwall-psiaxis)
        mask_lcfs = (psi_norm_grid >= 0.0) & (psi_norm_grid <= 1.0)
        if not np.any(mask):
            raise ValueError("Psi mask is empty. Check psiaxis and psiwall sign convention.")

        #bcenter
        bcenter = ftot.ev(raxis,zaxis) #[T]

        #min and max Btot within LCFS
        mask = psi <= psiwall
        bsub = btot[mask]
        bmin = np.amin(bsub)
        bmax = np.amax(bsub)

        #find R(lcfs,theta=0) and psiouter
        indr = np.argmin(np.abs(newr-raxis))
        indz = np.argmin(np.abs(newz-zaxis))
        subpsi = psi[:,indz] #psi at Z=zaxis
        indpsi = np.argmin(np.abs(subpsi-psiwall)) #R index close to psiwall
        rlcfs = newr[indpsi]
        router = newr[indr:]
        psiouter = subpsi[indr:]

        #outer midplane
        r_mid = newr[indr:]
        z_mid = np.full_like(r_mid,zaxis)
        psi_mid = psi[indr:,indz]
        btot_mid = psi[indr:,indz]
        bphi_mid = psi[indr:,indz]

        #keep only confined part
        psi_norm_mid = (psi_mid - psiaxis)/(psiwall - psiaxis)
        mid_mask = (psi_norm_mid >= 0.0) & (psi_norm_mid <= 1.0)
        r_mid = r_mid[mid_mask]
        z_mid = z_mid[mid_mask]
        psi_mid = psi_mid[mid_mask]
        btot_mid = btot_mid[mid_mask]
        bphi_mid = bphi_mid[mid_mask]

        #sort by R
        order = np.argsort(r_mid)
        r_mid = r_mid[order]
        z_mid = z_mid[order]
        psi_mid = psi_mid[order]
        btot_mid= btot_mid[order]
        bphi_mid = bphi_mid[order]
        
        bstr = {'bcenter':bcenter,'bmin':bmin,'bmax':bmax,
                'raxis':raxis,'zaxis':zaxis,'rlcfs':rlcfs,
                'router':router,'psiouter':psiouter,'psiwall':psiwall,
                'psiaxis':psiaxis,'fnorm':abs(bcenter),
                'psi_grid':psi,'btot_grid':btot,
                'bphi_grid':bphi,'mask_lcfs':mask_lcfs,
                'r_mid':r_mid,'z_mid':z_mid,'psi_mid':psi_mid,
                'btot_mid':btot_mid,'bphi_mid':bphi_mid}
        
        return bstr'''

    def get_kick_bfield(self, refine=10):
        """
        Read ASCOT 2D axisymmetric magnetic-field data and build the
        normalization/interpolation structure needed by the kick model.

        This implementation supports simple 2D gridded axisymmetric fields
        with br, bz, bphi, psi on an R-Z grid.

        Returns
        -------
        bstr : dict
        Magnetic-field and normalization data.
        """

        def _scalar(x):
            arr = np.asarray(x)
            return float(arr.ravel()[0])

        required_keys = [
            "axisr", "axisz",
            "br", "bz", "bphi", "psi",
            "rmin", "rmax", "nr",
            "zmin", "zmax", "nz",
            "psi0", "psi1",
        ]

        self._ascot.input_init(bfield=True)

        try:
            bout = self._ascot.data.bfield.active.read()
        finally:
            self._ascot.input_free()

        for key in required_keys:
            if key not in bout:
                raise KeyError(
                    f"Active ASCOT bfield is missing key '{key}'. "
                    "This get_kick_bfield() only supports simple 2D "
                    "axisymmetric gridded fields."
                )

        # Scalars.
        raxis = _scalar(bout["axisr"])
        zaxis = _scalar(bout["axisz"])
        
        rmin = _scalar(bout["rmin"])
        rmax = _scalar(bout["rmax"])
        nr = int(_scalar(bout["nr"]))

        zmin = _scalar(bout["zmin"])
        zmax = _scalar(bout["zmax"])
        nz = int(_scalar(bout["nz"]))

        psiaxis = _scalar(bout["psi0"])
        psiwall = _scalar(bout["psi1"])

        if np.isclose(psiwall, psiaxis):
            raise ValueError("psi1 and psi0 are equal; cannot normalize flux.")

        # Arrays.
        br = np.asarray(bout["br"], dtype=float)
        bz = np.asarray(bout["bz"], dtype=float)
        bphi = np.asarray(bout["bphi"], dtype=float)
        psi = np.asarray(bout["psi"], dtype=float)

        expected = (nr, nz)
        transposed = (nz, nr)

        if br.shape == expected:
            pass
        elif br.shape == transposed:
            br = br.T
            bz = bz.T
            bphi = bphi.T
            psi = psi.T
        else:
            raise ValueError(
                f"Unexpected bfield array shape {br.shape}; expected "
                f"{expected} or {transposed}."
            )

        btot = np.sqrt(br**2 + bz**2 + bphi**2)

        # Coarse grid.
        rarr = np.linspace(rmin, rmax, nr)
        zarr = np.linspace(zmin, zmax, nz)

        kx = min(3, nr - 1)
        ky = min(3, nz - 1)

        ftot = RectBivariateSpline(rarr, zarr, btot, kx=kx, ky=ky)
        fphi = RectBivariateSpline(rarr, zarr, bphi, kx=kx, ky=ky)
        fpsi = RectBivariateSpline(rarr, zarr, psi, kx=kx, ky=ky)

        # Refined grid.
        nr_f = max(refine * nr, nr)
        nz_f = max(refine * nz, nz)

        newr = np.linspace(rmin, rmax, nr_f)
        newz = np.linspace(zmin, zmax, nz_f)

        meshR, meshZ = np.meshgrid(newr, newz, indexing="ij")

        btot_f = ftot.ev(meshR, meshZ)
        bphi_f = fphi.ev(meshR, meshZ)
        psi_f = fpsi.ev(meshR, meshZ)

        # Axis field.
        bcenter = float(ftot.ev(raxis, zaxis)[0])

        if not np.isfinite(bcenter) or abs(bcenter) <= 0.0:
            raise ValueError(f"Invalid axis magnetic field bcenter={bcenter}")

        fnorm = abs(bcenter)

        # Normalized poloidal flux.
        psi_norm = (psi_f - psiaxis) / (psiwall - psiaxis)

        mask_lcfs = (psi_norm >= 0.0) & (psi_norm <= 1.0)

        if not np.any(mask_lcfs):
            raise ValueError(
                "LCFS mask is empty. Check psi0/psi1 convention or bfield data."
            )

        bsub = btot_f[mask_lcfs]
        bmin = float(np.nanmin(bsub))
        bmax = float(np.nanmax(bsub))

        # Midplane line.
        iz0 = int(np.argmin(np.abs(newz - zaxis)))
        ir_axis = int(np.argmin(np.abs(newr - raxis)))

        r_line = newr
        psi_line = psi_f[:, iz0]
        psi_norm_line = (psi_line - psiaxis) / (psiwall - psiaxis)
        btot_line = btot_f[:, iz0]
        bphi_line = bphi_f[:, iz0]

        def _make_branch(mask, name):
            r_b = r_line[mask]
            psi_b = psi_line[mask]
            psi_norm_b = psi_norm_line[mask]
            btot_b = btot_line[mask]
            bphi_b = bphi_line[mask]

            finite = (
                np.isfinite(r_b)
                & np.isfinite(psi_b)
                & np.isfinite(psi_norm_b)
                & np.isfinite(btot_b)
                & np.isfinite(bphi_b)
            )

            r_b = r_b[finite]
            psi_b = psi_b[finite]
            psi_norm_b = psi_norm_b[finite]
            btot_b = btot_b[finite]
            bphi_b = bphi_b[finite]

            if len(r_b) < 2:
                raise ValueError(f"Not enough points on {name} midplane branch.")

            # Sort by normalized flux for easy inversion psi_norm -> R.
            order = np.argsort(psi_norm_b)

            r_b = r_b[order]
            psi_b = psi_b[order]
            psi_norm_b = psi_norm_b[order]
            btot_b = btot_b[order]
            bphi_b = bphi_b[order]

            # Remove duplicate psi_norm points, because np.interp wants
            # a clean monotonic x-array.
            psi_unique, unique_idx = np.unique(psi_norm_b, return_index=True)

            r_b = r_b[unique_idx]
            psi_b = psi_b[unique_idx]
            btot_b = btot_b[unique_idx]
            bphi_b = bphi_b[unique_idx]

            return {
                "r": r_b,
                "psi": psi_b,
                "psi_norm": psi_unique,
                "btot": btot_b,
                "bphi": bphi_b,
            }

        confined_line = (psi_norm_line >= 0.0) & (psi_norm_line <= 1.0)
        
        outer_mask = confined_line & (r_line >= raxis)
        inner_mask = confined_line & (r_line <= raxis)
        
        outer = _make_branch(outer_mask, "outer")
        inner = _make_branch(inner_mask, "inner")

        rlcfs_outer = float(np.nanmax(outer["r"]))
        rlcfs_inner = float(np.nanmin(inner["r"]))

        bstr = {
            # Basic normalization.
            "bcenter": bcenter,
            "bmin": bmin,
            "bmax": bmax,
            "raxis": raxis,
            "zaxis": zaxis,
            "psiaxis": psiaxis,
            "psiwall": psiwall,
            "fnorm": fnorm,
            
            # Refined grid.
            "r_grid": newr,
            "z_grid": newz,
            "psi_grid": psi_f,
            "btot_grid": btot_f,
            "bphi_grid": bphi_f,
            "psi_norm_grid": psi_norm,
            "mask_lcfs": mask_lcfs,

            # Spline evaluators.
            "ftot": ftot,
            "fphi": fphi,
            "fpsi": fpsi,

            # Branches for R(psi) on midplane.
            "outer": outer,
            "inner": inner,
            
            # Backward-compatible names used by calc_pz_orbit().
            "router": outer["r"],
            "bphiouter":outer["bphi"],
            "psiouter": outer["psi"],
            "rlcfs": rlcfs_outer,
            "rlcfs_inner": rlcfs_inner}

        return bstr

    def find_gc_equiv(self,torb,eorb,pzorb,muorb,borb,ami,zmi):
        mass = ami*MP #[kg]
        charge = zmi*QE #[C]

        t = np.asarray(torb,dtype=float)
        b = np.asarray(borb,dtype=float)
        
        gyroT = 2.0*np.pi*mass/(charge*b)

        e_gc = np.full_like(eorb,np.nan,dtype=float)
        pz_gc = np.full_like(pzorb,np.nan,dtype=float)
        mu_gc = np.full_like(muorb,np.nan,dtype=float)

        i = 0
        while i < len(t)-1:
            t_next = t[i] + gyroT[i]
            j = np.searchsorted(t,t_next)

            if j <= i:
                j = i+1
                
            e_gc[i:j] = np.mean(eorb[i:j])
            pz_gc[i:j] = np.mean(pzorb[i:j])
            mu_gc[i:j] = np.mean(muorb[i:j])
            i = j

        gc_str = {'eorb_gc':e_gc,'pzorb_gc':pz_gc,'muorb_gc':mu_gc}
        
        return gc_str

    def energy_prime(self,energy_keV,bstr,anum=2.0,znum=1.0):
        """
        Convert physical energy in keV to ORBIT-normalized E'.
        
        E' = kE * E_keV
        
        kE = 1e3 * A * mp/qe * g0^2 / (R0 * Zp * B0)^2
        """
        energy_keV = np.asarray(energy_keV, dtype=float)

        b0 = float(bstr["bcenter"])
        r0 = float(bstr["raxis"])

        # If fnorm = B0, then g0 = R0
        g0 = r0

        mp = 1.67262192369e-27
        qe = 1.602176634e-19

        kE = 1.0e3 * anum * (mp / qe) * g0**2 / (r0 * znum * b0)**2

        return kE * energy_keV

    def calc_mu_norm(self,mu_eVT,energy_keV,bstr):
         """
         Convert ASCOT magnetic moment [eV/T] to normalized ORBIT/AEP mu.

         mu_norm = mu * B0 / E

         E should be the same physical energy convention used for the kick matrix.
         Here energy_keV is converted back to eV for consistency with mu_eVT.
         """
         mu_eVT = np.asarray(mu_eVT, dtype=float)
         energy_eV = np.asarray(energy_keV, dtype=float)*1.0e3

         b0 = float(bstr["bcenter"])

         return mu_eVT*b0/energy_eV

    def calc_pz_orbit(self,eprime,pitch,bnorm,bphi,psi,bstr,rlocal):
        """
        Compute ORBIT-normalized canonical toroidal momentum Pz.

        Inputs
        ------
        eprime : array
        ORBIT-normalized kinetic energy E'_k.
        pitch : array
        v_parallel / v.
        bnorm : array
        |B| at marker location [T].
        bphi : array
        Toroidal magnetic field at marker location [T].
        psi : array
        ASCOT poloidal flux at marker location [Wb].
        bstr : dict
        Magnetic normalization data.
        """

        eprime = np.asarray(eprime, dtype=float)
        pitch = np.asarray(pitch, dtype=float)
        bnorm = np.asarray(bnorm, dtype=float)
        bphi = np.asarray(bphi, dtype=float)
        psi = np.asarray(psi, dtype=float)
        rlocal = np.asarray(rlocal, dtype=float)

        b0 = float(bstr["bcenter"])
        r0 = float(bstr["raxis"])
        fnorm = float(bstr["fnorm"])
        psi_axis = float(bstr["psiaxis"])
        psi_lcfs = float(bstr["psiwall"])
       
        # Use relative flux because ASCOT psi may not be zero on-axis.
        psi_rel = psi - psi_axis
        psi_lcfs_rel = psi_lcfs - psi_axis

        # ORBIT relation: Psi_lcfs = Psi_w * fnorm
        psi_w = psi_lcfs_rel / fnorm

        # R at same psi on outer midplane.
        router = np.asarray(bstr["router"], dtype=float)
        psiouter = np.asarray(bstr["psiouter"], dtype=float)
        bphiouter = np.asarray(bstr["bphiouter"], dtype=float)

        gouter = router * bphiouter / fnorm

        # Ensure interpolation arrays are monotonically ordered.
        order = np.argsort(psiouter)
        r_mid = np.interp(psi,psiouter[order],router[order],left=np.nan,right=np.nan)
        bphi_mid = np.interp(psi,psiouter[order],bphiouter[order],left=np.nan,right=np.nan)
        g_mid = np.interp(psi,psiouter[order],gouter[order],left=np.nan,right=np.nan)

        # g(Psi)
        g = bphi/fnorm*r_mid
        g1 = rlocal*bphi/fnorm
        g2 = r_mid*bphi/fnorm
        g3 = r_mid*bphi_mid/fnorm
        g3 = g_mid

        # rho_parallel
        rho_parallel = pitch * np.sqrt(2.0 * eprime) * fnorm / bnorm

        # normalized flux term
        psi_norm = psi_rel / psi_lcfs_rel

        pz = rho_parallel * g / psi_w - psi_norm

        #pz1 = -rho_parallel * g1 / psi_w - psi_norm
        #pz2 = -rho_parallel * g2 / psi_w - psi_norm
        #pz3 = -rho_parallel * g3 / psi_w - psi_norm
        A1 = -rho_parallel * g1 / psi_w
        A2 = -rho_parallel * g2 / psi_w
        A3 = -rho_parallel * g3 / psi_w
        B = psi_norm
        pz1 = A1 - B
        pz2 = A2 - B
        pz3 = A3 - B
        print('term A',A3)
        print('term B',B)
        #print("A range/std:", np.nanmax(A1)-np.nanmin(A1), np.nanstd(A1))
        #print("B range/std:", np.nanmax(B)-np.nanmin(B), np.nanstd(B))
        #print("corr(A,B):", np.corrcoef(A1, B)[0,1])
        #print("slope dB/dA:", np.polyfit(A1, B, 1)[0])
        #print("A range/std:", np.nanmax(A2)-np.nanmin(A2), np.nanstd(A2))
        #print("B range/std:", np.nanmax(B)-np.nanmin(B), np.nanstd(B))
        #print("corr(A,B):", np.corrcoef(A2, B)[0,1])
        #print("slope dB/dA:", np.polyfit(A2, B, 1)[0])
        #print("A range/std:", np.nanmax(A3)-np.nanmin(A3), np.nanstd(A3))
        #print("B range/std:", np.nanmax(B)-np.nanmin(B), np.nanstd(B))
        #print("corr(A,B):", np.corrcoef(A3, B)[0,1])
        #print("slope dB/dA:", np.polyfit(A3, B, 1)[0])

        #print('whoa')
        #print(psi_norm)
        #print("psi_axis      =", bstr["psiaxis"])
        #print("psi_lcfs      =", bstr["psiwall"])
        #print("psi_lcfs_rel  =", bstr["psiwall"] - bstr["psiaxis"])
        #print("psi_norm min/max =", np.nanmin(psi_norm), np.nanmax(psi_norm))

        #print('whoa2')
        #print(g1)
        #print(g2)
        #print(g3)
        #print(pz1)
        #print(pz2)
        #print(pz3)
        #print(np.mean(pz1),np.std(pz1))
        #print(np.mean(pz2),np.std(pz2))
        #print(np.mean(pz3),np.std(pz3))

        #good = (
        #    np.isfinite(pz3)
        #    & np.isfinite(A3)
        #    & np.isfinite(B)
        #    & np.isfinite(g_mid)
        #    & np.isfinite(rho_parallel)
        #)
        #dpz3 = pz3[good]
        #dA3 = A3[good]
        #dB = B[good]
        #dg_mid = g_mid[good]
        #drho_parallel = rho_parallel[good]
        
        '''print("Pz range:", np.nanmax(dpz3)-np.nanmin(dpz3))
        print("psi_norm min/max:", np.nanmin(psi_norm), np.nanmax(psi_norm))
        print("psi raw min/max:", np.nanmin(psi), np.nanmax(psi))
        print("psiouter min/max:", np.nanmin(psiouter), np.nanmax(psiouter))
        print("r_mid finite fraction:", np.mean(np.isfinite(r_mid)))
        print( "bphi_mid finite fraction:", np.mean(np.isfinite(bphi_mid)))
        print("g_mid min/max:", np.nanmin(dg_mid), np.nanmax(dg_mid))
        corr = np.corrcoef(dA3, dB)[0, 1]
        slope = np.polyfit(dA3, dB, 1)[0]
        print("corr(A,B):", corr)
        print("slope dB/dA:", slope)
        print("pitch min/max:", np.nanmin(pitch), np.nanmax(pitch))
        print("bnorm min/max:", np.nanmin(bnorm), np.nanmax(bnorm))
        print("rho_parallel min/max:", np.nanmin(drho_parallel), np.nanmax(drho_parallel))
        print("Eprime:", np.nanmean(eprime))'''

        return pz3


    def check_bdry(self,e_max_keV,anum,znum,
                   mu_min,mu_max,mu_bins,
                   pz_min,pz_max,pz_bins,bstr):
        """
        mu_min : float [unitless]
            Minimum mag moment to calculate kicks. 
        mu_max : float [unitless]
            Minimum mag moment to calculate kicks. 
        mu_bins : int
            Number of bins in mag moment array.
        pz_min : float [unitless]
            Minimum tor momentum to calculate kicks.
        pz_max : float [unitless]
            Minimum tor momentum to calculate kicks.
        pz_bins : int
            Number of bins in momentum array.
        bstr : struct
            Output structure from get_kick_bfield()

        Check the range used for computing p(DE,DP|E,Pz,mu)
        and adjust the (Pz,mu) range on-the-fly to optimize
        the sampling
        """
        #print start
        print('Checking Pz,mu ranges...'+'\n')

        #threshold limit to alter ranges
        dthresh = 0.1

        #find boundary for mu and Pz based on energy range in pDEDP calc
        
        #max energy in Roscoe units
        dum_emax_prime = self.energy_prime(e_max_keV,bstr,anum=anum,znum=znum)

        #min and max b-field; rescale so Baxis=1.0 like ORBIT
        bcenter = abs(float(bstr['bcenter']))
        bmin = bstr['bmin']/bcenter
        bmax = bstr['bmax']/bcenter
        psiwall = bstr['psiwall']
        raxis = bstr['raxis']
        rlcfs = bstr['rlcfs']

        #redefine mu range with buffer
        dum_mumax = 1.0/bmin*(mu_bins+1.0)/mu_bins
        dum_mumax *= 1.05
        dum_mumin = 0.0

        #redefine pz range with buffer
        dum_pzmax = raxis/psiwall*np.sqrt(2.0*dum_emax_prime)*(pz_bins+1.0)/pz_bins
        dum_pzmax *= 1.05
        dum_pzmin = -1.0 - rlcfs/psiwall*np.sqrt(2.0*dum_emax_prime)/bmax*(pz_bins+1.0)/pz_bins
        dum_pzmin *= 1.05

        #check whether range needs to be adjusted to above
        fracMu = (mu_max-dum_mumax)/mu_max
        fracPz = np.amax([abs((pz_max-dum_pzmax)/pz_max),abs((pz_min-dum_pzmin)/pz_min)])

        if (fracMu > dthresh or fracPz > dthresh or
            dum_pzmin < pz_min or dum_pzmax > pz_max or
            dum_mumax > mu_max):
            print('New Pz,mu ranges computed:')
            print('original Pz range: '+str(pz_min)+', '+str(pz_max))
            print('updated Pz range: '+str(dum_pzmin)+', '+str(dum_pzmax))
            print('original mu range: '+str(mu_min)+', '+str(mu_max))
            print('updated mu range: '+str(dum_mumin)+', '+str(dum_mumax))
            print('')

            pz_min = dum_pzmin
            pz_max = dum_pzmax
            mu_max = dum_mumax
            mu_min = dum_mumin
        else:
            print('Pz,mu ranges look OK'+'\n')

        #form grid and return
        pz_arr = np.linspace(pz_min,pz_max,pz_bins)
        mu_arr = np.linspace(mu_min,mu_max,mu_bins)
        
        return pz_arr,mu_arr

    def check_DEDP(self,maxDE_kick,maxDPz_kick,
                   de_max_keV,de_min_keV,de_bins,
                   dpz_max,dpz_min,dpz_bins,
                   margin=1.1,edge_frac=0.9):
        """
        Parameters
        ----------
        maxDE_kick : float [eV]
            Max running DE kick calculated
        maxDPz_kick : float
            Max running DPz kick calculated
        de_min : float [eV]
            Minimum change in energy. 
        de_max : float [eV]
            Maximum change in energy.
        de_bins : int
            Number of bins in kick energy array. 
        dpz_min : float [unitless]
            Minimum change in momentum.
        dpz_max : float [unitless]
            Minimum change in  momentum.
        dpz_bins : int
            Number of bins in kick momentum array.

        Check the range used for computing p(DE,DP|E,Pz,mu)
        and adjust the (DE,DPz) range on-the-fly to optimize
        the sampling
        """
        #print start
        print('Checking (DE,DPz) boundaries...'+'\n')

        #Current half-widths
        de_half_old = max(abs(de_min_keV),abs(de_max_keV))
        dpz_half_old = max(abs(dpz_min),abs(dpz_max))

        #Resize if observed kicks are near or beyond the edge
        update_de = abs(maxDE_kick) >= edge_frac*de_half_old
        update_dpz = abs(maxDPz_kick) >= edge_frac*dpz_half_old

        #update energy array if needed
        if update_de:
            de_half_new = margin*abs(maxDE_kick)
            de_half_new = 1.0e-6*np.ceil(de_half_new/1.0e-6)
        else:
            de_half_new = de_half_old

        #update Pz array if needed
        if update_dpz:
            dpz_half_new = margin*abs(maxDPz_kick)
            dpz_half_new = 1.0e-8*np.ceil(dpz_half_new/1.0e-8)
        else:
            dpz_half_new = dpz_half_old

        #mirror them
        de_min_keV = -de_half_new
        de_max_keV =  de_half_new
        dpz_min = -dpz_half_new
        dpz_max =  dpz_half_new
        
        #print new grid info
        if update_de or update_dpz:
            print('New (DE,DPz) grid computed')
            print('original (DE,Dpz): ('+str(de_half_old)+', '+str(dpz_half_old)+')')
            print('updated (DE,Dpz): ('+str(de_half_new)+', '+str(dpz_half_new)+')')
            print('')

        #make grid
        de_arr = np.linspace(de_min_keV,de_max_keV,de_bins)
        dpz_arr = np.linspace(dpz_min,dpz_max,dpz_bins)

        #Force exact zero bins
        inde0 = np.argmin(np.abs(de_arr))
        indp0 = np.argmin(np.abs(dpz_arr))
        de_arr[inde0] = 0.0
        dpz_arr[indp0] = 0.0
        
        return de_arr,dpz_arr

    def read_pdedp_ufile(self,myfile='pDEDP.AEP'):
        """
        Parameters
        ----------
        myfile : string 
            Ufile containing orbit kicks. Default is pDEDP.AEP
        """
        #print start
        print('Reading kick matrices from '+myfile+'\n')
        
        with open(myfile) as fp:
            lines = fp.readlines()

            #read particle type
            spec = int(lines[3].split()[0])

            #read sampling time
            tsamp = float(lines[9].split()[0])/1000.0 #[s]

            #array sizes
            nde = int(lines[12].split()[0])
            ndpz = int(lines[13].split()[0])
            ne = int(lines[14].split()[0])
            npz = int(lines[15].split()[0])
            nmu = int(lines[16].split()[0])

            #storage
            e_arr = np.zeros(ne)
            pz_arr = np.zeros(npz)
            mu_arr = np.zeros(nmu)
            de_arr = np.zeros(nde)
            dpz_arr = np.zeros(ndpz)
            pdedp = np.zeros([ne,npz,nmu,nde,ndpz])

            #read 1D arrays
            lstart = 17 #skip header stuff
            dl = len(lines[17].split())
            de,lstart = self.read_arr1d(lines,lstart,dl,de_arr)
            dp,lstart = self.read_arr1d(lines,lstart,dl,dpz_arr)
            e,lstart = self.read_arr1d(lines,lstart,dl,e_arr)
            p,lstart = self.read_arr1d(lines,lstart,dl,pz_arr)
            m,lstart = self.read_arr1d(lines,lstart,dl,mu_arr)

            #read 5D matrix till footer
            for i in range(lstart,len(lines)-6):
                myline = lines[i].split()
                ind_de = int(myline[0])-1 #python indexing at 0
                ind_dpz = int(myline[1])-1
                ind_e = int(myline[2])-1
                ind_pz = int(myline[3])-1
                ind_mu = int(myline[4])-1
                val = float(myline[5])
                pdedp[ind_e,ind_pz,ind_mu,ind_de,ind_dpz] = val
                
        fp.close()

        #print end
        print('Finished reading '+myfile+'\n')
        
        struct = {'e_arr':e_arr,'pz_arr':pz_arr,'mu_arr':mu_arr,
                  'de_arr':de_arr,'dpz_arr':dpz_arr,'pdedp':pdedp,
                  'tsamp':tsamp,'spec':spec}
        
        return struct

    def read_arr1d(self,lines,lstart,dl,arr):
        """
        Parameters
        ----------
        lines : Array of strings of floats
            Lines to be read
        lstart : int
            Index at wwhich to start reading lines
        dl : int
            dl is maximum length of line to be read
        arr : float array
            Array of floats that was read

        Reads float valuess from lines to arr starting at line lstart
        until arr is completely filled
        """
        ix = 0
        lend = int(np.ceil(len(arr)/dl)) + lstart 
        for i in range(lstart,lend):
            if ix+dl > len(arr):
                arr[ix:] = [float(x) for x in lines[i].split()]
            else: #end case
                arr[ix:ix+dl] = [float(x) for x in lines[i].split()]
            ix = ix + dl
        return arr, lend

    def write_pdedp_ufile(self,dtsamp,e_arr,mu_arr,pz_arr,
                          de_arr,dpz_arr,pdedp,
                          ami=2,zmi=1,myfile='pDEDP.AEP'):
        """
        Parameters
        ----------
        dtsamp : float [s]
            Sampling time step in seconds
        e_arr : float array
            Array of energy values in keV units
        mu_arr : float array
            Array of mu values in Roscoe units
        pz_arr : float array
            Array of pz values in Roscoe units
        de_arr : float array
            Array of DE kicks in keV units
        dpz_arr : float array
            Array of DP kicks in Roscoe units
        pdedp : float array
            5D phase-space matrix
        ami : int
            Atomic mass number of ion. Default is 2 for deuterium
        zmi : int
            Atmoic charge number of ion. Default is 1 for deuterium
        myfile : string 
            Ufile name containing orbit kicks to write to. Default is pDEDP.AEP
        """
        #print start
        print('Writing kick output to '+myfile+'\n')
        
        #header information
        lshot=123456
        nd=5 #5-D data
        nq=0 #unknown parameter...
        nr=6 #number of decimal places f13.6
        npr=1 #representation, 0:full, 1:sparse 
        ns=1 #number  of scalars

        dev='DEV'
        labelx='DEstep                '	#c*20
        unitsx='   keV    '		#c*10
        labely='DPsteps               '	#c*20
        unitsy='          '		#c*10
        labelu='Evar                  '	#c*20
        unitsu='   keV    '		#c*10
        labelv='Pvar                  '	#c*20
        unitsv='          '		#c*10
        labelw='Muvar                '	#c*20
        unitsw='          '		#c*10

        #footer information
        com=';----END-OF-DATA-----------------COMMENTS:-----------'
        com2='UFILE WRITTEN BY ASCOT, see WRITE_PDEDP_UFILE'
        com3='SMOOTHING FACTORS, DELAY FACTORS:'
        com4='       NONE'
        com5='USER COMMENTS:'
        com6='       ASCOT FILE'

        #make date
        today = datetime.today()
        month = today.strftime("%b")
        day = today.strftime("%d")
        year = today.strftime("%y")
        date = year+'-'+month+'-'+day

        #create file
        f = open(myfile,"w")

        #line 1
        f.write(' '+str(lshot)+dev+str(nd)+'  '+str(nq)+' '+str(nr)+'          ')
        f.write(";-SHOT #- F(X) DATA -PDEDP_OUT- "+date+'\n')

        #line 2
        f.write('  '+date+'             ')
        f.write(';-SHOT DATE-  UFILES ASCII FILE SYSTEM'+'\n')

        #line 3
        f.write('   '+str(ns)+'                    ')
        f.write(';-NUMBER OF ASSOCIATED SCALAR QUANTITIES-'+'\n')

        #line 4
        #fast ion species
        if ami==1 and zmi==1:
            spec = 5 
            specstr = ';proton'
        elif ami==2 and zmi==1:
            spec = 1 
            specstr = 'deuterium'
        elif ami==3 and zmi==1:
            spec = 2 
            specstr = ';tritium'
        elif ami==3 and zmi==2:
            spec = 3 
            specstr = ';HE3 FUSN'
        elif ami==4 and zmi==2:
            spec = 4 
            specstr = ';HE4 FUSN'
        else:
            spec = 0 
            specstr = ';all fast ions'
        
        f.write(' '+str(spec)+'                        ')
        f.write(specstr+'\n')
        
        #line 5
        f.write(' '+labelx+unitsx+';-INDEPENDENT VARIABLE LABEL: X-'+'\n')

        #line 6
        f.write(' '+labely+unitsy+';-INDEPENDENT VARIABLE LABEL: Y-'+'\n')

        #line 7
        f.write(' '+labelu+unitsu+';-INDEPENDENT VARIABLE LABEL: U-'+'\n')

        #line 8
        f.write(' '+labelv+unitsv+';-INDEPENDENT VARIABLE LABEL: V-'+'\n')

        #line 9
        f.write(' '+labelw+unitsw+';-INDEPENDENT VARIABLE LABEL: W-'+'\n')

        #line 10
        dtsamp *= 1000.0 #[ms]
        f.write('  '+f'{dtsamp:13.6e}'+'          ')
        f.write('; TSTEPSIM  - TIME STEP USED IN SIMULATION [ms]'+'\n')

        #line 11
        f.write(' PROBABILITY DATA              ;-DEPENDENT VARIABLE LABEL-'+'\n')

        #line 12
        f.write(' 1                    ')
        f.write(';-REPRESENTATION - 0:FULL 1:SPARSE'+'\n')

        #line 13
        f.write('         '+str(len(de_arr))+'          ')
        f.write(';-# OF X PTS-'+'\n')

        #line 14
        f.write('         '+str(len(dpz_arr))+'          ')
        f.write(';-# OF Y PTS-'+'\n')

        #line 15
        f.write('         '+str(len(e_arr))+'          ')
        f.write(';-# OF U PTS-'+'\n')

        #line 16
        f.write('         '+str(len(pz_arr))+'          ')
        f.write(';-# OF V PTS-'+'\n')

        #line 17
        f.write('         '+str(len(mu_arr))+'          ')
        f.write(';-# OF W PTS- X,Y,U,V,W,F(X,Y,U,V,W) DATA FOLLOW:'+'\n')

        #make sure center bin for (DE,DP)=(0,0)
        inde = np.argmin(np.abs(de_arr))
        indp = np.argmin(np.abs(dpz_arr))
        de_arr[inde] = 0.0
        dpz_arr[indp] = 0.0

        #write 1D data
        for i in range(0,len(de_arr),6):
            f.write(''.join(f'{x:14.6e}' for x in de_arr[i:i+6]) +'\n')
        for i in range(0,len(dpz_arr),6):
            f.write(''.join(f'{x:14.6e}' for x in dpz_arr[i:i+6]) +'\n')
        for i in range(0,len(e_arr),6):
            f.write(''.join(f'{x:14.6e}' for x in e_arr[i:i+6]) +'\n')
        for i in range(0,len(pz_arr),6):
            f.write(''.join(f'{x:14.6e}' for x in pz_arr[i:i+6]) +'\n')
        for i in range(0,len(mu_arr),6):
            f.write(''.join(f'{x:14.6e}' for x in mu_arr[i:i+6]) +'\n')

        #write 5D matrix, only write non-zero elements, i.e. sparse matrix
        for i in range(0,len(e_arr)):
            for j in range(0,len(pz_arr)):
                for k in range(0,len(mu_arr)):
                    for m in range(0,len(de_arr)):
                        for n in range(0,len(dpz_arr)):
                            val = pdedp[i,j,k,m,n]
                            if val != 0.0:
                                #fortran indexing starts at 1!!
                                f.write(f"{m+1:6d}{n+1:6d}{i+1:6d}{j+1:6d}{k+1:6d}{val:14.6e}\n")

        #print footer information
        f.write(com+'\n')
        f.write(com2+'\n')
        f.write(com3+'\n')
        f.write(com4+'\n')
        f.write(com5+'\n')
        f.write(com6+'\n')
        
        f.close()

        #print end
        print('Finished writing kick output to '+myfile+'\n')
        
        return

    def write_amode_ufile(self,shot,tAmode,Amode,file_id,dev='D3D'):
        """
        Parameters
        ----------
        shot : integer
            Shot number for TRANSP run
        tAmode : float array
            1D array for corresponding time values in TRANSP run
        Amode : float array
            1D array with corresponding mode amplitude values. These
            values are relative to unity, e.g. Amode=1 corresponds to
            whatever the physical amplitude (db/B) was calculated in
            orbitkicks
        file_id : integer
            File ID suffix
        dev : string
            3 or 4 character device ID used by TRANSP
        """
        #print start
        prefx = 'FILEAMODE_'  # output file name, prefix
        suffx = '.AEP'  # output file name, suffix
        filename = prefx + str(file_id) + suffx
        print('Writing mode amplitude to '+filename+'\n')

        lshot = int(shot)
        ntm = len(tAmode)

        # --------------------------------------------------------

        # Get current date
        #make date
        today = datetime.today()
        month = today.strftime("%b")
        day = today.strftime("%d")
        year = today.strftime("%y")
        date = year+'-'+month+'-'+day
        
        #current_time = time.localtime()
        #month = time.strftime('%b', current_time)
        #day = time.strftime('%d', current_time)
        #year = time.strftime('%y', current_time)
        #date = f'{day}-{month}-{year}'

        nd = 1  # 1-D data
        nq = 0  # unknown
        nr = 6  # number of decimal places f13.6
        npr = 0  # process code
        ns = 1  # number of scalars
        labelx = 'time'
        unitsx = '   ms'

        com = ';----END-OF-DATA-----------------COMMENTS:-----------'
        com2 = "UFILE WRITTEN BY KORBPY'S write_amode_aep()"
        com3 = 'SMOOTHING FACTORS, DELAY FACTORS:'
        com4 = '       NONE'
        com5 = 'USER COMMENTS:'
        com6 = '        TEST FILE'

        # --------------------------------------------------------
        # Write Amode(t) data in UFILE format
        with open(filename, 'w') as f:
            # Write the data
            tag = f';-SHOT #- F(X) DATA -UF5DWR- {date}'
            f.write(f' {lshot:6d} {dev:<4}{nd:2d}{nq:2d}{nr:2d}          {tag}\n')

            tag = ';-SHOT DATE-  UFILES ASCII FILE SYSTEM'
            f.write('  '+date+'             ')
            #f.write(f' {date:>10}          {tag}\n')

            tag = ';-NUMBER OF ASSOCIATED SCALAR QUANTITIES-'
            f.write(f' {ns:3d}                    {tag}\n')

            f.write(' 0                           ;-SCALAR, LABEL FOLLOWS: \n')

            tag = ';-INDEPENDENT VARIABLE LABEL: X-'
            f.write(f' {labelx:<20}{unitsx:<10}{tag}\n')

            f.write(' MODE AMPL. DATA              ;-DEPENDENT VARIABLE LABEL-\n')

            tag = ';-PROC CODE- 0:RAW 1:AVG 2:SM 3:AVG+SM'
            f.write(f' {npr:1d}                    {tag}\n')

            tag = ';-# OF X PTS- X,F(X) DATA FOLLOW:'
            f.write(f' {ntm:10d}          {tag}\n')

            # Write time variable (tAmode) in 6e13.6 format
            for i in range(0, len(tAmode), 6):
                f.write(''.join(f'{x:13.6e}' for x in tAmode[i:i+6]) + '\n')

            # Write mode amplitude data (Amode>0.) in 6e13.6 format
            for i in range(0, len(Amode), 6):
                f.write(''.join(f'{x:13.6e}' for x in Amode[i:i+6]) + '\n')

            # Write comments
            f.write(f'{com}\n')
            f.write(f'{com2}\n')
            f.write(f'{com3}\n')
            f.write(f'{com4}\n')
            f.write(f'{com5}\n')
            f.write(f'{com6}\n')

            #print end
            print('Finished writing mode amplitude to '+filename+'\n')
            
            return
    
    def pdedp_finalize(self,pdedp,de_arr,dpz_arr):
        """
        Parameters
        ----------
        pdedp : float array
            5D phase-space matrix
        de_arr : float array
            Array of DE kicks in Roscoe units
        dpz_arr : float array
            Array of DP kicks in Roscoe units
        """
        #print start
        print('Finalizing pDEDP computation'+'\n')

        #make sure (DE,DP)=(0,0)
        inde0 = np.argmin(np.abs(de_arr))
        indp0 = np.argmin(np.abs(dpz_arr))
        de_arr[inde0] = 0.0
        dpz_arr[indp0] = 0.0

        #get dimensions
        n_e,n_pz,n_mu,_,_ = pdedp.shape

        #get average counts/bin from non-empty bins
        #fill in empty bins with unity
        n_filled = 0
        for i in range(0,n_e):
            for j in range(0,n_pz):
                for k in range(0,n_mu):
                    s = np.sum(pdedp[i,j,k,:,:])

                    if s > 0.0:
                        pdedp[i,j,k,:,:] /= s
                        n_filled += 1
                    else:
                        pdedp[i,j,k,inde0,indp0] = 1.0

        #print end
        print('pDEDP matrices normalized')
        print(f"Filled phase-space cells: {n_filled}/{n_e*n_pz*n_mu}"+'\n')
        
        return

    def empty_kick_dict(self):
        mystr = {"eavgs":np.asarray([]),
                 "pzavgs":np.asarray([]),
                 "muavgs":np.asarray([]),
                 "dekicks":np.asarray([]),
                 "dpzkicks":np.asarray([]),
                 "wgts":np.asarray([])}
        
        return mystr

    def pdedp_calc_kicks(self,dtsamp,dtav,torb,
                         eorb_keV,muorb_norm,pzorb_norm,
                         maxDE_kick,maxDPz_kick):
        """
        Calculate kick samples.

        eorb_keV      : physical energy in keV
        muorb_norm    : normalized ORBIT/AEP mu
        pzorb_norm  : normalized ORBIT/AEP Pz
        """
        t = np.asarray(torb, dtype=float)
        e = np.asarray(eorb_keV, dtype=float)
        mu = np.asarray(muorb_norm, dtype=float)
        pz = np.asarray(pzorb_norm, dtype=float)

        #filter out any possible NaNs
        valid = (np.isfinite(t) &
                 np.isfinite(e) &
                 np.isfinite(mu) &
                 np.isfinite(pz))
        t = t[valid]
        e = e[valid]
        mu = mu[valid]
        pz = pz[valid]

        #ensure time ordering
        order = np.argsort(t)
        t = t[order]
        e = e[order]
        mu = mu[order]
        pz = pz[order]

        #print('hey')
        #print(t)
        #print(e)
        #print(mu)
        #print(pz)
        #print('alarm bells')
        #for dummy in pz:
        #    print(str(dummy)+'\n')

        if len(t) < 3:
            return self.empty_kick_dict(), maxDE_kick, maxDPz_kick

        tstart = t[0] + 0.2*dtsamp

        eavgs = []
        pzavgs = []
        muavgs = []
        dekicks = []
        dpzkicks = []
        wgts = []

        starts = np.arange(tstart,t[-1]-dtsamp-dtav,dtsamp)
       
        for ts in starts:
            i0 = np.searchsorted(t, ts)
            i1 = np.searchsorted(t, ts + dtav)
            j0 = np.searchsorted(t, ts + dtsamp)
            j1 = np.searchsorted(t, ts + dtsamp + dtav)

            if i1 <= i0 or j1 <= j0 or j1 > len(t):
                continue

            e0 = np.mean(e[i0:i1])
            p0 = np.mean(pz[i0:i1])
            mu0 = np.mean(mu[i0:i1])

            e1 = np.mean(e[j0:j1])
            p1 = np.mean(pz[j0:j1])
            mu1 = np.mean(mu[j0:j1])

            de = e1 - e0          # keV
            dp = p1 - p0          # normalized
            dmu = mu1 - mu0
            print('test kick',dp,p1,p0)
            #potential diagnostic for mu conservation
            #if abs(dmu) > mutol:
                #something

            if abs(dp) > 0.3:
                print("\nLarge DPz kick detected")
                print(f"  dpz    = {dp:.6e}")
                print(f"  pz0    = {p0:.6e}")
                print(f"  pz1    = {p1:.6e}")
                print(f"  e0     = {e0:.6e} keV")
                print(f"  e1     = {e1:.6e} keV")
                print(f"  de     = {de:.6e} keV")
                print(f"  mu0    = {mu0:.6e}")
                print(f"  ts     = {ts:.6e}")
                print(f"  dtsamp = {dtsamp:.6e}")
                print(f"  dtav   = {dtav:.6e}")
                print(f"  nwin0  = {i1 - i0}")
                print(f"  nwin1  = {j1 - j0}")

            maxDE_kick = max(maxDE_kick,1.05*abs(de))
            maxDPz_kick = max(maxDPz_kick,1.05*abs(dp))

            eavgs.append(e0)
            pzavgs.append(p0)
            muavgs.append(mu0)
            dekicks.append(de)
            dpzkicks.append(dp)
            wgts.append(1.0) #use unit weight unless doing importance sampling

        kick_calc_str = {'eavgs':eavgs,'pzavgs':pzavgs,'muavgs':muavgs,
                         'dekicks':dekicks,'dpzkicks':dpzkicks,'wgts':wgts}
        
        return kick_calc_str,maxDE_kick,maxDPz_kick

    def pdedp_record_kicks(self,pdedp,e_arr,pz_arr,mu_arr,de_arr,dpz_arr,
                           all_kicks):
        """
        Parameters
        ----------
        pdedp : float array
            5D kick matrices
        e_arr : float array [eV]
            Energy array values
        pz_arr : float array
            Canonical momentum array values
        de_arr : float array [eV]
            Kick energy array values
        dpz_arr : float array
            Kick momentum array values
        all_kicks : list of structures
            List of structures from pdedp_calc_kicks() with calculated kick values
        """
        #print start
        print('Recording pDEDP to 5D matrix...'+'\n')

        #go through all kicks for each marker
        for i in range(0,len(all_kicks)):
            #unpack kick structure
            kick_calc_str = all_kicks[i]
            eavgs = kick_calc_str['eavgs']
            pzavgs = kick_calc_str['pzavgs']
            muavgs = kick_calc_str['muavgs']
            dekicks = kick_calc_str['dekicks']
            dpzkicks = kick_calc_str['dpzkicks']
            wgts = kick_calc_str['wgts']

            #histogram each individual kick calculation
            for j in range(0,len(eavgs)):
                #get (E,Pz,mu) bin indexes
                inde = np.argmin(np.abs(e_arr-eavgs[j]))
                indpz = np.argmin(np.abs(pz_arr-pzavgs[j]))
                indmu = np.argmin(np.abs(mu_arr-muavgs[j]))

                #get (DE,DP) bin indexes
                indde = np.argmin(np.abs(de_arr-dekicks[j]))
                inddpz = np.argmin(np.abs(dpz_arr-dpzkicks[j]))

                #update pdedp by weight [#]
                pdedp[inde,indpz,indmu,indde,inddpz] += wgts[j]

        #print end
        print('Finished recording pDEDP to 5D matrix'+'\n')
        
        return

    def uni_mrk_orbit_deposit(
            self,
            bstr,
            emin_keV=1.0,
            emax_keV=150.0,
            limit_psi=1.0,
            anum=2,
            znum=1,
            nprt=10000,
            seed=None,
    ):
        """
        Initialize markers using an ORBIT-kick-like full-deposition scheme.
        
        This mimics the FORTRAN logic:
        
        half particles:
            theta = 0, outer midplane, positive pitch

        half particles:
            theta = pi, inner midplane, negative pitch

        pol = 0.001 + 0.999*min(1,limit_psi)*random
        energy = uniform [emin_keV, emax_keV]
        zeta = random [0, 2pi]
        weight = 1

        This does not sample uniformly in constants-of-motion space directly.
        It samples ORBIT-like initial marker coordinates and later you bin the
        resulting trajectories in (E, Pphi, mu).
        """

        rng = np.random.default_rng(seed)
        
        if nprt < 2:
            raise ValueError("nprt must be at least 2.")

        n_outer = nprt // 2
        n_inner = nprt - n_outer

        zaxis = float(bstr["zaxis"])

        outer = bstr["outer"]
        inner = bstr["inner"]

        psi_outer_norm = np.asarray(outer["psi_norm"], dtype=float)
        r_outer = np.asarray(outer["r"], dtype=float)

        psi_inner_norm = np.asarray(inner["psi_norm"], dtype=float)
        r_inner = np.asarray(inner["r"], dtype=float)

        lim = min(1.0, float(limit_psi))

        # Match the FORTRAN expression:
        # pol = .001 + .999*min(1,limit_Psi)*ranx()
        # If limit_psi=1, this samples [0.001, 1.0].
        psi_norm_outer = 0.001 + 0.999 * lim * rng.random(n_outer)
        psi_norm_inner = 0.001 + 0.999 * lim * rng.random(n_inner)

        # Clip to branch availability.
        psi_norm_outer = np.clip(
            psi_norm_outer,
            np.nanmin(psi_outer_norm),
            np.nanmax(psi_outer_norm),
        )
        psi_norm_inner = np.clip(
            psi_norm_inner,
            np.nanmin(psi_inner_norm),
            np.nanmax(psi_inner_norm),
        )

        # Invert normalized flux -> R on each midplane branch.
        r1 = np.interp(psi_norm_outer, psi_outer_norm, r_outer)
        r2 = np.interp(psi_norm_inner, psi_inner_norm, r_inner)

        r = np.concatenate([r1, r2])
        z = np.full(nprt, zaxis)

        # Pitch.
        # Outer/co-moving: positive random pitch.
        # Inner/counter-moving: negative random pitch.
        pitch_outer = rng.random(n_outer)
        pitch_inner = -rng.random(n_inner)
        pitch = np.concatenate([pitch_outer, pitch_inner])

        # Energy.
        energy_keV = emin_keV + rng.random(nprt) * (emax_keV - emin_keV)
        energy_eV = energy_keV * 1.0e3

        # Angles.
        phi = rng.random(nprt) * 360.0
        zeta = rng.random(nprt) * 2.0 * np.pi

        # Species.
        anum_arr = np.full(nprt, anum, dtype=int)
        znum_arr = np.full(nprt, znum, dtype=int)

        # ASCOT marker mass is in amu, charge in elementary charges.
        mass_amu = anum*MP/AMU
        
        mass = np.full(nprt, mass_amu)
        charge = np.full(nprt, znum)

        ids = np.arange(1, nprt + 1, dtype=int)
        weight = np.ones(nprt)
        time = np.zeros(nprt)

        marker = {
            "n": nprt,
            "ids": ids,
            "r": r * u.m,
            "phi": phi * u.degree,
            "z": z * u.m,
            "pitch": pitch,
            "weight": weight,
            "time": time * u.s,
            "energy": energy_eV * u.eV,
            "zeta": zeta * u.rad,
            "mass": mass * u.amu,
            "charge": charge * u.e,
            "anum": anum_arr,
            "znum": znum_arr}

        return marker

    def calc_resfrac(self,myfile='pDEDP.AEP'):
        data = self.read_pdedp_ufile(myfile=myfile)
        
        ne = len(data['e_arr'])
        npp = len(data['pz_arr'])
        nm = len(data['mu_arr'])
        resfrac = np.zeros([ne,npp,nm])
        dprms = np.zeros([ne,npp,nm])
        derms = np.zeros([ne,npp,nm])
        asm = np.zeros([ne,npp,nm])

        de = data['de_arr']
        dp = data['dpz_arr']
        pdedp = data['pdedp']
        ide0 = np.argmin(np.abs(de))
        idp0 = np.argmin(np.abs(dp))
        CPdedp = np.sum(pdedp,axis=-2)
        C2Pdedp = np.sum(pdedp,axis=-1)
        for i in range(0,ne):
            for j in range(0,npp):
                for k in range(0,nm):
                    ttl = np.sum(CPdedp[i,j,k,:])
                    ttl2 = np.sum(C2Pdedp[i,j,k,:])
                    if ttl == 0:
                        derms[i,j,k] = 0.0
                        dprms[i,j,k] = 0.0
                        asm[i,j,k] = 0.0
                    else:
                        derms[i,j,k] = np.sum(np.abs(de*CPdedp[i,j,k,:]))/ttl
                        dprms[i,j,k] = np.sum(np.abs(dp*C2Pdedp[i,j,k,:]))/ttl2
                        dedumpos = np.array([x if x>0.0 else 0.0 for x in de])
                        dedumneg = np.array([x if x<0.0 else 0.0 for x in de])
                        depos = np.sum(dedumpos*CPdedp[i,j,k,:])
                        deneg = np.sum(dedumneg*CPdedp[i,j,k,:])
                        if (np.abs(depos)>0) and (np.abs(deneg)>0):
                            asm[i,j,k] = 100*(depos+deneg)/(depos-deneg)
                        else:
                            asm[i,j,k] = 0.0
                    if ttl > 0:
                        resfrac[i,j,k] = 1.0-pdedp[i,j,k,ide0,idp0]/np.sum(pdedp[i,j,k,:,:])
                            
        data = {'resfrac':resfrac,'derms':derms,'dprms':dprms,'asm':asm}
        return data

    def plot_quantity(self,
                      de=False,dp=False,
                      resfrac=False,asym=False,
                      myfile='pDEDP.AEP',
                      nlevels=30,
                      cmap="viridis",
                      ncols=4,
                      figsize_per_panel=(3.0,2.0)):
        """
        Plot P(dE,dPz) at fixed mu and Pz for multiple energy bins
        in one subplot grid.
        """
        if de==False and dp==False and resfrac==False and asym==False:
            print('Warning: Please choose de, dp, resfrac, or asym to plot!')
            return
        
        #get data
        data = self.read_pdedp_ufile(myfile=myfile)
        e_arr = np.asarray(data["e_arr"],dtype=float)
        pz_arr = np.asarray(data["pz_arr"],dtype=float)
        mu_arr = np.asarray(data["mu_arr"],dtype=float)
        de_arr = np.asarray(data["de_arr"],dtype=float)
        dpz_arr = np.asarray(data["dpz_arr"],dtype=float)
        pdedp = np.asarray(data["pdedp"],dtype=float)
        dt = np.asarray(data["tsamp"],dtype=float)
        tstep_str = str(round(data['tsamp']))

        #set new PZ and MU arrays for plot ranges
        ddp = (pz_arr[1]-pz_arr[0])/2.0
        ddm = (mu_arr[1]-mu_arr[0])/2.0
        xr = [np.amin(pz_arr)-2*ddp,np.amax(dpz_arr)+2*ddp]
        yr = [np.amin(mu_arr)-2*ddm,np.amax(mu_arr)+2*ddm]

        #make plot
        nplots = len(e_arr)
        nrows = int(np.ceil(nplots/ncols))
        fig,axarr = plt.subplots(nrows,ncols,
                                 figsize=(figsize_per_panel[0]*ncols,
                                          figsize_per_panel[1]*nrows),
                                 squeeze=False,sharex=True,sharey=True)

        #get desired signal to plot
        resdata = self.calc_resfrac(myfile=myfile)
        rfrac = resdata['resfrac']
        asm = resdata['asm']
        derms = resdata['derms']
        dprms = resdata['dprms']
        if de == True:
            val = derms
            fig.suptitle('$\mathrm{DE_{RMS}}$',fontsize=12)
            zmax = np.amax(val)
        elif dp == True:
            val = dprms
            fig.suptitle('$\mathrm{DPz_{RMS}}$',fontsize=12)
            zmax = np.amax(val)
        elif resfrac == True:
            val = rfrac
            fig.suptitle('$\mathrm{Resfrac}$',fontsize=12)
            zmax = np.amax(val)
        else:
            val = asm
            fig.suptitle('$\mathrm{Asymmetry}$',fontsize=12)
            zmax = np.abs(np.amax(val))
            zmin = np.abs(np.amin(val))

        #make plot labels
        for i in range(0,nrows):
            for j in range(0,ncols):
                #make y-labels
                if j == 0:
                    axarr[i,j].set_ylabel('$\mathrm{\\mu B_0/E}$',fontsize=12)
                #make x-labels
                if i == nrows-1:
                    axarr[i,j].set_xlabel('$\mathrm{P_{\\phi}}$',fontsize=12)

                #labelsizes
                axarr[i,j].tick_params(axis='x',labelsize=10)
                axarr[i,j].tick_params(axis='y',labelsize=10)

        for i in range(0,len(e_arr)):
            row = i//ncols
            col = i%ncols
            axarr[row,col].set_title('$\mathrm{E=%s}$'%str(round(e_arr[i],2)),fontsize=10)

            vplot = np.transpose(val[i,:,:])
            if asym == True:
                levels = np.linspace(-1*zmax,zmax,nlevels)
            else:
                levels = np.linspace(1e-3,zmax,nlevels)

            cs = axarr[row,col].contourf(pz_arr,mu_arr,vplot,
                                         levels=levels,cmap='jet')
            axarr[row,col].set_xlim(xr)
            axarr[row,col].set_ylim(yr)

        plt.tight_layout()
        plt.show()

        return

    def plot_kicks(self,
            mu_value,
            pz_value,
            myfile='pDEDP.AEP',
            nlevels=50,
            cmap="viridis",
            ncols=4,
            figsize_per_panel=(3.0,2.0)):
        """
        Plot P(dE,dPz) at fixed mu and Pz for multiple energy bins
        in one subplot grid.
        """
        #get data
        data = self.read_pdedp_ufile(myfile=myfile)
        e_arr = np.asarray(data["e_arr"],dtype=float)
        pz_arr = np.asarray(data["pz_arr"],dtype=float)
        mu_arr = np.asarray(data["mu_arr"],dtype=float)
        de_arr = np.asarray(data["de_arr"],dtype=float)
        dpz_arr = np.asarray(data["dpz_arr"],dtype=float)
        pdedp = np.asarray(data["pdedp"],dtype=float)
        dt = np.asarray(data["tsamp"],dtype=float)
        tstep_str = str(round(data['tsamp']))

        #find (Pz,mu) to plot at
        ipz = int(np.argmin(np.abs(pz_arr - pz_value)))
        imu = int(np.argmin(np.abs(mu_arr - mu_value)))

        pz_used = pz_arr[ipz]
        mu_used = mu_arr[imu]

        print(f"Requested Pz = {pz_value:.6e}, using nearest grid Pz = {pz_used:.6e}, index {ipz}")
        print(f"Requested mu = {mu_value:.6e}, using nearest grid mu = {mu_used:.6e}, index {imu}")

        #set new DE and DPZ arrays for plot ranges
        dde = (de_arr[1]-de_arr[0])/2.0
        ddp = (dpz_arr[1]-dpz_arr[0])/2.0
        xr = [np.amin(de_arr)-2*dde,np.amax(de_arr)+2*dde]
        yr = [np.amin(dpz_arr)-2*ddp,np.amax(dpz_arr)+2*ddp]

        #make plot
        nplots = len(e_arr)
        nrows = int(np.ceil(nplots/ncols))
        fig,axarr = plt.subplots(nrows,ncols,
                                 figsize=(figsize_per_panel[0]*ncols,
                                          figsize_per_panel[1]*nrows),
                                 squeeze=False,sharex=True,sharey=True)

        #plot title
        fig.suptitle('$\mathrm{Kick: \, P_{\\phi}=%s, \, \\mu=%s}$' %
                     (round(pz_used,2),round(mu_used,2)),fontsize=12)

        #make plot labels
        for i in range(0,nrows):
            for j in range(0,ncols):
                #make y-labels
                if j == 0:
                    axarr[i,j].set_ylabel('$\mathrm{\\Delta P_{\\phi} (1/%s \\mu s)}$'%tstep_str,
                                          fontsize=12)
                #make x-labels
                if i == nrows-1:
                    axarr[i,j].set_xlabel('$\mathrm{\\Delta E \, (keV/%s \\mu s)}$'%tstep_str,
                                          fontsize=12)

                #labelsizes
                axarr[i,j].tick_params(axis='x',labelsize=10)
                axarr[i,j].tick_params(axis='y',labelsize=10)

        for i in range(0,len(e_arr)):
            row = i//ncols
            col = i%ncols
            axarr[row,col].set_title('$\mathrm{E=%s}$'%str(round(e_arr[i],2)),fontsize=10)

            khold = np.transpose(pdedp[i,ipz,imu,:,:])
            zmax = np.amax(pdedp[i,ipz,imu,:,:])
            fkick = RectBivariateSpline(de_arr,dpz_arr,khold,kx=1,ky=1)
            kplot = fkick(de_arr,dpz_arr)
            levels = np.linspace(1e-3,zmax,nlevels)
            cs = axarr[row,col].contourf(de_arr,dpz_arr,kplot,levels=levels,cmap='jet')
            axarr[row,col].set_xlim(xr)
            axarr[row,col].set_ylim(yr)

        plt.tight_layout()
        plt.show()

        return
    
    '''def uni_mrk(self,emin=1000.0,emax=1.0e5,pmin=-1.0,pmax=1.0,
                rhomax=0.99,anum=2,znum=1,nprt=10000):
        #print start
        print('Initializing uniform markers...'+'\n')
        
        #constants
        pmass = 1.6726e-27 #kg
        q = 1.602e-19 #Coulomb
        amu = 1.6605e-27 #kg

        #marker ids
        ids = np.array(range(1,nprt+1))

        #default is deuterium
        anum = np.ones(nprt,int)*anum
        znum = np.ones(nprt,int)*znum

        #mass has units amu and charge in units e
        mass = np.ones(nprt)*anum*(pmass/amu)
        charge = np.ones(nprt)*znum

        #give equal weights
        weight = np.ones(nprt)

        #start time is t=0 for all
        time = np.zeros(nprt)
    
        #storage
        r = np.zeros(nprt)
        z = np.zeros(nprt)
        phi = np.zeros(nprt)
        zeta = np.zeros(nprt)
        vr = np.zeros(nprt)
        vz = np.zeros(nprt)
        vphi = np.zeros(nprt)
        pitch = np.zeros(nprt)
        energy = np.zeros(nprt)

        #uniform sampling
        for i in range(nprt):
            #toroidal angle (deg)
            phi[i] = np.random.uniform(0.0,360.0)
        
            #gyroangle (rad)
            zeta[i] = np.random.uniform(0.0,2.0*np.pi)

            #pitch (vpar/vtot)
            pitch[i] = np.random.uniform(pmin,pmax)
        
            #energy (eV); default is 1-100 keV
            energy[i] = np.random.uniform(emin,emax)

            #get random velocity vector
            vtot = np.sqrt(2.0*energy[i]*q/(mass[i]*amu)) #m/s
            vphi[i] = np.random.uniform(-1.0,1.0)
            if (vphi[i] == 1) or (vphi[i] == -1):
                vr[i] = 0
                vz[i] = 0
            else:
                v2 = np.sqrt(1.0-vphi[i]**2)
                vr[i] = np.random.uniform(-1.0*v2,v2)
                if (vr[i] == v2) or (vr[i] == -1*v2):
                    vz[i] = 0
                else:
                    vz[i] = np.sqrt(1.0-(vphi[i]**2 + vr[i]**2))

            #unnormalize by vtot
            vr[i] *= vtot #m/s
            vz[i] *= vtot #m/s
            vphi[i] *= vtot #m/s

            #get random R and Z
            rmin = 1.0
            rmax = 2.0
            zmin = -1
            zmax = 1
            r[i] = np.random.uniform(rmin,rmax) #m
            z[i] = np.random.uniform(zmin,zmax) #m
            
            #print status
            if (i+1)%1000==0:
                print('Sampled '+str(i+1)+' particles')
            
        mystr = {
            'n':nprt,
            'ids':ids,
            'r':r*u.m, #m
            'phi':phi*u.degree, #deg
            'z':z*u.m, #m
            'pitch':pitch, #vpar/vtot
            'weight':weight, #markers/s
            'time':time*u.s, #s
            'energy':energy*u.eV, #eV
            'zeta':zeta*u.rad, #rad
            'mass':mass*u.amu, #amu
            'charge':charge*u.e, #e
            'anum':anum,
            'znum':znum}
            #'vr':vr, #m/s
            #'vphi':vphi, #m/s
            #'vz':vz} #m/s

        #print end
        print('Marker initialization completed'+'\n')
    
        return mystr'''
