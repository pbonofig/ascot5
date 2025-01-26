"""orbitkicks: Calculates orbit "kicks" of fast ions in axisymmetric 
fusion devices. This is done by examining changes in the constants
of motion: energy and canonical toroidal momentum. It is assumed that
the magnetic moment is held constant (true for low freq. modes)
"""
import numpy as np
import unyt
from a5py.ascot5io.options import Opt
import subprocess

#from a5py.ascotpy.libascot import _LIBASCOT, STRUCT_DIST5D, STRUCT_AFSIDATA, \
#    STRUCT_AFSITHERMAL, PTR_REAL, AFSI_REACTIONS
#from a5py.exceptions import AscotNoDataException

class Orbitkicks():
    """
    Some explination will go here

    Attributes
    ----------
    _ascot : :class:`.Ascot`
        Ascot objecct used to run Orbitkicks
    """

    def __init__(self,ascot):
        self._ascot = ascot

    def simkick(self,dtsamp,nprt=10000,nloop=5,update=False,
                e_min=1000.0,e_max=150.0e3,e_bins=15,
                pz_min=-1.2,pz_max=1.0,pz_bins=40,
                mu_min=0.0,mu_max=1.4,mu_bins=16,
                de_min=1.0e-6,de_max=2.0e-6,de_bins=29,
                dpz_min=1.0e-8,dpz_max=2.0e-8,dpz_bins=29):
        """
        Parameters
        ----------
        dtsamp : float [s]
            Time over which to sample orbit kicks. Should be a little bigger than
            the mode period of interest.
        nprt : int
            Number of markers per iteration. Default is 10000
        nloop : int
            Number of iterations to perform, i.e. the total marker count will
            be nprt*nloop. Defualt is 5. 
        update : boolean
            If true, will read ufile in CWD and add statistics. Default is false.
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
            Number of bins in kick energy array. Default is 29
        dpz_min : float [unitless]
            Minimum change in momentum. Default 1e-8
        dpz_max : float [unitless]
            Minimum change in  momentum. Default 2e-8
        dpz_bins : int
            Number of bins in kick momentum array. Default is 29
        """
        #make storage for kicks
        if update == False:
            #new run
            e_arr = np.linspace(e_min,e_max,e_bins)
            mu_arr = np.linspace(mu_min,mu_max,mu_bins)
            pz_arr = np.linspace(pz_min,pz_max,pz_bins)
            de_arr = np.linspace(de_min,de_max,de_bins)
            dp_arr = np.linspace(dp_min,dp_max,dp_bins)
            pdedp = np.zeros(e_bins,pz_bins,mu_bins,de_bins,dpz_bins)
            pdedp_optimize = True
            pde_maxDE = 0.0
            pde_maxDPz = 0.0
        else:
            #build on previous Ufile
            data = read_pdedp_ufile(myfile='pDEDP.AEP')
            e_arr = data['E']
            mu_arr = data['mu']
            pz_arr = data['pphi']
            de_arr = data['dE']
            dp_arr = data['dP']
            pdedp = data['f']
            pdedp_optimize = False

        #check pdedp_bdry here before run

        #set some run options
        opt = self.data.options.active.read()
        opt.new({
            "SIM_MODE":sim_mode,"ENABLE_ADAPTIVE":1,
            "RECORD_MODE":sim_mode-1,"ENDCOND_SIMTIMELIM":1,
            "ENDCOND_MAX_MILEAGE":trun,"ENABLE_ORBIT_FOLLOWING":1,
            "ENABLE_ORBIT_COLLISIONS":0,"ENABLE_MHD":1
        })
        #self.data.create_input("opt",**opt,desc="ORBITKICK_ITER")

        #check for too short run
        trun = opt['ENDCOND_MAX_MILEAGE']
        if trun < 2.5*dtsamp:
            print('Warning')
            print('Simulation run time too short compared to dtsamp')
            print('Aborting')
            return

        #loop over sub-simulations and calculate kicks
        for iloop i range(0,nloop):
            #initialize markers
            #focusdep??

            #do simulation
            subprocess.run(["where is executable"])

            #get marker ids, mass, and lost times
            id_arr = self.data.active.getstate("ids")
            anum_arr = self.data.active.getstate("anum") #atomic mass num
            znum_arr = self.data.active.getstate("znum") #atomic charge num
            t_fin = self.data.active.getstate("time",state="end") #[s]

            #get CoM as function of time for every marker
            for j in range(0,len(id_arr)):
                #need to activate B-field to access ptor
                self.input_init(bfield=True)

                #get constants of motion vs. time
                eorb,torb,muorb,pzorb,rhoorb = self.data.active.getorbit("ekin",
                                                                         "time",
                                                                         "mu",
                                                                         "ptor",
                                                                         "rho",
                                                                         ids=id_arr[j])
                self.input_free()

                #limit calculations before marker is lost
                tind = np.where(torb <= t_fin)[0]
                eorb = eorb[tind] #[eV]
                muorb = muorb[tind] #[eV/T]
                pzorb = pzorb[tind] #[amu*m**2/s]
                rhoorb = rhoorb[tind] #psipol/psipol(a)

                #limit to rho<1; limi_psi in ORBIT
                rind = np.where(rhoorb < 1.0)[0]
                eorb = eorb[rind]
                muorb = muorb[rind]
                pzorb = pzorb[rind]
                rhoorb = rhoorb[rind]

                #convert to Roscoe units
                eorb = convert_en(eorb,anum=anum_arr[j],znum=znum_arr[j])
                muorb = convert_mu(muorb,eorb)
                pzorb = convert_pz(pzorb)

            #optimize only first loop

            #calculate and histogram kicks

        #loop has ended and we have finished all simulations
        
        #normalize matrices

        #sparse rep

        #write ufile
        
        return

    def check_en(myen,anum=2.0,znum=1.0):
        """
        Parameters
        ----------
        myen : float [eV]
            Energy to convert to Roscoe units.
        anum : int
            Atomic mass number of ion
        znum : int
            Atmoic charge of ion

        Eprime = ke*E, where E is in keV and
        ke = 1000*A*(mp/qe)*g_0**2/(R_0*Z_p*B_0)**2
        A = atomic mass number
        mp = proton mass
        qe = elementary charge
        g_0 = g-function = B_0/fnorm*R_0, where fnorm=B_0 setting B_0=1
        R_0 = major radius in [m] 
        Z_p = elementary charge of ion
        B_0 = mag field on axis [T]
        """
        qe = 1.602e-19 #[C]
        mp = 1.673e-27 #[kg]
        myen *= 1000.0 #[keV]

        ke = 1000.0*anum*(mp/qe)*(1.0/(znum*bcenter)**2)
        myen *= myen
        
        return myen

    def convert_mu(mymu,myen):
        """
        Parameters
        ----------
        mymu : float 
            Magnetic moment to convert to Roscoe units.
        myen : float 
            Energy already converted to Roscoe units.

        muprime = mu*B_0_E, where E is in Roscoe units (ke*E)
        ke = 1000*A*(mp/qe)*g_0**2/(R_0*Z_p*B_0)**2
        B_0 = mag field on axis [T]
        """
        mymu *= bcenter/myen
        
        return mymu

    def convert_pz(mypz,myen):
        """
        Parameters
        ----------
        mypz : float 
            Canonical tor momentum to convert to Roscoe units.
        myen : float 
            Energy already converted to Roscoe units.

        Pzprime = rho*g/psiw-psi, where
        g = Bphi/fnorm*R(psi,theta=0)
        Bphi = tor magnetic field [T]
        fnorm = B_0 = field on axis [T]
        R(psi,theta=0) = major radius [m] at outer mid-plane
        psiw = psi_lcfs/bcenter

        and

        rho = p*sqrt(2*Eprime)*fnorm/B(psi)
        p = particle pitch
        Eprime = Energy already converted to Roscoe units.
        B(psi) = magnetic field [T]
        """
        
        return

    def read_pdedp_ufile(myfile='pDEDP.AEP'):
        """
        Parameters
        ----------
        myfile : string 
            Ufile containing orbit kicks. Default is pDEDP.AEP
        """
        with open(myfile) as fp:
        lines = fp.readlines()

        tstep = float(lines[9].split()[0])*1000.0 #microsec
        
        nde = int(lines[12].split()[0])
        ndp = int(lines[13].split()[0])
        ne = int(lines[14].split()[0])
        npp = int(lines[15].split()[0])
        nm = int(lines[16].split()[0])

        e = np.zeros(ne)
        p = np.zeros(npp)
        m = np.zeros(nm)
        de = np.zeros(nde)
        dp = np.zeros(ndp)
        f = np.zeros([ne,npp,nm,nde,ndp])

        lstart = 17 #skip header stuff
        dl = len(lines[17].split())
        de,lstart = read_arr1d(lines,lstart,dl,de)
        dp,lstart = read_arr1d(lines,lstart,dl,dp)
        e,lstart = read_arr1d(lines,lstart,dl,e)
        p,lstart = read_arr1d(lines,lstart,dl,p)
        m,lstart = read_arr1d(lines,lstart,dl,m)
       
        for i in range(0,ne):
            for j in range(0,npp):
                for k in range(0,nm):
                    for l in range(0,nde):
                        f[i,j,k,l,:],lstart = read_arr1d(lines,lstart,dl,
                                                         f[i,j,k,l,:])
        fp.close()
        
        struct = {'E':e,'pphi':p,'mu':m,'dE':de,'dP':dp,'f':f,'tstep':tstep}
        
        return struct

    #def check_bdry():
    #    #threshold limit for when to adjust grid
    #    dthresh = 0.1
    #
    #find boundary for mu and Pz based on energy range in pdedp calcs
    #    pde_engn = convert_en(pde_emax) #convert to normalized energy units)
    #    
    #    return

    #def check_dedp():
    #    return

    #def write_kick_ufile():
    #    return
    
