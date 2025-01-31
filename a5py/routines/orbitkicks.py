"""orbitkicks: Calculates orbit "kicks" of fast ions in axisymmetric 
fusion devices. This is done by examining changes in the constants
of motion: energy and canonical toroidal momentum. It is assumed that
the magnetic moment is held constant (true for low freq. modes)
"""
import numpy as np
import unyt
from a5py.ascot5io.options import Opt
import subprocess
import fortranformat as ff

class Orbitkicks():
    """
    Class for computing orbit-kicks based on the ORBIT-kick
    code by M. Podesta (see PPCF 2014 and 2017).

    The main function is simkick which performs a series of
    ASCOT runs and calculates the "kick matrices" which
    are 5D phase-space matrices to compute P(DE,DP|E,P,mu).
    That is, the average change in energy and canonical
    toroidal momentum given the location in phase=space 
    (E.Pphi,Mu). Theses calculations are perform with 
    nprt number of markers for nloop iterations.

    All other functions withi this class are in support of 
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

    def simkick(self,dtsamp,tsim=0.0002,gcmode=True,
                nprt=10000,nloop=5,update=False,
                e_min=1000.0,e_max=150.0e3,e_bins=15,
                pz_min=-1.2,pz_max=1.0,pz_bins=40,
                mu_min=0.0,mu_max=1.4,mu_bins=16,
                de_min=-1.0e-6,de_max=1.0e-6,de_bins=29,
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
            Number of bins in kick energy array. Default is 29; must be odd!
        dpz_min : float [unitless]
            Minimum change in momentum. Default 1e-8
        dpz_max : float [unitless]
            Minimum change in  momentum. Default 2e-8
        dpz_bins : int
            Number of bins in kick momentum array. Default is 29; must be odd!
        """
        #some checks on simulation times
        #check for too long run
        if tsim > 0.0005:
            raise ValueError('Simulation run time exceeds 5 ms. Lower tsim')

        #check for too short run
        if tsim < 2.5*dtsamp:
            raise ValueError('Simulation run time too short compared to dtsamp')
        
        #make storage for kicks
        if update == False:
            #new run
            e_arr = np.linspace(e_min,e_max,e_bins)
            mu_arr = np.linspace(mu_min,mu_max,mu_bins)
            pz_arr = np.linspace(pz_min,pz_max,pz_bins)
            de_arr = np.linspace(de_min,de_max,de_bins)
            dpz_arr = np.linspace(dpz_min,dpz_max,dpz_bins)
            pdedp = np.zeros([e_bins,pz_bins,mu_bins,de_bins,dpz_bins])
            pdedp_optimize = True
            maxDE_kick = 0.0
            maxDPz_kick = 0.0
        else:
            #build on previous Ufile
            data = read_pdedp_ufile(myfile='pDEDP.AEP')
            e_arr = data['E']
            mu_arr = data['mu']
            pz_arr = data['pphi']
            de_arr = data['dE']
            dpz_arr = data['dP']
            pdedp = data['f']
            dtsamp_old = data['dtsamp']
            spec_old = data['spec']
            pdedp_optimize = False

            #check sampling time matches
            if dtsamp_old != dtsamp:
                raise ValueError('Reading from old pDEDP but sampling times do not match')

        #initialize inputs from hdf5 file
        self.simulation_initinputs()
            
        #make run options
        opt = Opt.get_default()
        set_simkick_opts(opt,tsim=tsim,gcmode=gcmode)
        self.simulation_initopt(**opt)

        #loop over sub-simulations and calculate kicks
        for iloop in range(0,nloop):
            #print beginning of loop
            print('Starting iteration '+str(iloop+1))

            #only optimize first loop to avoid over interpolation of kicks
            if iloop > 0:
                pdedp_optimize = False
        
            #initialize markers
            #mrk = get_markers()
            #self.simulation_initmarkers(**mrk)

            #do simulation
            vrun = self.simulation_run()

            #get marker ids, mass, charge and lost times
            id_arr = vrun.getstate("ids",state="ini") #particle IDs
            anum_arr = vrun.getstate("anum",state="ini") #atomic mass num
            znum_arr = vrun.getstate("znum",state="ini") #atomic charge num
            t_fin = vrun.getstate("time",state="end") #end time [s]

            #check same species if update=True
            if update == True:
                if check_spec(spec_old,anum_arr[0],znum_arr[0]):
                    pass
                else:
                    raise ValueError('Reading from old pDEDP but ion species does not match')

            #check kick ranges based on inputs only on first loop
            if pdedp_optimize = True:
                pz_arr,mu_arr = check_bdry(mu_min,mu_max,mu_bins,pz_min,pz_max,pz_bins)

            #need to activate B-field to access ptor and calc bcenter
            vrun.input_init(bfield=True)

            #calc B on axis
            bout = vrun.bfield.active.read()
            axisr = bout['axisr']
            axisz = bout = ['axisz']
            bcenter = vrun.input_eval(axisr,0,axisz,0,'bnorm') #[T]
            print('Bfield on axis: '+str(bcenter)+' T')
            print('')
            #function to get all bfield info of interst???
            #bcenter,bmin,bmax,psiwall

            #get CoM as function of time for every marker
            for j in range(0,len(id_arr)):
                #get constants of motion vs. time
                eorb,torb,muorb,pzorb,wgtorb = vrun.getorbit("ekin",
                                                             "time",
                                                             "mu",
                                                             "ptor",
                                                             "wgts"
                                                             ids=id_arr[j])

                #limit calculations before marker is terminated
                tind = np.where(torb <= t_fin)[0]
                eorb = eorb[tind] #[eV]
                muorb = muorb[tind] #[eV/T]
                pzorb = pzorb[tind] #[amu*m**2/s]
                wgtorb = wgts[tind] #[#/s]
                torb = torb[tind] #[s]

                #convert to Roscoe units
                eorb = convert_en(eorb,bcenter,anum=anum_arr[j],znum=znum_arr[j])
                muorb = convert_mu(muorb,eorb,bcenter=bcenter)
                pzorb = convert_pz(pzorb)

            #free bfield for next loop
            vrun.input_free()

            #calculate kicks
            kick_str = pdedp_calc_kicks(dtsamp,eorb,muorb,pzorb,wgtorb,torb,
                                        maxDE_kick,maxDPz_kick)

            #check (DE,DPz) ranges only on first loop after kick calcs
            if pdedp_optimize = True:
                de_arr,dpz_arr = check_DEDP(maxDE_kick,maxDPz_kick,
                                            de_max,de_min,de_bins,
                                            dpz_max,dpz_min,dpz_max)

            #record kicks
            pdedp_record_kicks(pdedp,e_arr,pz_arr,mu_arr,de_arr,dpz_arr,
                               kick_str)

            #free markers for next iteration
            self.simulation_free(markers=True,diagnostics=False)

            #print end of loop
            print('Completed iteration '+str(iloop+1)+'/'+str(nloop))
            print('')

        #loops have ended and we have finished all simulations
        
        #normalize matrices
        pdedp_finalize(pdedp,de_arr,dpz_arr)

        #write ufile
        write_pdedp_ufile(dtsamp,e_arr,mu_arr,pz_arr,de_arr,dpz_arr,
                         pdedp,ami=anum_arr[0],zmi=znum_arr[0])
        
        return

    def set_simkick_opts(opt,tsim=0.0002,gcmode=True):
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
        if gcmode == True: #GC
            sim_mode = 2
            rec_mode = 1
        else: #FO
            sim_mode = 1
            rec_mode = 0
            
        #change options from the default
        opt.update({
            "SIM_MODE":sim_mode,
            "RECORD_MODE":rec_mode,
            "ENDCOND_SIMTIMELIM":1, #end at max mileage
            "ENDCOND_RHOLIM":1, #end at rho>=1
            "ENDCOND_LIM_SIMTIME":10.0,
            "ENDCOND_MAX_MILEAGE":tsim, #mileage is actual sim time
            "ENDCOND_MAX_CPUTIME":,10.0,
            "ENDCOND_MAX_RHO":1.0,
            "ENABLE_ORBIT_FOLLOWING":1, #orbit following ON
            "ENABLE_MHD":1, #include MHD
            "ENABLE_ORBITWRITE":1, #store marker orbits
            "ORBITWRITE_MODE":1, #collect orbits on dt intervals
            "ORBITWRITE_NPOINT":50000, #max orbit collection points
            "ORBITWRITE_INTERVAL":1.0e-8, #time interval to collect marker info
        })
            
        return

    def get_kick_bfield(self):
        """
        Parameters
        ----------
        
        """
        self.input_init(bfield=True)
        bout = self.bfield.active.read()
        axisr = bout['axisr']
        axisz = bout = ['axisz']

        bcenter = vrun.input_eval(axisr,0,axisz,0,'bnorm') #[T]
    #        print('Bfield on axis: '+str(bcenter)+' T')
    #        print('')
        
        bout = {}
        
        return bout

    def convert_en(myen,bcenter,anum=2.0,znum=1.0):
        """
        Parameters
        ----------
        myen : float [eV]
            Energy to convert to Roscoe units.
        anum : int
            Atomic mass number of ion
        znum : int
            Atmoic charge of ion
        bcenter : float [T]
            Magnetic field on axis

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

    def convert_mu(mymu,myen,bcenter):
        """
        Parameters
        ----------
        mymu : float 
            Magnetic moment to convert to Roscoe units.
        myen : float 
            Energy already converted to Roscoe units.
        bcenter : float [T]
            Magnetic field on axis

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

    def check_spec(specid,anum,znum):
        """
        Parameters
        ----------
        specid : int
            TRANSP species ID to check
        anum : int
            Atomic mass number to compare to specid
        znum : int
            Atomic charge number to compare to specid
        """
        #check TRANSP species ID agrees with anum and znum
        #fast ion species
        if ami==1 and zmi==1:
            myspec = 5 
        elif ami=2 and zmi=1:
            myspec = 1 
        elif ami=3 andd zmi=1:
            myspec = 2 
        elif ami=3 and zmi=2:
            myspec = 3 
        elif ami=4 and zmi=2:
            myspec = 4 
        else:
            myspec = 0

        if myspec == specid:
            return True
        else:
            return False

    def check_bdry(mu_min,mu_max,mu_bins,pz_min,pz_max,pz_bins):
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

        Check the range used for computing p(DE,DP|E,Pz,mu)
        and adjust the (Pz,mu) range on-the-fly to optimize
        the sampling
        """
        #print start
        print('Checking Pz,mu ranges...')
        print('')

        #threshold limit to alter ranges
        dthresh = 0.1

        #find boundary for mu and Pz based on energy range in pDEDP calc

        #max energy in Roscoe units
        dum_emax = convert_en(e_max,bcenter,anum=anum,znum=znum)

        #min and max b-field; rescale so Baxis=1.0 like ORBIT
        bmin = 1/bcenter
        bmax = 2/bcenter

        #redefine mu range with buffer
        dum_mumax = 1.0/bmin*(mu_bins+1.0)/mu_bins
        dum_mumax *= 1.05
        dum_mumin = 0.0

        #redefine pz range with buffer
        dum_pzmax = axisr/psiwall*sqrt(2.0*dum_emax)*(pz_bins+1.0)/pz_bins
        dum_pzmax *= 1.05
        dum_pzmin = -1.0 - rlcfs/psiwall*sqrt(2.0*dum_emax)/bmax*(pz_bins+1.0)/pz_bins
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
            print('Pz,mu ranges look OK')
            print('')

        #form grid and return
        pz_arr = np.linspace(pz_min,pz_max,pz_bins)
        my_arr = np.linspace(mu_min,mu_max,mu_bins)
        
        return pz_arr,mu_arr

    def check_DEDP(maxDE_kick,maxDPz_kick,de_max,de_min,de_bins,
                   dpz_max,dpz_min,dpz_max):
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
        print('Checking (DE,DPz) boundaries...')
        print('')
        
        #threshold limit to alter grid
        dthresh = 0.1

        #check fractional size if DE,DPz ranges need to be updated
        fracE = (maxDE_kick-de_max)/de_max
        fracPz = (maxDPz_kick-dpz_max)/dpz_max

        if(abs(fracE)<dthresh and abs(fracPz)<dthresh and
           fracE <= 1.0 and fracPz <= 1.0):
            #grid okay
            print('(DE,DPz) grid looks ok - not updating')
            print('')
        else:
            #keep copy
            de_max_old = de_max
            dpz_max_old = dpz_max

            #new values
            de_max = (1.0+fracE)*de_max
            dpz_max = (1.0+fracPz)*dpz_max

            #round off (doesn't need to preserve precision)
            de_max = 1e-6*(int(np.ceil(1e6*de_max)))
            dpz_max = 1e-8*(int(np.ceil(1e8*dpz_max)))

            #symmetric grid
            de_min = -1*de_max
            dpz_min = -1*dpz_max

            #print new grid info
            print('New (DE,DPz) grid computed')
            print('original (DE,Dpz): ('+str(de_max_old)+,+str(dpz_max_old)+')')
            print('updated (DE,Dpz): ('+str(de_max)+,+str(dpz_max)+')')
            print('')

        #make grid
        de_arr = np.linspace(de_min,de_max,de_bins)
        dpz_arr = np.linspace(dpz_min,dpz_max,dpz_bins)
        
        return de_arr,dpz_arr

    def read_pdedp_ufile(myfile='pDEDP.AEP'):
        """
        Parameters
        ----------
        myfile : string 
            Ufile containing orbit kicks. Default is pDEDP.AEP
        """
        #print start
        print('Reading kick matrices from '+myfile)
        print('')
        
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
            p_arr = np.zeros(npz)
            mu_arr = np.zeros(nm)
            de_arr = np.zeros(nde)
            dpz_arr = np.zeros(ndpz)
            pdedp = np.zeros([ne,npz,nmu,nde,ndpz])

            #read 1D arrays
            lstart = 17 #skip header stuff
            dl = len(lines[17].split())
            de,lstart = read_arr1d(lines,lstart,dl,de_arr)
            dp,lstart = read_arr1d(lines,lstart,dl,dpz_arr)
            e,lstart = read_arr1d(lines,lstart,dl,e_arr)
            p,lstart = read_arr1d(lines,lstart,dl,pz_arr)
            m,lstart = read_arr1d(lines,lstart,dl,mu_arr)

            #read 5D matrix till footer
            for i in range(lstart,len(lines)-6):
                myline = lines[i].split()
                ind_de = int(myline[0])-1 #python indexing at 0
                ind_dpz = int(myline[1])-1
                ind_e = int(myline[2])-1
                ind_pz = int(myline[3])-1
                ind_mu = int(myline[4])-1
                val = float(myline[5])-1
                pdedp[ind_e,ind_pz,ind_mu,ind_de,ind_dpz] = val
                
        fp.close()

        #print end
        print('Finished reading '+myfile)
        print('')
        
        struct = {'e_arr':e_arr,'pz_arr':pz_arr,'mu_arr':mu_arrr,
                  'de_arr':de_arr,'dpz_arrr':dpz_arr,'pdedp':pdedp,
                  'tsamp':tsamp,'spec':spec}
        
        return struct

    def read_arr1d(lines,lstart,dl,arr):
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

    def write_pdedp_ufile(dtsamp,e_arr,mu_arr,pz_arr=29,
                         de_arr,dpz_arr,pdedp,ami=2,zmi=1,
                         myfile='pDEDP.AEP'):
        """
        Parameters
        ----------
        dtsamp : float [s]
            Sampling time step in seconds
        e_arr : float array
            Array of energy values in Roscoe units
        mu_arr : float array
            Array of mu values in Roscoe units
        pz_arr : float array
            Array of pz values in Roscoe units
        de_arr : float array
            Array of DE kicks in Roscoe units
        dpz_arr : float array
            Array of DP kicks in Roscoe units
        pdedp : float array
            5D phase-space matrix
        ami : int
            Atomic mass number of ion. Default is 2 for deuterium
        zmi : int
            Atmoic charge number of ion. Default is 1 for deuterium
        myfile : string 
            Ufile containing orbit kicks to write to. Default is pDEDP.AEP
        """
        #print start
        print('Writing kick output to '+myfile)
        print('')
        
        #header information
        lshot=123456
        nd=5	#5-D data
	nq=0	#unknown parameter...
	nr=6	#number of decimal places f13.6
	np=0	#representation, 0:full, 1:sparse (set below)
	ns=1	#number  of scalars

        dev='DEV'
	labelx='DEstep                '	#c*20
	unitsx='   kev    '		#c*10
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
	com2='UFILE WRITTEN BY ASCOT, see WRITE_KICK_UFILE'
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
        elif ami=2 and zmi=1:
            spec = 1 
            specstr = 'deuterium'
        elif ami=3 andd zmi=1:
            spec = 2 
            specstr = ';tritium'
        elif ami=3 and zmi=2:
            spec = 3 
            specstr = ';HE3 FUSN'
        elif ami=4 and zmi=2:
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
        dtsamp *= 1000.0
        hh = ff.FortranRecordWriter('(1e13.6)')
        f.write('  '+hh.write([dtsamp])+'          ')
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
        hh = ff.FortranRecordWriter('(6e14.6)')
        f.write(hh.write(de_arr))
        f.write('\n')
        f.write(hh.write(dpz_arr))
        f.write('\n')
        f.write(hh.write(e_arr))
        f.write('\n')
        f.write(hh.write(pz_arr))
        f.write('\n')
        f.write(hh.write(mu_arr))
        f.write('\n')

        #write 5D matrix, only write non-zero elements, i.e. sparse matrix
        for i in range(0,len(e_arr)):
            for j in range(0,len(pz_arr)):
                for k in range(0,len(mu_arr)):
                    for m in range(0,len(de_arr)):
                        for n in range(0,len(dpz_arr)):
                            val = pdedp[i,j,k,m,n]
                            if val != 0.0:
                                #fortran indexing at 1
                                f.write(m+1,n+1,i+1,j+1,k+1,hh.write([val]))

        #print footer information
        f.write(com+'\n')
        f.write(com2+'\n')
        f.write(com3+'\n')
        f.write(com4+'\n')
        f.write(com5+'\n')
        f.write(com6+'\n')
        
        f.close()

        #print end
        print('Finished writing kick output to '+myfile)
        print('')
        
        return
    
    def pdedp_finalize(pdedp,de_arr,dpz_arr):
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
        print('Finalizing pDEDP computation')
        print('')

        #make sure (DE,DP)=(0,0)
        inde = np.argmin(np.abs(de_arr))
        indp = np.argmin(np.abs(dpz_arr))
        de_arr[inde] = 0.0
        dpz_arr[indp] = 0.0

        #get dimensions
        n_e,n_pz,n_mu,n_de,n_dpz = pdedp.shape

        #get average counts/bin from non-empty bins
        #fill in empty bins with unity
        nbins = 0
        cnt_avg = 0
        sum_p = np.zeros([n_e,n_pz,n_mu])
        for i in range(0,n_e):
            for j in range(0,n_pz):
                for k in range(0,n_mu):
                    cnt = 0
                    sum_p[i,j,k] = 0
                    for m in range(0,n_de):
                        for n in range(0,n_dpz):
                            cnt += pdedp[i,j,k,m,n]

                    if cnt > 0:
                        cnt_avg += cnt
                        sum_p[i,j,k] += cnt
                        nbins += 1
                    else:
                        pdedp[i,j,k,inde,indp] = 1.0
                        sum_p[i,j,k] = 1.0

        if nbins > 0:
            cnt_avg /= nbins
                
        #Normalize all probabilities to average number of counts/bin
        for i in range(0,n_e):
            for j in range(0,n_pz):
                for k in range(0,n_mu):
                    pdedp[i,j,k,:,:] *= cnt_avg/sum_p[i,j,k]

        #print end
        print('pDEDP matrices normalized')
        print('Average number of counts: '+str(cnt_avg))
        print('')
        
        return

    def pdedp_calc_kicks(dtsamp,eorb,muorb,pzorb,wgtorb,torb,
                         maxDE_kick,maxDPz_kick):
        #print start
        print('Computing (DE,DP) kicks...')
        print('')

        #skip first nskip points
        dt = torb[1] - torb[0] #[s]
        ntot = len(torb)
        nskip = np.ceil(ntot/1e4)
        nskip = max(1,nskip)
        torb = torb[nskip:]
        eorb = eorb[nskip:]
        muorb = muorb[nskip:]
        pzorb = pzorb[nskip:]
        wgtorb = wgtorb[nskip:]

        ttot = torb[-1] - torb[0] #total time [s]
        nintv = ttot//tsamp #number of sampling intervals
        nav = (tsamp/dt)//50 #number of bins to smooth over in interval

        #storage to save values to record later
        #record later in case we need to optimize and re-bin
        eavgs = []
        pzavgs = []
        muavgs = []
        dekicks = []
        dpzkicks = []
        wgts = []
        
        #go through time array by sampling intervals
        for i in range(0,len(torb),nintv):
            #initialize to 0
            Eav = 0.0
            Pzav = 0.0
            Muav = 0.0
            Wgtav = 0.0
            newE = 0.0
            newPz = 0.0
            newMu = 0.0
            
            for j in range(i,i+nintv,nav):
                #find CoM avgs smoothed by nav
                Eav += eorb[j]
                Pzav += pzorb[j]
                Muav += np.abs(muorb[j])
                Wgtav += wgtorb[j]

                #find avg deltas smoothed by nav
                newE += eorb[j+nintv]
                newPz += pzorb[j+nintv]
                newMu += muorb[j+intv]

            #finish computing averages
            Eav /= nav
            Pzav /= nav
            Muav /= nav
            Wgtav /= nav
            Wgtav *= tsamp #units now [#]
            newE /= nav
            newPz /= nav
            newMu /= nav

            #calculate DE,DP,DM
            #dt_fact???
            dedum = newE - Eav
            dpzdum = newPz - Pzav
            dmudum = newMu - Muav

            #update max kicks
            maxDE_kick = max(1.05*dedum,maxDE_kick)
            maxDPz_kick = max(1.05*dpzdum,maxDPz_kick)

            #save it all
            eavgs.append(Eav)
            pzavgs.append(Pzav)
            muavgs.append(Muav)
            dekicks.append(dedum)
            dpzkicks.append(dpzdum)
            wgts.append(Wgtav)

        #print end
        print('Finished computing (DE,DP) kicks')
        print('')

        kick_calc_str = {'eavgs':eavgs,'pzavgs':pzavgs,'muavgs':muavgs,
                         'dekicks':dekicks,'dpzkicks':dpzkicks,'wgts':wgts}
        
        return kick_calc_str

    def pdedp_record_kicks(pdedp,e_arr,pz_arr,mu_arr,de_arr,dpz_arr,
                           kick_calc_str):
        """
        Parameters
        ----------
        
        """
        #print start
        print(print('Recording pDEDP to 5D matrix...')
        print('')
        
        #unpack kick structure
        eavgs = kick_calc_str['eavgs']
        pzavgs = kick_calc_str['pzavgs']
        muavgs = kick_calc_str['muavgs']
        dekicks = kick_calc_str['dekicks']
        dpzkicks = kick_calc_str['dpzkicks']
        wgts = kick_calc_str['wgts']

        #histogram each kick calculation
        for i in range(0,len(eavgs)):
            #get (E,Pz,mu) bin indexes
            inde = np.argmin(np.abs(e_arr-eavgs[i]))
            indpz = np.argmin(np.abs(pz_arr-pzavgs[i]))
            indmu = np.argmin(np.abs(mu_arr-muavgs[i]))

            #get (DE,DP) bin indexes
            indde = np.argmin(np.abs(de_arr-ekicks[i]))
            inddpz = np.argmin(np.abs(dpz_arr-pzkicks[i]))

            #update pdedp by weight [#]
            pdedp[inde,indpz,indmu,indde,inddpz] += wgts[i]

        #print end
        print('Finished recording pDEDP to 5D matrix')
        
        return

#R: energies be in eV, bfield be written from B_STS
#M: nothing
#E: Uniformly samples particles between emin and emax, within the LCFS,
# random pitch, random gyroradius, random tor angle. Default is deuterons.
def uni_mrk(fname,emin=1000.0,emax=1.0e5,pmin=-1.0,pmax=1.0,
            rhomax=0.99,anum=2,znum=1,nprt=10000):
    #constants
    pmass = 1.6726e-27 #kg
    q = 1.602e-19 #Coulomb
    amu = 1.6605e-27 #kg

    #marker ids
    ids = np.array(range(1,nprt+1))

    #default is deuterium
    anum = np.ones(nprt)*anum
    znum = np.ones(nprt)*znum

    #mass has units amu and charge in units e
    mass = np.ones(nprt)*anum*(pmass/amu)
    charge = np.ones(nprt)*znum

    #give equal weights
    weight = np.ones(nprt)

    #start time is t=0 for all
    time = np.zeros(nprt)

    #read B-field
    h5 = a5io(fname)
    bout = h5.data.bfield.active.read()
    phimin = bout['axis_phimin'][0] #deg
    phimax = bout['axis_phimax'][0] #deg
    nphi = bout['axis_nphi'][0]
    phiang = np.linspace(phimin,phimax,nphi,endpoint=False) #deg
    psi = bout['psi'] #Wb
    psi1 = bout['psi1'][0] #Wb at LCFS
    psi_rmin = bout['psi_rmin'][0] #m
    psi_rmax = bout['psi_rmax'][0] #m
    psi_nr = bout['psi_nr'][0]
    rmag = np.linspace(psi_rmin,psi_rmax,psi_nr)
    psi_zmin = bout['psi_zmin'][0] #m
    psi_zmax = bout['psi_zmax'][0] #m
    psi_nz = bout['psi_nz'][0]
    zmag = np.linspace(psi_zmin,psi_zmax,psi_nz)

    #get lcfs
    bout2 = np.load(fname[0:-3]+'_lcfs.npy',encoding="latin1",
                    allow_pickle=True).item()
    rlcfs = bout2['rlcfs'] #m
    zlcfs = bout2['zlcfs'] #m
    
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
        phi[i] = random.uniform(0.0,360.0)
        
        #gyroangle (rad)
        zeta[i] = random.uniform(0.0,2.0*np.pi)

        #pitch (vpar/vtot)
        #pitch[i] = random.uniform(-1.0,1.0)
        pitch[i] = random.uniform(pmin,pmax)
        
        #energy (eV); default is 1-100 keV
        energy[i] = random.uniform(emin,emax)

        #get random velocity vector
        vtot = np.sqrt(2.0*energy[i]*q/(mass[i]*amu)) #m/s
        vphi[i] = random.uniform(-1.0,1.0)
        if (vphi[i] == 1) or (vphi[i] == -1):
            vr[i] = 0
            vz[i] = 0
        else:
            v2 = np.sqrt(1.0-vphi[i]**2)
            vr[i] = random.uniform(-1.0*v2,v2)
            if (vr[i] == v2) or (vr[i] == -1*v2):
                vz[i] = 0
            else:
                vz[i] = np.sqrt(1.0-(vphi[i]**2 + vr[i]**2))

        #unnormalize by vtot
        vr[i] *= vtot #m/s
        vz[i] *= vtot #m/s
        vphi[i] *= vtot #m/s

        #get LCFS
        phind = np.argmin(np.abs(phiang-phi[i]))
        myrlcfs = rlcfs[:,phind]
        myzlcfs = zlcfs[:,phind]

        #find flux surface for rhomax
        #cs = plt.contour(rmag,zmag,np.transpose(psi[:,phind,:])/psi1,
        #                 colors='w',alpha=0,levels=100)
        #indlev = np.argmin(np.abs(cs.levels-rhomax))
        #mycont = cs.allsegs[indlev][0]
        mycont = ski.measure.find_contours(np.sqrt(psi[:,phind,:]/psi1),rhomax)
        mycont = mycont[0]
        fx = interp1d(np.arange(0,len(rmag)),rmag.flatten())
        fy = interp1d(np.arange(0,len(zmag)),zmag.flatten())
        mycont[:,0] = fx(mycont[:,0])
        mycont[:,1] = fy(mycont[:,1])

        #stay within rhomax
        #rmin = np.amin(myrlcfs)
        #rmax = np.amax(myrlcfs)
        #zmin = np.amin(myzlcfs)
        #zmax = np.amax(myrlcfs)
        rmin = np.amin(mycont[:,0])
        rmax = np.amax(mycont[:,0])
        zmin = np.amin(mycont[:,1])
        zmax = np.amax(mycont[:,1])

        #get random R and find intersection of vertical line with LCFS
        myr = random.uniform(rmin,rmax)
        line1 = sh.LineString(np.column_stack(([myr,myr],[zmin,zmax])))
        #line2 = sh.LineString(np.column_stack((myrlcfs,myzlcfs)))
        line2 = sh.LineString(np.column_stack((mycont[:,0],mycont[:,1])))
        intersection = line1.intersection(line2)
        zinters = []
        for segment in intersection.geoms:
            x,y = segment.xy
            zinters.append(y[0])

        #get random Z between intersection points
        myz = random.uniform(np.amin(zinters),np.amax(zinters))
        r[i] = myr #m
        z[i] = myz #m

        #debugging
        #plt.plot(myrlcfs,myzlcfs,color='k')
        #plt.plot(mycont[:,0],mycont[:,1],color='magenta',linestyle='--')
        #plt.plot([myr,myr],[np.amin(zinters),np.amax(zinters)],color='b')
        #plt.scatter(r[i],z[i],color='r',marker='x')
        #plt.show()
        #input()
            
        #print status
        if (i+1)%1000==0:
            print('Sampled '+str(i+1)+' particles')
    
    #close contour plot window
    plt.close()

    mystr = {'ids':ids,
             'mass':mass, #amu
             'charge':charge, #e
             'r':r, #m
             'phi':phi, #deg
             'z':z, #m
             'vr':vr, #m/s
             'vphi':vphi, #m/s
             'vz':vz, #m/s
             'anum':anum,
             'znum':znum,
             'weight':weight, #markers/s
             'time':time, #s
             'energy':energy, #eV
             'pitch':pitch, #vpar/vtot
             'zeta':zeta} #rad
    return mystr
