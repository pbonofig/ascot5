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
            Number of bins in kick energy array. Default is 29; must be odd!
        dpz_min : float [unitless]
            Minimum change in momentum. Default 1e-8
        dpz_max : float [unitless]
            Minimum change in  momentum. Default 2e-8
        dpz_bins : int
            Number of bins in kick momentum array. Default is 29; must be odd!
        """
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
            pde_maxDE = 0.0
            pde_maxDPz = 0.0
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

            pde_maxDE,pde_maxDPz = find_maxDEDP(pdedp,de_arr,dpz_arr)

            #check sampling time matches
            if dtsamp_old != dtsamp:
                print('Warning')
                print('Reading from old pDEDP but sampling times do not match')
                print('Aborting')
                return

        #check some run options??

        #check for too short run
        trun = opt['ENDCOND_MAX_MILEAGE']
        if trun < 2.5*dtsamp:
            print('Warning')
            print('Simulation run time too short compared to dtsamp')
            print('Aborting')
            return

        #loop over sub-simulations and calculate kicks
        for iloop in range(0,nloop):
            #print beginning of loop
            print('Starting iteration '+str(iloop+1))

            #only optimize first loop to avoid over interpolation of kicks
            if iloop > 0:
                pdep_optimize = False
        
            #initialize markers
            #focusdep??

            #do simulation
            subprocess.run(["where is executable"])

            #get marker ids, mass, and lost times
            id_arr = self.data.active.getstate("ids") #particle IDs
            anum_arr = self.data.active.getstate("anum") #atomic mass num
            znum_arr = self.data.active.getstate("znum") #atomic charge num
            t_fin = self.data.active.getstate("time",state="end") #end time [s]

            #check kick ranges based on inputs only on first loop
            #if pdedp_optimize = True:

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

                #limit to rho<1; limit_psi in ORBIT
                rind = np.where(rhoorb < 1.0)[0]
                eorb = eorb[rind]
                muorb = muorb[rind]
                pzorb = pzorb[rind]
                rhoorb = rhoorb[rind]

                #convert to Roscoe units
                eorb = convert_en(eorb,anum=anum_arr[j],znum=znum_arr[j],bcenter=bcenter)
                muorb = convert_mu(muorb,eorb,bcenter=bcenter)
                pzorb = convert_pz(pzorb)

            #calculate and histogram kicks

            #checkDEDPz ranges only on first loop after kick calcs
            #if pdedp_optimize = True:
            #pdedp_checkDEDP

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

    def convert_en(myen,anum=2.0,znum=1.0,bcenter):
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

    def find_maxDEDP(pdedp,de_arr,dpz_arr):
        """
        Parameters
        ----------
        pdedp : float array
            5D kick matrices
        de_arr : float array
            Array for dE kicks
        dpz_arr: float array
            Array for dPz kicks
        """
        #initialize variables
        pde_maxDE = 0.0
        pde_maxDPz = 0.0

        n_e,n_pz,n_mu,n_de,n_dpz = pdedp.shape

        for i in range(0,n_e):
            for j in range(0,n_pz):
                for k in range(0,n_mu):
                    for m in range(0,n_de):
                        for n in range(0,n_dpz):
                            if pdedp[i,j,k,n,m] > 0:
                                if abs(de_arr[m]) > pde_maxDE:
                                    pde_maxDE = de_arr[m]
                                if abs(dpz_arr[n]) > pde_maxDPz:
                                    pde_maxDPz = dpz_arr[n]
        
        return pde_maxDE,pde_maxDPz

    def check_pDEDP(pde_maxDE,pde_maxDPz,DEmax,DPzmax):
        """
        Parameters
        ----------
        pde_maxDE : float
        pde_maxDPz : float
        DEmax : float
        DPzmax : float
        
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
        fracE = (pde_maxDE-DEmax)/DEmax
        fracPz = (pde_maxDPz-DPzmax)/DPzmax

        if(abs(fracE)<dthresh and abs(fracPz)<dthresh and
           fracE <= 1.0 and fracPz <= 1.0):
            #grid okay
            print('(DE,DPz) grid looks ok - not updating')
            print('')
        else:
            #keep copy
            DEmax_old = DEmax
            DPzmax_old = DPzmax

            #new values
            DEmax = (1.0+fracE)*DEmax
            DPzmax = (1.0+fracPz)*DPzmax

            #round off
            DEmax = 1e-6*()
            DPzmax = 1e-8*()

            #symmetric grid
            DEmin = -1*DEmax
            DPzmin = -1*DPzmax

            #print new grid info
            print('New (DE,DPz) grid computed')
            print('original (DE,Dpz): ('+str(DEmax_old)+,+str(DPzmax_old)+')')
            print('updated (DE,Dpz): ('+str(DEmax)+,+str(DPzmax)+')')
            print('')

        #print end
        print()
        
        return

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
