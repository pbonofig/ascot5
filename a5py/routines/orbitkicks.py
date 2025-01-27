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

#PRINT UPDATE STATEMENTS

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

        #check some run options??

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

            #checkDEDPz after every loop

        #loop has ended and we have finished all simulations
        
        #normalize matrices

        #sparse rep

        #write ufile
        write_kick_ufile(dtsamp,e_arr,mu_arr,pz_arr,de_arr,dp_arr,
                         pdedp,ami=anum_arr[0],zmi=znum_arr[0])
        
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

    def write_kick_ufile(dtsamp,e_arr,mu_arr,pz_arr=29,
                         de_arr,dp_arr,pdedp,ami=2,zmi=1,
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
        dp_arr : float array
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
        print('Writing kick output to '+myfile+'\n')
        
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
        f.write('  '+str(dtsamp)+'          ')
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
        print('Finished writing kick output to '+myfile+'\n')
        
        return
    
    def pdedp_finalize(pdedp,de_arr,dpz_arr):
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
        dp_arr : float array
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
        print('Finalizing pDEDP computation'+'\n')

        #make sure (DE,DP)=(0,0)
        inde = np.argmin(np.abs(de_arr))
        indp = np.argmin(np.abs(dpz_arr))
        de_arr[inde] = 0.0
        dpz_arr[indp] = 0.0

        #get average counts/bin from non-empty bins
        #fill in empty bins with unity
        nbins = 0
        cnt_avg = 0
        sum_p = np.zeros(len(e_arr),len(pz_arr),len(mu_arr))
        for i in range(0,len(e_arr)):
            for j in range(0,len(pz_arr)):
                for k in range(0,len(mu_arr)):
                    cnt = 0
                    sum_p[i,j,k] = 0
                    for m in range(0,len(de_arr)):
                        for n in range(0,len(dpz_arr)):
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
        for i in range(0,len(e_arr)):
            for j in range(0,len(pz_arr)):
                for k in range(0,len(mu_arr)):
                    pdedp[i,j,k,:,:] *= cnt_avg/sum_p[i,j,k]

        #print end
        print('pDEDP matrices normalized'+'\n')
        print('Average number of counts: '+str(cnt_avg)+'\n')
        
        return
