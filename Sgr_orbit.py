#!/usr/bin/env python

import os, numpy as np, scipy.integrate, scipy.special, agama, matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G

plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


pi     = np.pi
massunit = (10**6 * (u.kpc/u.m) / (u.Msun / u.kg) / G.value).to(u.dimensionless_unscaled)
d2r    = pi/180
d0     = 27.0  # distance in kpc
#SGRfc0 = np.array([19.78, 2.40, -5.86, 222.7, -35.4, 197.3])  # LM10
#SGRfc0 = np.array([17.92, 2.56, -6.62, 239.5, -29.6,213.5])  #<<<for D=27 kpc
#SGRfc0 = np.array([17.44, 2.51, -6.50, 237.9, -24.3,209.0])  #<<<for D=26.5 kpc
#LMCfc0 = np.array([-0.57,-41.30,-27.12,-63.9,-213.8,206.6])  # present-day pos/vel of LMC

# Choose present-day phase space positions of Sgr and the LMC
SGRfc0 = agama.getGalactocentricFromGalactic(  5.61*d2r, -14.09*d2r, d0, -2.31 *4.74, 1.94 *4.74, 142)
LMCfc0 = agama.getGalactocentricFromGalactic(280.45*d2r, -32.85*d2r, 50, -0.672*4.74, 1.706*4.74, 260)

cleanup= True
Tbegin = -6.0  # initial evol time [Gyr]
Tbegin_Sgr = -3.0   # time at which Sgr mass is initmass
Tfinal =  0.   # current time
Tstep  = 1./128

# the simple LMC rewinding procedure does not quite match the actual N-body sim,
# and the difference in LMC acceleration is extracted from the sim and stored in this text file
acc_extra = np.loadtxt('acc_extra3.txt')

def simLMC(potmw, Mlmc):
    '''
    compute the past trajectories of MW and LMC under mutual gravitational acceleration,
    and return the combined time-dependent MW-centered potential
    (including acceleration from the non-inertial reference frame) and the LMC trajectory
    '''
    if Mlmc==0: return agama.Potential(type='logarithmic',v0=0), agama.Potential(type='logarithmic',v0=0)
    Rlmc = (Mlmc*232500e-11)**0.6 * 8.5; Rcut = Rlmc*10
    lmcparams=dict(type='spheroid', gamma=1, beta=3, alpha=1,
        scaleradius=Rlmc, mass=Mlmc, outercutoffradius=Rcut)
    potlmc=agama.Potential(lmcparams)
    #print('LMC mass=%.4g, scaleradius=%.4g, Vcirc at 5kpc=%.4g, 15kpc=%.4g' % (Mlmc*232500, Rlmc, (-potlmc.force(5,0,0)[0]*5)**0.5, (-potlmc.force(15,0,0)[0]*15)**0.5))

    def difeq(t, vars):
        x0=vars[0:3]  # MW pos
        v0=vars[3:6]  # MW vel
        x1=vars[6:9]  # LMC pos
        v1=vars[9:12] # LMC vel
        dx=x1-x0
        dr=sum(dx**2)**0.5
        f0=potlmc.force(-dx)
        f1=potmw.force(dx)
        # add extra acceleration due to dynamical friction, deformation of both galaxies, etc.,
        # assuming it scales linearly with the LMC mass
        f0[0]+=np.interp(t, acc_extra[:,0], acc_extra[:,1]) * (Mlmc/645000)
        f0[1]+=np.interp(t, acc_extra[:,0], acc_extra[:,2]) * (Mlmc/645000)
        f0[2]+=np.interp(t, acc_extra[:,0], acc_extra[:,3]) * (Mlmc/645000)
        f1[0]+=np.interp(t, acc_extra[:,0], acc_extra[:,4]) * (Mlmc/645000)
        f1[1]+=np.interp(t, acc_extra[:,0], acc_extra[:,5]) * (Mlmc/645000)
        f1[2]+=np.interp(t, acc_extra[:,0], acc_extra[:,6]) * (Mlmc/645000)
        return np.hstack((v0, f0, v1, f1))

    tgrid = np.linspace(Tbegin, Tfinal, round((Tfinal-Tbegin)/Tstep)+1)
    ic = np.hstack((np.zeros(6), LMCfc0))
    sol = scipy.integrate.solve_ivp(difeq, (Tfinal, Tbegin), ic, t_eval=tgrid[::-1], max_step=Tstep, rtol=1e-12,
        method='LSODA').y.T[::-1]
    rr=np.sum((sol[:,6:9]-sol[:,0:3])**2, axis=1)**0.5
    vr=np.sum((sol[:,6:9]-sol[:,0:3]) * (sol[:,9:12]-sol[:,3:6]), axis=1) / rr
    #print('LMC initial distance: %g, vr: %g' % (rr[0],vr[0]))
    #print sol[0,6:12]-sol[0,0:6]
    #if not (np.all(vr[:-16]<0) or rr[0]>200): raise RuntimeError('LMC is not unbound')
    if not cleanup:
        tgrid = np.linspace(Tbegin, Tfinal+0.125, round((Tfinal+0.125-Tbegin)/Tstep)+1)
        sol = scipy.integrate.solve_ivp(difeq, (Tbegin, Tfinal+0.125), sol[0], t_eval=tgrid, max_step=Tstep, rtol=1e-12,
            method='LSODA').y.T
    mwx = agama.CubicSpline(tgrid, sol[:,0], der=sol[:,3])
    mwy = agama.CubicSpline(tgrid, sol[:,1], der=sol[:,4])
    mwz = agama.CubicSpline(tgrid, sol[:,2], der=sol[:,5])
    accfile = 'accel%g' % os.getpid()
    lmcfile = 'trajlmc%g' % os.getpid()
    trajlmc = np.column_stack((tgrid, sol[:,6:12]-sol[:,0:6]))  # LMC trajectory in the MW-centered reference frame
    np.savetxt(accfile, np.column_stack((tgrid, -mwx(tgrid,2), -mwy(tgrid,2), -mwz(tgrid,2))), '%g')
    np.savetxt(lmcfile, trajlmc, '%g')
    potacc = agama.Potential(type='UniformAcceleration', file=accfile)
    potlmc = agama.Potential(center=lmcfile, **lmcparams)
    if cleanup:
        os.remove(accfile)
        os.remove(lmcfile)
    else:
        np.savetxt(accfile+'_', np.column_stack((-tgrid, -mwx(tgrid,2), -mwy(tgrid,2), -mwz(tgrid,2)))[::-1], '%g')
        np.savetxt(lmcfile+'_', np.column_stack((-tgrid, sol[:,6:12]-sol[:,0:6]))[::-1], '%g')
        np.savetxt('trajboth', np.column_stack((tgrid,sol)), '%g')
    return potlmc, potacc, trajlmc


def simSgr(potmw, potlmc, potacc, decay_rate, finalmass_Sgr=4e8, reflex=True):
    '''
    compute the past trajectory of Sgr in the MW+LMC potential,
    incl.dynamical friction with a prescribed mass(t),
    and return the time-dependent and moving potential of Sgr
    '''

    def difeq(t, xv):
        #return np.hstack((xv[3:6], pot.force(xv[0:3])))
        vel   = xv[3:6]
        vmag  = sum(vel**2)**0.5
        rho   = pot.density(xv[0:3])
        sigma = 120.0
        couLog= 3#.5
        X     = vmag / (sigma * 2**.5)
        drag  = -4*pi * rho / vmag * \
            (scipy.special.erf(X) - 2/pi**.5 * X * np.exp(-X*X)) * couLog * massfnc(t) / vmag**2
        force = pot.force(xv[0:3], t=t) + vel * drag
        return np.hstack((xv[3:6], force))

    #decay_rate = 0.4
    
    finalmass = finalmass_Sgr/massunit
    
    
    if reflex==True:
        pot = agama.Potential(potmw, potlmc, potacc)
    else:
        pot = agama.Potential(potmw, potlmc)
    
    finished = False
    n_orbit = 0
    
    t_node = np.array([Tfinal])
    mass_node = np.array([finalmass])
    
    while finished == False:
        massfnc = lambda t: np.exp(np.interp(t, t_node, np.log(mass_node)))
        tgrid = np.linspace(Tbegin, Tfinal, round((Tfinal-Tbegin)/Tstep)+1)
        sol = scipy.integrate.solve_ivp(difeq, (Tfinal, Tbegin), SGRfc0, t_eval=tgrid[::-1], max_step=Tstep, rtol=1e-12,
            method='LSODA').y.T[::-1]
        trajsgr = np.column_stack((tgrid, sol))
        vr = np.sum(sol[:,0:3]*sol[:,3:6], axis=1) / np.sum(sol[:,0:3]**2, axis=1)**0.5
        iperi = np.where( (vr[:-1]<0) * (vr[1:]>0) )[0]
        tperi = tgrid[iperi]
        
        iapo = np.where( (vr[:-1]>0) * (vr[1:]<0) )[0]
        tapo = tgrid[iapo]
        
        try:
            t_node = np.concatenate(([tperi[-n_orbit-2], tapo[-n_orbit-1]], t_node))
            mass_node = np.concatenate(([finalmass * 10**(decay_rate*(n_orbit + 1)), finalmass * 10**(decay_rate*n_orbit)], mass_node))
        
            n_orbit += 1
        
        except:
            #print('finished')
            finished = True
    
    
    if len(tperi) < 3:
        raise RuntimeError('Not enough pericentre passages for Sgr: '+str(tperi))
    sgrtrjfile = 'trajsgr%g.txt' % os.getpid()
    np.savetxt(sgrtrjfile, trajsgr, '%g')

    #radi0, radi1, radi2, radi3 = 2.0, 1.8, 1.4, 1.2  # plummer scale radius in kpc
    Tstep_pot = 0.05
    tpot = np.linspace(Tbegin, Tfinal, round((Tfinal-Tbegin)/Tstep_pot)+1)
    sgrtime = tpot#t_node#np.array([tperi[-3], tapo[-2], tperi[-2], tapo[-1], tperi[-1], 0.])  
    sgrmass = massfnc(tpot)#mass_node#np.array([mass0, mass1, mass1, mass2, mass2, mass3])
    sgrrad  = np.maximum(1.4 * (sgrmass*massunit / 4e8)**(1/3), 1.4)#np.array([radi0, radi1, radi1, radi2, radi2, radi3])
    
    mass_Sgr = massfnc.__call__(tgrid)*232500
    # mass_pot_func = massfnc#interp1d(sgrtime, sgrmass, bounds_error=False, fill_value=(sgrmass[0], sgrmass[0]))
    # mass_pot = mass_pot_func.__call__(tgrid)*232500
    
    sgrpotfile = 'potsgr%g' % os.getpid()
    fo = open(sgrpotfile+'.pot', 'w')
    fo.write('[Potential]\ntype=Evolving\ncenter=%s\nlinearinterp=True\nTimestamps\n' % sgrtrjfile)
    for i in range(len(sgrtime)):
        fp = open(sgrpotfile+'%i.pot'%i, 'w')
        fp.write('[Potential]\ntype=Dehnen\ngamma=1\nscaleradius=%g\nmass=%g\n' % (sgrrad[i], sgrmass[i]))
        fp.close()
        fo.write('%g %s%i.pot\n' % (sgrtime[i], sgrpotfile, i))
    fo.close()
    potsgr = agama.Potential(sgrpotfile+'.pot')
    for i in range(len(sgrtime)):
        os.remove(sgrpotfile+'%i.pot' % i)
    os.remove(sgrpotfile+'.pot')
    if cleanup: os.remove(sgrtrjfile)
    
    # Include acceleration due to reflex motion from Sgr
    accsgr = -potsgr.force(np.zeros((len(trajsgr[:,0]), 3)), t=trajsgr[:,0])
    
    accsgrfile = 'accelsgr%g' % os.getpid()
    np.savetxt(accsgrfile, np.column_stack((tgrid, accsgr)), '%g')
    potaccsgr = agama.Potential(type='UniformAcceleration', file=accsgrfile)
    if cleanup:
        os.remove(accsgrfile)
    
    potsgr = agama.Potential(potsgr, potaccsgr)
    
    return potsgr, trajsgr, tgrid, mass_Sgr



potbulge = dict(Type='Spheroid', mass=51600, scaleRadius=0.2, outerCutoffRadius=1.8, gamma=0.0, beta=1.8)
potdisk  = dict(Type='Disk', SurfaceDensity=3803.5, scaleRadius=3.0, scaleHeight=-0.4)
potbary  = agama.Potential(potbulge, potdisk)

def makeMW(scaleRadius, gamma, beta, alpha, q_in, q_out, p_out, alpha_q, beta_q, gamma_q, shapeRadius):
    '''
    construct the MW potential with prescribed properties
    '''
    params = dict(Type='Spheroid', outerCutoffRadius=200, cutoffStrength=2,
        gamma=gamma, beta=beta, alpha=alpha, scaleRadius=scaleRadius, densityNorm=1)
    pot0 = agama.Potential(**params)
    r0   = 8.0
    vcr0 = (-r0*pot0.force(r0,0,0)[0])**0.5
    params['densityNorm'] = (155.0/vcr0)**2
    dens = agama.Density(params)
    ca,sa= np.cos(alpha_q), np.sin(alpha_q)
    cg,sg= np.cos(gamma_q), np.sin(gamma_q)
    def mydens(x):
        r2 = np.sum(x**2) #, axis=1)
        ch = 1 / (1 + r2 / shapeRadius**2)  # changeover fnc: 1 at small radii, 0 at large radii
        q  = q_out + (q_in-q_out) * ch
        p  = p_out + ( 1  -p_out) * ch
        cb,sb= np.cos(beta_q*(1-ch)), np.sin(beta_q*(1-ch))
        xp, yp, zp = np.array( [
            [ ca*cg - sa*cb*sg,  sa*cg + ca*cb*sg, sb*sg],
            [-ca*sg - sa*cb*cg, -sa*sg + ca*cb*cg, sb*cg],
            [         sa*sb   ,         -ca*sb   , cb   ] ]).dot(x.T)
        rt = (xp**2 + (yp/p)**2 + (zp/q)**2)**0.5 * (p*q)**(1./3)
        return dens.density(np.column_stack((rt, rt*0, rt*0)))
    lmax = 4
    pothalo = agama.Potential(type='multipole', density=mydens,
        lmax=lmax, mmax=lmax, symmetry='r', rmin=1e-2, rmax=1e3)
    potmw = agama.Potential(potbary[0], agama.Potential(type='multipole',
        density=agama.Density(pothalo, potbary[1]), lmax=16, mmax=lmax, symmetry='r', rmin=1e-2, rmax=1e3))
    
    return potmw


# Combine potentials

def createPotential(params, decay_rate, finalmass_Sgr=4e8, reflex=True):
    Mlmc  = params[-1]*1e10
    potmw = makeMW(*params[:-1])
    potlmc, potacc, trajlmc = simLMC(potmw, Mlmc/232500)
    if reflex==True:
        potsgr, trajsgr, tgrid, mass_Sgr = simSgr(potmw, potlmc, potacc, decay_rate, finalmass_Sgr=finalmass_Sgr, reflex=True)
        pottot = agama.Potential(potmw, potlmc, potacc, potsgr)
    else:
        potsgr, trajsgr, tgrid, mass_Sgr = simSgr(potmw, potlmc, potacc, decay_rate, finalmass_Sgr=finalmass_Sgr, reflex=False)
        pottot = agama.Potential(potmw, potlmc, potsgr)
    
    return pottot, trajlmc, trajsgr, tgrid, mass_Sgr

params = [ 7.36, 1.20, 2.40, 2.40, 0.64, 1.45, 1.37, -25*d2r, 0.0, 0.0, 54.0, 15.0]  # params from Tango for Three

# Repeat with multiple Sgr decay rates
decay_rates = np.array([0.2, 0.4, 0.6, 0.7, 1.])



fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6,8))
plt.subplots_adjust(hspace=0.)

for i in range(len(decay_rates)):
    
    decay_rate = decay_rates[i]
    
    
    # Calculate potential, trajectories, mass of Sgr
    pottot, trajlmc, trajsgr, t, mass_Sgr = createPotential(params, decay_rate, finalmass_Sgr=4e8, reflex=True)
    
    
    # Galactocentric radii of the LMC and Sgr
    r_lmc = np.linalg.norm(trajlmc[:,1:4], axis=1)
    r_Sgr = np.linalg.norm(trajsgr[:,1:4], axis=1)
    
    ax1 = axs[1]
    ax1.plot(t, r_Sgr, label=r'$r$')
    #ax1.plot(t, r_lmc, label=r'$r$, LMC')
    
    ax0 = axs[0]
    ax0.plot(t, mass_Sgr, label=r'$\delta=$ '+str(decay_rate))
    ax0.set_yscale('log')
    
    ax1.set_xlabel(r'$t\;[\mathrm{Gyr}]$', fontsize=16)
    ax1.set_ylabel(r'$r_\mathrm{Sgr}(t)\;[\mathrm{kpc}]$', fontsize=16)
    ax0.set_ylabel(r'$M_\mathrm{Sgr}(t)\;[M_\odot]$', fontsize=16)
    

ax1.text(-1.3, 150, 'LMC', rotation=-72, c='grey', fontsize=16, weight='bold')

ax1.plot(t, r_lmc, c='grey', ls='--')
ax1.set_xlim(-6.,0.)
ax1.set_ylim(0, 199.)
ax0.set_ylim(3e8, 1e11)

for ax in axs.flat:
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in', which='both', labelsize=14)

ax0.legend(fontsize=14)

plt.suptitle('Mass decay profiles and orbits for\ndifferent models of Sgr', fontsize=18)
plt.show()
plt.close()
