#!/usr/bin/env python

import os, numpy as np, scipy.integrate, scipy.special, agama, matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import G
import astropy.coordinates as coord
from galpy.orbit import Orbit

plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

np.random.seed(0)


pi     = np.pi
massunit = (10**6 * (u.kpc/u.m) / (u.Msun / u.kg) / G.value).to(u.dimensionless_unscaled)
d2r    = pi/180
d0     = 27.0  # distance in kpc
#SGRfc0 = np.array([19.78, 2.40, -5.86, 222.7, -35.4, 197.3])  # LM10
#SGRfc0 = np.array([17.92, 2.56, -6.62, 239.5, -29.6,213.5])  #<<<for D=27 kpc
#SGRfc0 = np.array([17.44, 2.51, -6.50, 237.9, -24.3,209.0])  #<<<for D=26.5 kpc
#LMCfc0 = np.array([-0.57,-41.30,-27.12,-63.9,-213.8,206.6])  # present-day pos/vel of LMC
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
    incl.dynamical friction with a prescribed mass(t) calibrated from N-body sims,
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
    
    return potsgr, trajsgr


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
    
    #potmw = pot_McMillan17
    #potmw = pot_MW2014
    #potmw = pot_PriceWhelan19
    
    #print('MW Vcirc at 8kpc=%g, at 20kpc=%g, at 50kpc=%g' %
        #((-potmw.force(8,0,0)[0]*8)**0.5, (-potmw.force(20,0,0)[0]*20)**0.5, (-potmw.force(50,0,0)[0]*50)**0.5))
    return potmw

def createPotential(params, decay_rate, finalmass_Sgr=4e8, reflex=True):
    Mlmc  = params[-1]*1e10
    potmw = makeMW(*params[:-1])
    potlmc, potacc, trajlmc = simLMC(potmw, Mlmc/232500)
    if reflex==True:
        potsgr, trajsgr = simSgr(potmw, potlmc, potacc, decay_rate, finalmass_Sgr=finalmass_Sgr, reflex=True)
        pottot = agama.Potential(potmw, potlmc, potacc, potsgr)
    else:
        potsgr, trajsgr = simSgr(potmw, potlmc, potacc, decay_rate, finalmass_Sgr=finalmass_Sgr, reflex=False)
        pottot = agama.Potential(potmw, potlmc, potsgr)
    
    return pottot, trajlmc, trajsgr

params = [ 7.36, 1.20, 2.40, 2.40, 0.64, 1.45, 1.37, -25*d2r, 0.0, 0.0, 54.0, 15.0]  # params from the paper



# Define function to generate stream similar to GD-1
def lagrange_cloud_strip(fc):
    
    trajsize = int(np.round((Tfinal-Tbegin)/Tstep)+1)
    
    # Integrate orbit of progenitor
    prog_orbit = agama.orbit(ic=fc, potential=pottot, timestart=Tfinal, time=Tbegin-Tfinal, trajsize=trajsize)
    
    times = np.flip(prog_orbit[0])
    trajsize = len(times)
    prog_traj = np.flip(prog_orbit[1], axis=0)
    prog_traj_array = np.hstack((np.array([times]).T, prog_traj))
    
    #print(prog_traj_array.shape)
    
    np.savetxt('prog_traj_'+str(os.getpid())+'.txt', prog_traj_array[:, 0:4])
    
    # Construct potential of progenitor and combine with total potential
    potprog = agama.Potential(type='Plummer', mass=M_s/232500, scaleRadius=a_s, center='prog_traj_'+str(os.getpid())+'.txt')
    pottot_prog = agama.Potential(pottot, potprog)
    
    # Calculate angular speed of progenitor (in units of km/s/kpc)
    r_prog = np.linalg.norm(prog_traj[:,0:3], axis=1)
    L_prog = np.linalg.norm(np.cross(prog_traj[:,0:3], prog_traj[:,3:6]), axis=1)
    Omega_prog = L_prog / r_prog**2


    # Calculate 2nd derivative of external potential
    force, deriv = pottot.forceDeriv(prog_traj[:,0:3], t=times)
    
    # Calculate Hessian matrix and 2nd derivative of potential
    Hessian = np.zeros((trajsize, 3, 3))
    Hessian[:, 0, :] = -np.array([deriv[:, 0], deriv[:, 3], deriv[:, 5]]).T
    Hessian[:, 1, :] = -np.array([deriv[:, 3], deriv[:, 1], deriv[:, 4]]).T
    Hessian[:, 2, :] = -np.array([deriv[:, 5], deriv[:, 4], deriv[:, 2]]).T
    
    r_hat = prog_traj[:, 0:3] / r_prog[:,None]
    d2Phi_d2r = np.einsum('ki,kij,kj->k', r_hat, Hessian, r_hat)
    
    
    # Calculate tidal radius
    r_t_max = 1 # Cap on tidal radius (used to replace nans).
    r_t = np.nan_to_num( (((G * M_s / (Omega_prog**2 - d2Phi_d2r))*u.Msun/(u.km/u.s/u.kpc)**2)**(1/3)).to(u.kpc).value, nan=r_t_max)
    r_t = np.minimum(r_t, r_t_max)    # Limit r_t to chosen value
    
    #print('1 = ', time.process_time()-start_rt)
    
    # Calculate positions and velocities of points of particle release
    source_coords_in = prog_traj[:, 0:3] - lambda_source * r_t[:, None] * r_hat
    source_coords_out = prog_traj[:, 0:3] + lambda_source * r_t[:, None] * r_hat
    
    prog_velocity_r = np.sum(prog_traj[:, 3:6]*r_hat, axis=1)
    prog_velocity_tan = prog_traj[:, 3:6] - prog_velocity_r[:, None] * r_hat
    
    source_velocity_tan_in = prog_velocity_tan * (1 - 0.5 * r_t / r_prog)[:, None]
    source_velocity_tan_out = prog_velocity_tan * (1 + 0.5 * r_t / r_prog)[:, None]
    
    source_velocity_in = prog_velocity_r[:, None] * r_hat + source_velocity_tan_in
    source_velocity_out = prog_velocity_r[:, None] * r_hat + source_velocity_tan_out
    
    #print('2 = ', time.process_time()-start_prog)
    
    source_coords = np.zeros((len(source_coords_in)*2, 3))
    source_coords[::2] = source_coords_in
    source_coords[1::2] = source_coords_out
    
    source_velocity = np.zeros((len(source_velocity_in)*2, 3))
    source_velocity[::2] = source_velocity_in
    source_velocity[1::2] = source_velocity_out
    
    ic_source_coords = np.repeat(source_coords, strip_rate/2, axis=0)
    ic_source_velocities = np.repeat(source_velocity, strip_rate/2, axis=0)
    
    np.random.seed(0)
    ic = np.hstack((ic_source_coords, ic_source_velocities)) + np.hstack((np.zeros((trajsize*strip_rate, 3)), np.random.randn(trajsize*strip_rate, 3)*sigma_s))
    
    start_times = np.repeat(times, strip_rate)
    
    trajsizes = np.repeat(np.arange(trajsize, 0, -1), strip_rate)
    
    
    # Integrate orbits
    result = agama.orbit(ic=ic, potential=pottot_prog, timestart=start_times, time=Tfinal-start_times, trajsize=trajsizes)[:-strip_rate]

    
    trajs = np.zeros((trajsize, len(result), 6))
    for part_index in range(len(result)):
        trajsize_part = trajsizes[part_index]
        traj_part = trajs[:, part_index, :]
        
        traj_part[0:-trajsize_part, :] = prog_traj[0:-trajsize_part]
        traj_part[-trajsize_part:, :] = result[:, 1][part_index]

    
    return trajs, prog_traj




# Define function to calculate streamtrack
def str_trajs_to_track(trajs):
    str_coords = trajs[:,:,0:3].swapaxes(0,1)
    str_velocities = trajs[:,:,3:6].swapaxes(0,1)

    # Create SkyCoord object for stream
    str_track_0_galcen = coord.SkyCoord(x=str_coords[:,-1,0]*u.kpc, y=str_coords[:,-1,1]*u.kpc, z=str_coords[:,-1,2]*u.kpc, v_x=str_velocities[:,-1,0]*u.km/u.s, v_y=str_velocities[:,-1,1]*u.km/u.s, v_z=str_velocities[:,-1,2]*u.km/u.s, frame='galactocentric')
    str_track_0 = str_track_0_galcen.transform_to('gd1koposov10')
    
    return str_coords, str_velocities, str_track_0


# Function to return galactocentric longitude and latitude in frame following stream
def str_coord_transform(trajs, prog_traj):
    L_prog = np.cross(prog_traj[:,0:3], prog_traj[:,3:6])
    
    r_prog = np.linalg.norm(prog_traj[:,0:3], axis=1)
    
    z_prime_hat = L_prog / np.linalg.norm(L_prog, axis=1)[:, None]
    x_prime_hat = prog_traj[:,0:3] / r_prog[:,None]
    y_prime_hat = np.cross(z_prime_hat, x_prime_hat)
    
    M = np.stack((x_prime_hat, y_prime_hat, z_prime_hat), axis=1)
    
    str_part_coords_prime = np.einsum('lij,lkj->lki', M, trajs[:,:,0:3])
    
    str_part_r_prime = np.linalg.norm(str_part_coords_prime, axis=2)
    
    theta = 90 - np.arccos(str_part_coords_prime[:,:,2] / str_part_r_prime) * 180/np.pi
    
    phi = np.arctan2(str_part_coords_prime[:,:,1], str_part_coords_prime[:,:,0]) * 180/np.pi
    
    return phi, theta
    



# Mass of stream progenitor
M_s = 2e4

# Scale radius
a_s = 2e-3

# Velocity dispersion of released particles
sigma_s = 0.5

# Position of release compared to Lagrange point
lambda_source = 1.2

# Number of particles stripped per timestep
# Needs to be even
strip_rate = 2


finalmasses_Sgr = np.array([4e8, 4e8, 4e8, 4e8, 4e8])#np.array([4e8, 4e8, 4e8, 4e8, 4e8])#, 4e8])#, 4e8])
decay_rates = np.array([0.2, 0.4, 0.6, 0.7, 1])#np.array([0.2, 0.4, 0.6, 0.7, 1.])


N_masses = len(decay_rates)



# Load GC ICs (DR2)
o = Orbit.from_name('MW globular clusters')
fc_galcen_GCs = np.array([-o.x(), o.y(), o.z(), -o.vx(), o.vy(), o.vz()]).T

GC_names = np.array(o.name, dtype='object')



# Calculate potential and trajectories of LMC and Sgr
pottot, trajlmc, trajsgr = createPotential(params, decay_rate=0, finalmass_Sgr=0, reflex=True)

trajsize = int(np.round((Tfinal-Tbegin)/Tstep)+1)

# Integrate orbit of progenitor
prog_orbits = agama.orbit(ic=fc_galcen_GCs, potential=pottot, timestart=Tfinal, time=Tbegin-Tfinal, trajsize=trajsize)

prog_trajs = prog_orbits[:, 1]

peri_apo_array = np.zeros((len(prog_trajs), 2))

for i in range(len(prog_trajs)):
    posvel = prog_trajs[i]
    r = np.linalg.norm(posvel[:, 0:3], axis=1)
    
    try:
        peri_apo_array[i, 0] = np.min(r)
        peri_apo_array[i, 1] = np.max(r)
    
    except:
        peri_apo_array[i, 0] = 0.
        peri_apo_array[i, 1] = 0.

# Select minimum and maximum radii of orbits for inclusion
r_min = 10.
r_max = 27.

# Cut GCs based on radial orbital ranges
GC_cut = np.where((peri_apo_array[:, 0] > r_min) & (peri_apo_array[:, 1] < r_max))[0]

N_GC = len(GC_cut)

fc_galcen_GCs = fc_galcen_GCs[GC_cut]

GC_names_cut = GC_names[GC_cut]

# Number of stream initialisations
init_str_number = N_GC




str_coords_array = []
str_velocities_array = []
str_coords_cut_array = []
str_velocities_cut_array = []
prog_coords_array = []
sat_coords_array = []
sat_velocities_array = []
N_str_part_array = []
lon_array = np.empty((N_masses, N_GC), dtype='object')
lat_array = np.empty((N_masses, N_GC), dtype='object')


        
for i in range(len(decay_rates)):
    
    decay_rate = decay_rates[i]
    finalmass_Sgr = finalmasses_Sgr[i]
    
    #fc_initguess = np.load(data_path+'fc_optimized/lagrange_cloud_strip/triaxial/0_0_6.npy')
    
    pottot, trajlmc, trajsgr = createPotential(params, decay_rate = decay_rate, finalmass_Sgr=finalmass_Sgr, reflex=True)

    for init_str_count in range(init_str_number):

        fc = fc_galcen_GCs[init_str_count]
        
        # Calculate trajectories of stream and Sgr
        trajs, trajprog = lagrange_cloud_strip(fc)
        
        lon, lat = str_coord_transform(trajs, trajprog)
        
        lon_array[i, init_str_count] = lon
        lat_array[i, init_str_count] = lat
        
        # Convert to stream track
        str_coords, str_velocities, str_track_0 = str_trajs_to_track(trajs)
        
        str_coords_array.append(str_coords)
        str_velocities_array.append(str_velocities)
        prog_coords_array.append(trajprog[:,0:3])
        
        N_str_part = len(str_coords)
        N_str_part_array.append(N_str_part)
    
        
        # Store time
        trajsize = len(trajs)
        ts = np.linspace(Tbegin, Tfinal, trajsize)
        
    
        # Store coords and velocities of Sgr
        sat_coords = trajsgr[:,1:4]
        sat_velocities = trajsgr[:,4:7]
        
        sat_coords_array.append(sat_coords)
        sat_velocities_array.append(sat_velocities)

        start_times = np.repeat(ts, strip_rate)[:-strip_rate]


# Plot GC streams at present day (This must be modified if a different potential is used)
fig, axs = plt.subplots(len(decay_rates), 3, sharex=True, sharey=True, figsize=(12,6))
plt.subplots_adjust(hspace=0, wspace=0)


titles = ['Pal 1', 'Pal 5', 'BH 176']

for i in range(len(decay_rates)):
    for j in range(3):
        plot = axs[i,j].scatter(lon_array[i,j-1][-1], lat_array[i,j-1][-1], marker='.', s=1, c=start_times)
        
        if i == 0:
            axs[i,j].set_title(titles[j], fontsize=18)
        
        if i == len(decay_rates)-1:
            axs[i,j].set_xlabel(r'$\psi_1\;[^\circ]$', fontsize=16)
        
        if j == 0:
            axs[i,j].set_ylabel(r'$\delta=$ '+str(decay_rates[i])+'\n'+r'$\psi_2\;[^\circ]$', fontsize=16)
        
for ax in axs.flat:
    ax.set_xlim(90, -90)
    ax.set_ylim(-19.99, 19.99)   
    ax.tick_params(bottom=True, top=True, left=True, right=True, direction='in', labelsize=14)
    ax.set_rasterization_zorder(10)     


cb = fig.colorbar(plot, ax=axs, location='right', aspect=50, pad=0.01)
cb.set_label(r'Time of stripping [Gyr]', fontsize=16)

plt.show()
plt.close()
