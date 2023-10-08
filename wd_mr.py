#%%
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as c
import astropy.units as u

#%%
def M_rho_deriv_degen(r_in_rsun, M_lrho, Z_on_A=0.5):
    """Find the Mass and density derivatives with respect to radius 
    for solving a degenerate star equation of state problem. 
    We have log(rho) as an input to help with numerics, as rho can never be 
    negative.
    
    Parameters
    ----------
    M_lrho: numpy array-like, including M in solar units, rho in g/cm^3.
    
    Returns
    -------
    derivatives: Derivatives of M in solar units, log(rho in g/cm^3), with respect 
        to r_in_rsun, as a numpy array-like variable.
    """
    M_in_Msun, lrho = M_lrho
    
    #Lets create a variable rho in real units
    rho = np.exp(lrho)*u.g/u.cm**3
    
    #Mass continuity equation! Convert to dimensionless units
    dM_in_Msundr = float(4*np.pi*r_in_rsun**2*rho * c.R_sun**3/c.M_sun)
    
    #Fermi momentum - a useful intermediate quantity in the semi-relativistic
    #equation of state
    p_F = ( c.h/2*(3*rho*Z_on_A/np.pi/u.u)**(1/3) ).cgs
    
    #Derivative of pressure with respect to density. 
    #dPdrho - should have units of (velocity)^2
    dPdrho = ( p_F**2*c.c/3/u.u/np.sqrt(p_F**2 + c.m_e**2*c.c**2) * Z_on_A ).cgs
    
    #drho/dr
    if r_in_rsun==0:
        dlrhodr = 0
    else:
        dlrhodr = -(c.G*M_in_Msun*c.M_sun/(r_in_rsun*c.R_sun)**2 / dPdrho)\
            .to(1/c.R_sun).value
    
    return np.array([dM_in_Msundr, dlrhodr])
    
#%%
#The following function may be slightly confusing for people who aren't python or 
#object oriented programming experts, as variables (properties) are added to a function. 
#We can actually always add additional properties to functions, as all variables and
#functions in python are objects.
def near_surface(r_in_rsun, M_lrho):
    """Determine a surface condition by the surface becoming too cool. In practice, 
    our adiabatic approximation is likely to break before this!"""
    return M_lrho[1] - np.log(1e-2)
near_surface.terminal = True
near_surface.direction = -1

#%%

try:
    from scipy.integrate import solve_ivp
    solve_ivp_available = True
except:
    from scipy.integrate import ode
    solve_ivp_available = False
        
def wd_structure(rho_c):
    """Assuming a fully degenerate equation of state, compute the interior structure
    of a white dwarf, including its outer radius and mass.
    
    Parameters
    ----------
    rho_c: Central density, including units from astropy.units
    """
    #Start the problem at the white dwarf center.
    y0 = [0, np.log(rho_c.to(u.g/u.cm**3).value)]
    
    #Don't go past 1 R_sun!
    rspan = [0,1] 
    
    if solve_ivp_available:
        #Solve the initial value problem!
        result = solve_ivp(M_rho_deriv_degen, [0,100], y0, events=[near_surface], method='RK23') 
    
        #Extract the results
        r_in_rsun = result.t
        M_in_Msun = result.y[0]
        rho = np.exp(result.y[1])*u.g/u.cm**3
    else:
        #A little more tricky if we have an old version of scipy. Things are a little more
        #manual... Set the maximum step and initial step to be equal to 0.01 M_sun equivalent. 
        #As the integration continues, this will be set to a factor of e in density.
        dr_max = ((0.01*c.M_sun/rho_c/(4/3*np.pi))**(1/3)).to(u.R_sun).value
        #The default integrator... dopri5 works just as well but seems marginally slower.
        integrator = ode(M_rho_deriv_degen).set_integrator('vode', first_step=1e-5, max_step=dr_max) 
        integrator.set_initial_value(y0, 0)
        drho_max = 1.0
        dr = dr_max
        rs = [0]
        M_lrhos = [np.array(y0)]
        while (near_surface(rs[-1],M_lrhos[-1])>0) and integrator.successful():
            rs.append(integrator.t + dr)
            M_lrhos.append(integrator.integrate(integrator.t+dr)) 
            #Lets see if dr has to be changed. We do this manually.
            dr = np.min([-drho_max/M_rho_deriv_degen(rs[-1], M_lrhos[-1])[1], dr_max])
        M_lrhos = np.array(M_lrhos)
        r_in_rsun = np.array(rs)
        M_in_Msun = M_lrhos[:,0]
        rho = np.exp(M_lrhos[:,1])*u.g/u.cm**3
    return r_in_rsun, M_in_Msun, rho
    
    
    

# %%
rho_c = 1e7*u.g/u.cm**3

plt.plot(wd_structure(rho_c)[0], wd_structure(rho_c)[2])
plt.show()

#%%

# Convert density to number density using n_e = 



# %%
