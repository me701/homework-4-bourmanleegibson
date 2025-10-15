import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix
from starter_code import construct_system

def Cool_can(N_r, t_initial, t_final, t_step, T0=None):
    """
    Simulates transient cooling of "beverage" can using finite volume discretization 
    in cylindrical coordinates and solves the heat equation using implicit time integration.

    The simulation assumes no internal heat generation and convection-based boundary 
    conditions on all surfaces of the can. The domain is discretized radially and axially, 
    and the resulting system of ODEs is solved using `scipy.integrate.solve_ivp`.

    Parameters:
    ----------
    N_r : int
        Number of discretization cells in the radial direction. Axial discretization is calculated 
        assuming square cells given the can dimensions
    t_initial : float
        Initial time for the simulation [seconds].
    t_final : float
        Final time for the simulation [seconds].
    t_step : float
        Time step for evaluation [seconds].
    T0 : int, float, or np.ndarray, optional
        Initial temperature [°C]. Can be a scalar (uniform temperature) or a 1D array matching
        the number of spatial cells (N_z * N_r) for non-uniform initial conditions. 
        If None, a default of 35°C is used.

    Returns:
    -------
    t_initial : float
        The initial simulation time [seconds]. (Useful for autimated tests defined elswhere)
    t_final : float
        The final simulation time [seconds]. (Useful for autimated tests defined elswhere)
    T_bar : np.ndarray
        Spatially averaged temperature of the can over time [°C], accounting for cylindrical 
        volume weighting.
    T_map : np.ndarray
        3D array of shape (time_steps, N_z, N_r) representing the temperature distribution 
        over time within the can.

    Notes:
    -----
    - Material properties (thermal conductivity, specific heat, viscosity) are hardcoded 
      for water at 10°C.
    - The heat transfer coefficient is calculated from a Nusselt number correlation assuming 
      external convection. Correlation from:
      K. M. Becker, "Measurements of convective heat transfer from a horizontal cylinder 
      rotating in a tank of water," Int. J. Heat Mass Transf., vol. 6, no. 12, pp. 1053–1062, 
      1963, doi: 10.1016/0017-9310(63)90006-1
    - The function internally constructs the discretized spatial system using a call to 
      `construct_system()`, which must be defined elsewhere.
    - Boundary conditions are Robin-type (convection) on all external surfaces.
    """
    def T_prime(t, T, A, b, rho, c_p):
        return (b - A@T)/rho/c_p
    
    h = 0.12
    r = 0.03
    d = 2*r
    
    kfun = lambda r, z : 0.58 #[W m-1 K-1] @ 10 C not assuming effective conductivity
    qppp = lambda r, z : 0 # no heat gen in can
    
    k = kfun(0,0)
    mu = 17.916e-4 # {Pa*s}
    rho = 1000 # ~{kg/m^3}
    w = 2*np.pi*d # {m/s}
    c_p = 4.18 * 1000 # {J/kg*K}
    
    Re = rho * w * d / mu
    Pr = c_p * mu / k
    Nu = 0.133 * Re**(2/3) * Pr**(1/3)
    HTC = Nu*k/d
    T_oo = 0 # {C}
    
    #set Boundary Conditions
    BC_z = ('robin', HTC, T_oo)
    BC_r_top = ('robin', HTC, T_oo)
    BC_r_bottom = ('robin', HTC, T_oo)

    #discritization
    #N_r = 30
    N_z = int((h/r)*N_r) #equal delta in h and r directions

    r_edge = np.linspace(0, r, N_r+1)
    z_edge = np.linspace(0, h, N_z+1)

    r_center = (r_edge[:-1]+r_edge[1:])/2
    z_center = z_edge[:-1]+z_edge[1:]/2

    A, b = construct_system(r_center, z_center, kfun, qppp,
                        BC_z, BC_r_top, BC_r_bottom)
    #Use sparse matricies for calculations to speed up solving
    A_spc = csr_matrix(A)
    
    #Logic to set initial temperature using assummed value, given value, or map (usded if extending the soluton time)
    try:
        if T0 == None: T0 = 35
    except: pass 
    if type(T0) == int: T0 = T0 * np.ones(A.shape[0]) # {deg C}
    elif type(T0) == np.ndarray: T0 = T0.flatten()
    else: raise "Missing Initial Temperature Map"

    t_eval = np.arange(t_initial, t_final+t_step, t_step, dtype=np.float64) # Evaluate at times listed (t_final is last in the list)
    
    sol = solve_ivp(T_prime, [t_initial, t_final+t_step], T0, 
                    args=(A_spc, b, rho, c_p),
                    t_eval=t_eval,
                    method='BDF')

    # Properly weight the cells as a pie slice for average temperature calculation
    r_weights = np.array([r_edge[-(i+1)]**2 - r_edge[-(i+2)]**2 for i in range(N_r)])
    r_weights_norm = np.flip(r_weights/sum(r_weights))
    # Calculate T_bar
    T_bar = np.zeros_like(t_eval)
    for i in range(len(T_bar)):
        T_bar[i] = sum(sum(sol.y[:,i].reshape((N_z, N_r))*r_weights_norm))/N_z

    return t_initial, t_final, T_bar, sol.y.T.reshape((-1, N_z, N_r))


def valCross(ind,dep,ths):
    """
    Interpolates to find the value of `dep` corresponding to a threshold `ths` 
    in the independent variable `ind`, assuming `ind` is in descending order.

    The function searches for the first point where `ind` drops below `ths`, 
    and linearly interpolates between surrounding points to estimate `dep(ths)`.

    Parameters
    ----------
    ind : array-like
        Independent variable values (must be in descending order).
    dep : array-like
        Dependent variable values corresponding to `ind`.
    ths : float
        Threshold value to locate within `ind`.

    Returns
    -------
    float
        Interpolated value of `dep` at the threshold `ths`.

    Raises
    ------
    ValueError
        If `ths` is not crossed by any value in `ind`.

    """
    for i in range(len(ind)):
        if ind[i] < ths:
            return (dep[i-1]-dep[i])/(ind[i-1]-ind[i])*(ths-ind[i]) + dep[i]
        elif ind[i] == ths:
            return dep[i]
    raise ValueError("Threshold was not crossed by ind.")


def genPlots(vals, errs, N_r):
    """
    Generates a two-panel plot to visualize the results of a convergence analysis.

    The function creates:
      - A top subplot showing the time at which a temperature threshold is reached 
        as a function of the number of mesh points (N_r).
      - A bottom subplot showing the corresponding relative errors on a logarithmic scale.

    Parameters
    ----------
    vals : array-like
        Time values (e.g., when a temperature threshold is crossed), typically from 
        a convergence test.
    errs : array-like
        Relative errors corresponding to the values in `vals`.
    N_r : array-like
        Number of mesh points used in the radial direction for each test case.

    Returns
    -------
    None
        This function only displays the plot and does not return any values.
    """
    # --- 1. Plot temperature and relative error on shared x-axis ---
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6), gridspec_kw={'height_ratios': [2, 1]})
    
    # Top plot: Temperature values
    ax1.plot(N_r, vals, marker='o', color='steelblue', label='Temperature')
    ax1.set_ylabel("Time (s)")
    ax1.set_title("Convergence Analysis")
    ax1.grid(True)
    ax1.legend()

    # Bottom plot: Relative errors
    ax2.semilogy(N_r, errs, marker='s', color='darkorange', label='Relative Error')
    ax2.set_xlabel("Number of Mesh Points (N_r)")
    ax2.set_ylabel("Relative Error")
    ax2.grid(True)
    ax2.legend()


    plt.tight_layout()
    plt.show()


def convergenceTest(N_r_initial=2, t_initial=0, t_final=900, t_step=20, diff_percent=0.1, max_itr=20, gen_plots=False):
    """
    Performs a convergence test to determine the required mesh density (N_r)
    for accurately predicting when the average temperature of the "beverage" can
    reaches a specified threshold.

    The test iteratively increases the number of radial mesh points until the 
    change in time (at which T_bar crosses T_thres) between successive refinements is within
    a specified relative difference.

    If the temperature threshold is not reached within the initial simulation 
    time, `t_final` is extended automatically with the final condition set as the new
    initial temperature map (now spacially dependent)

    Parameters
    ----------
    N_r_initial : int, optional
        Initial number of mesh points in the radial direction. Default is 2.
    t_initial : float, optional
        Start time of the simulation [seconds]. Default is 0.
    t_final : float, optional
        Initial end time of the simulation [seconds]. Default is 900.
    t_step : float, optional
        Time step for evaluating the solution [seconds]. Default is 20.
    diff_percent : float, optional
        Maximum allowable relative difference (%) in threshold-crossing time
        between iterations for convergence. Default is 0.1.
    max_itr : int, optional
        Maximum number of refinement iterations. Default is 20.
    gen_plots : bool, optional
        If True, generates convergence plots of threshold-crossing time and 
        relative error vs. mesh density. Default is False.

    Returns
    -------
    t_start : float
        The initial simulation time [seconds] given for the final call of Cool_can(). Not necessarily the default initial
        time if the simulation time is extended to reach a solution.
    t_end : float
        The final simulation time [seconds] of the converged answer.
    T_bar : np.ndarray
        Spatially averaged temperature of the can over time [°C], accounting for cylindrical 
        volume weighting.
    T_map : np.ndarray
        3D array of shape (time_steps, N_z, N_r), representing the temperature 
        distribution in the can over time at the converged mesh resolution.

    Raises
    ------
    ValueError
        If the convergence condition is not met within `max_itr` iterations.

    Notes
    -----
    - The function assumes that the can's average temperature is used for 
      determining when the threshold is crossed.
    - Uses `Cool_can()` for simulation and `valCross()` for interpolation.
    - Threshold temperature is hardcoded to 10°C per problem statement.
    """
    
    T_thres = 10 #deg C set by problem statement
    extend_mesh = 2 # number of mesh elements is increased by value after each iteration
    t_extend = 200 #if the threshold temp is not reached, t_final is extended and T is recalculated
    N_r = [N_r_initial]

    rel_diff = [] #
    t_cross = [1] #Time at which the temperature threshold is crossed
    
    for i in range(max_itr):
        t_start, t_end, T_bar, T_map = Cool_can(N_r[i], t_initial,
                                                         t_final, t_step)
        while T_bar[-1] > T_thres:
            print("Iteration {}: threshold not crossed...increasing time to {}s".format(i,t_end+t_extend))
            t_start, t_end, T_bar, T_map = Cool_can(N_r[i], t_end, 
                                                             t_end+t_extend, t_step,
                                                             T0=T_map[-1])
        t_eval = np.arange(t_start, t_end+t_step, t_step, dtype=float)
        t_cross.append(valCross(T_bar,t_eval,T_thres))
        rel_diff.append((abs(t_cross[i+1]-t_cross[i])/t_cross[i]*100))
        if rel_diff[i] < diff_percent:
            print("System converged on step {} at N_r={}".format(i+1,N_r[-1]))
            print("The average temperature of the bevi reached {} C in {:.1f} seconds.".format(T_thres,t_cross[-1]))
            if gen_plots: genPlots(np.array(t_cross[1:]), np.array(rel_diff), N_r)
            return t_start, t_end, T_bar,T_map

        N_r.append(N_r[-1] + extend_mesh)
        
    raise ValueError("Max number of iterations exceeded: value did not converge.")