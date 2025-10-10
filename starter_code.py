import numpy as np

def construct_system(r_points, z_points, k_fun, qppp,
                     BC_z, BC_r_top, Bc_r_bottom):
    """
    Build dense A, b for -div(k grad T) = q''' on an axisymmetric (r,z) grid.

    Parameters
    ----------
    r_points : 1D array (Nr,) of radial cell centers, uniform, r[0] ~ Δr/2
    z_points : 1D array (Nz,) of axial cell centers, uniform, z[0] ~ Δz/2
    k_fun    : callable(r, z [, T]) -> k(r,z). If it accepts T, pass None or
               close over your iterate in the callable when doing nonlinear solves.
    qppp     : callable(r, z) -> volumetric source (W/m^3)
    BC_z : tuple for the OUTER radial boundary (r = r_{Nr+1/2}).
               ('dirichlet', T) OR ('neumann', q'') OR ('robin', h, T_inf)
    BC_r_top     : tuple for the TOP axial boundary (z = z_{Nz+1/2}).
               ('dirichlet', T) OR ('neumann', q'') OR ('robin', h, T_inf)
    Bc_r_bottom : tuple for the BOTTOM axial boundary (z = z_{1-1/2}).
               ('dirichlet', T) OR ('neumann', q'') OR ('robin', h, T_inf)

    Returns
    -------
    A : (Nr*Nz, Nr*Nz) dense ndarray
    b : (Nr*Nz,) dense ndarray

    Notes
    -----
    - Coefficients correspond to the equations (E1)–(E5x) in the summary.
    - Axis condition at r=0 is enforced naturally (D_W=0 in (E2)).
    - Uniform grid required here to keep it “bare bones.”
    """
    r = np.asarray(r_points, dtype=float)
    z = np.asarray(z_points, dtype=float)
    Nr, Nz = r.size, z.size
    if Nr < 1 or Nz < 1:
        raise ValueError("r_points and z_points must be non-empty.")

    # Uniform spacing check
    drs = np.diff(r)
    dzs = np.diff(z)
    if not (np.allclose(drs, drs[0]) and np.allclose(dzs, dzs[0])):
        raise ValueError("This starter code assumes UNIFORM Δr and Δz.")
    dr = drs[0]
    dz = dzs[0]

    # Face radii: r_{i-1/2}, r_{i+1/2}; axis is r_{1-1/2}=0
    r_faces = np.hstack([0.0, r + 0.5*dr])  # length Nr+1

    # Convenience: linear index p = i + j*Nr (0-based)
    def idx(i, j):  # 0 <= i < Nr, 0 <= j < Nz
        return i + j*Nr

    # Evaluate k and q''' on cell centers
    K = np.zeros((Nr, Nz))
    Q = np.zeros((Nr, Nz))
    for i in range(Nr):
        for j in range(Nz):
            try:
                K[i, j] = k_fun(r[i], z[j])
            except TypeError:
                K[i, j] = k_fun(r[i], z[j], None)
            Q[i, j] = qppp(r[i], z[j])

    # Helpers
    def harm(a, b):
        return 2.0*a*b/(a+b) if (a > 0.0 and b > 0.0) else 0.0

    A = np.zeros((Nr*Nz, Nr*Nz))
    b = np.zeros(Nr*Nz)

    # --- Loop over cells and assemble (E1)–(E5x) ---
    for j in range(Nz):
        for i in range(Nr):
            p = idx(i, j)
            ri = r[i]
            rw = r_faces[i]       # r_{i-1/2}
            re = r_faces[i+1]     # r_{i+1/2}

            # Interior-like D's (set to zero where neighbor doesn't exist).
            # See (E1) for definitions.
            DW = 0.0
            if i > 0:  # west face harmonic k
                kw = harm(K[i-1, j], K[i, j])
                DW = (rw * kw) / (ri * dr*dr)

            DE = 0.0
            if i < Nr-1:  # east face harmonic k
                ke = harm(K[i, j], K[i+1, j])
                DE = (re * ke) / (ri * dr*dr)

            DS = 0.0
            if j > 0:
                ks = harm(K[i, j-1], K[i, j])
                DS = ks / (dz*dz)

            DN = 0.0
            if j < Nz-1:
                kn = harm(K[i, j], K[i, j+1])
                DN = kn / (dz*dz)

            # Start from interior stencil (E1)
            diag = DW + DE + DS + DN
            A[p, p] += diag
            if i > 0:       A[p, idx(i-1, j)] -= DW
            if i < Nr-1:    A[p, idx(i+1, j)] -= DE
            if j > 0:       A[p, idx(i, j-1)] -= DS
            if j < Nz-1:    A[p, idx(i, j+1)] -= DN
            b[p] = Q[i, j]

            # ---- Axis handling (E2): nothing extra; DW already 0 at i=0 ----

            # ---- Outer radial boundary at i = Nr-1: (E3D/E3N/E3R) ----
            if i == Nr-1:
                kind = BC_z[0].lower()
                if kind == 'dirichlet':  # (E3D)
                    A[p, :] = 0.0
                    A[p, p] = 1.0
                    b[p] = BC_z[1]
                    continue  # row done

                elif kind == 'neumann':  # (E3N): add + (re/(ri*dr)) * qR'' to b, remove E coupling
                    # Remove any accidental east coupling
                    if i < Nr-1:
                        A[p, idx(i+1, j)] = 0.0
                    # Diagonal should NOT include DE for a boundary row
                    A[p, p] -= DE
                    # RHS add from known east-face flux
                    qR = BC_z[1]
                    b[p] += (re / (ri * dr)) * qR

                elif kind == 'robin':    # (E3R): diag += DE*alpha, b += DE*gamma
                    h, Tinf = BC_z[1], BC_z[2]
                    k_face = K[i, j]          # k_E^* (simple choice)
                    DE_star = (re * k_face) / (ri * dr*dr)
                    beta = h * dr / max(k_face, 1e-300)
                    alpha = (1.0 - beta) / (1.0 + beta)
                    gamma = (2.0 * beta / (1.0 + beta)) * Tinf
                    # Remove interior DE (there isn’t an east cell), then apply Robin augments
                    A[p, p] -= DE
                    if i < Nr-1:
                        A[p, idx(i+1, j)] = 0.0
                    A[p, p] += DE_star * (1-alpha)
                    b[p]    += DE_star * gamma

                else:
                    raise ValueError(f"Unknown BC kind for outer radius: {BC_r_top}")

            # ---- TOP axial boundary at j = Nz-1: (E4D/E4N/E4R) ----
            if j == Nz-1:
                kind = BC_r_top[0].lower()
                if kind == 'dirichlet':  # (E4D)
                    A[p, :] = 0.0
                    A[p, p] = 1.0
                    b[p] = BC_r_top[1]
                    continue

                elif kind == 'neumann':  # (E4N): add + q_t''/dz to b, remove N coupling
                    A[p, p] -= DN
                    if j < Nz-1:
                        A[p, idx(i, j+1)] = 0.0
                    qt = BC_r_top[1]
                    b[p] += qt / dz

                elif kind == 'robin':    # (E4R): diag += DN*alpha_N, b += DN*gamma_N (using k_face = K[i,j])
                    h, Tinf = BC_r_top[1], BC_r_top[2]
                    k_face = K[i, j]
                    DN_star = k_face / (dz*dz)
                    beta = h * dz / max(k_face, 1e-300)
                    alpha = (1.0 - beta) / (1.0 + beta)
                    gamma = (2.0 * beta / (1.0 + beta)) * Tinf
                    A[p, p] -= DN
                    if j < Nz-1:
                        A[p, idx(i, j+1)] = 0.0
                    A[p, p] += DN_star * (1-alpha)
                    b[p]    += DN_star * gamma

                else:
                    raise ValueError(f"Unknown BC kind for top z boundary: {BC_z}")

            # ---- BOTTOM axial boundary at j = 0: (E5D/E5N/E5R) ----
            if j == 0:
                kind = Bc_r_bottom[0].lower()
                if kind == 'dirichlet':  # (E5D)
                    A[p, :] = 0.0
                    A[p, p] = 1.0
                    b[p] = Bc_r_bottom[1]
                    continue

                elif kind == 'neumann':  # (E5N): add + q_b''/dz to b, remove S coupling
                    A[p, p] -= DS
                    if j > 0:
                        A[p, idx(i, j-1)] = 0.0
                    qb = Bc_r_bottom[1]
                    b[p] += qb / dz

                elif kind == 'robin':    # (E5R): diag += DS*alpha_S, b += DS*gamma_S
                    h, Tinf = Bc_r_bottom[1], Bc_r_bottom[2]
                    k_face = K[i, j]
                    DS_star = k_face / (dz*dz)
                    beta = h * dz / max(k_face, 1e-300)
                    alpha = (1.0 - beta) / (1.0 + beta)
                    gamma = (2.0 * beta / (1.0 + beta)) * Tinf
                    A[p, p] -= DS
                    if j > 0:
                        A[p, idx(i, j-1)] = 0.0
                    A[p, p] += DS_star * (1-alpha)
                    b[p]    += DS_star * gamma

                else:
                    raise ValueError(f"Unknown BC kind for bottom z boundary: {Bc_r_bottom}")

    return A, b

