"""
MDCPP: Multi-robot Dynamic Coverage Path Planning for Workload Adaptation
Implementation of the algorithm described in Chen, Chen & Park, arXiv:2509.23705.

Core components implemented:
  - Ground-truth target distribution as a Gaussian Mixture Model (Eq. 5)
  - Task-dependent coverage velocity v_cov via density interpolation (Sec. IV-B)
  - Goal initialization by a power-diagram variant of Lloyd's algorithm (Alg. 1)
  - Capacity-constrained Voronoi / power-diagram cell assignment, Problem 2 (Alg. 2)
  - Shortest coverage path via nearest-neighbour TSP heuristic, Problem 3
  - Online GMM target-distribution estimation: K-means + fit-scoring + sigma search (Sec. III-A)
  - The full MDCPP loop with dynamic re-partitioning (Alg. 3)
  - A sweeping baseline and a "Dynamic, no-prediction" ablation for comparison

The simulator is event/cell-based: robots advance cell-by-cell along their
planned paths; travel time between cells uses v_max, coverage time within a cell
uses the density-dependent v_cov.  Mission completion time = max over robots.
"""

import numpy as np

CELL_M = 10.0            # physical cell width in metres
GRID = 40               # 40 x 40 grid
N_CELLS = GRID * GRID    # 1600 cells


# --------------------------------------------------------------------------- #
#  Target distribution (Gaussian mixture)                                     #
# --------------------------------------------------------------------------- #
class GroundTruthGMM:
    """Target distribution over the grid (index space), Eq. (5)."""

    def __init__(self, centers, sigma=2.5, amp=1.0):
        self.centers = np.asarray(centers, dtype=float)   # in cell-index units
        self.sigma = float(sigma)
        self.amp = float(amp)

    def density_grid(self):
        ix, iy = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
        d = np.zeros((GRID, GRID))
        for (cx, cy) in self.centers:
            d += self.amp * np.exp(-((ix - cx) ** 2 + (iy - cy) ** 2) /
                                   (2.0 * self.sigma ** 2))
        return d


def normalize01(d):
    """Map a density grid into [0,1] for velocity interpolation."""
    m = d.max()
    return d / m if m > 0 else d


# --------------------------------------------------------------------------- #
#  Geometry helpers                                                           #
# --------------------------------------------------------------------------- #
def cell_centers_m():
    """Physical (metre) centre coordinates of every cell, shape (N_CELLS, 2)."""
    ix, iy = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    xs = (ix.ravel() + 0.5) * CELL_M
    ys = (iy.ravel() + 0.5) * CELL_M
    return np.stack([xs, ys], axis=1)


def idx_to_xy(idx):
    return idx // GRID, idx % GRID


CENTERS_M = cell_centers_m()
_DIST = np.linalg.norm(CENTERS_M[:, None, :] - CENTERS_M[None, :, :], axis=2)


# --------------------------------------------------------------------------- #
#  Robots                                                                      #
# --------------------------------------------------------------------------- #
class Robot:
    def __init__(self, rid, v_max, v_min, alpha=1.0):
        self.id = rid
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.alpha = float(alpha)

    def v_cov(self, norm_density):
        """Coverage speed: v_max at density 0, v_min at density 1 (Sec IV-B)."""
        return self.v_max - norm_density * (self.v_max - self.v_min)

    def cover_time(self, cell_norm_density):
        """Time to cover one cell (sweep its width) at v_cov."""
        return CELL_M / self.v_cov(cell_norm_density)


# --------------------------------------------------------------------------- #
#  Coverage-time matrix:  C[i, cell] for robot i covering `cell`              #
# --------------------------------------------------------------------------- #
def coverage_time_matrix(robots, norm_density_flat):
    C = np.zeros((len(robots), N_CELLS))
    for i, r in enumerate(robots):
        C[i] = CELL_M / r.v_cov(norm_density_flat)
    return C


# --------------------------------------------------------------------------- #
#  Goal initialization (Algorithm 1) — power-diagram Lloyd                    #
# --------------------------------------------------------------------------- #
def lloyd_initial_goals(robots, weight_flat, iters=30, start=None, seed=0):
    """
    Spread robots over the area via a power-diagram variant of Lloyd's method.
    `weight_flat` (>=0) biases cell mass (estimated density * alpha).
    Returns array of robot seed positions in metres, shape (n_r, 2).
    """
    rng = np.random.default_rng(seed)
    n = len(robots)
    if start is None:
        # paper: robots begin clustered at lower-left corner
        q = np.tile(CENTERS_M[0] + rng.normal(0, 1.0, 2), (n, 1))
    else:
        q = np.array(start, dtype=float)

    w = np.maximum(weight_flat, 1e-6)
    pw = np.array([r.alpha for r in robots])           # power weights ~ capability
    for _ in range(iters):
        # power assignment: argmin ||g-q_i||^2 - pw_i
        d2 = np.sum((CENTERS_M[:, None, :] - q[None, :, :]) ** 2, axis=2)
        d2 -= pw[None, :]
        assign = np.argmin(d2, axis=1)
        newq = q.copy()
        for i in range(n):
            mask = assign == i
            if mask.any():
                mass = w[mask]
                newq[i] = np.average(CENTERS_M[mask], axis=0, weights=mass)
        if np.max(np.linalg.norm(newq - q, axis=1)) < 1e-3:
            q = newq
            break
        q = newq
    return q


# --------------------------------------------------------------------------- #
#  Capacity-constrained assignment (Algorithm 2 / Problem 2)                  #
#  Power-diagram weights are tuned so per-robot workloads (coverage time)     #
#  balance — i.e. minimise the maximum workload (Eq. 4).                      #
# --------------------------------------------------------------------------- #
def capacity_constrained_assignment(robots, seeds, cells, cover_time,
                                    iters=60, lr=0.15):
    """
    Assign each cell in `cells` (list of flat indices) to a robot.
    Returns dict {robot_index: [cell indices]}.

    cover_time[i, cell] = time for robot i to cover that cell.
    We use additively-weighted (power) Voronoi over seed positions and adapt
    the weights so realized workloads equalise -> capacity-constrained Voronoi.
    """
    cells = np.asarray(cells)
    n = len(robots)
    if len(cells) == 0:
        return {i: [] for i in range(n)}

    pos = CENTERS_M[cells]                                   # (m,2)
    d2 = np.sum((pos[:, None, :] - seeds[None, :, :]) ** 2, axis=2)  # (m,n)
    W = np.zeros(n)
    # typical workload scale, used to normalise the weight update
    scale = cover_time[:, cells].mean() * len(cells) / n + 1e-9

    best_assign, best_spread = None, np.inf
    for _ in range(iters):
        assign = np.argmin(d2 - W[None, :], axis=1)
        loads = np.array([cover_time[i, cells[assign == i]].sum()
                          for i in range(n)])
        spread = loads.max() - loads.min()
        if spread < best_spread:
            best_spread, best_assign = spread, assign.copy()
        target = loads.mean()
        # increase weight (grow region) for under-loaded robots, shrink overloaded
        W += lr * (target - loads) / scale * (CELL_M ** 2)
    assign = best_assign
    return {i: cells[assign == i].tolist() for i in range(n)}


# --------------------------------------------------------------------------- #
#  Shortest coverage path (Problem 3) — nearest-neighbour TSP heuristic       #
# --------------------------------------------------------------------------- #
def nearest_neighbour_path(start_cell, cells):
    cells = list(cells)
    if not cells:
        return []
    remaining = set(cells)
    path = []
    cur = start_cell if start_cell in remaining else min(
        remaining, key=lambda c: _DIST[start_cell, c])
    remaining.discard(cur)
    path.append(cur)
    while remaining:
        nxt = min(remaining, key=lambda c: _DIST[cur, c])
        remaining.discard(nxt)
        path.append(nxt)
        cur = nxt
    return path


def path_travel_time(start_cell, path, v_max):
    """Travel time between consecutive cells along path (covering excluded)."""
    if not path:
        return 0.0
    t = _DIST[start_cell, path[0]] / v_max
    for a, b in zip(path[:-1], path[1:]):
        t += _DIST[a, b] / v_max
    return t


# --------------------------------------------------------------------------- #
#  Online GMM estimation (Section III-A)                                      #
# --------------------------------------------------------------------------- #
# estimation length-scales scale with the grid (tuned on the original 20x20)
_GS = GRID / 20.0
def kmeans(points, k, seed=0, iters=50):
    rng = np.random.default_rng(seed)
    if len(points) < k:
        return points.copy(), np.arange(len(points))
    idx = rng.choice(len(points), k, replace=False)
    cent = points[idx].astype(float)
    for _ in range(iters):
        d = np.linalg.norm(points[:, None, :] - cent[None, :, :], axis=2)
        lab = d.argmin(axis=1)
        new = np.array([points[lab == j].mean(axis=0) if (lab == j).any()
                        else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return cent, lab


def estimate_gmm(observations, theta=0.6, k_candidates=(1, 2, 3, 4),
                 sigma_candidates=np.linspace(2.5 * _GS, 5.0 * _GS, 11),
                 corr_radius=4.0 * _GS):
    """
    observations: dict {cell_idx: observed_density}
    Returns (centers[index space], sigmas, rho_est_grid normalized to [0,1]).
    Implements: filtering (theta), K-means clustering over K candidates,
    fit-scoring (Eq. 8-9), sigma grid-search minimising MSE (Eq. 11).
    """
    if not observations:
        return np.zeros((0, 2)), np.array([]), np.zeros((GRID, GRID))

    obs_cells = np.array(list(observations.keys()))
    obs_vals = np.array([observations[c] for c in obs_cells])
    obs_xy = np.stack(idx_to_xy(obs_cells), axis=1).astype(float)  # index space

    # ---- Filtering: keep high-density observations near Gaussian centres
    hi = obs_vals > theta
    if hi.sum() < 1:
        # not enough signal yet -> flat estimate
        return np.zeros((0, 2)), np.array([]), np.zeros((GRID, GRID))
    Z = obs_xy[hi]

    # ---- model selection over K via fit score (Eq. 8-9)
    best = None
    for K in k_candidates:
        if K > len(Z):
            continue
        cent, _ = kmeans(Z, K, seed=K)
        scores = []
        for mu in cent:
            # rho_est at the candidate centre (max over observed near it)
            near = np.linalg.norm(obs_xy - mu, axis=1) < corr_radius
            if near.sum() == 0:
                scores.append(0.0)
                continue
            rho_center = obs_vals[near].max()
            n_explored = int(near.sum())
            # 2-D Pearson correlation between observed density and a unit Gaussian
            gauss = np.exp(-np.sum((obs_xy[near] - mu) ** 2, axis=1) /
                           (2 * (3.0 * _GS) ** 2))
            ov = obs_vals[near]
            if ov.std() < 1e-9 or gauss.std() < 1e-9:
                e = 0.0
            else:
                e = abs(np.corrcoef(ov, gauss)[0, 1])
            scores.append(rho_center * n_explored * e)
        S = float(np.mean(scores)) if scores else 0.0
        if best is None or S > best[0]:
            best = (S, cent)
    if best is None:
        return np.zeros((0, 2)), np.array([]), np.zeros((GRID, GRID))
    centers = best[1]

    # ---- variance estimation per centre (Eq. 10-11)
    sigmas = []
    for mu in centers:
        near = np.linalg.norm(obs_xy - mu, axis=1) < corr_radius + 2 * _GS
        if near.sum() == 0:
            sigmas.append(sigma_candidates.mean())
            continue
        xy, val = obs_xy[near], obs_vals[near]
        best_sig, best_mse = sigma_candidates[0], np.inf
        for s in sigma_candidates:
            pred = np.exp(-np.sum((xy - mu) ** 2, axis=1) / (2 * s ** 2))
            mse = np.mean((val - pred) ** 2)
            if mse < best_mse:
                best_mse, best_sig = mse, s
        sigmas.append(best_sig)
    sigmas = np.array(sigmas)

    # ---- build estimated density grid, Eq. (12): max over components
    ix, iy = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    rho = np.zeros((GRID, GRID))
    for mu, s in zip(centers, sigmas):
        g = np.exp(-((ix - mu[0]) ** 2 + (iy - mu[1]) ** 2) / (2 * s ** 2))
        rho = np.maximum(rho, g)
    return centers, sigmas, rho


# --------------------------------------------------------------------------- #
#  Sliced Wasserstein distance between two density grids (Sec IV-B.1)          #
# --------------------------------------------------------------------------- #
def sliced_wasserstein(a, b, n_proj=64, seed=0):
    """SWD between two non-negative grids treated as 2-D distributions."""
    ix, iy = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="ij")
    pts = np.stack([ix.ravel(), iy.ravel()], axis=1).astype(float)
    wa = a.ravel().copy(); wb = b.ravel().copy()
    if wa.sum() <= 0 or wb.sum() <= 0:
        return 1.0
    wa /= wa.sum(); wb /= wb.sum()
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_proj):
        theta = rng.uniform(0, np.pi)
        d = np.array([np.cos(theta), np.sin(theta)])
        proj = pts @ d
        order = np.argsort(proj)
        pa = proj[order]; ca = np.cumsum(wa[order])
        cb = np.cumsum(wb[order])
        # 1-D Wasserstein along this slice
        total += np.sum(np.abs(ca - cb) * np.diff(np.concatenate([[pa[0]], pa])))
    return total / n_proj


# --------------------------------------------------------------------------- #
#  Simulators                                                                  #
# --------------------------------------------------------------------------- #
def simulate_mdcpp(robots, gmm, use_prediction=True, n0=2, seed=0, record=False):
    """
    Event/cell-based MDCPP simulation (Algorithm 3).

    use_prediction=True  -> MDCPP (assignment uses online GMM estimate)
    use_prediction=False -> "Dynamic" ablation (flat density, reactive only)

    Returns dict with completion_time, path_length, per-robot times,
    repartition count, swd_history, and (optionally) coverage record.
    """
    true_norm = normalize01(gmm.density_grid()).ravel()      # realized speeds
    true_grid = gmm.density_grid()

    uncovered = set(range(N_CELLS))
    covered_by = -np.ones(N_CELLS, dtype=int)                # which robot covered
    observations = {}                                        # cell -> observed density
    swd_history = []
    repartitions = 0
    n = len(robots)

    robot_time = np.zeros(n)
    robot_pathlen = np.zeros(n)
    robot_pos = None

    def current_estimate_flat():
        if use_prediction:
            _, _, rho = estimate_gmm(observations)
            if rho.max() <= 0:
                return np.zeros(N_CELLS)
            return normalize01(rho).ravel()
        else:
            return np.zeros(N_CELLS)   # no prediction -> uniform

    # ---- initial goals (Alg 1) + initial assignment (Alg 2)
    est = current_estimate_flat()
    # record the pre-observation prediction error so swd_history captures the
    # full convergence curve (flat prior -> learned GMM), not just the tail.
    swd_history.append(sliced_wasserstein(est.reshape(GRID, GRID), true_grid))
    weight = est + 1e-3
    seeds = lloyd_initial_goals(robots, weight, start=None, seed=seed)
    robot_pos = seeds.copy()

    cover_time_est = coverage_time_matrix(robots, np.maximum(est, 0.0))
    assign = capacity_constrained_assignment(robots, seeds, list(uncovered),
                                             cover_time_est)

    # plan initial paths
    paths = {}
    start_cells = {}
    for i in range(n):
        sc = int(np.argmin(np.linalg.norm(CENTERS_M - robot_pos[i], axis=1)))
        start_cells[i] = sc
        paths[i] = nearest_neighbour_path(sc, assign[i])

    record_frames = [] if record else None

    def replan(active_robots):
        nonlocal repartitions, assign, paths, start_cells
        repartitions += 1
        est = current_estimate_flat()
        swd_history.append(sliced_wasserstein(
            (est.reshape(GRID, GRID)), true_grid))
        seeds_now = robot_pos.copy()
        ct = coverage_time_matrix(robots, np.maximum(est, 0.0))
        new_assign = capacity_constrained_assignment(
            robots, seeds_now, list(uncovered), ct)
        for i in range(n):
            assign[i] = new_assign[i]
            sc = int(np.argmin(np.linalg.norm(CENTERS_M - robot_pos[i], axis=1)))
            start_cells[i] = sc
            paths[i] = nearest_neighbour_path(sc, assign[i])

    # ---- main loop: advance the robot that is currently "earliest in time"
    #      cell by cell, until all cells covered.
    MAX_STEPS = N_CELLS * 4
    steps = 0
    while uncovered and steps < MAX_STEPS:
        steps += 1
        # choose the robot with least accumulated time that still has a path
        candidates = [i for i in range(n) if paths[i]]
        if not candidates:
            # everyone empty but cells remain -> force a replan
            if uncovered:
                replan(list(range(n)))
                candidates = [i for i in range(n) if paths[i]]
                if not candidates:
                    break
            else:
                break
        i = min(candidates, key=lambda r: robot_time[r])

        nxt = paths[i][0]
        if nxt not in uncovered:
            paths[i].pop(0)
            continue

        # travel to the cell + cover it
        d = _DIST[start_cells[i], nxt]
        robot_time[i] += d / robots[i].v_max
        robot_pathlen[i] += d
        nd = true_norm[nxt]
        robot_time[i] += robots[i].cover_time(nd)

        # observe + mark covered
        observations[nxt] = float(true_grid.ravel()[nxt])
        uncovered.discard(nxt)
        covered_by[nxt] = i
        robot_pos[i] = CENTERS_M[nxt]
        start_cells[i] = nxt
        paths[i].pop(0)

        if record:
            record_frames.append((nxt, i, covered_by.copy()))

        # re-partition trigger (Alg 3, line 7): a robot is about to finish
        remaining_i = len([c for c in paths[i] if c in uncovered])
        if uncovered and remaining_i < n0:
            replan(list(range(n)))

    completion_time = robot_time.max()
    result = dict(
        completion_time=completion_time,
        path_length=robot_pathlen.sum(),
        robot_times=robot_time.copy(),
        repartitions=repartitions,
        swd_history=swd_history,
        covered_by=covered_by,
        all_covered=(len(uncovered) == 0),
        n_covered=N_CELLS - len(uncovered),
    )
    if record:
        result["frames"] = record_frames
    return result


def simulate_sweeping(robots, gmm):
    """
    Baseline: split E0 into 4 equal vertical strips (one per robot), each robot
    boustrophedon-sweeps its strip at its own speeds. No info exchange, static
    assignment.  Returns the same result dict shape.
    """
    true_norm = normalize01(gmm.density_grid()).ravel()
    n = len(robots)
    cols_per = GRID // n
    robot_time = np.zeros(n)
    robot_pathlen = np.zeros(n)
    covered_by = -np.ones(N_CELLS, dtype=int)

    for i, r in enumerate(robots):
        c0 = i * cols_per
        c1 = GRID if i == n - 1 else (i + 1) * cols_per
        # boustrophedon order over the strip
        order = []
        for cx in range(c0, c1):
            ys = range(GRID) if (cx - c0) % 2 == 0 else range(GRID - 1, -1, -1)
            for cy in ys:
                order.append(cx * GRID + cy)
        start = order[0]
        prev = start
        # travel from lower-left of the strip
        robot_time[i] += _DIST[c0 * GRID, start] / r.v_max
        for cell in order:
            d = _DIST[prev, cell]
            robot_time[i] += d / r.v_max
            robot_pathlen[i] += d
            robot_time[i] += r.cover_time(true_norm[cell])
            covered_by[cell] = i
            prev = cell

    return dict(
        completion_time=robot_time.max(),
        path_length=robot_pathlen.sum(),
        robot_times=robot_time.copy(),
        repartitions=0,
        swd_history=[],
        covered_by=covered_by,
        all_covered=bool((covered_by >= 0).all()),
        n_covered=int((covered_by >= 0).sum()),
    )
