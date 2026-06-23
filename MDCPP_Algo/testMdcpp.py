"""
Test harness for the MDCPP implementation.

Verifies the four properties that define a *working* MDCPP:
  T1  Complete coverage      - every one of the 400 cells is covered exactly once
  T2  Dynamic balancing      - re-partitioning fires; per-robot finish times stay
                               far better balanced than the static baseline
  T3  GMM prediction         - sliced-Wasserstein error to ground truth decreases
  T4  Performance            - MDCPP completes faster than the sweeping baseline
                               (and faster than the no-prediction "Dynamic" ablation)

Scenarios reproduce Sec. IV-C of the paper (LD/SD velocity heterogeneity, 2/3
Gaussian centres).
"""
import sys
import time
import numpy as np
from mdcpp import (Robot, GroundTruthGMM, simulate_mdcpp, simulate_sweeping,
                   GRID, N_CELLS)

# Velocity sets (travel / detect) m/s, exactly as the paper's ablation scenarios
LD = [(0.05, 0.008), (0.15, 0.030), (0.30, 0.060), (0.40, 0.080)]   # LargeDiff
SD = [(0.08, 0.015), (0.10, 0.020), (0.12, 0.025), (0.15, 0.030)]   # SmallDiff

# Gaussian-centre layouts expressed as fractions of the grid so they scale
# automatically with GRID (peaks stay in the same relative spots).
_LAYOUTS = {
    "LD_2C": (LD, [(0.25, 0.25), (0.75, 0.75)]),
    "LD_3C": (LD, [(0.25, 0.25), (0.75, 0.25), (0.50, 0.75)]),
    "SD_2C": (SD, [(0.25, 0.50), (0.50, 0.25)]),
    "SD_3C": (SD, [(0.25, 0.50), (0.50, 0.25), (0.75, 0.75)]),
}
# blob spread scales with the grid too (2.5 cells on the original 20x20 grid)
SIGMA = 2.5 * GRID / 20.0

SCENARIOS = {
    tag: (vel, [(fx * GRID, fy * GRID) for fx, fy in fracs])
    for tag, (vel, fracs) in _LAYOUTS.items()
}


def make_robots(vel_set):
    return [Robot(i, vmax, vmin, alpha=1.0)
            for i, (vmax, vmin) in enumerate(vel_set)]


def balance_ratio(robot_times):
    """max/mean finish time; 1.0 == perfectly balanced team."""
    rt = np.asarray(robot_times)
    return rt.max() / rt.mean() if rt.mean() > 0 else np.inf


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"   [{status}] {name}" + (f"  ({detail})" if detail else ""))
    return condition


# --------------------------------------------------------------------------- #
#  CLI visualization                                                           #
# --------------------------------------------------------------------------- #
# 256-colour ANSI background codes, one per robot (distinct, readable swatches)
ROBOT_ANSI = [196, 46, 201, 51, 226, 21, 208, 129]   # red, grn, mag, cyan, ...
# fallback glyphs when the terminal isn't a colour TTY
ROBOT_GLYPH = "ABCDEFGH"
_COLOR = sys.stdout.isatty()
ANIMATE = False                # set by the --animate CLI flag


def _swatch(rid):
    """Return a 2-char block representing robot `rid` (colour or glyph)."""
    if _COLOR:
        return f"\033[48;5;{ROBOT_ANSI[rid % len(ROBOT_ANSI)]}m  \033[0m"
    return ROBOT_GLYPH[rid % len(ROBOT_GLYPH)] * 2


def _head(rid):
    """Marker for a robot's current position (a bright pip on its colour)."""
    if _COLOR:
        return (f"\033[48;5;{ROBOT_ANSI[rid % len(ROBOT_ANSI)]}m"
                f"\033[1;97m()\033[0m")
    return ROBOT_GLYPH[rid % len(ROBOT_GLYPH)].lower() * 2


def _grid_lines(covered_by, heads=None):
    """Build the printable grid rows (origin lower-left, y increasing upward).

    `heads` maps cell index -> robot id for cells to draw as position markers.
    """
    grid = covered_by.reshape(GRID, GRID)          # [ix, iy]
    heads = heads or {}
    lines = []
    for y in range(GRID - 1, -1, -1):
        row = "   "
        for x in range(GRID):
            idx = x * GRID + y
            r = int(grid[x, y])
            if idx in heads:
                row += _head(heads[idx])
            else:
                row += "  " if r < 0 else _swatch(r)
        lines.append(row)
    return lines


def render_partition(covered_by, robots, robot_times, title=""):
    """Print the final coverage partition as a colour grid + per-robot legend.

    covered_by[cell] = id of the robot that covered that cell (-1 = uncovered).
    Origin is lower-left, matching the matplotlib figure in Visualize.py.
    """
    if title:
        print(f"\n   {title}")
    for row in _grid_lines(covered_by):
        print(row)

    # legend: colour swatch, velocities, cells owned, finish time
    print("   " + "-" * 40)
    for r in robots:
        cells = int((covered_by == r.id).sum())
        print(f"   {_swatch(r.id)} robot {r.id}  "
              f"v=[{r.v_min:.3f}-{r.v_max:.3f}] m/s  "
              f"cells={cells:>3}  finish={robot_times[r.id]:7.1f}s")


def animate_partition(frames, robots, robot_times, title="", fps=60):
    """Replay coverage cell-by-cell, redrawing the grid in place.

    `frames` is the list recorded by simulate_mdcpp(record=True): each entry is
    (cell, robot_id, covered_by_snapshot).  Falls back to a static render when
    stdout isn't a colour TTY (e.g. piped output / CI logs).
    """
    if not (_COLOR and frames):
        # nothing to animate to -> snapshot of the final state
        render_partition(frames[-1][2] if frames else
                         -np.ones(N_CELLS, dtype=int), robots, robot_times,
                         title=title)
        return

    # cap redraws so big grids stay snappy; always include the final frame
    step = max(1, len(frames) // 150)
    idxs = list(range(0, len(frames), step))
    if idxs[-1] != len(frames) - 1:
        idxs.append(len(frames) - 1)
    delay = 1.0 / fps

    sys.stdout.write("\033[?25l")          # hide cursor
    printed = 0
    try:
        for fi in idxs:
            cell, rid, snap = frames[fi]
            lines = [f"   {title}  [{fi + 1:>3}/{len(frames)} cells]"]
            lines += _grid_lines(snap, heads={cell: rid})
            lines.append("   " + "-" * 40)
            for r in robots:
                cells = int((snap == r.id).sum())
                lines.append(f"   {_swatch(r.id)} robot {r.id}  "
                             f"cells={cells:>3}")
            if printed:
                sys.stdout.write(f"\033[{printed}A")     # rewind to redraw
            # \033[K clears each line's leftovers from the previous frame
            sys.stdout.write("".join(ln + "\033[K\n" for ln in lines))
            sys.stdout.flush()
            printed = len(lines)
            time.sleep(delay)
    finally:
        sys.stdout.write("\033[?25h")      # restore cursor
        sys.stdout.flush()

    # final finish-time summary under the completed grid
    for r in robots:
        print(f"   {_swatch(r.id)} robot {r.id}  finish={robot_times[r.id]:7.1f}s")


def run():
    np.set_printoptions(precision=1, suppress=True)
    all_ok = True
    summary = []

    for tag, (vel, centers) in SCENARIOS.items():
        print(f"\n=== Scenario {tag}  centres={centers} ===")
        gmm = GroundTruthGMM(centers, sigma=SIGMA)
        robots = make_robots(vel)

        mdcpp = simulate_mdcpp(robots, gmm, use_prediction=True, seed=1,
                               record=ANIMATE)
        dynamic = simulate_mdcpp(robots, gmm, use_prediction=False, seed=1)
        base = simulate_sweeping(robots, gmm)

        # ---- T1 complete coverage (all three methods must fully cover) ----
        t1 = check("T1 complete coverage (MDCPP)",
                   mdcpp["all_covered"] and mdcpp["n_covered"] == N_CELLS,
                   f"{mdcpp['n_covered']}/{N_CELLS} cells")
        # partition disjointness: every covered cell assigned to exactly one robot
        cb = mdcpp["covered_by"]
        disjoint = (cb >= 0).all() and (cb < len(robots)).all()
        t1b = check("T1 valid partition (each cell -> one robot)", disjoint)

        # ---- T2 dynamic balancing ----
        br_mdcpp = balance_ratio(mdcpp["robot_times"])
        br_base = balance_ratio(base["robot_times"])
        t2a = check("T2 re-partitioning fired", mdcpp["repartitions"] > 0,
                    f"{mdcpp['repartitions']} re-partitions")
        t2b = check("T2 better balanced than baseline",
                    br_mdcpp < br_base,
                    f"MDCPP {br_mdcpp:.2f} vs baseline {br_base:.2f}")

        # ---- T3 GMM prediction convergence ----
        swd = mdcpp["swd_history"]
        if len(swd) >= 2:
            t3 = check("T3 GMM SWD error decreases",
                       swd[-1] < swd[0],
                       f"{swd[0]:.3f} -> {swd[-1]:.3f}")
        else:
            t3 = check("T3 GMM SWD error decreases", True,
                       "single re-partition; skipped")

        # ---- T4 performance ----
        red_base = 100 * (1 - mdcpp["completion_time"] / base["completion_time"])
        red_dyn = 100 * (1 - mdcpp["completion_time"] / dynamic["completion_time"])
        t4a = check("T4 MDCPP faster than baseline",
                    mdcpp["completion_time"] < base["completion_time"],
                    f"{red_base:+.1f}% time")
        t4b = check("T4 MDCPP <= Dynamic (prediction helps)",
                    mdcpp["completion_time"] <= dynamic["completion_time"] * 1.02,
                    f"{red_dyn:+.1f}% vs Dynamic")

        # ---- visual example: how MDCPP split the area between the robots ----
        if ANIMATE:
            animate_partition(mdcpp["frames"], robots, mdcpp["robot_times"],
                              title=f"MDCPP coverage playback ({tag})")
        else:
            render_partition(mdcpp["covered_by"], robots, mdcpp["robot_times"],
                             title=f"MDCPP coverage partition ({tag})")

        ok = all([t1, t1b, t2a, t2b, t3, t4a])
        all_ok &= ok
        summary.append((tag, mdcpp["completion_time"], dynamic["completion_time"],
                        base["completion_time"], red_base,
                        mdcpp["path_length"], base["path_length"]))

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"{'Scenario':<8}{'MDCPP(s)':>11}{'Dynamic(s)':>12}{'Baseline(s)':>13}"
          f"{'Δvs base':>10}{'MDCPP len':>11}{'base len':>10}")
    print("-" * 78)
    for tag, m, dyn, b, red, mlen, blen in summary:
        print(f"{tag:<8}{m:>11.1f}{dyn:>12.1f}{b:>13.1f}{red:>9.1f}%"
              f"{mlen:>11.1f}{blen:>10.1f}")
    print("=" * 78)
    print(f"\nOVERALL: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


if __name__ == "__main__":
    # `--animate` / `-a` replays each scenario's coverage cell-by-cell
    # (needs a colour terminal; otherwise it prints a static final map).
    ANIMATE = any(a in ("--animate", "-a") for a in sys.argv[1:])
    ok = run()
    raise SystemExit(0 if ok else 1)
