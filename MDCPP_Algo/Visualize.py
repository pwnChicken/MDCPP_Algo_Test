import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mdcpp import (Robot, GroundTruthGMM, simulate_mdcpp, simulate_sweeping, estimate_gmm, normalize01, GRID, idx_to_xy)

LD = [(0.05, 0.008), (0.15, 0.030), (0.30, 0.060), (0.40, 0.080)]
# centres / spread expressed relative to GRID so they scale with the grid size
centers = [(0.25 * GRID, 0.25 * GRID), (0.75 * GRID, 0.25 * GRID),
           (0.50 * GRID, 0.75 * GRID)]
robots = [Robot(i, vmax, vmin) for i, (vmax, vmin) in enumerate(LD)]
gmm = GroundTruthGMM(centers, sigma=2.5 * GRID / 20.0)
 
res = simulate_mdcpp(robots, gmm, use_prediction=True, seed=1, record=True)
base = simulate_sweeping(robots, gmm)
dyn = simulate_mdcpp(robots, gmm, use_prediction=False, seed=1)

obs = {f: float(gmm.density_grid().ravel()[f]) for f in range(GRID * GRID)}
_, _, rho_est = estimate_gmm(obs)
 
fig, ax = plt.subplots(2, 2, figsize=(11, 9))
robot_cols = ["#e6194B", "#3cb44b", "#911eb4", "#42d4f4"]
cmap = ListedColormap(robot_cols)

part = res["covered_by"].reshape(GRID, GRID)
ax[0, 0].imshow(part.T, origin="lower", cmap=cmap, vmin=0, vmax=3)
ax[0, 0].set_title("(a) Final coverage partition (4 robots)")
ax[0, 0].set_xlabel("cell x"); ax[0, 0].set_ylabel("cell y")

gt = normalize01(gmm.density_grid())
ax[0, 1].imshow(gt.T, origin="lower", cmap="Blues")
ax[0, 1].contour(rho_est.T, levels=5, colors="darkorange", linewidths=1.2)
ax[0, 1].set_title("(b) Target density: truth (blue) vs GMM estimate (orange)")
ax[0, 1].set_xlabel("cell x"); ax[0, 1].set_ylabel("cell y")

swd = res["swd_history"]
ax[1, 0].plot(range(1, len(swd) + 1), swd, "-o", color="#444")
ax[1, 0].set_title("(c) GMM prediction error (sliced Wasserstein)")
ax[1, 0].set_xlabel("re-partition event"); ax[1, 0].set_ylabel("SWD to ground truth")
ax[1, 0].grid(alpha=0.3)

labels = ["MDCPP\n(Bayesian)", "Dynamic\n(no pred.)", "Baseline\n(sweep)"]
vals = [res["completion_time"], dyn["completion_time"], base["completion_time"]]
bars = ax[1, 1].bar(labels, vals, color=["#2a9d8f", "#e9c46a", "#e76f51"])
ax[1, 1].set_title("(d) Mission completion time (LD_3C)")
ax[1, 1].set_ylabel("time (s)")
for b, v in zip(bars, vals):
    ax[1, 1].text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                  ha="center", va="bottom")
 
plt.tight_layout()
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "mdcpp_verification.png")
plt.savefig(_out, dpi=130)
print("saved figure; completion:", {k: round(v, 1) for k, v in
      zip(["MDCPP", "Dynamic", "Baseline"], vals)})
print("all_covered:", res["all_covered"], "cells:", res["n_covered"])
