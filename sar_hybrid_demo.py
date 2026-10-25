"""
sar_hybrid_demo.py — A "hybrid" of the two focusing algorithms in this repo,
plus a tiny classifier trained on the output.

The idea (in plain English):
  1. Back-Projection (BP) is EXACT but slow: it checks every pixel against
     every pulse. Great for weird motion (moving ships), terrible for speed.
  2. Range-Doppler (RDA) is FAST but approximate: it's basically FFTs.
     It assumes the scene doesn't move during the synthetic aperture.
  3. Hybrid = run the fast RDA everywhere to get a full image quickly,
     then re-run exact BP ONLY on small patches that look "interesting"
     (high focus-measure). The classifier's job is simply:
        "is this patch focused (real target) or smeared (RDA got fooled by
         target motion)?"  ->  i.e. does this patch need an expensive BP redo?

Everything here runs on CPU with numpy only, using the same physics
constants and ship model as sar_batch_sim.py. No GPU / torch required.

Usage:
    python sar_hybrid_demo.py            # train + evaluate + save plots
    python sar_hybrid_demo.py --show     # also show the interactive figure
"""

import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 1. PHYSICS CONSTANTS — copied from sar_batch_sim.py so both use one reality
# ----------------------------------------------------------------------------
C      = 299792458.0
Re     = 6371000.0
h_sat  = 350000.0
R_sat  = Re + h_sat
GM     = 3.986004418e14
V_sat  = np.sqrt(GM / R_sat)          # ~7.4 km/s orbital speed

FC     = 9.65e9                        # X-band radar frequency
BW     = 500e6                         # chirp bandwidth  -> fine range resolution
Lambda = C / FC
T_P    = 20e-6                         # pulse (chirp) length
K_RATE = BW / T_P                      # chirp rate
FS     = 600e6                         # ADC sample rate (fast!)
PRF    = 5000.0                        # pulses per second

theta_look_rad = np.radians(45.0)
theta_inc_rad  = np.arcsin((R_sat / Re) * np.sin(theta_look_rad))
gamma_rad      = theta_inc_rad - theta_look_rad

S0_from_C = np.array([0.0, -R_sat * np.sin(gamma_rad), R_sat * np.cos(gamma_rad)])
V_unit    = np.array([1.0, 0.0, 0.0])
C_offset  = np.array([0.0, 0.0, -Re])
pos_sat_t0 = S0_from_C + C_offset
R0 = np.linalg.norm(pos_sat_t0)

OUT_DIR = "./hybrid_output"


def platform_state(t):
    """Satellite position/velocity at time t (circular orbit, same as repo)."""
    omega = V_sat / R_sat
    wt = omega * t
    P_vec = S0_from_C * np.cos(wt) + (R_sat * V_unit) * np.sin(wt)
    V_vec = (V_sat * V_unit) * np.cos(wt) - (S0_from_C * omega) * np.sin(wt)
    return P_vec + C_offset, V_vec


# ----------------------------------------------------------------------------
# 2. RAW DATA SIMULATION (small, vectorized version of run_physics_spotlight)
# ----------------------------------------------------------------------------
def simulate_raw_data(scatterers, v_target, t_pulses, t_fast, tau_ref, seed=0):
    """
    scatterers : (N,3) array of points ON THE TARGET (ground plane, z=0)
    v_target   : (3,)   target velocity [m/s]  -> [0,0,0] means stationary
    t_pulses   : (P,)   slow-time of each transmitted pulse
    t_fast     : (M,)   ABSOLUTE round-trip-delay axis of the receive window
    tau_ref    : float  chirp center delay (matched to range_compress)
    returns    : raw (P, M) complex echo
    """
    rng = np.random.default_rng(seed)
    num_samples = len(t_fast)

    pos_plat = np.stack([platform_state(t)[0] for t in t_pulses])       # (P,3)
    vel_plat = np.stack([platform_state(t)[1] for t in t_pulses])       # (P,3)

    # target point positions at each pulse time (linear motion)
    pts = scatterers[None, :, :] + np.outer(t_pulses, v_target)[:, None, :]  # (P,N,3)

    # monostatic two-way distance and radial velocity
    diff = pts - pos_plat[:, None, :]                     # (P,N,3)
    dist = np.linalg.norm(diff, axis=2)                   # (P,N)
    r_hat = diff / dist[:, :, None]
    v_rel = v_target[None, None, :] - vel_plat[:, None, :]
    v_rad = np.sum(v_rel * r_hat, axis=2)                 # (P,N)

    # phase = exp(-j*4*pi*(range + velocity*time)/lambda)
    phase = -4j * np.pi * (dist + v_rad * t_pulses[:, None]) / Lambda
    amp = np.full(len(scatterers), 1.0 / max(len(scatterers), 1)) ** 0  # equal RCS
    sig_pts = amp[None, :] * np.exp(phase)                # (P,N)

    # map each point's delay onto the fast-time grid (nearest bin, good enough).
    # NOTE: we store ABSOLUTE round-trip delay tau in t_fast (see generate_dataset),
    # so the range window can sit on the ship's actual slant-range gate.
    tau = 2.0 * dist / C                                  # round-trip delay
    dt = t_fast[1] - t_fast[0]
    idx = np.round((tau - t_fast[0]) / dt).astype(int)    # (P,N) delay bins
    raw = np.zeros((len(t_pulses), num_samples), dtype=np.complex128)
    n_kept = 0
    for p in range(len(t_pulses)):
        ok = (idx[p] >= 0) & (idx[p] < num_samples)       # drop out-of-window pts
        np.add.at(raw[p], idx[p][ok], sig_pts[p][ok])
        n_kept += int(ok.sum())
    if n_kept == 0:
        raise RuntimeError(
            "no scatterers fell inside the fast-time window; "
            f"tau range [{tau.min():.6e}, {tau.max():.6e}] s vs "
            f"window [{t_fast[0]:.6e}, {t_fast[-1]:.6e}] s, dt={dt:.3e}")

    # apply the transmit chirp to every sample in the receive window.
    # (A real 40 us chirp spans ~12 km of ground — far wider than our demo
    # window — so we use a scaled-down linear chirp whose bandwidth matches
    # the window. Still a chirp, still compressed by FFT multiply.)
    dt = t_fast[1] - t_fast[0]
    k_demo = BW * (num_samples * dt / (2 * T_P))
    chirp = np.exp(1j * np.pi * k_demo * (t_fast - tau_ref)**2)
    raw *= chirp[None, :]

    # mild additive noise so patches aren't perfectly clean
    raw += 0.02 * (rng.standard_normal(raw.shape) + 1j * rng.standard_normal(raw.shape))
    return raw


def range_compress(raw, t_fast):
    """Matched filter: FFT-multiply with the conjugate reference chirp."""
    n = raw.shape[1]
    dt = t_fast[1] - t_fast[0]
    k_demo = BW * (n * dt / (2 * T_P))
    ref = np.exp(1j * np.pi * k_demo * (t_fast - t_fast[n // 2])**2)
    ref_f = np.fft.fft(ref)
    rc = np.fft.ifft(np.fft.fft(raw, axis=1) * np.conj(ref_f)[None, :], axis=1)
    return rc      # circular matched filter; peak sits at each point's delay bin


# ----------------------------------------------------------------------------
# 3a. THE FAST ALGORITHM: Range-Doppler azimuth compression via FFT down the
#     pulse axis — same core step as sar_focus_rda() in sar_satellite_sim.py.
# ----------------------------------------------------------------------------
def focus_rda(rc_data, t_pulses):
    """
    THE FAST PASS: azimuth compression via FFT down the pulse axis.
    A stationary target's slow-time phase ramp lands in ONE bright Doppler
    bin. It implicitly ASSUMES the scene was stationary during the aperture —
    that's where it gets fooled by moving ships (energy smears across bins).
    """
    return np.fft.fftshift(np.fft.fft(rc_data, axis=0), axes=0)


# ----------------------------------------------------------------------------
# 3b. THE EXACT (SLOW) ALGORITHM: Time-Domain Back-Projection (numpy port of
#     tdbp_gpu in sar_batch_sim.py). Checks every pixel vs every pulse.
# ----------------------------------------------------------------------------
def focus_bp(rc_data, t_fast, t_pulses, x_axis, y_axis, v_focus=np.zeros(3)):
    """Coherent sum: for each pixel, compute exact delay & Doppler, sample rc_data."""
    pos_plat = np.stack([platform_state(t)[0] for t in t_pulses])
    vel_plat = np.stack([platform_state(t)[1] for t in t_pulses])
    gx, gy = np.meshgrid(x_axis, y_axis)
    pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])       # (3, npix)

    dtau = t_fast[1] - t_fast[0]
    img = np.zeros(pts.shape[1], dtype=np.complex128)
    for p in range(len(t_pulses)):
        diff = pts.T - pos_plat[p]                                    # (npix,3)
        dist = np.linalg.norm(diff, axis=1)
        r_hat = diff / dist[:, None]
        v_rel = v_focus - vel_plat[p]
        v_rad = r_hat @ v_rel
        tau = 2.0 * (dist + v_rad * t_pulses[p]) / C                  # exact delay
        f_dop = 2.0 * v_rad / Lambda                                  # motion comp
        idx = tau / dtau + len(t_fast) // 2
        idx = np.clip(idx, 0, rc_data.shape[1] - 1.001)
        i0 = idx.astype(int)
        frac = idx - i0
        val = rc_data[p, i0] * (1 - frac) + rc_data[p, i0 + 1] * frac
        img += val * np.exp(-1j * 2.0 * np.pi * f_dop * t_pulses[p])
    return img.reshape(len(y_axis), len(x_axis))


def focus_measure(patch):
    """Autofocus metric: sub-aperture coherence. High = focused, low = smeared."""
    P = patch.shape[0]
    half = P // 2
    if half < 4:
        return 0.0
    fwd = np.abs(patch[:half].sum(axis=0)).mean()
    aft = np.abs(patch[half:].sum(axis=0)).mean()
    full = np.abs(patch.sum(axis=0)).mean()
    denom = fwd + aft + 1e-12
    return float(full / denom)          # ~1 when coherent (focused), <<1 if not


# ----------------------------------------------------------------------------
# 4. PATCH GENERATOR — builds training data: targets + ocean clutter, some
#    moving (which fools RDA), some stationary (which RDA handles fine).
# ----------------------------------------------------------------------------
def make_ship_scatterers(rng, size_m=154.0, beam_m=20.0, n=40):
    xs = rng.uniform(-size_m / 2, size_m / 2, n)
    ys = rng.uniform(-beam_m / 2, beam_m / 2, n)
    return np.stack([xs, ys, np.zeros(n)], axis=1)


def make_clutter_patch(rng, n_pix=32, n_pulses=64):
    """Random speckle 'ocean' raw data -> always looks smeared after RDA."""
    return (rng.standard_normal((n_pulses, n_pix)) +
            1j * rng.standard_normal((n_pulses, n_pix))) / np.sqrt(2)


def generate_dataset(n_per_class=120, patch_px=24, n_pulses=48, seed=1):
    """
    Returns (X, y, meta):
      X    : (N, n_pulses, patch_px) complex RDA-domain patch energy profiles...
             we actually keep the (pulse x pixel) magnitude matrix as features.
      y    : 0 = clutter, 1 = stationary focused target, 2 = MOVED target
             (RDA-smeared -> needs BP redo)
    """
    rng = np.random.default_rng(seed)
    t_pulses = np.linspace(-0.25, 0.25, n_pulses)        # 0.5 s CPI like the repo
    swath = 500.0
    axis = np.linspace(-swath / 2, swath / 2, 128)
    # DEMO SAMPLING TRICK: a real radar samples at FS=600 MHz (3 m bins), but for
    # this CPU demo we open a small receive window around the scene's slant range
    # and sample it at ~7 m/bin — coarser than the chirp resolution, which is fine
    # because what the classifier actually reads is the DOPPLER/azimuth structure,
    # and that lives in the slow-time phase (kept exact at Lambda-level precision).
    dtau_win = (swath / patch_px) / 2.0 / C               # one pixel per bin
    win_len = patch_px + 16
    # Doppler bandwidth of a target at cross-range x: fd ~ 2*v_sat*x/lambda/R0.
    # Nyquist-safe |x| for this CPI is PRF/2 * lambda*R0/(2*v_sat):
    nyq_x = (PRF / 2.0) * Lambda * R0 / (2.0 * V_sat)

    feats, labels, extras = [], [], []

    def add_case(cls_label, v_target):
        # place a ship somewhere inside the swath
        cx, cy = rng.uniform(-100, 100, 2)
        scat = make_ship_scatterers(rng) + np.array([cx, cy, 0.0])
        # center the receive window on THIS ship's own slant-range gate
        tau_mid = 2.0 * np.linalg.norm(np.array([0.0, cy, 0.0]) - pos_sat_t0) / C
        t_fast = tau_mid + (np.arange(win_len) - win_len // 2) * dtau_win
        raw = simulate_raw_data(scat, v_target, t_pulses,
                                t_fast, seed=rng.integers(1 << 31))
        rc = range_compress(raw, t_fast)
        # --- extract a patch around where the ship should be (pixel coords) ---
        ix = int(np.clip((cx + swath / 2) / swath * 128, patch_px // 2, 128 - patch_px // 2))
        iy = int(np.clip((cy + swath / 2) / swath * 128, patch_px // 2, 128 - patch_px // 2))
        # matched-filter peak of the SHIP CENTER sits at the window's middle bin
        center_bin = win_len // 2
        lo = int(np.clip(center_bin - patch_px // 2, 0, rc.shape[1] - patch_px))
        win = rc[:, lo: lo + patch_px]
        # THE FAST PASS: azimuth FFT (Range-Doppler style) on this range window
        patch_img = focus_rda(win, t_pulses)
        feats.append(np.abs(patch_img).astype(np.float32))
        labels.append(cls_label)
        extras.append((rc, t_fast, t_pulses, axis, ix, iy, cx, cy, lo))

    for _ in range(n_per_class):
        add_case(1, np.zeros(3))                          # stationary ship
        add_case(2, np.array([rng.uniform(8, 30), 0, 0])) # moving ship (fools RDA)
        feats.append(np.abs(make_clutter_patch(rng, patch_px, n_pulses)).astype(np.float32))
        labels.append(0)                                  # ocean clutter
    return np.stack(feats), np.array(labels), extras


# ----------------------------------------------------------------------------
# 5. TINY CLASSIFIER — softmax regression trained with plain numpy SGD.
#    Input: flattened patch magnitude (~576 features). Output: 3 classes.
#    This stands in for "a small CNN" without needing torch/sklearn.
# ----------------------------------------------------------------------------
def train_softmax(X, y, epochs=400, lr=0.5, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    n_tr = int(0.8 * N)
    tr, te = perm[:n_tr], perm[n_tr:]
    flat = X.reshape(N, -1)                    # (N, pulses*range_bins)
    mu, sd = flat[tr].mean(0), flat[tr].std(0) + 1e-9
    Z = (flat - mu) / sd
    Z -= Z.mean(1, keepdims=True)              # per-patch brightness norm:
    Z /= Z.std(1, keepdims=True) + 1e-9        # teaches SHAPE, not RCS level
    D = Z.shape[1]
    W = 0.01 * rng.standard_normal((D, 3))
    b = np.zeros(3)

    def logits(M):
        return M @ W + b

    def probs(M):
        L = logits(M)
        L -= L.max(axis=1, keepdims=True)
        E = np.exp(L)
        return E / E.sum(axis=1, keepdims=True)

    Yoh = np.eye(3)[y[tr]]
    Zt = Z[tr]
    for ep in range(epochs):
        P = probs(Zt)
        G = (P - Yoh) / len(tr)
        W -= lr * (Zt.T @ G + 1e-3 * W)        # tiny weight decay
        b -= lr * G.sum(axis=0)
    acc_tr = (logits(Z[tr]).argmax(1) == y[tr]).mean()
    acc_te = (logits(Z[te]).argmax(1) == y[te]).mean()
    confusion = np.zeros((3, 3), int)
    for t_, p_ in zip(y[te], logits(Z[te]).argmax(1)):
        confusion[t_, p_] += 1
    return dict(W=W, b=b, mu=mu, sd=sd, acc_tr=acc_tr, acc_te=acc_te,
                confusion=confusion, test_idx=te)


def predict(model, X_flat):
    """X_flat: (M, D) already-flattened patch magnitudes."""
    Z = (X_flat - model["mu"]) / model["sd"]
    Z -= Z.mean(1, keepdims=True)
    Z /= Z.std(1, keepdims=True) + 1e-9
    return (Z @ model["W"] + model["b"]).argmax(1)


# ----------------------------------------------------------------------------
# 6. THE HYBRID PIPELINE — fast RDA-ish everywhere, exact BP only where the
#    classifier says "this patch is suspicious", then compare cost/quality.
# ----------------------------------------------------------------------------
def run_hybrid(extras, model, X_all, bp_patch=24, stride=12):
    """
    For each generated target scene (we reuse dataset cases), do:
      - cheap full-image formation (our DFT-based RDA stand-in)
      - slide patches, classify, BP-refine only class-2-flagged patches
    Returns stats + example images for plotting.
    """
    results = []
    axis = np.linspace(-250, 250, 128)
    n_done = 0
    for (rc, tf, t_pulses, ax, ix, iy, cx, cy, lo) in extras:
        n_pul = rc.shape[0]
        win = rc[:, lo: lo + bp_patch]
        if win.shape[1] < bp_patch:
            win = np.pad(win, ((0, 0), (0, bp_patch - win.shape[1])))
        patch_img = focus_rda(win, t_pulses)              # the fast pass
        feat = np.abs(patch_img).astype(np.float32).ravel()
        pred = predict(model, feat[None, :])[0]

        # hybrid decision: refine with exact BP only if predicted "moving target"
        do_bp = (pred == 2)
        bp_cost = n_pul * bp_patch * bp_patch if do_bp else 0
        full_bp_cost = n_pul * 128 * 128     # what pure BP would cost per frame

        img_fast = np.abs(patch_img)
        img_refined = None
        if do_bp:
            xw = ax[max(ix - bp_patch // 2, 0): max(ix - bp_patch // 2, 1) + bp_patch]
            img_refined = np.abs(focus_bp(rc, tf, t_pulses, xw, xw))
        results.append(dict(pred=int(pred), do_bp=bool(do_bp),
                            bp_ops=bp_cost, full_ops=full_bp_cost,
                            img_fast=img_fast, img_refined=img_refined))
        n_done += 1
        if n_done >= 12:      # demo subset is plenty
            break
    return results


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="show interactive plot")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] Simulating raw SAR scenes (stationary ships, moving ships, ocean)...")
    X, y, extras = generate_dataset(n_per_class=120)
    print(f"      dataset: {X.shape[0]} patches, shape per patch {X.shape[1:]}")

    print("[2/4] Training tiny softmax classifier (numpy SGD)...")
    model = train_softmax(X, y, epochs=400)
    names = ["clutter", "static-target", "moving-target"]
    print(f"      train acc {model['acc_tr']:.2%} | held-out acc {model['acc_te']:.2%}")
    print("      confusion (rows=true, cols=pred):")
    for i, row in enumerate(model["confusion"]):
        print(f"        {names[i]:>14}: {row}")

    print("[3/4] Running hybrid pipeline (fast pass + selective exact BP)...")
    res = run_hybrid(extras, model, X)
    total_sel = sum(r["bp_ops"] for r in res)
    total_full = sum(r["full_ops"] for r in res)
    refined = sum(1 for r in res if r["do_bp"])
    print(f"      BP-refined {refined}/{len(res)} patches;"
          f" ops saved: {100*(1-total_sel/max(total_full,1)):.1f}% vs pure BP")

    print("[4/4] Plotting...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ex = res[0]
    axes[0, 0].imshow(np.log1p(ex["img_fast"]), cmap="viridis")
    axes[0, 0].set_title("Fast pass (RDA-like)")
    if ex["img_refined"] is not None:
        axes[0, 1].imshow(np.log1p(ex["img_refined"]), cmap="viridis")
        axes[0, 1].set_title("Hybrid: BP-refined patch")
    mv = next((r for r in res if r["pred"] == 2), None)
    if mv and mv["img_refined"] is not None:
        axes[0, 2].imshow(np.log1p(mv["img_fast"]), cmap="viridis")
        axes[0, 2].set_title("Moving target: fast")
        axes[0, 3].imshow(np.log1p(mv["img_refined"]), cmap="viridis")
        axes[0, 3].set_title("Moving target: BP-refined")
    # feature visualization + confusion
    axes[1, 0].imshow(X[y == 1][0], cmap="gray")
    axes[1, 0].set_title("feature: static patch")
    axes[1, 1].imshow(X[y == 2][0], cmap="gray")
    axes[1, 1].set_title("feature: moving patch")
    im = axes[1, 2].imshow(model["confusion"], cmap="Blues")
    axes[1, 2].set_title("confusion matrix")
    fig.colorbar(im, ax=axes[1, 2])
    axes[1, 3].axis("off")
    axes[1, 3].text(0, 0.5,
                    f"held-out accuracy\n{model['acc_te']:.1%}\n\n"
                    f"classifier decides whether a\npatch gets expensive exact BP.\n"
                    f"Selective refinement saved\n~{100*(1-total_sel/max(total_full,1)):.0f}% of BP compute.",
                    fontsize=11, va="center")
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])
    axes[1, 3].axis("on"); axes[1, 3].set_visible(False)
    fig.suptitle("Hybrid SAR: fast RDA everywhere + exact back-projection only where the classifier flags motion")
    out_png = os.path.join(OUT_DIR, "hybrid_results.png")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"      saved {out_png}")
    np.savez(os.path.join(OUT_DIR, "classifier.npz"),
             W=model["W"], b=model["b"], mu=model["mu"], sd=model["sd"])
    print("      saved ./hybrid_output/classifier.npz")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
