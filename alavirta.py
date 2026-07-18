"""
Alavirta — the descending stream that closes the predictive loop.

PerceptionLab / Antti Luode (Helsinki), with Claude (Opus 4.8). July 2026.

    Do not hype. Do not lie. Just show.

The parent, CorticalLoop, already dead-reckons through a sensory blackout: it has
the metric anchor (grid) and the comparator (CA1 theta gate). What it did NOT have
is the one wire that lets a top-down state and a bottom-up sensory stream be
compared AT ALL: the descending renderer that puts the held internal state into the
sensor's own coordinate frame, so that

        residual = prediction - reality

is even a defined subtraction. Bottom-up lives in Takens delay space; the internal
generator lives in an abstract latent. You cannot subtract a delay vector from a
hidden state. This file builds the missing map g: h -> v_hat, closes the loop on it,
weights the residual by a DYNAMIC precision (not a fixed clock), and hands the error
signal a steering wheel (active inference).

Everything printed by run() is measured by the engine, not assumed. Pure numpy.
"""

import numpy as np

# --------------------------------------------------------------------------- #
#  1. The world (reality). A scalar oscillator we only ever observe as s(t).   #
#     Segmented so each mechanism gets its own falsifiable test.               #
# --------------------------------------------------------------------------- #

def make_world(cfg, rng):
    """Return s (observed scalar, with per-segment noise), phi_true, omega_true,
    and boolean masks for the fog and blackout segments."""
    T = cfg["T"]
    s        = np.zeros(T)
    phi_true = np.zeros(T)
    om_true  = np.zeros(T)
    fog      = np.zeros(T, bool)
    blackout = np.zeros(T, bool)

    phi = 0.0
    a   = 1.0
    for t in range(T):
        # frequency STEP at t_step: the world changes its own rhythm.
        om = cfg["omega0"] if t < cfg["t_step"] else cfg["omega1"]
        phi += om
        phi_true[t] = phi
        om_true[t]  = om
        s_clean = a * np.cos(phi)

        # sensory noise: quiet everywhere, loud in the fog band.
        in_fog = cfg["fog"][0] <= t < cfg["fog"][1]
        sig = cfg["sigma_fog"] if in_fog else cfg["sigma_clean"]
        fog[t] = in_fog
        s[t] = s_clean + rng.normal(0, sig)

        blackout[t] = cfg["blk"][0] <= t < cfg["blk"][1]

    return s, phi_true, om_true, fog, blackout


def delay_vector(s, t, m, tau):
    """Bottom-up recognizer: the Takens delay embedding v_t = [s(t), s(t-tau), ...].
    Returns None before enough history exists."""
    if t < (m - 1) * tau:
        return None
    return np.array([s[t - k * tau] for k in range(m)])


# --------------------------------------------------------------------------- #
#  2. The internal generative model + the DESCENDING RENDERER.                 #
#     State theta = (phi, omega, log_a). The renderer g(theta) produces a      #
#     PREDICTED delay vector v_hat in the SAME space as the recognizer's v.    #
# --------------------------------------------------------------------------- #

class Alavirta:
    def __init__(self, cfg):
        self.m   = cfg["m"]
        self.tau = cfg["tau"]
        self.phi = 0.0
        self.om  = cfg["omega0"] * 0.6        # deliberately WRONG start freq
        self.la  = 0.0                         # log amplitude (a = 1)
        # inference gains
        self.eta_phi = cfg["eta_phi"]
        self.eta_om  = cfg["eta_om"]
        self.eta_a   = cfg["eta_a"]
        # precision as a reliability weight in (0,1], driven by SENSORY noise
        self.sig_ref = cfg["sig_ref"]
        # switches for the ablations
        self.freq_adapt = cfg.get("freq_adapt", True)
        self.fixed_prec = cfg.get("fixed_prec", None)   # None => dynamic
        self.closed     = cfg.get("closed", True)       # False => no residual ever

    # ---- the descending renderer g(theta) -> v_hat  (this is the missing wire) ---
    #      renders the abstract state into a PREDICTED Takens delay vector, so the
    #      residual below is a defined subtraction in the recognizer's own space.
    def render(self):
        k  = np.arange(self.m)
        ph = self.phi - k * self.tau * self.om
        a  = np.exp(self.la)
        return a * np.cos(ph), ph, a

    def precision(self, noise_est):
        """Reliability of the sensory channel: high when the signal is clean, low
        in fog. Estimated from signal roughness, NOT from the model's own error, so
        an unlocked start cannot suppress its own means of locking."""
        if self.fixed_prec is not None:
            return self.fixed_prec
        return 1.0 / (1.0 + (noise_est / self.sig_ref) ** 2)

    def step(self, v_real, noise_est):
        """One tick. The prior generative flow always advances the hallucination.
        If the loop is closed and input is present, form the residual in delay
        space, invert it to a phase error, and apply a precision-weighted type-II
        correction (phase nudge + frequency integrator)."""
        # --- prior generative flow: the hallucination advances on its own ---
        self.phi += self.om

        Pi = 0.0
        e_rms = np.nan
        if self.closed and v_real is not None:
            v_hat, ph, a = self.render()
            e  = v_hat - v_real                    # residual, IN THE SHARED SPACE
            Pi = self.precision(noise_est)
            e_rms = float(np.sqrt(np.mean(e * e)))
            # first-order inversion of the delay-space residual -> phase error.
            #   d/dphi (1/2||v_hat-v_real||^2) = sum(e * -a sin ph);  curvature ~ (m/2)a^2
            g_phi = np.sum(e * (-a * np.sin(ph)))
            curv  = 0.5 * self.m * a * a + 1e-6
            d_est = -g_phi / curv                  # ~ (phi_true - phi), read off the residual
            self.phi += self.eta_phi * Pi * d_est
            if self.freq_adapt:                    # type-II: integrate phase error into omega
                self.om += self.eta_om * Pi * d_est
            # light amplitude relaxation (keeps the renderer honest, not the point)
            self.la -= self.eta_a * Pi * np.sum(e * v_hat) / (self.m * a * a + 1e-6)
        return Pi, e_rms


# --------------------------------------------------------------------------- #
#  3. Run + score. Circular phase error, per segment.                          #
# --------------------------------------------------------------------------- #

def circ_err(phi_a, phi_b):
    """Absolute circular difference in [0, pi]."""
    d = np.angle(np.exp(1j * (phi_a - phi_b)))
    return np.abs(d)


def roughness(s, t, w=12):
    """Lock-independent sensory-noise estimate: RMS of the second difference over a
    short window. For white noise of std sigma this is ~sqrt(6)*sigma; for a smooth
    oscillation the second difference is O(omega^2), negligible. So it measures the
    noise, not the model's error."""
    lo = max(2, t - w)
    if t < 2:
        return 0.0
    d2 = s[lo:t+1][2:] - 2 * s[lo:t+1][1:-1] + s[lo:t+1][:-2]
    if len(d2) == 0:
        return 0.0
    return float(np.sqrt(np.mean(d2 * d2) / 6.0))


def run(cfg, rng, **override):
    c = dict(cfg); c.update(override)
    s, phi_true, om_true, fog, blackout = make_world(c, rng)
    model = Alavirta(c)

    T = c["T"]
    perr = np.full(T, np.nan)
    oerr = np.full(T, np.nan)          # omega error
    prec = np.full(T, np.nan)
    omhat= np.full(T, np.nan)
    for t in range(T):
        v = None if blackout[t] else delay_vector(s, t, c["m"], c["tau"])
        n_est = roughness(s, t)
        Pi, _ = model.step(v, n_est)
        perr[t] = circ_err(model.phi, phi_true[t])
        oerr[t] = abs(model.om - om_true[t])
        prec[t] = Pi
        omhat[t]= model.om
    return dict(s=s, phi_true=phi_true, om_true=om_true, fog=fog,
                blackout=blackout, perr=perr, oerr=oerr, prec=prec, omhat=omhat)


# --------------------------------------------------------------------------- #
#  4. Active inference — the efferent arm.                                     #
#     Perception cannot change reality's amplitude; only ACTION can. The agent #
#     has a preferred observation a* and drives a motor u to make the world's  #
#     amplitude match it. residual -> muscle, not residual -> belief.          #
# --------------------------------------------------------------------------- #

def active_inference(cfg, rng, act_on):
    T   = cfg["ai_T"]
    a_star = cfg["a_star"]          # preferred amplitude (a prior over sensation)
    kappa  = cfg["ai_kappa"]        # world: a_true relaxes toward the actuator u
    eta_u  = cfg["ai_eta"]
    a_true = cfg["ai_a0"]           # reality starts far from preference
    u      = a_true                 # motor variable
    phi    = 0.0
    surprise = np.zeros(T)
    a_hist   = np.zeros(T)
    for t in range(T):
        phi += cfg["omega0"]
        a_obs = a_true + rng.normal(0, cfg["sigma_clean"])   # what the agent sees
        # precision-weighted preference error
        err = a_obs - a_star
        Pi  = 1.0 / (cfg["sigma_clean"]**2 + 1e-6)
        if act_on:
            u -= eta_u * Pi * err * 0.001        # error drives the steering wheel
        # world responds to the actuator (the only route to changing reality)
        a_true += kappa * (u - a_true)
        surprise[t] = abs(a_obs - a_star)
        a_hist[t]   = a_true
    return surprise, a_hist


# --------------------------------------------------------------------------- #
#  5. Config + main.                                                           #
# --------------------------------------------------------------------------- #

CFG = dict(
    T=2600, omega0=0.15, omega1=0.22,
    t_step=700,
    fog=(1100, 1450),
    blk=(1800, 2150),
    sigma_clean=0.03, sigma_fog=0.55,
    m=8, tau=3,
    eta_phi=0.30, eta_om=0.020, eta_a=0.05,
    sig_ref=0.09,                 # reliability half-point (~3x clean noise)
    # active inference
    ai_T=900, a_star=1.0, ai_a0=0.25, ai_kappa=0.04, ai_eta=6.0,
)


def seg_mean(x, lo, hi, mask=None):
    sl = slice(lo, hi)
    v = x[sl]
    if mask is not None:
        v = v[mask[sl]]
    v = v[~np.isnan(v)]
    return float(np.mean(v)) if len(v) else float("nan")


def make_figure(closed, open_, nofreq, fixhi, ai, path="alavirta.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s_on, a_on, s_off, a_off = ai
    T = CFG["T"]; t = np.arange(T)
    fig, ax = plt.subplots(5, 1, figsize=(11, 13), constrained_layout=True)

    def shade(a):
        a.axvspan(*CFG["fog"], color="0.85", label="fog")
        a.axvspan(*CFG["blk"], color="0.75", label="blackout")
        a.axvline(CFG["t_step"], color="crimson", ls=":", lw=1, label="freq step")

    ax[0].plot(t, closed["s"], lw=.5, color="0.5")
    ax[0].set_title("reality  s(t)  — the only thing the loop ever observes")
    shade(ax[0]); ax[0].set_xlim(0, T); ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(t, closed["perr"], lw=1, color="C0", label="closed (descending renderer)")
    ax[1].plot(t, open_["perr"],  lw=1, color="C3", alpha=.6, label="open (no residual)")
    ax[1].plot(t, nofreq["perr"], lw=1, color="C1", alpha=.6, label="no freq-adapt")
    ax[1].set_title("[A][B][D]  phase-track error (rad)"); ax[1].set_ylim(0, 3.2)
    shade(ax[1]); ax[1].set_xlim(0, T); ax[1].legend(loc="upper left", fontsize=8)

    ax[2].plot(t, closed["prec"], lw=1, color="C2")
    ax[2].set_title("[C]  dynamic precision  — reliability of the sensory channel (drops in fog)")
    shade(ax[2]); ax[2].set_xlim(0, T); ax[2].set_ylim(0, 1.05)

    ax[3].plot(t, closed["om_true"], color="k", lw=2, label="true omega")
    ax[3].plot(t, closed["omhat"], color="C0", lw=1, label="adaptive estimate")
    ax[3].plot(t, nofreq["omhat"], color="C1", lw=1, alpha=.7, label="frozen (no adapt)")
    ax[3].set_title("[D]  frequency estimate tracking the world's rhythm change")
    shade(ax[3]); ax[3].set_xlim(0, T); ax[3].legend(loc="lower right", fontsize=8)

    ta = np.arange(CFG["ai_T"])
    ax[4].plot(ta, a_on,  color="C0", label="action ON  (moves world to prior)")
    ax[4].plot(ta, a_off, color="C3", label="action OFF (passive)")
    ax[4].axhline(CFG["a_star"], color="k", ls="--", lw=1, label="preferred a*")
    ax[4].set_title("[E]  active inference — world amplitude, driven toward the prior by action")
    ax[4].set_xlim(0, CFG["ai_T"]); ax[4].legend(loc="lower right", fontsize=8)

    fig.savefig(path, dpi=110)
    print(f"\nwrote {path}")


def main():
    rng = np.random.default_rng(0)

    closed = run(CFG, np.random.default_rng(1))
    open_  = run(CFG, np.random.default_rng(1), closed=False)
    nofreq = run(CFG, np.random.default_rng(1), freq_adapt=False)
    # fixed precision pinned to full trust -> must chase the fog noise
    fixhi  = run(CFG, np.random.default_rng(1), fixed_prec=1.0)

    T = CFG["T"]; fb = CFG["fog"]; bb = CFG["blk"]; ts = CFG["t_step"]

    print("=" * 66)
    print("ALAVIRTA — closing the predictive loop through the descending stream")
    print("=" * 66)

    print("\n[A] Descending renderer closes the loop (phase-track error, rad)")
    lock = (ts - 200, ts)   # settled window before the frequency step
    print(f"    settled, closed : {seg_mean(closed['perr'], *lock):.4f}")
    print(f"    settled, open   : {seg_mean(open_['perr'],  *lock):.4f}   (no residual, free-run)")

    print("\n[B] Dead-reckoning through the blackout (phase-track error, rad)")
    print(f"    in blackout, closed : {seg_mean(closed['perr'], *bb):.4f}")
    print(f"    in blackout, open   : {seg_mean(open_['perr'],  *bb):.4f}")
    post = (bb[1], bb[1] + 150)
    print(f"    re-lock after,closed: {seg_mean(closed['perr'], *post):.4f}")

    print("\n[C] Dynamic precision vs fixed-high gain, IN THE FOG (phase err, rad)")
    print(f"    dynamic precision : {seg_mean(closed['perr'], *fb):.4f}")
    print(f"    fixed-high gain   : {seg_mean(fixhi['perr'],  *fb):.4f}   (chases the noise)")
    # and both must still be good on clean data afterwards
    clean_after = (fb[1] + 100, bb[0])
    print(f"    (clean after fog) dynamic {seg_mean(closed['perr'], *clean_after):.4f}"
          f" / fixed {seg_mean(fixhi['perr'], *clean_after):.4f}")

    print("\n[D] Frequency-step tracking (omega error after the step)")
    aft = (ts + 250, fb[0])
    print(f"    with freq adapt : {seg_mean(closed['oerr'], *aft):.5f}")
    print(f"    no freq adapt   : {seg_mean(nofreq['oerr'], *aft):.5f}   (stuck at old rhythm)")
    print(f"    (also as phase err) adapt {seg_mean(closed['perr'], *aft):.4f}"
          f" / frozen {seg_mean(nofreq['perr'], *aft):.4f}")

    print("\n[E] Active inference — the efferent arm (mean surprise |a_obs - a*|)")
    s_on,  a_on  = active_inference(CFG, np.random.default_rng(2), act_on=True)
    s_off, a_off = active_inference(CFG, np.random.default_rng(2), act_on=False)
    tail = slice(CFG["ai_T"] - 200, CFG["ai_T"])
    print(f"    action ON  : {np.mean(s_on[tail]):.4f}   (moves the world to fit the prior)")
    print(f"    action OFF : {np.mean(s_off[tail]):.4f}  (paralyzed brain in a jar)")

    print("\n" + "=" * 66)
    ai = (s_on, a_on, s_off, a_off)
    make_figure(closed, open_, nofreq, fixhi, ai)
    return closed, open_, nofreq, fixhi, ai


if __name__ == "__main__":
    main()
