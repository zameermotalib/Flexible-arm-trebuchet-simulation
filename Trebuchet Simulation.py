from __future__ import annotations
from dataclasses import dataclass
import contextlib
import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
np.seterr(over="raise", invalid="raise", divide="raise")

@dataclass
class TrebuchetParams:
    # Geometry and masses from the Assignment 4
    L_B: float = 0.245       # counterweight-side beam length [m]
    L_A: float = 2.82        # arm length from pivot to rigid tip A [m]
    L_P: float = 2.50        # sling/projectile length from A' to projectile [m]
    L_G: float = 0.96        # distance from pivot to beam centre of mass [m]
    m_P: float = 0.50        # projectile mass [kg]
    m_C: float = 86.0        # counterweight mass [kg]
    m_b: float = 4.20        # total beam mass [kg]
    m_A: float = 0.30        # lumped flexible tip mass [kg]; chosen/tunable
    I_C: float = 0.0         # counterweight inertia about its own centre [kg m^2]
    I_b: float = 2.909       # beam inertia about its own centre [kg m^2]

    # Flexible-arm spring
    k: float = 100.0      # transverse spring stiffness [N/m]

    # Gravity and soft-contact ground penalty & Integration
    g: float = 9.81          # normal gravitational acceleration [m/s^2]
    alpha: float = 3000.0    # penalty factor used in projectile gravity term [1/m]
    y0: float = -3.0e-4      # chute/ground reference level [m]
    pivot_y: float = 1.994   # pivot height used only for plotting and soft-contact height [m]
    dt: float = 0.0004       # timestep [s]
    t_end: float = 2.0       # maximum simulated time [s]

    # Initial conditions
    theta0_deg: float | None = None  # initial beam angle [deg]; None = solve from initial_projectile_x
    delta0: float = 0.0              # initial flexible tip deflection [m]
    phi0_deg: float | None = None    # if None, code places projectile on ground/chute
    initial_projectile_x: float | None = -0.20  # x-location of projectile at t=0 [m]
    theta0_guess_deg: float = -35.0  # used only when theta0_deg is None
    theta_dot0: float = 0.0
    delta_dot0: float = 0.0
    phi_dot0: float = 0.0

    # Release/stop condition for the attached-sling simulation
    stop_at_release: bool = True

    # Stop when the projectile v makes this angle with the ground.
    release_launch_angle_deg: float = 45.0
    release_direction: str = "left"    # "left", "right", or "either"
    min_release_time: float = 0.05       # avoid accidental release at t = 0
    min_release_speed: float = 1.0       # ignore near-zero velocity directions [m/s]
    ground_clearance: float = 0.02       # projectile must be above y0 by this much [m]

    # Plotting
    frame_stride: int = 10              # plot every nth timestep
    playback_interval_ms: int = 20      # animation update interval
    visual_deflection_scale: float = 1.0  # set >1 only if deflection is too small to see

    @property
    def m_eq(self) -> float:
        return self.m_b - self.m_A   # Beam mass left over after the flexible tip mass is treated separately

    @property
    def I_eq(self) -> float:
        # Effective rotational inertia used in the beam moment equation
        return self.I_b + self.m_eq * self.L_G**2 + self.I_C + self.m_C * self.L_B**2

@dataclass
class SimulationResult:
    t: np.ndarray
    q: np.ndarray        # columns: theta, delta, phi
    qd: np.ndarray       # columns: theta_dot, delta_dot, phi_dot
    qdd: np.ndarray      # columns: theta_ddot, delta_ddot, phi_ddot
    loads: np.ndarray    # columns: T, F_r, M
    gmod: np.ndarray
    projectile_xy: np.ndarray
    release_index: int
    release_speed: float
    release_angle_deg: float       # global angle from +x [deg]
    release_angle_to_ground_deg: float  # acute launch angle above horizontal ground [deg]

# Unit vectors and geometry
def unit_vectors(theta: float, phi: float):
    """Return e_r, e_theta, e_s, e_phi."""
    e_r = np.array([np.cos(theta), np.sin(theta)])
    e_theta = np.array([-np.sin(theta), np.cos(theta)])

    e_s = np.array([np.cos(phi), np.sin(phi)])
    e_phi = np.array([-np.sin(phi), np.cos(phi)])
    return e_r, e_theta, e_s, e_phi

def positions(q: np.ndarray, p: TrebuchetParams, *, visual: bool = False):
    """Compute important points. If visual=True, delta is optionally scaled for beam drawing only."""
    theta, delta, phi = q
    if visual:
        delta = p.visual_deflection_scale * delta

    e_r, e_theta, e_s, _ = unit_vectors(theta, phi)
    O = np.array([0.0, p.pivot_y])
    B = O - p.L_B * e_r
    A = O + p.L_A * e_r
    A_deflected = A - delta * e_theta
    P = A_deflected + p.L_P * e_s
    G = O + p.L_G * e_r
    return O, B, G, A, A_deflected, P

def projectile_velocity(q: np.ndarray, qd: np.ndarray, p: TrebuchetParams) -> np.ndarray:
    """Velocity of projectile while it is still attached to the sling."""
    theta, delta, phi = q
    theta_dot, delta_dot, phi_dot = qd
    e_r, e_theta, _, e_phi = unit_vectors(theta, phi)
    v_A = p.L_A * theta_dot * e_theta
    v_A_deflected_relative = -delta_dot * e_theta + delta * theta_dot * e_r
    v_P_relative = p.L_P * phi_dot * e_phi
    return v_A + v_A_deflected_relative + v_P_relative

def flexible_beam_curve(q: np.ndarray, p: TrebuchetParams, n: int = 80) -> np.ndarray:
    """ Curve used only for visualising the flexible beam."""
    theta, delta, phi = q
    delta = p.visual_deflection_scale * delta
    e_r, e_theta, _, _ = unit_vectors(theta, phi)
    O = np.array([0.0, p.pivot_y])
    s_values = np.linspace(-p.L_B, p.L_A, n)
    pts = []
    for s in s_values:
        if s <= 0.0:
            pts.append(O + s * e_r)
        else:
            shape = (s / p.L_A) ** 2
            pts.append(O + s * e_r - delta * shape * e_theta)
    return np.array(pts)

def rigid_beam_line(q: np.ndarray, p: TrebuchetParams, n: int = 20) -> np.ndarray:
    theta = q[0]
    e_r = np.array([np.cos(theta), np.sin(theta)])
    O = np.array([0.0, p.pivot_y])
    s_values = np.linspace(-p.L_B, p.L_A, n)
    return np.array([O + s * e_r for s in s_values])

# Dynamics: 6 x 6 simultaneous equation system
def projectile_penalty_gravity(y_projectile: float, p: TrebuchetParams) -> float:
    """Soft-contact gravity used ONLY in the projectile equations. When the projectile is on/near the chute level, this reduces its effective gravity term."""
    exponent = -p.alpha * (y_projectile - p.y0)
    exponent = np.clip(exponent, -50.0, 50.0)  # prevents overflow
    return p.g * (1.0 - np.exp(exponent))

def solve_state(q: np.ndarray, qd: np.ndarray, p: TrebuchetParams):
    """Solve for accelerations and internal loads."""
    theta, delta, phi = q
    theta_dot, delta_dot, phi_dot = qd
    c = np.cos(theta - phi)
    s = np.sin(theta - phi)
    _, _, _, _, _, P = positions(q, p)
    g_mod = projectile_penalty_gravity(P[1], p)

    # Matrix from the derived 3-DoF flexible-arm equations
    A = np.array([
        [delta * c - p.L_A * s,       s,       0.0,       1.0 / p.m_P,       0.0,  0.0],
        [p.L_A * c + delta * s,      -c,       p.L_P,     0.0,               0.0,  0.0],
        [-p.m_A * p.L_A,              p.m_A,   0.0,      -s,                 0.0,  0.0],
        [-p.m_A * delta,              0.0,     0.0,       c,                -1.0,  0.0],
        [p.m_A * delta**2,            0.0,     0.0,      -delta * c,         0.0, -1.0],
        [p.I_eq,                      0.0,     0.0,       0.0,               0.0,  1.0],
    ], dtype=float)

    b = np.array([
        delta * theta_dot**2 * s
        - (2.0 * delta_dot * theta_dot - p.L_A * theta_dot**2) * c
        + p.L_P * phi_dot**2
        - g_mod * np.sin(phi),

        p.L_A * theta_dot**2 * s
        - g_mod * np.cos(phi)
        - delta * theta_dot**2 * c
        - 2.0 * delta_dot * theta_dot * s,

        p.m_A * p.g * np.cos(theta)
        + p.m_A * delta * theta_dot**2
        - p.k * delta,

        p.m_A * p.g * np.sin(theta)
        + 2.0 * p.m_A * delta_dot * theta_dot
        - p.m_A * p.L_A * theta_dot**2,

        -p.m_A * p.g * np.sin(theta) * delta,

        p.m_C * p.g * p.L_B * np.cos(theta)
        - p.m_eq * p.g * p.L_G * np.cos(theta)
        - p.L_A * p.k * delta,
    ], dtype=float)

    x = np.linalg.solve(A, b)

    qdd = x[:3]
    loads = x[3:]
    return qdd, loads, g_mod

# Multi-step temporal integration: AB for velocity, AM for position
def next_velocity_AB(qd_hist: np.ndarray, qdd_hist: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Explicit Adams-Bashforth integration of acceleration to velocity."""
    if n == 0:
        # 1-point Adams-Bashforth / Forward Euler
        return qd_hist[n] + dt * qdd_hist[n]
    if n == 1:
        # 2-point Adams-Bashforth
        return qd_hist[n] + 0.5 * dt * (3.0 * qdd_hist[n] - qdd_hist[n - 1])

    # 3-point Adams-Bashforth
    return qd_hist[n] + (dt / 12.0) * (
        23.0 * qdd_hist[n]
        - 16.0 * qdd_hist[n - 1]
        + 5.0 * qdd_hist[n - 2]
    )

def next_position_AM(q_hist: np.ndarray, qd_hist: np.ndarray, qd_next: np.ndarray, n: int, dt: float) -> np.ndarray:
    """Implicit Adams-Moulton integration of velocity to position"""
    if n == 0:
        # 2-point Adams-Moulton / Trapezoidal rule
        return q_hist[n] + 0.5 * dt * (qd_next + qd_hist[n])
    if n == 1:
        # 3-point Adams-Moulton
        return q_hist[n] + (dt / 12.0) * (
            5.0 * qd_next
            + 8.0 * qd_hist[n]
            - qd_hist[n - 1]
        )

    # 4-point Adams-Moulton
    return q_hist[n] + (dt / 24.0) * (
        9.0 * qd_next
        + 19.0 * qd_hist[n]
        - 5.0 * qd_hist[n - 1]
        + qd_hist[n - 2]
    )

# Simulation driver
def _wrap_to_pi(angle: float) -> float:
    """Return equivalent angle in the range [-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))

def _tip_position_from_theta(theta: float, delta: float, p: TrebuchetParams) -> np.ndarray:
    """Deflected tip A' position for a trial theta and delta."""
    e_r = np.array([np.cos(theta), np.sin(theta)])
    e_theta = np.array([-np.sin(theta), np.cos(theta)])
    O = np.array([0.0, p.pivot_y])
    return O + p.L_A * e_r - delta * e_theta

def _solve_theta_for_ground_projectile(p: TrebuchetParams, delta0: float) -> float:
    """Find an initial theta such that the sling of length L_P reaches the chosen projectile ground point."""
    if p.initial_projectile_x is None:
        raise ValueError("initial_projectile_x must be set when solving theta automatically.")

    target = np.array([p.initial_projectile_x, p.y0])

    def residual(theta: float) -> float:
        Ap = _tip_position_from_theta(theta, delta0, p)
        return float(np.linalg.norm(target - Ap) - p.L_P)

    # Search a practical trebuchet start range.
    candidates = []
    theta_grid = np.deg2rad(np.linspace(-85.0, 25.0, 1200))
    r_grid = np.array([residual(th) for th in theta_grid])

    for i in range(len(theta_grid) - 1):
        r0 = r_grid[i]
        r1 = r_grid[i + 1]
        if not np.isfinite(r0) or not np.isfinite(r1):
            continue
        if r0 == 0.0:
            candidates.append(theta_grid[i])
        elif r0 * r1 < 0.0:
            lo = theta_grid[i]
            hi = theta_grid[i + 1]
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                rm = residual(mid)
                if residual(lo) * rm <= 0.0:
                    hi = mid
                else:
                    lo = mid
            candidates.append(0.5 * (lo + hi))


    guess = np.deg2rad(p.theta0_guess_deg)
    return float(min(candidates, key=lambda th: abs(th - guess)))

def make_initial_state(p: TrebuchetParams):
    delta0 = p.delta0

    # If theta0_deg is None, solve theta0 so that P starts at
    if p.theta0_deg is None and p.phi0_deg is None and p.initial_projectile_x is not None:
        theta0 = _solve_theta_for_ground_projectile(p, delta0)
        Ap = _tip_position_from_theta(theta0, delta0, p)
        target = np.array([p.initial_projectile_x, p.y0])
        d = target - Ap
        phi0 = np.arctan2(d[1], d[0])
    else:
        theta0 = np.deg2rad(0.0 if p.theta0_deg is None else p.theta0_deg)

        if p.phi0_deg is None:
            # Choose phi0 so that the projectile starts on the chute/ground level. Use the left-hand sling branch, so the projectile starts before/near
            A_deflected_y = p.pivot_y + p.L_A * np.sin(theta0) - delta0 * np.cos(theta0)
            arg = (p.y0 - A_deflected_y) / p.L_P
            if abs(arg) <= 1.0:
                base = np.arcsin(arg)
                phi0 = _wrap_to_pi(np.pi - base)  # left-hand branch, cos(phi) < 0
            else:
                phi0 = np.deg2rad(-150.0)
        else:
            phi0 = np.deg2rad(p.phi0_deg)

    q0 = np.array([theta0, delta0, phi0], dtype=float)
    qd0 = np.array([p.theta_dot0, p.delta_dot0, p.phi_dot0], dtype=float)
    return q0, qd0

def _projectile_launch_angles(q_now: np.ndarray, qd_now: np.ndarray, p: TrebuchetParams):
    """Return projectile speed, global velocity angle, and launch angle above the ground."""
    v = projectile_velocity(q_now, qd_now, p)
    vx, vy = float(v[0]), float(v[1])
    speed = float(np.linalg.norm(v))
    global_angle_deg = float(np.rad2deg(np.arctan2(vy, vx)))
    angle_to_ground_deg = float(np.rad2deg(np.arctan2(vy, abs(vx)))) if speed > 0.0 else 0.0
    return speed, global_angle_deg, angle_to_ground_deg, vx, vy

def _release_ready(q_now: np.ndarray, qd_now: np.ndarray, xy_now: np.ndarray, t_now: float, p: TrebuchetParams) -> bool:
    """Basic checks before the 45 degree release angle is allowed to trigger."""
    if not p.stop_at_release:
        return False
    if t_now < p.min_release_time:
        return False
    if xy_now[1] <= p.y0 + p.ground_clearance:
        return False

    speed, _, _, vx, vy = _projectile_launch_angles(q_now, qd_now, p)
    if speed < p.min_release_speed or vy <= 0.0:
        return False

    direction = p.release_direction.lower().strip()
    if direction == "right" and vx <= 0.0:
        return False
    if direction == "left" and vx >= 0.0:
        return False
    if direction not in ("left", "right", "either"):
        raise ValueError('release_direction must be "left", "right", or "either".')

    return True

def _release_crossing_fraction(q0: np.ndarray, qd0: np.ndarray, xy0: np.ndarray, t0: float, q1: np.ndarray, qd1: np.ndarray, xy1: np.ndarray, t1: float, p: TrebuchetParams):
    """Return interpolation fraction lambda in [0, 1] if the projectile launch angle crosses the target release angle during the step from state 0 to state 1."""
    if not (_release_ready(q0, qd0, xy0, t0, p) or _release_ready(q1, qd1, xy1, t1, p)):
        return None

    target = float(p.release_launch_angle_deg)
    _, _, angle0, _, _ = _projectile_launch_angles(q0, qd0, p)
    _, _, angle1, _, _ = _projectile_launch_angles(q1, qd1, p)
    f0 = angle0 - target
    f1 = angle1 - target

    # Exact hit at the start/end of a step.
    if abs(f0) < 1.0e-12 and _release_ready(q0, qd0, xy0, t0, p):
        return 0.0
    if abs(f1) < 1.0e-12 and _release_ready(q1, qd1, xy1, t1, p):
        return 1.0

    # Release on the first crossing of the chosen launch angle.
    if f0 * f1 > 0.0:
        return None

    denom = angle1 - angle0
    if abs(denom) < 1.0e-12:
        return None

    lam = (target - angle0) / denom
    if 0.0 <= lam <= 1.0:
        # Check the interpolated state still satisfies direction, speed, and clearance.
        q_rel = (1.0 - lam) * q0 + lam * q1
        qd_rel = (1.0 - lam) * qd0 + lam * qd1
        xy_rel = (1.0 - lam) * xy0 + lam * xy1
        t_rel = (1.0 - lam) * t0 + lam * t1
        if _release_ready(q_rel, qd_rel, xy_rel, t_rel, p):
            return float(lam)

    return None

def simulate(p: TrebuchetParams) -> SimulationResult:
    n_max = int(np.ceil(p.t_end / p.dt)) + 1
    t = np.zeros(n_max)
    q = np.zeros((n_max, 3))
    qd = np.zeros((n_max, 3))
    qdd = np.zeros((n_max, 3))
    loads = np.zeros((n_max, 3))
    gmod = np.zeros(n_max)
    projectile_xy = np.zeros((n_max, 2))
    q[0], qd[0] = make_initial_state(p)
    _, _, _, _, _, projectile_xy[0] = positions(q[0], p)

    release_index = n_max - 1

    for n in range(n_max - 1):
        t[n] = n * p.dt
        qdd[n], loads[n], gmod[n] = solve_state(q[n], qd[n], p)
        _, _, _, _, _, projectile_xy[n] = positions(q[n], p)

        qd_next = next_velocity_AB(qd, qdd, n, p.dt)
        q_next = next_position_AM(q, qd, qd_next, n, p.dt)
        t_next = (n + 1) * p.dt
        _, _, _, _, _, xy_next = positions(q_next, p)

        # Check whether the projectile velocity crosses the requested 45 degree
        # If it does, interpolate to the crossing instead of stopping at a timestep that is only approximately 45 deg.
        lam = _release_crossing_fraction(q[n], qd[n], projectile_xy[n], t[n], q_next, qd_next, xy_next, t_next, p)
        if lam is not None:
            q[n + 1] = (1.0 - lam) * q[n] + lam * q_next
            qd[n + 1] = (1.0 - lam) * qd[n] + lam * qd_next
            t[n + 1] = (1.0 - lam) * t[n] + lam * t_next
            _, _, _, _, _, projectile_xy[n + 1] = positions(q[n + 1], p)
            release_index = n + 1
            break

        q[n + 1] = q_next
        qd[n + 1] = qd_next
        t[n + 1] = t_next
        projectile_xy[n + 1] = xy_next

    # Trim arrays to simulated length
    final = release_index + 1
    t = t[:final]
    q = q[:final]
    qd = qd[:final]
    qdd = qdd[:final]
    loads = loads[:final]
    gmod = gmod[:final]
    projectile_xy = projectile_xy[:final]

    # Fill final diagnostic values if release happened before the last acceleration solve
    qdd[-1], loads[-1], gmod[-1] = solve_state(q[-1], qd[-1], p)
    _, _, _, _, _, projectile_xy[-1] = positions(q[-1], p)
    release_speed, release_angle_deg, release_angle_to_ground_deg, _, _ = _projectile_launch_angles(q[-1], qd[-1], p)

    return SimulationResult(
        t=t,
        q=q,
        qd=qd,
        qdd=qdd,
        loads=loads,
        gmod=gmod,
        projectile_xy=projectile_xy,
        release_index=len(t) - 1,
        release_speed=release_speed,
        release_angle_deg=release_angle_deg,
        release_angle_to_ground_deg=release_angle_to_ground_deg,
    )

# Animation
def animate(result: SimulationResult, p: TrebuchetParams):
    idx = np.arange(0, len(result.t), max(1, p.frame_stride))
    if idx[-1] != len(result.t) - 1:
        idx = np.append(idx, len(result.t) - 1)

    # Precompute axis limits from all plotted positions
    all_pts = []
    for i in idx:
        O, B, G, A, Ap, P = positions(result.q[i], p)
        all_pts.extend([O, B, A, Ap, P])
    all_pts = np.array(all_pts)
    x_min, y_min = np.min(all_pts, axis=0) - 1.0
    x_max, y_max = np.max(all_pts, axis=0) + 1.0
    y_min = min(y_min, p.y0 - 1.0)

    fig, ax = plt.subplots(figsize=(13, 7))
    plt.subplots_adjust(left=0.08, right=0.68, bottom=0.16, top=0.90)

    ax.set_title("3-DoF Flexible-Arm FCW Trebuchet Motion", fontsize=14, fontweight="bold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ground_line, = ax.plot([x_min, x_max], [p.y0, p.y0], "k-", lw=1, alpha=0.25, label="Ground/chute")
    beam_line, = ax.plot([], [], "b-", lw=4, label="Flexible beam")
    rigid_line, = ax.plot([], [], "b--", lw=1.5, alpha=0.45, label="Rigid beam reference")
    sling_line, = ax.plot([], [], "g-", lw=2.5, label="Sling")
    traj_line, = ax.plot([], [], "r-", lw=1.2, alpha=0.5, label="Projectile trajectory")
    projectile_dot, = ax.plot([], [], "ro", ms=8, label="Projectile")
    counterweight_dot, = ax.plot([], [], "ko", ms=13, label="Counterweight")
    pivot_dot, = ax.plot([], [], "ko", ms=5, label="Pivot")
    tip_dot, = ax.plot([], [], "o", ms=5, color="tab:purple", label="Deflected tip A'")

    info_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75),
        fontsize=10,
    )

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        borderaxespad=0.0,
        frameon=True,
        title="Key",
    )

    button_ax = fig.add_axes([0.42, 0.045, 0.16, 0.065])
    play_button = Button(button_ax, "Pause")
    is_playing = {"value": True}

    def draw(frame_number: int):
        frame_number = int(frame_number) % len(idx)
        i = idx[frame_number]
        q_i = result.q[i]
        qd_i = result.qd[i]

        O, B, G, A, Ap, P = positions(q_i, p)
        beam_pts = flexible_beam_curve(q_i, p)
        rigid_pts = rigid_beam_line(q_i, p)

        beam_line.set_data(beam_pts[:, 0], beam_pts[:, 1])
        rigid_line.set_data(rigid_pts[:, 0], rigid_pts[:, 1])
        sling_line.set_data([Ap[0], P[0]], [Ap[1], P[1]])
        projectile_dot.set_data([P[0]], [P[1]])
        counterweight_dot.set_data([B[0]], [B[1]])
        pivot_dot.set_data([O[0]], [O[1]])
        tip_dot.set_data([Ap[0]], [Ap[1]])

        traj = result.projectile_xy[: i + 1]
        traj_line.set_data(traj[:, 0], traj[:, 1])

        theta_deg = np.rad2deg(q_i[0])
        phi_deg = np.rad2deg(q_i[2])
        delta_mm = 1000.0 * q_i[1]
        omega = qd_i[0]
        v_proj = np.linalg.norm(projectile_velocity(q_i, qd_i, p))

        info_text.set_text(
            f"t = {result.t[i]:.3f} s\n"
            f"theta = {theta_deg:.1f} deg\n"
            f"phi = {phi_deg:.1f} deg\n"
            f"delta = {delta_mm:.2f} mm\n"
            f"beam omega = {omega:.2f} rad/s\n"
            f"v_proj = {v_proj:.2f} m/s\n"
            f"g_mod = {result.gmod[i]:.2f} m/s²"
        )
        return (
            beam_line, rigid_line, sling_line, projectile_dot, counterweight_dot,
            pivot_dot, tip_dot, traj_line, info_text, ground_line
        )

    def update_animation(frame_number: int):
        return draw(frame_number)

    ani = FuncAnimation(
        fig,
        update_animation,
        frames=len(idx),
        interval=p.playback_interval_ms,
        blit=False,
        repeat=True,
    )

    def on_button_clicked(event):
        is_playing["value"] = not is_playing["value"]
        if is_playing["value"]:
            ani.event_source.start()
            play_button.label.set_text("Pause")
        else:
            ani.event_source.stop()
            play_button.label.set_text("Play")
        fig.canvas.draw_idle()

    button_callback_id = play_button.on_clicked(on_button_clicked)

    def on_key_press(event):
        # Press the space bar as another reliable way to pause/play.
        if event.key == " ":
            on_button_clicked(event)

    key_callback_id = fig.canvas.mpl_connect("key_press_event", on_key_press)

    # Keep references alive so the animation and widgets do not get garbage-collected.
    fig._trebuchet_animation = ani
    fig._trebuchet_play_button = play_button
    fig._trebuchet_button_callback_id = button_callback_id
    fig._trebuchet_key_callback_id = key_callback_id
    plt.show()

# Velocity plot
def sling_tip_speed_history(result: SimulationResult, p: TrebuchetParams) -> np.ndarray:
    """Speed of the sling tip/projectile while the projectile is still attached."""
    return np.array([
        np.linalg.norm(projectile_velocity(result.q[i], result.qd[i], p))
        for i in range(len(result.t))
    ])

def plot_beam_omega_and_sling_tip_velocity(result: SimulationResult, p: TrebuchetParams):
    """Plot beam angular velocity and sling tip velocity versus time."""
    t = result.t
    beam_omega = np.abs(result.qd[:, 0])
    sling_tip_speed = sling_tip_speed_history(result, p)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    line1, = ax1.plot(t, beam_omega, "b-", lw=2.0, label="Beam Angular Velocity")
    line2, = ax2.plot(t, sling_tip_speed, "g-", lw=2.0, label="Sling Tip Velocity")
    ax1.set_title(f"Beam Angular Velocity vs Sling Tip Velocity  —  k = {p.k:,.0f} N/m", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Beam Angular Velocity [rad/s]", color="b")
    ax2.set_ylabel("Sling Tip Velocity [m/s]", color="g")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2.tick_params(axis="y", labelcolor="g")
    ax1.grid(True, alpha=0.3)
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")

    # Mark the final release point on the sling-tip velocity curve.
    ax2.plot(t[-1], sling_tip_speed[-1], "go", ms=5)
    ax2.annotate(
        f"release\n{t[-1]:.3f} s, {sling_tip_speed[-1]:.2f} m/s",
        xy=(t[-1], sling_tip_speed[-1]),
        xytext=(-95, -35),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", lw=1.0),
        fontsize=9,
    )

    fig.tight_layout()
    plt.show()

# Energy calculations and plot
def energy_components(result: SimulationResult, p: TrebuchetParams):
    """Compute energy terms for the flexible-arm trebuchet."""
    n = len(result.t)

    PE_arm_cw = np.zeros(n)
    KE_arm_rot = np.zeros(n)
    KE_tip = np.zeros(n)
    PE_tip = np.zeros(n)
    KE_projectile = np.zeros(n)
    PE_projectile = np.zeros(n)
    PE_spring = np.zeros(n)
    E_total = np.zeros(n)

    for i in range(n):
        theta, delta, phi = result.q[i]
        theta_dot, delta_dot, phi_dot = result.qd[i]

        # Kinetic energies
        KE_arm_rot[i] = 0.5 * p.I_eq * theta_dot**2

        v_Ap_sq = delta**2 * theta_dot**2 + (p.L_A * theta_dot - delta_dot)**2
        KE_tip[i] = 0.5 * p.m_A * v_Ap_sq

        vP = projectile_velocity(result.q[i], result.qd[i], p)
        KE_projectile[i] = 0.5 * p.m_P * float(np.dot(vP, vP))

        # Shifted potential energies
        PE_arm_cw[i] = p.g * (p.m_C * p.L_B - p.m_eq * p.L_G) * (1.0 - np.sin(theta))

        # Flexible tip mass gravitational PE above the chute/ground reference.
        y_Ap = p.pivot_y + p.L_A * np.sin(theta) - delta * np.cos(theta)
        PE_tip[i] = p.m_A * p.g * (y_Ap - p.y0)

        # Projectile gravitational PE above the chute/ground, using the same penalty form that appears in the projectile EoMs.
        y_P = result.projectile_xy[i, 1]
        z = y_P - p.y0
        exponent = np.clip(-p.alpha * z, -50.0, 50.0)
        PE_projectile[i] = p.m_P * p.g * (z + (np.exp(exponent) - 1.0) / p.alpha)

        # Spring strain energy.
        PE_spring[i] = 0.5 * p.k * delta**2

        E_total[i] = (
            PE_arm_cw[i] + KE_arm_rot[i] + KE_tip[i] + PE_tip[i]
            + KE_projectile[i] + PE_projectile[i] + PE_spring[i]
        )

    return {
        "PE_arm_cw": PE_arm_cw,
        "KE_arm_rot": KE_arm_rot,
        "KE_tip": KE_tip,
        "PE_tip": PE_tip,
        "KE_projectile": KE_projectile,
        "PE_projectile": PE_projectile,
        "PE_spring": PE_spring,
        "E_total": E_total,
    }

def plot_energy_distribution(result: SimulationResult, p: TrebuchetParams):
    """Plot the energy distribution"""
    terms = energy_components(result, p)
    t = result.t

    E0 = terms["E_total"][0]
    scale = E0 if abs(E0) > 1.0e-12 else 1.0

    def pct(arr):
        return 100.0 * arr / scale

    fig, ax = plt.subplots(figsize=(13, 6.5))

    ax.plot(t, pct(terms["PE_arm_cw"]), "b-", lw=1.8, label="PE (Arm+CW)")
    ax.plot(t, pct(terms["KE_arm_rot"]), "r-", lw=1.8, label="KE (Arm Rotation)")
    ax.plot(t, pct(terms["KE_tip"]), "c-", lw=1.8, label="KE (Tip Mass)")
    ax.plot(t, pct(terms["PE_tip"]), "m-", lw=1.8, label="PE (Tip Mass)")
    ax.plot(t, pct(terms["KE_projectile"]), "g-", lw=1.8, label="KE (Projectile)")
    ax.plot(t, pct(terms["PE_spring"]), color="olive", ls="--", lw=1.8, label="PE (Spring)")
    ax.plot(t, pct(terms["E_total"]), "k-", lw=2.0, label="Total Energy (100%)")
    ax.set_title(f"Energy Distribution  —  k = {p.k:,.0f} N/m", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy [% of initial total]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    plt.show()

def plot_energy_error(result: SimulationResult, p: TrebuchetParams):
    """Plot absolute total mechanical energy error percentage versus time0"""
    terms = energy_components(result, p)
    t = result.t

    E_total = terms["E_total"]
    E0 = E_total[0]
    scale = abs(E0) if abs(E0) > 1.0e-12 else 1.0
    signed_energy_error_percent = 100.0 * (E_total - E0) / scale
    absolute_energy_error_percent = np.abs(signed_energy_error_percent)
    max_abs_error = float(np.max(absolute_energy_error_percent))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(t, absolute_energy_error_percent, "r-", lw=2.0, label="Absolute Energy Error")
    ax.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    ax.set_title(
        f"Absolute Energy Error Percentage vs Time  —  k = {p.k:,.0f} N/m"
        f"\nMax error = {max_abs_error:.4f}%",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Absolute Energy Error [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    plt.show()

    return absolute_energy_error_percent

# Verification helper functions
def max_absolute_energy_error_percent(result: SimulationResult, p: TrebuchetParams) -> float:
    """Return max(100*|E(t)-E(0)|/|E(0)|) for one simulation."""
    terms = energy_components(result, p)
    E = terms["E_total"]
    E0 = E[0]
    scale = abs(E0) if abs(E0) > 1.0e-12 else 1.0
    return float(np.max(100.0 * np.abs(E - E0) / scale))

def copy_params(p: TrebuchetParams, **changes) -> TrebuchetParams:
    """ Create a copy of the current parameter set with selected values changed"""
    data = p.__dict__.copy()
    data.update(changes)
    return TrebuchetParams(**data)

def summarize_simulation_run(p: TrebuchetParams) -> dict:
    """Run one simulation and return the small set of values needed for verification tables """
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = simulate(p)

        terms = energy_components(result, p)
        values = np.concatenate((
            result.q.ravel(), result.qd.ravel(), result.qdd.ravel(),
            result.loads.ravel(), terms["E_total"], terms["PE_spring"]
        ))
        state_values = np.concatenate((result.q.ravel(), result.qd.ravel(), result.qdd.ravel(), result.loads.ravel()))
        if not np.all(np.isfinite(values)) or np.max(np.abs(state_values)) > 1.0e8:
            raise FloatingPointError

        release_error = abs(result.release_angle_to_ground_deg - p.release_launch_angle_deg)
        released_at_target_angle = release_error < 0.1 and result.t[-1] < p.t_end - 0.5 * p.dt

        return {
            "dt": p.dt,
            "k": p.k,
            "t_release": float(result.t[-1]),
            "v_release": float(result.release_speed),
            "theta_release_deg": float(np.rad2deg(result.q[-1, 0])),
            "phi_release_deg": float(np.rad2deg(result.q[-1, 2])),
            "delta_release_mm": float(1000.0 * result.q[-1, 1]),
            "delta_max_mm": float(1000.0 * np.max(np.abs(result.q[:, 1]))),
            "max_spring_energy_J": float(np.max(terms["PE_spring"])),
            "max_energy_error_percent": max_absolute_energy_error_percent(result, p),
            "release_angle_deg": float(result.release_angle_to_ground_deg),
            "released": released_at_target_angle,
            "stable": True,
        }
    except (FloatingPointError, np.linalg.LinAlgError):
        print(f"Numerical instability: k = {p.k:,.0f} N/m, dt = {p.dt:.6f} s; values are unstable and cannot be solved.")
        return {
            "dt": p.dt,
            "k": p.k,
            "t_release": np.nan,
            "v_release": np.nan,
            "theta_release_deg": np.nan,
            "phi_release_deg": np.nan,
            "delta_release_mm": np.nan,
            "delta_max_mm": np.nan,
            "max_spring_energy_J": np.nan,
            "max_energy_error_percent": np.nan,
            "release_angle_deg": np.nan,
            "released": False,
            "stable": False,
        }

def _cell(value: float, spec: str) -> str:
    return format(value, spec) if np.isfinite(value) else "UNSTABLE"

def print_main_simulation_summary(result: SimulationResult, p: TrebuchetParams):
    """Print a summary for the chosen visual simulation."""
    summary = summarize_simulation_run(p)
    if not summary["stable"]:
        print("Chosen visual simulation is unstable and will not be plotted.")
        return False

    print("=" * 78)
    print("CHOSEN VISUAL SIMULATION")
    print("=" * 78)
    print("This is the parameter set used for the animation and the three main plots.")
    print(f"Spring stiffness, k             : {p.k:,.0f} N/m")
    print(f"Flexible tip mass, m_A          : {p.m_A:.3f} kg")
    print(f"Timestep, dt                    : {p.dt:.6f} s")
    print(f"Initial projectile position     : x = {result.projectile_xy[0,0]:.3f} m, y = {result.projectile_xy[0,1]:.4f} m")
    print(f"Initial theta, phi              : {np.rad2deg(result.q[0,0]):.2f} deg, {np.rad2deg(result.q[0,2]):.2f} deg")
    print("-" * 78)
    print(f"Release time                    : {result.t[-1]:.4f} s")
    print(f"Release theta, phi              : {np.rad2deg(result.q[-1,0]):.2f} deg, {np.rad2deg(result.q[-1,2]):.2f} deg")
    print(f"Release delta                   : {1000.0 * result.q[-1,1]:.3f} mm")
    print(f"Maximum |delta|                 : {summary['delta_max_mm']:.3f} mm")
    print(f"Projectile release speed        : {result.release_speed:.3f} m/s")
    print(f"Projectile launch angle         : {result.release_angle_to_ground_deg:.2f} deg above ground")
    print(f"Energy norm, max absolute error : {summary['max_energy_error_percent']:.5f} %")
    print("=" * 78)
    print()
    return True

def _print_table(title: str, subtitle: str, headers: list[str], rows: list[list[str]]):
    """Small fixed-width table printer for terminal output."""
    print("=" * 78)
    print(title)
    print("=" * 78)
    if subtitle:
        print(subtitle)
        print()

    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(str(cell)))

    header_line = "  ".join(h.ljust(widths[j]) for j, h in enumerate(headers))
    sep_line = "  ".join("-" * widths[j] for j in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(str(cell).ljust(widths[j]) for j, cell in enumerate(row)))
    print()

def print_verification_tables(p: TrebuchetParams):
    """Print verification data"""
    print("=" * 78)
    print("NUMERICAL SOLUTION VERIFICATION")
    print("=" * 78)
    print("The tables below are supporting checks for the report.")
    print("They are separate from the chosen visual simulation above.")
    print("No animations are generated for these extra runs.")
    print()

    # 1) Energy norm for the chosen case
    chosen = summarize_simulation_run(p)
    _print_table(
        "1) ENERGY NORM FOR THE CHOSEN CASE",
        "This is the same parameter set used for the animation/plots.",
        ["k [N/m]", "dt [s]", "t_rel [s]", "v_rel [m/s]", "max |delta| [mm]", "max |E err| [%]"],
        [[
            _cell(chosen["k"], ",.0f"),
            _cell(chosen["dt"], ".6f"),
            _cell(chosen["t_release"], ".4f"),
            _cell(chosen["v_release"], ".3f"),
            _cell(chosen["delta_max_mm"], ".3f"),
            _cell(chosen["max_energy_error_percent"], ".5f"),
        ]],
    )

    # 2) Timestep convergence at the chosen stiffness
    dt_values = [0.01, 0.008, 0.004, 0.002, 0.001, 0.0008, 0.0004, 0.0002, 0.0001]
    dt_values = sorted(set(float(dt) for dt in dt_values if dt > 0.0), reverse=True)

    dt_summaries = [summarize_simulation_run(copy_params(p, dt=dt)) for dt in dt_values]
    stable_dt_summaries = [s for s in dt_summaries if s["stable"]]
    ref = min(stable_dt_summaries, key=lambda item: item["dt"]) if stable_dt_summaries else None

    dt_rows = []
    for s in dt_summaries:
        if s["stable"] and ref is not None:
            v_error = abs((s["v_release"] - ref["v_release"]) / ref["v_release"]) * 100.0
            delta_error = abs((s["delta_max_mm"] - ref["delta_max_mm"]) / ref["delta_max_mm"]) * 100.0
        else:
            v_error = np.nan
            delta_error = np.nan
        dt_rows.append([
            _cell(s["dt"], ".6f"),
            _cell(s["t_release"], ".4f"),
            _cell(s["v_release"], ".3f"),
            _cell(s["delta_max_mm"], ".3f"),
            _cell(v_error, ".4f"),
            _cell(delta_error, ".4f"),
            _cell(s["max_energy_error_percent"], ".5f"),
        ])

    _print_table(
        "2) TIMESTEP CONVERGENCE CHECK",
        f"Spring stiffness is kept at k = {p.k:,.0f} N/m. ",
        ["dt [s]", "t_rel [s]", "v_rel [m/s]", "max |delta| [mm]", "v error [%]", "delta error [%]", "max |E err| [%]"],
        dt_rows,
    )

    # 3) Stiffness sensitivity for the new flexible-arm DoF
    stiffness_values = [25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 20000] #100.0, 1_000.0, 10_000.0, p.k, 100_000.0, 500_000.0, 1_500_000.0, 2_350_000.0
    stiffness_values = sorted(set(float(k) for k in stiffness_values if k > 0.0))
    k_rows = []
    for k in stiffness_values:
        s = summarize_simulation_run(copy_params(p, k=k))
        k_rows.append([
            _cell(s["k"], ",.0f"),
            _cell(s["t_release"], ".4f"),
            _cell(s["v_release"], ".3f"),
            _cell(s["delta_max_mm"], ".3f"),
            _cell(s["max_spring_energy_J"], ".4f"),
            _cell(s["max_energy_error_percent"], ".5f"),
        ])

    _print_table(
        "3) STIFFNESS SENSITIVITY CHECK",
        "Only k is changed here. This shows how the added flexible-arm DoF affects the solution.",
        ["k [N/m]", "t_rel [s]", "v_rel [m/s]", "max |delta| [mm]", "max PE spring [J]", "max |E err| [%]"],
        k_rows,
    )

    # 4) Rigid-arm limit / reduction toward 2-DoF behaviour
    rigid_k_values = [22000, 22500, 23000, 23500, 24000, 24500, 25000, 25500, 26000, 26500] #2_200_000.0, 2_300_000.0, 2_350_000.0, 2_400_000.0, 2_450_000.0, 2_500_000.0
    rigid_rows = []
    for k in rigid_k_values:
        s = summarize_simulation_run(copy_params(p, k=k))
        rigid_rows.append([
            _cell(s["k"], ",.0f"),
            _cell(s["t_release"], ".4f"),
            _cell(s["v_release"], ".3f"),
            _cell(s["delta_max_mm"], ".4e"),
            _cell(s["max_energy_error_percent"], ".5f"),
        ])

    _print_table(
        "4) RIGID-LIMIT / 2-DOF REDUCTION CHECK",
        (
            "As k becomes large, the flexible deflection should become very small, but too large value causes instability"
        ),
        ["k [N/m]", "t_rel [s]", "v_rel [m/s]", "max |delta| [mm]", "max |E err| [%]"],
        rigid_rows,
    )
    print("=" * 78)
    print()

# Main script
if __name__ == "__main__":
    params = TrebuchetParams(
        # This explicitly makes the projectile start just before the pivot
        # and on the ground/chute. Change this value if you want it further left/right.
        initial_projectile_x=-0.20,
        theta0_deg=None,
        phi0_deg=None,
    )
    try:
        result = simulate(params)
        main_stable = print_main_simulation_summary(result, params)
    except (FloatingPointError, np.linalg.LinAlgError):
        result = None
        main_stable = False
        print("Numerical instability: chosen visual simulation values are unstable and cannot be solved.")
    print_verification_tables(params)

    # Show the main verification-style plots and the animation every time for now.
    if main_stable:
        animate(result, params)
        plot_beam_omega_and_sling_tip_velocity(result, params)
        plot_energy_distribution(result, params)
        plot_energy_error(result, params)

