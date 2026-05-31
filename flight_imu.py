import airsim
import zmq
import json
import time
import threading
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================

HOST = "localhost"
PORT = 5556

DT = 0.05

TARGET_Z = -5.0

MAX_VEL = 0.6

BOUNDARY_MIN = -10
BOUNDARY_MAX = 10

LOOKAHEAD_DIST = 1.0
WAYPOINT_HYSTERESIS = 0.3

NUM_WAYPOINTS = 30
MIN_WAYPOINT_DIST = 3.0

MAX_FLIGHT_TIME = 300

# =========================================================
# PID XY
# =========================================================

KP = 1.0
KI = 0.03
KD = 0.35

INTEGRAL_LIMIT = 0.5
INTEGRAL_DEADZONE = 0.15

# =========================================================
# PID Z
# =========================================================

Z_KP = 0.12
Z_KI = 0.01

Z_INTEGRAL_LIMIT = 0.3

# =========================================================
# FILTERS
# =========================================================

FILTER_WINDOW = 5

LOOKAHEAD_TIME = 0.15

# =========================================================
# SCALE
# =========================================================

INITIAL_SCALE_TIME = 8.0
INITIAL_SCALE_SPEED = 1.0

MIN_SCALE = 0.05
MAX_SCALE = 5000.0

ONLINE_SCALE_ALPHA = 0.01

MIN_GT_SPEED = 0.08
MIN_ORB_SPEED = 1e-4

MAX_SCALE_JUMP = 1.5

# =========================================================
# SAFETY
# =========================================================

MAX_DATA_AGE = 0.3

# =========================================================
# LOGGING
# =========================================================

LOG_ENABLED = True

log_gt = []
log_slam = []

# =========================================================
# AIRSIM
# =========================================================

print("Connecting to AirSim...")

client = airsim.MultirotorClient()

client.confirmConnection()

client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

client.moveToZAsync(TARGET_Z, 2).join()

print(f"Takeoff complete | Z={TARGET_Z}")

# =========================================================
# STATE
# =========================================================

orb_data = {
    "pose": np.zeros(3),
    "timestamp": 0.0
}

lock = threading.Lock()

position_buffer = deque(maxlen=FILTER_WINDOW)

scale = 1.0

prev_filtered_xy = None

prev_err_xy = np.zeros(2)
integral_err_xy = np.zeros(2)

integral_err_z = 0.0

waypoints = []

# online scale state
prev_orb_for_scale = None
prev_gt_for_scale = None
prev_scale_time = None

# =========================================================
# ZMQ
# =========================================================

def zmq_listener():

    socket = zmq.Context().socket(zmq.PULL)

    socket.setsockopt(zmq.RCVTIMEO, 100)

    socket.connect(f"tcp://{HOST}:{PORT}")

    print(f"ZMQ connected: {HOST}:{PORT}")

    while True:

        try:

            msg = socket.recv_string()

            data = json.loads(msg)

            raw = np.array([
                data.get("z", 0.0),
                data.get("x", 0.0),
                data.get("y", 0.0)
            ], dtype=np.float32)

            with lock:

                orb_data["pose"] = raw
                orb_data["timestamp"] = time.time()

        except zmq.Again:
            continue

        except Exception as e:

            print(f"\nZMQ ERROR: {e}")

            time.sleep(0.01)

threading.Thread(
    target=zmq_listener,
    daemon=True
).start()

# =========================================================
# HELPERS
# =========================================================

def clamp_scale(v):

    return np.clip(
        v,
        MIN_SCALE,
        MAX_SCALE
    )

def is_data_fresh(ts):

    return (time.time() - ts) < MAX_DATA_AGE

def apply_moving_average(v):

    position_buffer.append(v.copy())

    return np.mean(position_buffer, axis=0)

def estimate_velocity(cur, prev, dt):

    if prev is None:
        return np.zeros(2)

    return (cur - prev) / dt

def clamp_to_boundary(pos, margin=1.0):

    return np.clip(
        pos,
        BOUNDARY_MIN + margin,
        BOUNDARY_MAX - margin
    )

# =========================================================
# WAYPOINTS
# =========================================================

def generate_chaotic_waypoints():

    wps = []

    attempts = 0

    while (
        len(wps) < NUM_WAYPOINTS and
        attempts < NUM_WAYPOINTS * 100
    ):

        attempts += 1

        x = np.random.uniform(
            BOUNDARY_MIN + 1,
            BOUNDARY_MAX - 1
        )

        y = np.random.uniform(
            BOUNDARY_MIN + 1,
            BOUNDARY_MAX - 1
        )

        candidate = np.array([x, y])

        valid = True

        for wp in wps:

            if np.linalg.norm(candidate - wp[:2]) < MIN_WAYPOINT_DIST:
                valid = False
                break

        if valid:

            wps.append(
                np.array([x, y, TARGET_Z])
            )

    print(f"Generated {len(wps)} waypoints")

    return wps

# =========================================================
# INITIAL SCALE CALIBRATION
# =========================================================

def calibrate_initial_scale():

    print("\nINITIAL SCALE CALIBRATION")

    samples = []

    prev_orb = None
    prev_t = None

    client.moveByVelocityAsync(
        INITIAL_SCALE_SPEED,
        0,
        0,
        INITIAL_SCALE_TIME,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, 0)
    )

    start = time.time()

    while time.time() - start < INITIAL_SCALE_TIME:

        now = time.time()

        with lock:

            orb_pose = orb_data["pose"].copy()
            orb_ts = orb_data["timestamp"]

        if not is_data_fresh(orb_ts):

            time.sleep(DT)
            continue

        orb_xy = orb_pose[:2]

        state = client.getMultirotorState()

        gt_vel = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val
        ])

        gt_speed = np.linalg.norm(gt_vel)

        if prev_orb is not None:

            dt = now - prev_t

            if dt > 1e-3:

                orb_vel = (orb_xy - prev_orb) / dt

                orb_speed = np.linalg.norm(orb_vel)

                if (
                    orb_speed > MIN_ORB_SPEED and
                    gt_speed > MIN_GT_SPEED
                ):

                    s = gt_speed / orb_speed

                    if np.isfinite(s):

                        s = clamp_scale(s)

                        samples.append(s)

        prev_orb = orb_xy.copy()
        prev_t = now

        time.sleep(DT)

    if len(samples) < 10:

        print("Bad calibration")

        final_scale = 50.0

    else:

        arr = np.array(samples)

        median = np.median(arr)

        good = np.abs(arr - median) < median * 0.3

        final_scale = np.mean(arr[good])

    final_scale = clamp_scale(final_scale)

    print(f"INITIAL SCALE = {final_scale:.3f}")

    client.moveToPositionAsync(
        0,
        0,
        TARGET_Z,
        1
    ).join()

    time.sleep(1)

    return final_scale

# =========================================================
# ONLINE SCALE CORRECTION
# =========================================================

def update_online_scale(
    current_scale,
    orb_xy,
    gt_xy,
    now
):

    global prev_orb_for_scale
    global prev_gt_for_scale
    global prev_scale_time

    if prev_orb_for_scale is None:

        prev_orb_for_scale = orb_xy.copy()
        prev_gt_for_scale = gt_xy.copy()
        prev_scale_time = now

        return current_scale

    dt = now - prev_scale_time

    if dt < 1e-3:
        return current_scale

    # -----------------------------------------------------
    # ORB velocity
    # -----------------------------------------------------

    orb_vel = (orb_xy - prev_orb_for_scale) / dt

    orb_speed = np.linalg.norm(orb_vel)

    # -----------------------------------------------------
    # GT velocity
    # -----------------------------------------------------

    gt_vel = (gt_xy - prev_gt_for_scale) / dt

    gt_speed = np.linalg.norm(gt_vel)

    prev_orb_for_scale = orb_xy.copy()
    prev_gt_for_scale = gt_xy.copy()
    prev_scale_time = now

    # -----------------------------------------------------
    # validity
    # -----------------------------------------------------

    if orb_speed < MIN_ORB_SPEED:
        return current_scale

    if gt_speed < MIN_GT_SPEED:
        return current_scale

    instant_scale = gt_speed / orb_speed

    if not np.isfinite(instant_scale):
        return current_scale

    instant_scale = clamp_scale(instant_scale)

    # -----------------------------------------------------
    # reject violent jumps
    # -----------------------------------------------------

    ratio = instant_scale / current_scale

    if ratio > MAX_SCALE_JUMP:
        return current_scale

    if ratio < (1.0 / MAX_SCALE_JUMP):
        return current_scale

    # -----------------------------------------------------
    # EMA update
    # -----------------------------------------------------

    new_scale = (
        (1.0 - ONLINE_SCALE_ALPHA) * current_scale +
        ONLINE_SCALE_ALPHA * instant_scale
    )

    new_scale = clamp_scale(new_scale)

    return new_scale

# =========================================================
# LOGGING
# =========================================================

def log_position(t, gt, slam):

    if not LOG_ENABLED:
        return

    log_gt.append([
        t,
        gt[0],
        gt[1],
        gt[2]
    ])

    log_slam.append([
        t,
        slam[0],
        slam[1],
        slam[2]
    ])

# =========================================================
# PLOTS
# =========================================================

def plot_trajectories():

    if len(log_gt) < 10:
        return

    gt = np.array(log_gt)
    slam = np.array(log_slam)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    ax1.plot(
        gt[:,1],
        gt[:,2],
        label="Ground Truth"
    )

    ax1.plot(
        slam[:,1],
        slam[:,2],
        "--",
        label="ORB_SLAM3"
    )

    if len(waypoints) > 0:

        wp = np.array(waypoints)

        ax1.scatter(
            wp[:,0],
            wp[:,1],
            s=20,
            label="Waypoints"
        )

    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect("equal")
    ax1.set_title("XY trajectory")

    err = np.linalg.norm(
        gt[:,1:3] - slam[:,1:3],
        axis=1
    )

    ax2.plot(err)

    ax2.axhline(
        np.mean(err),
        linestyle=":"
    )

    ax2.grid(True)

    ax2.set_title(
        f"Mean error = {np.mean(err):.2f}m"
    )

    plt.tight_layout()

    plt.savefig(
        "trajectory_analysis.png",
        dpi=150
    )

    plt.show()

# =========================================================
# CONTROL LOOP
# =========================================================

def run_navigation(initial_scale):

    global scale
    global prev_filtered_xy
    global prev_err_xy
    global integral_err_xy
    global integral_err_z
    global waypoints

    scale = initial_scale

    waypoints = generate_chaotic_waypoints()

    if len(waypoints) == 0:
        return

    print("\nSTART NAVIGATION")
    print(f"INITIAL SCALE = {scale:.3f}")
    print("-" * 60)

    start_time = time.time()

    wp = 0

    while True:

        loop_start = time.time()

        elapsed_total = time.time() - start_time

        if elapsed_total > MAX_FLIGHT_TIME:

            print("\nTime limit")
            break

        if wp >= len(waypoints):

            print("\nMission complete")
            break

        # =================================================
        # ORB
        # =================================================

        with lock:

            orb_raw = orb_data["pose"].copy()
            orb_ts = orb_data["timestamp"]

        if not is_data_fresh(orb_ts):

            print("\nSLAM LOST")

            client.hoverAsync().join()

            time.sleep(DT)

            continue

        # =================================================
        # FILTER
        # =================================================

        filtered_xy = apply_moving_average(
            orb_raw[:2]
        )

        vel_est = estimate_velocity(
            filtered_xy,
            prev_filtered_xy,
            DT
        )

        predicted_xy = (
            filtered_xy +
            vel_est * LOOKAHEAD_TIME
        )

        prev_filtered_xy = filtered_xy.copy()

        # =================================================
        # GT
        # =================================================

        state = client.getMultirotorState()

        gt_xy = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val
        ])

        # =================================================
        # ONLINE SCALE UPDATE
        # =================================================

        scale = update_online_scale(
            scale,
            predicted_xy,
            gt_xy,
            time.time()
        )

        # =================================================
        # CURRENT POSITION
        # =================================================

        current_xy = predicted_xy * scale

        current_xy = clamp_to_boundary(current_xy)

        # =================================================
        # TARGET
        # =================================================

        target = waypoints[wp]

        err_xy = target[:2] - current_xy

        # integral
        if np.linalg.norm(err_xy) > INTEGRAL_DEADZONE:

            integral_err_xy += err_xy * DT

            integral_err_xy = np.clip(
                integral_err_xy,
                -INTEGRAL_LIMIT,
                INTEGRAL_LIMIT
            )

        else:

            integral_err_xy *= 0.98

        # PID
        vel_xy = (
            KP * err_xy +
            KI * integral_err_xy +
            KD * (err_xy - prev_err_xy) / DT
        )

        prev_err_xy = err_xy.copy()

        speed = np.linalg.norm(vel_xy)

        if speed > MAX_VEL:

            vel_xy = vel_xy / speed * MAX_VEL

        # =================================================
        # Z HOLD
        # =================================================

        current_z = (
            state.kinematics_estimated.position.z_val
        )

        err_z = TARGET_Z - current_z

        if abs(err_z) > 0.05:

            integral_err_z += err_z * DT

            integral_err_z = np.clip(
                integral_err_z,
                -Z_INTEGRAL_LIMIT,
                Z_INTEGRAL_LIMIT
            )

        else:

            integral_err_z *= 0.99

        vel_z = (
            Z_KP * err_z +
            Z_KI * integral_err_z
        )

        vel_z = np.clip(
            vel_z,
            -0.2,
            0.2
        )

        # =================================================
        # SEND COMMAND
        # =================================================

        client.moveByVelocityAsync(
            vel_xy[0],
            vel_xy[1],
            vel_z,
            DT,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        )

        # =================================================
        # WAYPOINT SWITCH
        # =================================================

        err_2d = np.linalg.norm(err_xy)

        if err_2d < (
            LOOKAHEAD_DIST -
            WAYPOINT_HYSTERESIS
        ):

            wp += 1

            integral_err_xy *= 0.5

            if wp < len(waypoints):

                print(
                    f"\nWP {wp}/{len(waypoints)}"
                )

        # =================================================
        # LOGGING
        # =================================================

        gt_pos = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ])

        slam_pos = np.array([
            current_xy[0],
            current_xy[1],
            orb_raw[2] * scale
        ])

        log_position(
            elapsed_total,
            gt_pos,
            slam_pos
        )

        # =================================================
        # PRINT
        # =================================================

        print(
            f"\r"
            f"WP={wp}/{len(waypoints)} | "
            f"ERR={err_2d:.2f}m | "
            f"SCALE={scale:.2f} | "
            f"CUR=[{current_xy[0]:+.2f},{current_xy[1]:+.2f}]",
            end="",
            flush=True
        )

        elapsed = time.time() - loop_start

        if elapsed < DT:

            time.sleep(DT - elapsed)

    print("\n\nFINISHED")

    client.hoverAsync().join()

    time.sleep(1)

    if LOG_ENABLED:
        plot_trajectories()

# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    try:

        initial_scale = calibrate_initial_scale()

        run_navigation(initial_scale)

    except KeyboardInterrupt:

        print("\nInterrupted")

        client.hoverAsync().join()

    except Exception as e:

        print(f"\nERROR: {e}")

        import traceback
        traceback.print_exc()

        client.hoverAsync().join()

    finally:

        client.enableApiControl(False)

        print("Done")

