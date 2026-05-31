import airsim
import zmq
import json
import time
import threading
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
plt.ion()  # Интерактивный режим для live-обновления

# =========================
# CONFIG
# =========================

HOST = "localhost"
PORT = 5556
DT = 0.05  # Частота контроллера: 20 Гц

MAX_VEL = 0.5              # Макс. горизонтальная скорость (м/с)
CIRCLE_RADIUS = 10.0       # Радиус окружности (метры)
CIRCLE_CENTER = np.array([0.0, 0.0])  # Центр окружности в горизонтальной плоскости
LOOKAHEAD_ANGLE = 0.15     # Угол опережения на круге (радианы) ~8.6°
ANGLE_HYSTERESIS = 0.05    # Гистерезис для плавного движения по кругу

# PID для горизонтали
KP, KD = 1.2, 0.4

# Калибровка
CALIBRATION_TIME = 10.0
CALIBRATION_SPEED = 1.0

# Z-LOCK: фиксированная высота
TARGET_Z = -5.0

# ФИЛЬТР И МАСШТАБ
FILTER_WINDOW = 5
SCALE_CORRECTION_RATE = 0.0
MIN_SCALE = 0.5
MAX_SCALE = 5000.0
LOOKAHEAD_TIME = 0.15

# БЕЗОПАСНОСТЬ
MAX_DATA_AGE = 0.3
MIN_ORB_CONFIDENCE = 0.1

# ПАРАМЕТРЫ ПОЛЁТА ПО КРУГУ
CIRCLE_POINTS = 200
MAX_LAPS = 3

# ВИЗУАЛИЗАЦИЯ
PLOT_ENABLED = True          # Включить пост-полётные графики
LIVE_PLOT_ENABLED = False    # Включить live-обновление графика (требует GUI)
PLOT_UPDATE_INTERVAL = 1.0   # Интервал обновления live-графика (сек)

print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToZAsync(TARGET_Z, 2).join()
print(f"Takeoff complete, holding Z={TARGET_Z}m")

# =========================
# STATE
# =========================

orb_data = {"pose": np.zeros(3), "timestamp": 0.0}
lock = threading.Lock()

scale = 1.0

# Буферы для фильтрации
position_buffer = deque(maxlen=FILTER_WINDOW)
velocity_buffer = deque(maxlen=3)

# Состояние контроллера
prev_err_xy = np.zeros(2)
prev_filtered_xy = None
last_scale_update = 0

# Состояние полёта по кругу
current_angle = 0.0
laps_completed = 0

# LOGGING ARRAYS
log_time = []
log_gt_xy = []
log_slam_xy = []
log_radius_error = []
log_scale = []
log_pos_error = []

# LIVE PLOT STATE
live_plot_fig = None
live_plot_ax = None
live_plot_last_update = 0

# =========================
# ZMQ LISTENER
# =========================

def zmq_listener():
    """Поток получения данных от ORB-SLAM3 через ZMQ"""
    socket = zmq.Context().socket(zmq.PULL)
    socket.setsockopt(zmq.RCVTIMEO, 100)
    socket.connect(f"tcp://{HOST}:{PORT}")
    print(f"ZMQ listener connected to {HOST}:{PORT}")

    while True:
        try:
            msg = socket.recv_string()
            data = json.loads(msg)

            # Конвертация: ORB (z,x,y) -> AirSim/NED (x,y,z)
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
            print(f"ZMQ error: {e}")
            time.sleep(0.01)

threading.Thread(target=zmq_listener, daemon=True, name="ZMQ_Listener").start()

# =========================
# FILTER HELPERS
# =========================

def apply_moving_average(new_value):
    """Скользящее среднее для сглаживания позиции"""
    position_buffer.append(new_value.copy())
    return np.mean(position_buffer, axis=0)

def estimate_velocity(current_xy, prev_xy, dt):
    """Оценка скорости для экстраполяции"""
    if prev_xy is None:
        return np.zeros(2)
    return (current_xy - prev_xy) / dt

def is_data_fresh(timestamp, max_age=MAX_DATA_AGE):
    """Проверка актуальности данных от SLAM"""
    return (time.time() - timestamp) < max_age

def clamp_scale(s, min_s=MIN_SCALE, max_s=MAX_SCALE):
    """Ограничение масштаба разумными пределами"""
    return np.clip(s, min_s, max_s)

# =========================
# CIRCLE TRAJECTORY
# =========================

def generate_circle_path(radius, center, n_points):
    """Генерирует список точек окружности для дискретного представления"""
    path = []
    for i in range(n_points):
        angle = 2 * np.pi * i / n_points
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        path.append(np.array([x, y, TARGET_Z]))
    return path

def get_lookahead_target(current_xy, center, radius, current_angle, lookahead_angle):
    """Вычисляет целевую точку с опережением на окружности"""
    dx = current_xy[0] - center[0]
    dy = current_xy[1] - center[1]
    angle = np.arctan2(dy, dx)
    target_angle = angle + lookahead_angle
    target_x = center[0] + radius * np.cos(target_angle)
    target_y = center[1] + radius * np.sin(target_angle)
    return np.array([target_x, target_y]), target_angle

def update_angle_progress(current_xy, center, radius, prev_angle):
    """Отслеживает прогресс движения по кругу"""
    dx = current_xy[0] - center[0]
    dy = current_xy[1] - center[1]
    current_angle = np.arctan2(dy, dx)
    
    if prev_angle is not None:
        delta = current_angle - prev_angle
        if delta > np.pi:
            delta -= 2 * np.pi
        elif delta < -np.pi:
            delta += 2 * np.pi
        if abs(delta) < np.pi / 2:
            return current_angle, delta
        else:
            return current_angle, 0.0
    return current_angle, 0.0

# =========================
# CALIBRATION PHASE
# =========================

def calibrate_scale():
    """Калибровка масштаба: пролет по горизонтали"""
    print("\nCALIBRATION START (Horizontal only)")
    print(f"   Flying at {CALIBRATION_SPEED} m/s for {CALIBRATION_TIME}s...")

    orb_dist = 0.0
    real_dist = 0.0
    start = time.time()
    prev_orb = None
    prev_real = None

    client.moveByVelocityAsync(
        CALIBRATION_SPEED, 0, 0, CALIBRATION_TIME,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, 0)
    )

    while time.time() - start < CALIBRATION_TIME:
        time.sleep(DT)

        with lock:
            p = orb_data["pose"].copy()
        
        if prev_orb is not None:
            orb_dist += np.linalg.norm(p[:2] - prev_orb[:2])
        prev_orb = p.copy()

        state = client.getMultirotorState().kinematics_estimated.position
        real = np.array([state.x_val, state.y_val, state.z_val])
        
        if prev_real is not None:
            real_dist += np.linalg.norm(real[:2] - prev_real[:2])
        prev_real = real.copy()

    if orb_dist > MIN_ORB_CONFIDENCE:
        scale = 50.0
    else:
        print("ORB motion too small, using default scale=1.0")
        scale = 50.0

    print("\n========================")
    print("CALIBRATION RESULT")
    print(f"   ORB horizontal distance : {orb_dist:.3f} (arb. units)")
    print(f"   REAL horizontal distance: {real_dist:.3f} (meters)")
    print(f"   SCALE                   : {scale:.3f} (m/unit)")
    print("========================\n")
    
    start_pos = CIRCLE_CENTER + np.array([CIRCLE_RADIUS, 0])
    client.moveToPositionAsync(start_pos[0], start_pos[1], TARGET_Z, 1).join()
    time.sleep(1)
    
    return scale

# =========================
# PLOTTING FUNCTIONS
# =========================

def init_live_plot():
    """Инициализация live-графика"""
    global live_plot_fig, live_plot_ax
    if not LIVE_PLOT_ENABLED:
        return
    
    live_plot_fig, live_plot_ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Траектория
    live_plot_ax[0].set_title("Position Estimate (Live)")
    live_plot_ax[0].set_xlabel("X [m]")
    live_plot_ax[0].set_ylabel("Y [m]")
    live_plot_ax[0].grid(True)
    live_plot_ax[0].set_aspect('equal')
    
    # Ошибка
    live_plot_ax[1].set_title("Position Error (Live)")
    live_plot_ax[1].set_xlabel("Time [s]")
    live_plot_ax[1].set_ylabel("Error [m]")
    live_plot_ax[1].grid(True)
    
    plt.tight_layout()
    plt.show(block=False)

def update_live_plot():
    """Обновление live-графика"""
    global live_plot_last_update
    if not LIVE_PLOT_ENABLED or live_plot_fig is None:
        return
    
    now = time.time()
    if now - live_plot_last_update < PLOT_UPDATE_INTERVAL:
        return
    live_plot_last_update = now
    
    if len(log_gt_xy) < 10:
        return
    
    gt = np.array(log_gt_xy)
    slam = np.array(log_slam_xy)
    times = np.array(log_time)
    errors = np.array(log_pos_error)
    
    # Траектория
    live_plot_ax[0].cla()
    circle = plt.Circle(CIRCLE_CENTER, CIRCLE_RADIUS, color='gray', 
                       linestyle='--', fill=False, alpha=0.3, label='Target circle')
    live_plot_ax[0].add_patch(circle)
    live_plot_ax[0].plot(gt[:, 0], gt[:, 1], 'b-', label='Ground Truth', linewidth=1.5)
    live_plot_ax[0].plot(slam[:, 0], slam[:, 1], 'r--', label='SLAM Estimate', linewidth=1)
    live_plot_ax[0].scatter(slam[-1, 0], slam[-1, 1], c='red', s=30, zorder=5)
    live_plot_ax[0].set_xlabel("X [m]")
    live_plot_ax[0].set_ylabel("Y [m]")
    live_plot_ax[0].set_title("Position Estimate (Live)")
    live_plot_ax[0].legend(fontsize=8)
    live_plot_ax[0].grid(True)
    live_plot_ax[0].set_aspect('equal')
    
    # Ошибка
    live_plot_ax[1].cla()
    live_plot_ax[1].plot(times, errors, 'm-', label='Position error')
    live_plot_ax[1].axhline(np.mean(errors[-50:]) if len(errors) >= 50 else np.mean(errors), 
                           color='orange', linestyle=':', label=f'Avg: {np.mean(errors):.2f}m')
    live_plot_ax[1].set_xlabel("Time [s]")
    live_plot_ax[1].set_ylabel("Error [m]")
    live_plot_ax[1].set_title("Position Error (Live)")
    live_plot_ax[1].legend(fontsize=8)
    live_plot_ax[1].grid(True)
    
    plt.tight_layout()
    live_plot_fig.canvas.draw_idle()
    live_plot_fig.canvas.flush_events()

def plot_final_results():
    """Пост-полётная визуализация результатов"""
    if not PLOT_ENABLED or len(log_gt_xy) < 20:
        print("Not enough data for plotting")
        return
    
    print("\nGenerating final plots...")
    
    gt = np.array(log_gt_xy)
    slam = np.array(log_slam_xy)
    times = np.array(log_time)
    radius_errors = np.array(log_radius_error)
    scales = np.array(log_scale)
    pos_errors = np.array(log_pos_error)
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Траектория (2D)
    ax1 = fig.add_subplot(221)
    circle = plt.Circle(CIRCLE_CENTER, CIRCLE_RADIUS, color='gray', 
                       linestyle='--', fill=False, alpha=0.3, label='Target circle')
    ax1.add_patch(circle)
    ax1.plot(gt[:, 0], gt[:, 1], 'b-', label='Ground Truth', linewidth=2, alpha=0.8)
    ax1.plot(slam[:, 0], slam[:, 1], 'r--', label='SLAM Estimate', linewidth=1.5, alpha=0.8)
    ax1.scatter(gt[0, 0], gt[0, 1], c='blue', s=50, marker='o', label='Start', zorder=5)
    ax1.scatter(gt[-1, 0], gt[-1, 1], c='darkblue', s=50, marker='x', label='End', zorder=5)
    ax1.set_xlabel("X [m]", fontsize=10)
    ax1.set_ylabel("Y [m]", fontsize=10)
    ax1.set_title("2D Trajectory Comparison", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 2. Ошибка позиции во времени
    ax2 = fig.add_subplot(222)
    ax2.plot(times, pos_errors, 'm-', linewidth=1.5, label='Euclidean error')
    ax2.axhline(np.mean(pos_errors), color='orange', linestyle=':', 
               label=f'Mean: {np.mean(pos_errors):.3f}m')
    ax2.axhline(np.std(pos_errors), color='red', linestyle=':', alpha=0.7,
               label=f'Std: {np.std(pos_errors):.3f}m')
    ax2.fill_between(times, 
                    np.mean(pos_errors) - np.std(pos_errors),
                    np.mean(pos_errors) + np.std(pos_errors),
                    color='orange', alpha=0.1)
    ax2.set_xlabel("Time [s]", fontsize=10)
    ax2.set_ylabel("Error [m]", fontsize=10)
    ax2.set_title("Position Error Over Time", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Ошибка радиуса
    ax3 = fig.add_subplot(223)
    ax3.plot(times, radius_errors, 'g-', linewidth=1.5, label='Radius error')
    ax3.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax3.axhline(np.mean(radius_errors), color='orange', linestyle=':', 
               label=f'Mean: {np.mean(radius_errors):.3f}m')
    ax3.set_xlabel("Time [s]", fontsize=10)
    ax3.set_ylabel("Error [m]", fontsize=10)
    ax3.set_title(f"Radius Error (Target: {CIRCLE_RADIUS}m)", fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Эволюция масштаба
    ax4 = fig.add_subplot(224)
    ax4.plot(times, scales, 'c-', linewidth=1.5, label='Estimated scale')
    ax4.axhline(np.mean(scales[-100:]) if len(scales) >= 100 else np.mean(scales), 
               color='orange', linestyle=':', label=f'Final avg: {np.mean(scales[-100:]):.2f}' if len(scales)>=100 else f'Mean: {np.mean(scales):.2f}')
    ax4.set_xlabel("Time [s]", fontsize=10)
    ax4.set_ylabel("Scale [m/unit]", fontsize=10)
    ax4.set_title("Scale Estimation Over Time", fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('circle_trajectory_analysis.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'circle_trajectory_analysis.png'")
    
    # Статистика в консоль
    print(f"\nFINAL METRICS:")
    print(f"   Mean position error: {np.mean(pos_errors):.3f} +/- {np.std(pos_errors):.3f} m")
    print(f"   Mean radius error:   {np.mean(radius_errors):.3f} +/- {np.std(radius_errors):.3f} m")
    print(f"   Final scale:         {np.mean(scales[-50:]):.3f} (last 50 samples)")
    print(f"   Total samples:       {len(log_time)}")
    
    plt.show()

def log_position(t, gt_xy, slam_xy, radius_err, pos_err, current_scale):
    """Логирование данных для графиков"""
    if not PLOT_ENABLED and not LIVE_PLOT_ENABLED:
        return
    log_time.append(t)
    log_gt_xy.append(gt_xy.copy())
    log_slam_xy.append(slam_xy.copy())
    log_radius_error.append(radius_err)
    log_pos_error.append(pos_err)
    log_scale.append(current_scale)

# =========================
# CONTROL LOOP (CIRCLE)
# =========================

def run_circle(initial_scale):
    """Основной цикл навигации по окружности"""
    
    global scale, prev_err_xy, prev_filtered_xy, last_scale_update
    global current_angle, laps_completed
    scale = initial_scale
    current_angle = 0.0
    laps_completed = 0
    prev_angle = None
    
    print("NAVIGATION START: CIRCLE TRAJECTORY")
    print(f"   Center: {CIRCLE_CENTER}, Radius: {CIRCLE_RADIUS}m, Height: {TARGET_Z}m")
    print(f"   Lookahead angle: {np.degrees(LOOKAHEAD_ANGLE):.1f} deg, Max laps: {MAX_LAPS if MAX_LAPS > 0 else 'inf'}")
    print("-" * 70)

    start_pos = CIRCLE_CENTER + np.array([CIRCLE_RADIUS, 0])
    print(f"Starting at: {start_pos}")
    
    # Инициализация live-графика
    init_live_plot()

    while True:
        loop_start = time.time()
        
        # === 1. Получение и валидация данных ===
        with lock:
            orb_raw = orb_data["pose"].copy()
            orb_ts = orb_data["timestamp"]
        
        if not is_data_fresh(orb_ts):
            print(f"\nSLAM DATA LOST! Hovering... (age={time.time()-orb_ts:.2f}s)")
            client.hoverAsync().join()
            time.sleep(DT)
            continue
        
        # === 2. Фильтрация позиции ===
        filtered_xy = apply_moving_average(orb_raw[:2])
        
        # === 3. Экстраполяция для компенсации задержки ===
        vel_est = estimate_velocity(filtered_xy, prev_filtered_xy, DT)
        predicted_xy = filtered_xy + vel_est * LOOKAHEAD_TIME
        prev_filtered_xy = filtered_xy.copy()
        
        # === 4. Динамическая коррекция масштаба ===
        if abs(orb_raw[2]) > MIN_ORB_CONFIDENCE:
            scale_z = TARGET_Z / orb_raw[2]
            scale_z = clamp_scale(scale_z)
            
            if time.time() - last_scale_update > 0.2:
                scale = (1 - SCALE_CORRECTION_RATE) * scale + SCALE_CORRECTION_RATE * scale_z
                last_scale_update = time.time()
        
        current_xy = predicted_xy * scale
        
        # === 5. Вычисление целевой точки на окружности с опережением ===
        target_xy, target_angle = get_lookahead_target(
            current_xy, CIRCLE_CENTER, CIRCLE_RADIUS, current_angle, LOOKAHEAD_ANGLE
        )
        
        # === 6. ПИД-регулятор (только для XY) ===
        err_xy = target_xy - current_xy
        
        vel_xy = KP * err_xy + KD * (err_xy - prev_err_xy) / DT
        prev_err_xy = err_xy.copy()
        
        # Ограничение скорости
        speed_xy = np.linalg.norm(vel_xy)
        if speed_xy > MAX_VEL:
            vel_xy = vel_xy / speed_xy * MAX_VEL
        
        # === 7. Формирование команды: Z = 0 ===
        vel_cmd = np.array([vel_xy[0], vel_xy[1], 0.0])
        
        # === 8. Отслеживание прогресса по кругу ===
        current_angle, angle_delta = update_angle_progress(current_xy, CIRCLE_CENTER, CIRCLE_RADIUS, prev_angle)
        prev_angle = current_angle
        
        if angle_delta is not None:
            laps_completed += angle_delta / (2 * np.pi)
        
        # === 9. Отправка команды на дрон ===
        client.moveByVelocityAsync(
            vel_cmd[0], vel_cmd[1], vel_cmd[2],
            duration=DT,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)
        )
        
        # === 10. Вычисление метрик для логирования ===
        dx = current_xy[0] - CIRCLE_CENTER[0]
        dy = current_xy[1] - CIRCLE_CENTER[1]
        actual_radius = np.sqrt(dx**2 + dy**2)
        radius_error = actual_radius - CIRCLE_RADIUS
        
        # Ground truth позиция
        state = client.getMultirotorState()
        gt_xy = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val
        ])
        
        # Ошибка позиции (между GT и оценкой)
        pos_error = np.linalg.norm(gt_xy - current_xy)
        
        # === 11. Логирование ===
        elapsed = time.time() - loop_start  # approximate time
        log_position(
            time.time(), 
            gt_xy, 
            current_xy, 
            radius_error, 
            pos_error, 
            scale
        )
        
        # === 12. Live plot update ===
        update_live_plot()
        
        # === 13. Консольный вывод ===
        err_z = TARGET_Z - (orb_raw[2] * scale)
        print(
            f"\rLAP={laps_completed:4.2f} | "
            f"R_ERR={radius_error:+5.2f}m | "
            f"Z_ERR={err_z:+5.2f}m | "
            f"SCALE={scale:6.2f} | "
            f"POS_ERR={pos_error:.2f}m | "
            f"VEL=[{vel_xy[0]:+.2f}, {vel_xy[1]:+.2f}]  ",
            end="", flush=True
        )
        
        # === 14. Проверка завершения ===
        if MAX_LAPS > 0 and laps_completed >= MAX_LAPS:
            print(f"\n\nCIRCLE COMPLETE: {laps_completed:.2f} laps done!")
            print(f"   Final scale: {scale:.3f}")
            print(f"   Avg radius error: {np.mean(log_radius_error[-100:]):.2f}m")
            client.hoverAsync().join()
            break
        
        # Поддержание частоты цикла
        elapsed = time.time() - loop_start
        if elapsed < DT:
            time.sleep(DT - elapsed)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    try:
        # 1. Калибровка масштаба
        initial_scale = calibrate_scale()
        
        # 2. Запуск навигации по кругу
        run_circle(initial_scale)
        
    except KeyboardInterrupt:
        print("\n\nManual stop requested")
        client.hoverAsync().join()
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        client.hoverAsync().join()
        
    finally:
        # Пост-обработка: финальные графики
        if PLOT_ENABLED:
            plot_final_results()
        
        print("Disconnecting...")
        client.enableApiControl(False)
        print("Done")
