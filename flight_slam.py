import airsim
import zmq
import json
import time
import threading
import numpy as np
from collections import deque

# =========================
# CONFIG
# =========================

HOST = "localhost"
PORT = 5556
DT = 0.05  # Частота контроллера: 20 Гц

MAX_VEL = 0.5           # Макс. горизонтальная скорость (м/с)
LOOKAHEAD_DIST = 1.0    # Радиус переключения вейпоинта (м)
WAYPOINT_HYSTERESIS = 0.3  # Гистерезис, чтобы не «дергаться» на границе

# PID для горизонтали (немного снижены из-за фильтра)
KP, KD = 1.2, 0.4

# Калибровка
CALIBRATION_TIME = 10.0
CALIBRATION_SPEED = 1.0

# Z-LOCK: фиксированная высота (должна совпадать с точками пути!)
TARGET_Z = -5.0

# ФИЛЬТР И МАСШТАБ
FILTER_WINDOW = 5                    # Окно скользящего среднего (~250 мс)
SCALE_CORRECTION_RATE = 0.02        # Скорость адаптации масштаба (0.01-0.05)
MIN_SCALE = 0.5                      # Ограничители масштаба (защита от «взрыва»)
MAX_SCALE = 5000.0
LOOKAHEAD_TIME = 0.15               # Экстраполяция на задержку SLAM (сек)

# БЕЗОПАСНОСТЬ
MAX_DATA_AGE = 0.3                   # Если данных нет >300 мс — аварийное зависание
MIN_ORB_CONFIDENCE = 0.1             # Мин. движение для коррекции масштаба (защита от деления на шум)

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

# orb_data: pose + timestamp + флаг качества (опционально)
orb_data = {"pose": np.zeros(3), "timestamp": 0.0}
lock = threading.Lock()

# Начальный масштаб (будет переопределён калибровкой)
scale = 1.0

# Буферы для фильтрации
position_buffer = deque(maxlen=FILTER_WINDOW)  # Для скользящего среднего
velocity_buffer = deque(maxlen=3)              # Для оценки скорости (экстраполяция)

# Состояние контроллера
prev_err_xy = np.zeros(2)
prev_filtered_xy = None
last_scale_update = 0

# =========================
# ZMQ LISTENER
# =========================

def zmq_listener():
    """Поток получения данных от ORB-SLAM3 через ZMQ"""
    socket = zmq.Context().socket(zmq.PULL)
    socket.setsockopt(zmq.RCVTIMEO, 100)  # Таймаут 100 мс для неблокирующего приёма
    socket.connect(f"tcp://{HOST}:{PORT}")
    print(f"ZMQ listener connected to {HOST}:{PORT}")

    while True:
        try:
            msg = socket.recv_string()
            data = json.loads(msg)

            # Конвертация координат: ORB (z,x,y) -> AirSim/NED (x,y,z)
            raw = np.array([
                data.get("z", 0.0),
                data.get("x", 0.0),
                data.get("y", 0.0)
            ], dtype=np.float32)

            with lock:
                orb_data["pose"] = raw
                orb_data["timestamp"] = time.time()
                
        except zmq.Again:
            # Нет новых данных — это нормально, просто ждём следующий цикл
            continue
        except Exception as e:
            print(f"ZMQ error: {e}")
            time.sleep(0.01)

# Запускаем слушатель в отдельном потоке
threading.Thread(target=zmq_listener, daemon=True, name="ZMQ_Listener").start()

# =========================
# FILTER HELPERS
# =========================

def apply_moving_average(new_value):
    """Добавляет значение в буфер и возвращает скользящее среднее"""
    position_buffer.append(new_value.copy())
    return np.mean(position_buffer, axis=0)

def estimate_velocity(current_xy, prev_xy, dt):
    """Простая оценка скорости для экстраполяции"""
    if prev_xy is None:
        return np.zeros(2)
    return (current_xy - prev_xy) / dt

def is_data_fresh(timestamp, max_age=MAX_DATA_AGE):
    """Проверяет актуальность данных от SLAM"""
    return (time.time() - timestamp) < max_age

def clamp_scale(s, min_s=MIN_SCALE, max_s=MAX_SCALE):
    """Ограничивает масштаб разумными пределами"""
    return np.clip(s, min_s, max_s)

# =========================
# CALIBRATION PHASE
# =========================

def calibrate_scale():
    """Калибровка масштаба: пролет по горизонтали для оценки коэффициента"""
    print("\nCALIBRATION START (Horizontal only)")
    print(f"   Flying at {CALIBRATION_SPEED} m/s for {CALIBRATION_TIME}s...")

    orb_dist = 0.0
    real_dist = 0.0
    start = time.time()
    prev_orb = None
    prev_real = None

    # Команда полёта: только по оси X, Y и Z = 0 (не меняем высоту)
    client.moveByVelocityAsync(
        CALIBRATION_SPEED, 0, 0, CALIBRATION_TIME,
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, 0)
    )

    while time.time() - start < CALIBRATION_TIME:
        time.sleep(DT)

        # === ORB ===
        with lock:
            p = orb_data["pose"].copy()
        
        if prev_orb is not None:
            # Считаем только горизонтальное перемещение
            orb_dist += np.linalg.norm(p[:2] - prev_orb[:2])
        prev_orb = p.copy()

        # === AirSim Ground Truth (только для калибровки!) ===
        state = client.getMultirotorState().kinematics_estimated.position
        real = np.array([state.x_val, state.y_val, state.z_val])
        
        if prev_real is not None:
            real_dist += np.linalg.norm(real[:2] - prev_real[:2])
        prev_real = real.copy()

    # Вычисляем масштаб
    if orb_dist > MIN_ORB_CONFIDENCE:
        scale = real_dist / orb_dist
        scale = clamp_scale(scale)
    else:
        print("ORB motion too small, using default scale=1.0")
        scale = 1.0

    print("\n========================")
    print("CALIBRATION RESULT")
    print(f"   ORB horizontal distance : {orb_dist:.3f} (arb. units)")
    print(f"   REAL horizontal distance: {real_dist:.3f} (meters)")
    print(f"   SCALE                   : {scale:.3f} (m/unit)")
    print("========================\n")
    
    # Возвращаем дрон в исходную точку (опционально)
    client.moveToPositionAsync(0, 0, TARGET_Z, 1).join()
    time.sleep(1)
    
    return scale

# =========================
# CONTROL LOOP
# =========================

def run(initial_scale):
    """Основной цикл навигации с коррекцией масштаба и фильтрацией"""
    
    # Маршрут: квадрат 20×20 м на высоте TARGET_Z
    path = [
        np.array([-10, -10, TARGET_Z]),
        np.array([ 10, -10, TARGET_Z]),
        np.array([ 10,  10, TARGET_Z]),
        np.array([-10,  10, TARGET_Z]),
        np.array([-10, -10, TARGET_Z]),  # Замыкаем круг
    ]
    
    wp = 0
    global scale, prev_err_xy, prev_filtered_xy, last_scale_update
    scale = initial_scale
    
    print("NAVIGATION START (Z-LOCKED, Scale-Adaptive)")
    print(f"   Target: {len(path)-1} waypoints, Z={TARGET_Z}m")
    print(f"   Max velocity: {MAX_VEL} m/s, Lookahead: {LOOKAHEAD_DIST}m")
    print("-" * 60)

    while True:
        loop_start = time.time()
        
        # === 1. Получение и валидация данных ===
        with lock:
            orb_raw = orb_data["pose"].copy()
            orb_ts = orb_data["timestamp"]
        
        # Watchdog: если данных нет — аварийное зависание
        if not is_data_fresh(orb_ts):
            print(f"\nSLAM DATA LOST! Hovering... (age={time.time()-orb_ts:.2f}s)")
            client.hoverAsync().join()
            time.sleep(DT)
            continue
        
        # === 2. Фильтрация позиции (скользящее среднее) ===
        # Работаем только с горизонтальными координатами
        filtered_xy = apply_moving_average(orb_raw[:2])
        
        # === 3. Экстраполяция для компенсации задержки ===
        vel_est = estimate_velocity(filtered_xy, prev_filtered_xy, DT)
        predicted_xy = filtered_xy + vel_est * LOOKAHEAD_TIME
        prev_filtered_xy = filtered_xy.copy()
        
        # === 4. Динамическая коррекция масштаба ===
        # Используем фиксированную высоту как «якорь» для масштаба
        if abs(orb_raw[2]) > MIN_ORB_CONFIDENCE:
            # Оцениваем масштаб по вертикали (предполагаем изотропность)
            scale_z = TARGET_Z / orb_raw[2]
            scale_z = clamp_scale(scale_z)
            
            # Плавное обновление (низкочастотная коррекция)
            if time.time() - last_scale_update > 0.2:  # Обновляем не чаще 5 Гц
                scale = (1 - SCALE_CORRECTION_RATE) * scale + SCALE_CORRECTION_RATE * scale_z
                last_scale_update = time.time()
        
        # Применяем масштаб к горизонтальным координатам
        current_xy = predicted_xy * scale
        
        # === 5. ПИД-регулятор (только для XY) ===
        target = path[wp]
        err_xy = target[:2] - current_xy
        
        # P + D контроль
        vel_xy = KP * err_xy + KD * (err_xy - prev_err_xy) / DT
        prev_err_xy = err_xy.copy()
        
        # Ограничение скорости
        speed_xy = np.linalg.norm(vel_xy)
        if speed_xy > MAX_VEL:
            vel_xy = vel_xy / speed_xy * MAX_VEL
        
        # === 6. Формирование команды: Z = 0 (дрон держит высоту сам) ===
        vel_cmd = np.array([vel_xy[0], vel_xy[1], 0.0])
        
        # === 7. Переключение вейпоинтов с гистерезисом ===
        err_2d = np.linalg.norm(err_xy)
        
        # Переключаемся, если близко к цели И не только что переключились
        if err_2d < LOOKAHEAD_DIST:
            # Гистерезис: не переключаемся обратно, если ошибка чуть выросла
            if wp < len(path) - 1 and err_2d < (LOOKAHEAD_DIST - WAYPOINT_HYSTERESIS):
                wp += 1
                print(f"\nWaypoint {wp} reached, next: {path[wp][:2]}")
        
        # === 8. Отправка команды на дрон ===
        client.moveByVelocityAsync(
            vel_cmd[0], vel_cmd[1], vel_cmd[2],
            duration=DT,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, 0)  # Можно добавить управление по курсу
        )
        
        # === 9. Логирование ===
        err_z = TARGET_Z - (orb_raw[2] * scale)  # Для отладки
        print(
            f"\rWP={wp}/{len(path)-1} | "
            f"ERR_2D={err_2d:4.2f}m | "
            f"ERR_Z={err_z:5.2f}m | "
            f"SCALE={scale:6.2f} | "
            f"VEL=[{vel_xy[0]:+.2f}, {vel_xy[1]:+.2f}]",
            end="", flush=True
        )
        
        # === 10. Проверка завершения маршрута ===
        if wp >= len(path) - 1 and err_2d < 0.3:
            print(f"\n\nTRAJECTORY COMPLETE!")
            print(f"   Final scale: {scale:.3f}")
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
        
        # 2. Запуск навигации
        run(initial_scale)
        
    except KeyboardInterrupt:
        print("\n\nManual stop requested")
        client.hoverAsync().join()
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        client.hoverAsync().join()
        
    finally:
        print("Disconnecting...")
        client.enableApiControl(False)
        print("Done")
