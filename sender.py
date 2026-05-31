import airsim
import zmq
import numpy as np
import cv2
import time
import struct

# Настройки
ZMQ_HOST = "0.0.0.0"  # 0.0.0.0 для BIND
ZMQ_PORT = 5555
JPEG_QUALITY = 75
MIN_JPEG_SIZE = 2000
SEND_SIZE_HEADER = True
TARGET_FPS = 30
TARGET_WIDTH = 256
TARGET_HEIGHT = 144

def main():
    print("AirSim Sender starting...")
    
    # ZeroMQ - BIND
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUSH)
    sock.setsockopt(zmq.SNDHWM, 1)
    sock.bind(f"tcp://{ZMQ_HOST}:{ZMQ_PORT}")  
    print(f"ZMQ bound to tcp://{ZMQ_HOST}:{ZMQ_PORT}")

    # Подключение к AirSim
    try:
        client = airsim.MultirotorClient()
        client.confirmConnection()
        print("AirSim connected")
    except Exception as e:
        print(f"Failed to connect to AirSim: {e}")
        return

    frame_cnt = 0
    t_start = time.time()
    last_send_time = 0

    print(f"Streaming: {TARGET_WIDTH}x{TARGET_HEIGHT} @ {TARGET_FPS} FPS")

    while True:
        try:
            if TARGET_FPS > 0:
                elapsed = time.time() - last_send_time
                target_frame_time = 1.0 / TARGET_FPS
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
            
            resp = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False)
            ])[0]
            
            img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            img_rgb = img.reshape(resp.height, resp.width, 3)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            success, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not success or buf.size < MIN_JPEG_SIZE:
                continue
            
            if SEND_SIZE_HEADER:
                size_header = struct.pack('I', buf.size)
                packet = size_header + buf.tobytes()
            else:
                packet = buf.tobytes()
            
            sock.send(packet)
            last_send_time = time.time()
            
            frame_cnt += 1
            if frame_cnt % 30 == 0:
                fps = frame_cnt / (time.time() - t_start)
                print(f"Sent: {frame_cnt} | FPS: {fps:.1f} | JPEG: {buf.size}B")
            
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)

    sock.close()
    ctx.term()

if __name__ == "__main__":
    main()
