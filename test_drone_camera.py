import sys
import time
import cv2
from djitellopy import Tello

def main():
    print("=" * 50)
    print("  DJI TELLO: CAMERA & BATTERY TEST")
    print("=" * 50)
    print("Connecting to Tello over Wi-Fi...")
    
    tello = Tello()
    try:
        tello.connect()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Tello: {e}")
        print("Please check: Is Tello turned ON and laptop connected to TELLO-XXXXXX Wi-Fi?")
        return 1

    battery = tello.get_battery()
    temp = tello.get_temperature() if hasattr(tello, "get_temperature") else "N/A"
    print(f"\n>>> Tello Battery: {battery}%")
    print(f">>> Tello Temp:    {temp}")

    if battery < 15:
        print("\n[WARNING] Battery is critically low! Drone camera may freeze or turn off.")

    print("\nStarting video stream...")
    tello.streamon()
    frame_read = tello.get_frame_read()

    print("Opening camera window... Press 'q' on the video window to close.")
    while True:
        frame = frame_read.frame
        if frame is not None:
            # frame is RGB in djitellopy, convert to BGR for opencv
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(frame_bgr, f"Battery: {battery}%", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if battery > 20 else (0, 0, 255), 2)
            cv2.imshow("Tello Drone Live Camera (Press 'q' to exit)", frame_bgr)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    print("Closing stream...")
    try:
        tello.streamoff()
    except Exception:
        pass
    cv2.destroyAllWindows()
    return 0

if __name__ == "__main__":
    sys.exit(main())
