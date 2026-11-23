import sys
import time
from djitellopy import Tello

def main():
    print("=" * 60)
    print("  CONNECT TELLO TO HOME WI-FI (STATION MODE)")
    print("=" * 60)
    ssid = sys.argv[1] if len(sys.argv) > 1 else input("Enter your Wi-Fi SSID name: ").strip()
    password = sys.argv[2] if len(sys.argv) > 2 else input("Enter your Wi-Fi Password: ").strip()
    print(f"Target Wi-Fi: {ssid}")
    print("\nConnecting to Tello over its current TELLO-XXXXXX Wi-Fi...")
    
    tello = Tello()
    try:
        tello.connect()
    except Exception as e:
        print(f"\n[ERROR] Could not connect to Tello: {e}")
        print("Make sure your laptop Wi-Fi is connected to TELLO-XXXXXX right now.")
        return 1

    print("\nSending Station Mode command to Tello...")
    try:
        # Command: ap <ssid> <password>
        response = tello.send_control_command(f"ap {ssid} {password}")
        print(f"Tello response: {response}")
        print(f"\nSUCCESS! Tello will now reboot and join your home Wi-Fi ('{ssid}').")
        print("\n[NOTE] If you ever want to reset Tello back to TELLO-XXXXXX,")
        print("just hold Tello's power button for 5 seconds while it is ON.")
    except Exception as e:
        print(f"\n[NOTE] Response or timeout: {e}")
        print("(Tello typically reboots immediately upon receiving this command)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
