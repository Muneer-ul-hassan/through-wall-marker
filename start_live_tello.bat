@echo off
cd /d "%~dp0"
echo ========================================================
echo   WALL-MARKER: REAL TELLO DRONE LIVE SERVER (HTTPS)
echo ========================================================
echo.
echo 1. Turn ON your DJI Tello drone.
echo 2. Connect your Laptop Wi-Fi to TELLO-XXXXXX.
echo 3. Connect your Mobile Phone Wi-Fi to the SAME TELLO-XXXXXX.
echo.
echo Your Laptop IP on Tello network:
ipconfig | findstr /i "IPv4"
echo.
echo On your Phone browser (Safari or Chrome), open:
echo   https://192.168.10.2:8443/viewer
echo (if your laptop IP differs from 192.168.10.2, use that IP above)
echo.
"C:\Users\Muneer\AppData\Local\Python\bin\python.exe" -m server.main --mode live --port 8443 --https --cert certs/cert.pem --key certs/key.pem
pause
