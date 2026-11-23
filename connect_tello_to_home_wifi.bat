@echo off
cd /d "%~dp0"
echo ========================================================
echo Connecting Tello to Home Wi-Fi (Station Mode)
echo ========================================================
echo.
echo Make sure Laptop Wi-Fi is currently connected to TELLO-XXXXXX.
echo.
pause
"C:\Users\Muneer\AppData\Local\Python\bin\python.exe" connect_tello_to_home_wifi.py
echo.
echo Done. You can now reconnect your Laptop back to 'Net ni ata'.
pause
