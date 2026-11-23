@echo off
cd /d "%~dp0"
echo Connecting to Tello Camera...
"C:\Users\Muneer\AppData\Local\Python\bin\python.exe" test_drone_camera.py
pause
