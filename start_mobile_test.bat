@echo off
cd /d "%~dp0"
echo ========================================================
echo   WALL-MARKER: VIRTUAL TEST SERVER (HTTPS)
echo ========================================================
echo.
echo Finding your laptop IP address:
ipconfig | findstr /i "IPv4"
echo.
echo On your phone browser (Safari or Chrome), open:
echo   https://^<LAPTOP-IP^>:8443/viewer
echo.
echo (Accept the browser self-signed certificate warning to allow camera access)
echo.
"C:\Users\Muneer\AppData\Local\Python\bin\python.exe" -m tests.virtual_server --scenario s1_basic --port 8443 --https --cert certs/cert.pem --key certs/key.pem
pause
