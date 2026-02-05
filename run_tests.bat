@echo off
cd /d D:\AI.Voice.Detector
timeout /t 3 /nobreak
powershell -ExecutionPolicy Bypass -NoProfile -File test_validation_simple.ps1
pause
