@echo off
REM Test FastAPI Validation and Authentication

setlocal enabledelayedexpansion

set "BASE_URL=http://127.0.0.1:8000"
set "API_KEY=my-secret-key"

echo.
echo ========================================================================
echo Testing FastAPI Validation and Authentication
echo ========================================================================

REM Test 1: Missing all required fields
echo.
echo [Test 1] Missing all required fields (422 expected):
curl -X POST "%BASE_URL%/api/voice-detection" ^
  -H "x-api-key: %API_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{}" ^
  -w "\nStatus: %%{http_code}\n" 2>nul

REM Test 2: Missing audioBase64
echo.
echo [Test 2] Missing audioBase64 field (422 expected):
curl -X POST "%BASE_URL%/api/voice-detection" ^
  -H "x-api-key: %API_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"english\",\"audioFormat\":\"wav\"}" ^
  -w "\nStatus: %%{http_code}\n" 2>nul

REM Test 3: Invalid API Key
echo.
echo [Test 3] Invalid API Key (401 expected):
curl -X POST "%BASE_URL%/api/voice-detection" ^
  -H "x-api-key: wrong-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"english\",\"audioFormat\":\"wav\",\"audioBase64\":\"dGVzdA==\"}" ^
  -w "\nStatus: %%{http_code}\n" 2>nul

REM Test 4: Valid API Key
echo.
echo [Test 4] Valid API Key:
curl -X POST "%BASE_URL%/api/voice-detection" ^
  -H "x-api-key: %API_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"english\",\"audioFormat\":\"wav\",\"audioBase64\":\"dGVzdA==\"}" ^
  -w "\nStatus: %%{http_code}\n" 2>nul

REM Test 5: Valid API Key in UPPERCASE (case-insensitive)
echo.
echo [Test 5] Valid API Key in UPPERCASE (case-insensitive):
curl -X POST "%BASE_URL%/api/voice-detection" ^
  -H "x-api-key: my-secret-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"english\",\"audioFormat\":\"wav\",\"audioBase64\":\"dGVzdA==\"}" ^
  -w "\nStatus: %%{http_code}\n" 2>nul

REM Test 6: Health Check
echo.
echo [Test 6] Health Check (no auth required):
curl -X GET "%BASE_URL%/health" ^
  -w "\nStatus: %%{http_code}\n" 2>nul

echo.
echo ========================================================================
echo Testing complete. Check Swagger at: http://127.0.0.1:8000/docs
echo ========================================================================
echo.

endlocal
