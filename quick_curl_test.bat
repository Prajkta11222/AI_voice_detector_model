@echo off
REM Quick test of API with curl

echo.
echo ============================================================
echo Testing FastAPI with curl
echo ============================================================
echo.

REM Test 1: Valid request
echo Test 1: Valid request with all fields
curl -X POST "http://127.0.0.1:8000/api/voice-detection" ^
  -H "x-api-key: your-secret-api-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"english\",\"audioFormat\":\"wav\",\"audioBase64\":\"//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB\"}"
echo.
echo.

REM Test 2: Missing field
echo Test 2: Missing audioBase64 field (should get 422)
curl -X POST "http://127.0.0.1:8000/api/voice-detection" ^
  -H "x-api-key: your-secret-api-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"english\",\"audioFormat\":\"wav\"}"
echo.
echo.

REM Test 3: Invalid API key
echo Test 3: Invalid API key (should get 401)
curl -X POST "http://127.0.0.1:8000/api/voice-detection" ^
  -H "x-api-key: wrong-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"english\",\"audioFormat\":\"wav\",\"audioBase64\":\"//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB\"}"
echo.
echo.

echo Test 4: UPPERCASE language (should work - returns 200)
curl -X POST "http://127.0.0.1:8000/api/voice-detection" ^
  -H "x-api-key: your-secret-api-key" ^
  -H "Content-Type: application/json" ^
  -d "{\"language\":\"ENGLISH\",\"audioFormat\":\"wav\",\"audioBase64\":\"//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB\"}"
echo.
echo.

echo ============================================================
echo Test Summary
echo ============================================================
echo.
echo Status Code Legend:
echo   200 = Success
echo   401 = Unauthorized (invalid API key)
echo   422 = Validation Error (invalid/missing fields)
echo.
pause
