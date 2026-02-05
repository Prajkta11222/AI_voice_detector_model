#!/usr/bin/env powershell
<#
Test validation and authentication for FastAPI endpoints
#>

$BaseUrl = "http://127.0.0.1:8000"
$ApiKey = "your-secret-api-key"

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "Testing FastAPI Validation and Authentication" -ForegroundColor Cyan
Write-Host "========================================================================" -ForegroundColor Cyan

# Test 1: Missing all required fields
Write-Host "`n[Test 1] Missing all required fields (422 expected):" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/voice-detection" `
        -Method POST `
        -Headers @{"x-api-key" = $ApiKey} `
        -ContentType "application/json" `
        -Body '{}' `
        -ErrorAction Stop
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
} catch {
    $statusCode = $_.Exception.Response.StatusCode.Value__
    $body = $_.Exception.Response.Content | ConvertFrom-Json
    Write-Host "Status: $statusCode" -ForegroundColor Red
    Write-Host "Response: $(ConvertTo-Json $body -Depth 10)" -ForegroundColor Gray
}

# Test 2: Missing audioBase64 field
Write-Host "`n[Test 2] Missing audioBase64 field (422 expected):" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/voice-detection" `
        -Method POST `
        -Headers @{"x-api-key" = $ApiKey} `
        -ContentType "application/json" `
        -Body '{"language":"english","audioFormat":"wav"}' `
        -ErrorAction Stop
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
} catch {
    $statusCode = $_.Exception.Response.StatusCode.Value__
    Write-Host "Status: $statusCode" -ForegroundColor Red
    Write-Host "Headers: $(ConvertTo-Json $_.Exception.Response.Headers -Depth 10)" -ForegroundColor Gray
}

# Test 3: Invalid API Key (401 expected)
Write-Host "`n[Test 3] Invalid API Key (401 expected):" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/voice-detection" `
        -Method POST `
        -Headers @{"x-api-key" = "wrong-key"} `
        -ContentType "application/json" `
        -Body '{"language":"english","audioFormat":"wav","audioBase64":"dGVzdA=="}' `
        -ErrorAction Stop
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
} catch {
    $statusCode = $_.Exception.Response.StatusCode.Value__
    $body = $_.Exception.Response.Content | ConvertFrom-Json
    Write-Host "Status: $statusCode" -ForegroundColor Red
    Write-Host "Response: $(ConvertTo-Json $body -Depth 10)" -ForegroundColor Gray
}

# Test 4: Valid API Key (correct case)
Write-Host "`n[Test 4] Valid API Key - Correct Case:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/voice-detection" `
        -Method POST `
        -Headers @{"x-api-key" = $ApiKey} `
        -ContentType "application/json" `
        -Body '{"language":"english","audioFormat":"wav","audioBase64":"dGVzdA=="}' `
        -ErrorAction Stop
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
} catch {
    $statusCode = $_.Exception.Response.StatusCode.Value__
    $body = $_.Exception.Response.Content | ConvertFrom-Json
    Write-Host "Status: $statusCode (Invalid base64 is expected)" -ForegroundColor Green
    Write-Host "Message: $($body.detail)" -ForegroundColor Gray
}

# Test 5: Valid API Key in uppercase (case-insensitive test)
Write-Host "`n[Test 5] Valid API Key - UPPERCASE (case-insensitive test):" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/api/voice-detection" `
        -Method POST `
        -Headers @{"x-api-key" = $ApiKey.ToUpper()} `
        -ContentType "application/json" `
        -Body '{"language":"english","audioFormat":"wav","audioBase64":"dGVzdA=="}' `
        -ErrorAction Stop
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
} catch {
    $statusCode = $_.Exception.Response.StatusCode.Value__
    $body = $_.Exception.Response.Content | ConvertFrom-Json
    Write-Host "Status: $statusCode (Invalid base64 is expected)" -ForegroundColor Green
    Write-Host "Message: $($body.detail)" -ForegroundColor Gray
}

# Test 6: Health check (no auth required)
Write-Host "`n[Test 6] Health Check (no auth required):" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "$BaseUrl/health" -Method GET -ErrorAction Stop
    Write-Host "Status: $($response.StatusCode)" -ForegroundColor Green
    $body = ConvertFrom-Json $response.Content
    Write-Host "Response: $(ConvertTo-Json $body -Depth 10)" -ForegroundColor Gray
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}

# Test 7: Swagger UI (no auth required)
Write-Host "`n[Test 7] Swagger UI Documentation:" -ForegroundColor Yellow
Write-Host "URL: http://127.0.0.1:8000/docs" -ForegroundColor Cyan

Write-Host "`n========================================================================" -ForegroundColor Cyan
Write-Host "✓ Testing Complete - Check Swagger at http://127.0.0.1:8000/docs" -ForegroundColor Cyan
Write-Host "========================================================================`n" -ForegroundColor Cyan
