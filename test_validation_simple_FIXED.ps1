#!/usr/bin/env powershell
<#
    FastAPI Validation Test Script - FIXED VERSION
    Tests all 422 and 401 scenarios with proper header handling
    
    FIXES APPLIED:
    1. Fresh headers hashtable each call (avoids mutation issues)
    2. Explicit header passing (clear intent for test cases)
    3. Safe response content reading (fixes ConvertFrom-Json null error)
    4. Better error reporting (shows validation details)
#>

param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ApiKey = "your-secret-api-key"
)

$TestsPassed = 0
$TestsFailed = 0

function Test-API {
    param(
        [string]$Name,
        [object]$Body,
        [hashtable]$CustomHeaders = $null,
        [int]$ExpectedStatus = 200
    )
    
    # BUILD HEADERS FRESH EACH TIME - Don't modify a hashtable passed in
    # This avoids PowerShell hashtable mutation issues with Invoke-WebRequest
    $Headers = @{
        "Content-Type" = "application/json"
    }
    
    # Add custom headers if provided, otherwise use default API key
    if ($CustomHeaders -and $CustomHeaders.Count -gt 0) {
        foreach ($key in $CustomHeaders.Keys) {
            $Headers[$key] = $CustomHeaders[$key]
        }
    } else {
        # Default: add the API key with proper casing
        $Headers["x-api-key"] = $ApiKey
    }
    
    Write-Host "`nTest: $Name"
    Write-Host "  Expected: $ExpectedStatus"
    
    # DEBUG: Show headers being sent (helps diagnose header issues)
    $ApiKeyStatus = if ($Headers.ContainsKey('x-api-key')) { 
        $Headers['x-api-key'] 
    } else { 
        'NOT SET' 
    }
    Write-Host "  Sending x-api-key: $ApiKeyStatus" -ForegroundColor Gray
    
    try {
        $Response = Invoke-WebRequest `
            -Uri "$BaseUrl/api/voice-detection" `
            -Method POST `
            -Headers $Headers `
            -Body (ConvertTo-Json $Body -Depth 10) `
            -ErrorAction Stop
        
        $Status = $Response.StatusCode
        if ($Status -eq $ExpectedStatus) {
            Write-Host "  ✓ PASS - Status: $Status" -ForegroundColor Green
            $script:TestsPassed++
        } else {
            Write-Host "  ✗ FAIL - Got $Status, expected $ExpectedStatus" -ForegroundColor Red
            $script:TestsFailed++
        }
    }
    catch {
        # Get status code safely - handle different exception types
        $Status = if ($_.Exception.Response) {
            $_.Exception.Response.StatusCode.Value__
        } else {
            0
        }
        
        if ($Status -eq $ExpectedStatus) {
            Write-Host "  ✓ PASS - Status: $Status" -ForegroundColor Green
            $script:TestsPassed++
            
            # Show error details for 422 validation errors
            if ($Status -eq 422) {
                Show-ValidationErrors $_
            }
        } else {
            Write-Host "  ✗ FAIL - Got $Status, expected $ExpectedStatus" -ForegroundColor Red
            $script:TestsFailed++
        }
    }
}

function Show-ValidationErrors {
    param([System.Management.Automation.ErrorRecord]$Error)
    
    try {
        # FIXED: Safely read response content using StreamReader
        # This avoids the ConvertFrom-Json null reference error
        $responseContent = $Error.Exception.Response.Content
        
        if ($responseContent -and $responseContent.Length -gt 0) {
            # Read the stream properly
            $stream = New-Object System.IO.StreamReader($responseContent)
            $responseText = $stream.ReadToEnd()
            $stream.Close()
            
            if ($responseText -and $responseText.Length -gt 0) {
                # Try to parse JSON safely
                $ErrorBody = ConvertFrom-Json $responseText -ErrorAction SilentlyContinue
                
                if ($ErrorBody.detail -and $ErrorBody.detail.Count -gt 0) {
                    Write-Host "    Validation Errors:" -ForegroundColor Yellow
                    foreach ($ValidationError in $ErrorBody.detail) {
                        if ($ValidationError.loc -and $ValidationError.loc.Count -gt 0) {
                            $Field = $ValidationError.loc[-1]
                            $Msg = $ValidationError.msg
                            Write-Host "      • $Field : $Msg" -ForegroundColor Yellow
                        }
                    }
                }
            }
        }
    }
    catch {
        Write-Host "    (Could not parse error details)" -ForegroundColor Gray
    }
}

# ============================================================================
# TEST EXECUTION
# ============================================================================

Write-Host "`n===================================================================="
Write-Host "FASTAPI VALIDATION TEST SUITE" -ForegroundColor Cyan
Write-Host "Fixed Version: Proper header handling + safe error parsing" -ForegroundColor Cyan
Write-Host "====================================================================" -ForegroundColor Cyan

Write-Host "`nSCHEMA REQUIREMENTS:"
Write-Host "  • language: required string, min_length 1"
Write-Host "  • audioFormat: required string, min_length 1"
Write-Host "  • audioBase64: required string, min_length 1"
Write-Host "  • x-api-key header: required for authentication"

Write-Host "`n" + ("="*68) -ForegroundColor Green
Write-Host "SECTION 1: SCHEMA VALIDATION" -ForegroundColor Green
Write-Host ("="*68) -ForegroundColor Green

Test-API -Name "Valid request - all fields correct" `
    -Body @{language="english"; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Test-API -Name "Missing audioBase64 field" `
    -Body @{language="english"; audioFormat="wav"} `
    -ExpectedStatus 422

Test-API -Name "Missing audioFormat field" `
    -Body @{language="english"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 422

Test-API -Name "Missing language field" `
    -Body @{audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 422

Test-API -Name "Empty request body" `
    -Body @{} `
    -ExpectedStatus 422

Write-Host "`n" + ("="*68) -ForegroundColor Green
Write-Host "SECTION 2: FIELD VALUE VALIDATION" -ForegroundColor Green
Write-Host ("="*68) -ForegroundColor Green

Test-API -Name "Empty string in language field" `
    -Body @{language=""; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 422

Test-API -Name "Whitespace-only language field" `
    -Body @{language="   "; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 422

Test-API -Name "NULL value in language field" `
    -Body @{language=$null; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 422

Test-API -Name "Wrong data type - integer instead of string" `
    -Body @{language=123; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 422

Write-Host "`n" + ("="*68) -ForegroundColor Green
Write-Host "SECTION 3: CASE SENSITIVITY" -ForegroundColor Green
Write-Host ("="*68) -ForegroundColor Green

Test-API -Name "UPPERCASE language field" `
    -Body @{language="ENGLISH"; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Test-API -Name "UPPERCASE audioFormat field" `
    -Body @{language="english"; audioFormat="WAV"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Test-API -Name "Mixed case both fields" `
    -Body @{language="English"; audioFormat="Wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Write-Host "`n" + ("="*68) -ForegroundColor Green
Write-Host "SECTION 4: AUTHENTICATION" -ForegroundColor Green
Write-Host ("="*68) -ForegroundColor Green

Test-API -Name "Missing x-api-key header" `
    -CustomHeaders @{} `
    -Body @{language="english"; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 422

Test-API -Name "Invalid x-api-key value" `
    -CustomHeaders @{"x-api-key"="wrong-key"} `
    -Body @{language="english"; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 401

Test-API -Name "Valid x-api-key (correct case)" `
    -CustomHeaders @{"x-api-key"=$ApiKey} `
    -Body @{language="english"; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Test-API -Name "Valid x-api-key (UPPERCASE)" `
    -CustomHeaders @{"x-api-key"=$ApiKey.ToUpper()} `
    -Body @{language="english"; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

# ============================================================================
# SUMMARY
# ============================================================================

$Total = $TestsPassed + $TestsFailed
$PercentPass = if ($Total -gt 0) { [math]::Round(($TestsPassed / $Total) * 100) } else { 0 }

Write-Host "`n" + ("="*68) -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host ("="*68) -ForegroundColor Cyan

if ($TestsFailed -eq 0) {
    Write-Host "✓ All Tests Passed!" -ForegroundColor Green -BackgroundColor Black
} else {
    Write-Host "Some tests failed. See details above." -ForegroundColor Yellow
}

Write-Host "  Passed: $TestsPassed" -ForegroundColor Green
Write-Host "  Failed: $TestsFailed" -ForegroundColor $(if ($TestsFailed -gt 0) { "Red" } else { "Green" })
Write-Host "  Total:  $Total" -ForegroundColor Cyan
Write-Host "  Pass Rate: $PercentPass%" -ForegroundColor $(if ($PercentPass -eq 100) { "Green" } else { "Yellow" })

Write-Host "`nDOCUMENTATION & RESOURCES:" -ForegroundColor Cyan
Write-Host "  • Swagger UI: http://127.0.0.1:8000/docs"
Write-Host "  • Python tests: python debug_422_validation.py"
Write-Host "  • FastAPI docs: http://127.0.0.1:8000/redoc"
Write-Host "  • Fix guide: POWERSHELL_401_FIX.md"

Write-Host "`nNOTE: This is the FIXED version with:" -ForegroundColor Gray
Write-Host "  • Fresh header hashtable each call" -ForegroundColor Gray
Write-Host "  • Safe ConvertFrom-Json error handling" -ForegroundColor Gray
Write-Host "  • Better diagnostic output" -ForegroundColor Gray

Write-Host ""
