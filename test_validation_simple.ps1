#!/usr/bin/env powershell
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
    
    $Headers = @{
        "Content-Type" = "application/json"
    }
    
    if ($CustomHeaders -and $CustomHeaders.Count -gt 0) {
        foreach ($key in $CustomHeaders.Keys) {
            $Headers[$key] = $CustomHeaders[$key]
        }
    } else {
        $Headers["x-api-key"] = $ApiKey
    }
    
    Write-Host "`nTest: $Name"
    Write-Host "  Expected: $ExpectedStatus"
    
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
            Write-Host "  + PASS - Status: $Status" -ForegroundColor Green
            $script:TestsPassed++
        } else {
            Write-Host "  - FAIL - Got $Status, expected $ExpectedStatus" -ForegroundColor Red
            $script:TestsFailed++
        }
    }
    catch {
        $Status = if ($_.Exception.Response) {
            $_.Exception.Response.StatusCode.Value__
        } else {
            0
        }
        
        if ($Status -eq $ExpectedStatus) {
            Write-Host "  + PASS - Status: $Status" -ForegroundColor Green
            $script:TestsPassed++
            
            if ($Status -eq 422) {
                Show-ValidationErrors $_
            }
        } else {
            Write-Host "  - FAIL - Got $Status, expected $ExpectedStatus" -ForegroundColor Red
            $script:TestsFailed++
        }
    }
}

function Show-ValidationErrors {
    param([System.Management.Automation.ErrorRecord]$Error)
    
    try {
        $responseContent = $Error.Exception.Response.Content
        
        if ($responseContent -and $responseContent.Length -gt 0) {
            $stream = New-Object System.IO.StreamReader($responseContent)
            $responseText = $stream.ReadToEnd()
            $stream.Close()
            
            if ($responseText -and $responseText.Length -gt 0) {
                $ErrorBody = ConvertFrom-Json $responseText -ErrorAction SilentlyContinue
                
                if ($ErrorBody.detail -and $ErrorBody.detail.Count -gt 0) {
                    Write-Host "    Validation Errors:" -ForegroundColor Yellow
                    foreach ($ValidationError in $ErrorBody.detail) {
                        if ($ValidationError.loc -and $ValidationError.loc.Count -gt 0) {
                            $Field = $ValidationError.loc[-1]
                            $Msg = $ValidationError.msg
                            Write-Host "      - $Field : $Msg" -ForegroundColor Yellow
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

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "FASTAPI VALIDATION TEST SUITE - FIXED VERSION" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "`nSCHEMA REQUIREMENTS:"
Write-Host "  - language: required string, min_length 1"
Write-Host "  - audioFormat: required string, min_length 1"
Write-Host "  - audioBase64: required string, min_length 1"
Write-Host "  - x-api-key header: required for authentication"

Write-Host "`n======== SECTION 1: SCHEMA VALIDATION ========" -ForegroundColor Green

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

Write-Host "`n======== SECTION 2: FIELD VALUE VALIDATION ========" -ForegroundColor Green

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

Write-Host "`n======== SECTION 3: CASE SENSITIVITY ========" -ForegroundColor Green

Test-API -Name "UPPERCASE language field" `
    -Body @{language="ENGLISH"; audioFormat="wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Test-API -Name "UPPERCASE audioFormat field" `
    -Body @{language="english"; audioFormat="WAV"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Test-API -Name "Mixed case both fields" `
    -Body @{language="English"; audioFormat="Wav"; audioBase64="//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"} `
    -ExpectedStatus 200

Write-Host "`n======== SECTION 4: AUTHENTICATION ========" -ForegroundColor Green

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

$Total = $TestsPassed + $TestsFailed
$PercentPass = if ($Total -gt 0) { [math]::Round(($TestsPassed / $Total) * 100) } else { 0 }

Write-Host "`n======================================================================" -ForegroundColor Cyan
Write-Host "TEST SUMMARY" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

if ($TestsFailed -eq 0) {
    Write-Host "SUCCESS: All Tests Passed!" -ForegroundColor Green
} else {
    Write-Host "NOTICE: Some tests failed. See details above." -ForegroundColor Yellow
}

Write-Host "  Passed: $TestsPassed" -ForegroundColor Green
Write-Host "  Failed: $TestsFailed" -ForegroundColor $(if ($TestsFailed -gt 0) { "Red" } else { "Green" })
Write-Host "  Total:  $Total" -ForegroundColor Cyan
Write-Host "  Pass Rate: $PercentPass%" -ForegroundColor $(if ($PercentPass -eq 100) { "Green" } else { "Yellow" })

Write-Host "`nIMPORTANT FIXES APPLIED:" -ForegroundColor Cyan
Write-Host "  [1] Fresh header hashtable each call (fixes 401 errors)"
Write-Host "  [2] Safe ConvertFrom-Json error handling (fixes null crashes)"
Write-Host "  [3] Better diagnostic output (shows validation details)"

Write-Host ""
