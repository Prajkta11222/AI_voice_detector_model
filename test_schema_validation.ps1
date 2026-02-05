#!/usr/bin/env powershell
<#
.SYNOPSIS
    Systematic validation testing for FastAPI voice-detection endpoint
    
.DESCRIPTION
    Tests request schema, required fields, validation errors, and authentication.
    Displays colored output and detailed validation error information.

.EXAMPLE
    .\test_schema_validation.ps1
    
.NOTES
    Requires: PowerShell 5.0+, server running on http://127.0.0.1:8000
#>

param(
    [string]$BaseUrl = "http://127.0.0.1:8000",
    [string]$ApiKey = "your-secret-api-key"
)

# Color definitions
$Colors = @{
    'Reset'   = "`e[0m"
    'Bold'    = "`e[1m"
    'Green'   = "`e[32m"
    'Red'     = "`e[31m"
    'Yellow'  = "`e[33m"
    'Cyan'    = "`e[36m"
    'Blue'    = "`e[34m"
}

# Test counters
$TestsPassed = 0
$TestsFailed = 0
$TestsTotal = 0

function Write-ColorOutput {
    param(
        [string]$Text,
        [string]$Color = "Reset"
    )
    Write-Host $Text -ForegroundColor $Color -NoNewline
}

function Write-Section {
    param([string]$Title)
    Write-Host "`n$($Colors.Bold)$('='*75)$($Colors.Reset)"
    Write-Host "$($Colors.Bold)$Title$($Colors.Reset)"
    Write-Host "$($Colors.Bold)$('='*75)$($Colors.Reset)`n"
}

function Write-TestHeader {
    param(
        [string]$TestName,
        [string]$ExpectedStatus
    )
    Write-Host "`n$($Colors.Bold)Test: $TestName$($Colors.Reset)"
    Write-Host "$($Colors.Bold)Expected: $ExpectedStatus$($Colors.Reset)"
}

function Test-Request {
    param(
        [string]$Name,
        [object]$Body,
        [hashtable]$Headers = @{},
        [string]$ExpectedStatus = "200",
        [switch]$ShowFullResponse
    )
    
    $Script:TestsTotal++
    
    Write-TestHeader -TestName $Name -ExpectedStatus $ExpectedStatus
    
    # Set default header
    if (-not $Headers.ContainsKey("x-api-key")) {
        $Headers["x-api-key"] = $ApiKey
    }
    
    try {
        $Response = Invoke-WebRequest `
            -Uri "$BaseUrl/api/voice-detection" `
            -Method POST `
            -Headers $Headers `
            -ContentType "application/json" `
            -Body (ConvertTo-Json $Body -Depth 10) `
            -ErrorAction Stop
        
        $StatusCode = $Response.StatusCode
        $ResponseBody = $Response.Content | ConvertFrom-Json
        
        if ($StatusCode -eq [int]$ExpectedStatus) {
            Write-Host "$($Colors.Green)✓ PASS$($Colors.Reset) - Status: $StatusCode"
            $Script:TestsPassed++
        } else {
            Write-Host "$($Colors.Red)✗ FAIL$($Colors.Reset) - Status: $StatusCode (expected $ExpectedStatus)"
            $Script:TestsFailed++
        }
        
        if ($ShowFullResponse) {
            Write-Host "$($Colors.Cyan)Response:$($Colors.Reset)"
            $ResponseBody | ConvertTo-Json -Depth 10 | Write-Host
        }
    }
    catch {
        $StatusCode = $_.Exception.Response.StatusCode.Value__
        $ErrorBody = $_.Exception.Response.Content | ConvertFrom-Json -ErrorAction SilentlyContinue
        
        if ($StatusCode -eq [int]$ExpectedStatus) {
            Write-Host "$($Colors.Green)✓ PASS$($Colors.Reset) - Status: $StatusCode"
            $Script:TestsPassed++
        } else {
            Write-Host "$($Colors.Red)✗ FAIL$($Colors.Reset) - Status: $StatusCode (expected $ExpectedStatus)"
            $Script:TestsFailed++
        }
        
        # Show validation errors for 422
        if ($StatusCode -eq 422) {
            Write-Host "$($Colors.Red)Validation Errors:$($Colors.Reset)"
            
            if ($ErrorBody -is [array]) {
                foreach ($Error in $ErrorBody) {
                    $Field = $Error.loc[-1]
                    $Message = $Error.msg
                    Write-Host "  • $Field`: $Message"
                }
            } elseif ($ErrorBody.detail -is [array]) {
                foreach ($Error in $ErrorBody.detail) {
                    $Field = $Error.loc[-1]
                    $Message = $Error.msg
                    Write-Host "  • $Field`: $Message"
                }
            }
        }
        
        if ($ShowFullResponse -and $ErrorBody) {
            Write-Host "$($Colors.Cyan)Full Response:$($Colors.Reset)"
            $ErrorBody | ConvertTo-Json -Depth 10 | Write-Host
        }
    }
}

function Test-HealthCheck {
    Write-Host "`n$($Colors.Cyan)Testing: Health Check (no auth required)$($Colors.Reset)"
    try {
        $Response = Invoke-WebRequest -Uri "$BaseUrl/health" -ErrorAction Stop
        Write-Host "$($Colors.Green)✓ Status: $($Response.StatusCode)$($Colors.Reset)"
        $Response.Content | ConvertFrom-Json | ConvertTo-Json | Write-Host
    }
    catch {
        Write-Host "$($Colors.Red)✗ Failed to reach health endpoint$($Colors.Reset)"
        Write-Host "$($_.Exception.Message)"
    }
}

function Write-SchemaInfo {
    Write-Host "REQUEST SCHEMA SPECIFICATION"
    Write-Host ""
    Write-Host "Endpoint:  POST /api/voice-detection"
    Write-Host "Auth:      x-api-key: [your-api-key] - case-insensitive"
    Write-Host ""
    Write-Host "REQUIRED FIELDS:"
    Write-Host "  1. language - type string, min_length 1"
    Write-Host "     - Valid: english, hindi, malayalam, telugu, tamil"
    Write-Host "     - Auto-converted to lowercase"
    Write-Host ""
    Write-Host "  2. audioFormat - type string, min_length 1"
    Write-Host "     - Valid: wav, mp3, flac, ogg"
    Write-Host "     - Auto-converted to lowercase"
    Write-Host ""
    Write-Host "  3. audioBase64 - type string, min_length 1"
    Write-Host "     - Must be valid base64 encoded audio"
    Write-Host "     - Decoded at endpoint, not at validation"
    Write-Host ""
    Write-Host "VALIDATION RULES:"
    Write-Host "  - All fields required (no defaults)"
    Write-Host "  - Empty strings rejected - min_length 1"
    Write-Host "  - NULL values rejected"
    Write-Host "  - Whitespace-only strings rejected by validator"
    Write-Host "  - Wrong data types rejected - returns 422"
}

function Write-TestSummary {
    Write-Host "`n$($Colors.Bold)$('='*75)$($Colors.Reset)"
    Write-Host "$($Colors.Bold)TEST SUMMARY$($Colors.Reset)"
    Write-Host "$($Colors.Bold)$('='*75)$($Colors.Reset)`n"
    
    Write-Host "Total Tests:  $TestsTotal"
    Write-Host "$($Colors.Green)Passed:       $TestsPassed$($Colors.Reset)"
    Write-Host "$($Colors.Red)Failed:       $TestsFailed$($Colors.Reset)"
    
    if ($TestsFailed -eq 0) {
        Write-Host "`n$($Colors.Green)✓ All tests passed!$($Colors.Reset)"
    } else {
        Write-Host "`n$($Colors.Red)✗ Some tests failed - see details above$($Colors.Reset)"
    }
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

Clear-Host

Write-Host "$($Colors.Bold)$($Colors.Cyan)"
Write-Host @"
╔════════════════════════════════════════════════════════════════════════════╗
║                  FASTAPI VOICE-DETECTION VALIDATION TESTS                 ║
║                   Schema | Fields | Auth | Error Handling                 ║
╚════════════════════════════════════════════════════════════════════════════╝
"@
Write-Host "$($Colors.Reset)"

Write-SchemaInfo

# Check server connectivity
Write-Section "PRE-TEST: Server Connectivity Check"
Test-HealthCheck

# ============================================================================
# SECTION 1: SCHEMA VALIDATION
# ============================================================================

Write-Section "SECTION 1: SCHEMA VALIDATION"

Test-Request `
    -Name "✓ Valid Request - All Required Fields" `
    -Body @{
        language = "english"
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "200"

Test-Request `
    -Name "✗ Empty Request Body" `
    -Body @{} `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Missing language Field" `
    -Body @{
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Missing audioFormat Field" `
    -Body @{
        language = "english"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Missing audioBase64 Field" `
    -Body @{
        language = "english"
        audioFormat = "wav"
    } `
    -ExpectedStatus "422"

# ============================================================================
# SECTION 2: FIELD VALUE VALIDATION
# ============================================================================

Write-Section "SECTION 2: FIELD VALUE VALIDATION"

Test-Request `
    -Name "✗ Empty String - language" `
    -Body @{
        language = ""
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Empty String - audioFormat" `
    -Body @{
        language = "english"
        audioFormat = ""
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Empty String - audioBase64" `
    -Body @{
        language = "english"
        audioFormat = "wav"
        audioBase64 = ""
    } `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Whitespace Only - language" `
    -Body @{
        language = "   "
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ NULL Value - language" `
    -Body @{
        language = $null
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Wrong Type - language (integer)" `
    -Body @{
        language = 123
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "422"

# ============================================================================
# SECTION 3: CASE SENSITIVITY
# ============================================================================

Write-Section "SECTION 3: CASE SENSITIVITY & AUTO-CONVERSION"

Test-Request `
    -Name "✓ UPPERCASE language (auto-converted to lowercase)" `
    -Body @{
        language = "ENGLISH"
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "200"

Test-Request `
    -Name "✓ MixedCase audioFormat (auto-converted to lowercase)" `
    -Body @{
        language = "english"
        audioFormat = "WAV"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "200"

Test-Request `
    -Name "✓ Mixed Case language and format" `
    -Body @{
        language = "English"
        audioFormat = "Wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -ExpectedStatus "200"

# ============================================================================
# SECTION 4: AUTHENTICATION
# ============================================================================

Write-Section "SECTION 4: AUTHENTICATION"

Test-Request `
    -Name "✗ Missing x-api-key Header" `
    -Body @{
        language = "english"
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -Headers @{} `
    -ExpectedStatus "422"

Test-Request `
    -Name "✗ Invalid x-api-key Value" `
    -Body @{
        language = "english"
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -Headers @{"x-api-key" = "wrong-key"} `
    -ExpectedStatus "401"

Test-Request `
    -Name "✓ Valid API Key - Correct Case" `
    -Body @{
        language = "english"
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -Headers @{"x-api-key" = $ApiKey} `
    -ExpectedStatus "200"

Test-Request `
    -Name "✓ Valid API Key - UPPERCASE (case-insensitive)" `
    -Body @{
        language = "english"
        audioFormat = "wav"
        audioBase64 = "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
    } `
    -Headers @{"x-api-key" = $ApiKey.ToUpper()} `
    -ExpectedStatus "200"

# ============================================================================
# SUMMARY
# ============================================================================

Write-TestSummary

Write-Host "`n$($Colors.Cyan)$($Colors.Bold)NEXT STEPS:$($Colors.Reset)"
Write-Host "1. Swagger Testing:"
Write-Host "   - Visit: $BaseUrl/docs"
Write-Host "   - Try the /api/voice-detection endpoint interactively"
Write-Host ""
Write-Host "2. Detailed Debugging:"
Write-Host "   - Run: python debug_422_validation.py"
Write-Host ""
Write-Host "3. Documentation:"
Write-Host "   - See: VALIDATION_DEBUGGING_GUIDE.md"
Write-Host ""
Write-Host "4. Before Deployment:"
Write-Host "   - Verify all test results above"
Write-Host "   - Test with real audio files (not test base64)"
Write-Host "   - Check with production API key"
Write-Host "   - Verify model prediction accuracy"

Write-Host "`n"
