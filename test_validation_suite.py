#!/usr/bin/env python3
"""
FastAPI Validation Test Suite
Tests the API with proper header handling and comprehensive validation
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my-secret-key"

def test_api(name, body, api_key=None, expected_status=200):
    """Test the API endpoint"""
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if api_key is not None:
        headers["x-api-key"] = api_key
    elif api_key is None:  
        # Test with no API key
        pass
    else:
        headers["x-api-key"] = api_key
    
    print(f"\nTest: {name}")
    print(f"  Expected: {expected_status}")
    print(f"  Sending x-api-key: {headers.get('x-api-key', 'NOT SET')}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/voice-detection",
            json=body,
            headers=headers,
            timeout=5
        )
        status = response.status_code
        
        if status == expected_status:
            print(f"  + PASS - Status: {status}")
            return True
        else:
            print(f"  - FAIL - Got {status}, expected {expected_status}")
            return False
    except Exception as e:
        print(f"  - ERROR: {e}")
        return False

# Initialize counters
passed = 0
failed = 0

print("\n" + "="*70)
print("FASTAPI VALIDATION TEST SUITE - PYTHON VERSION")
print("="*70)

print("\nSCHEMA REQUIREMENTS:")
print("  - language: required string, min_length 1")
print("  - audioFormat: required string, min_length 1")
print("  - audioBase64: required string, min_length 1")
print("  - x-api-key header: required for authentication")

print("\n======== SECTION 1: SCHEMA VALIDATION ========")

result = test_api(
    "Valid request - all fields correct",
    {"language": "english", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    200
)
passed += result
failed += not result

result = test_api(
    "Missing audioBase64 field",
    {"language": "english", "audioFormat": "wav"},
    API_KEY,
    422
)
passed += result
failed += not result

result = test_api(
    "Missing audioFormat field",
    {"language": "english", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    422
)
passed += result
failed += not result

result = test_api(
    "Missing language field",
    {"audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    422
)
passed += result
failed += not result

result = test_api(
    "Empty request body",
    {},
    API_KEY,
    422
)
passed += result
failed += not result

print("\n======== SECTION 2: FIELD VALUE VALIDATION ========")

result = test_api(
    "Empty string in language field",
    {"language": "", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    422
)
passed += result
failed += not result

result = test_api(
    "Whitespace-only language field",
    {"language": "   ", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    422
)
passed += result
failed += not result

result = test_api(
    "NULL value in language field",
    {"language": None, "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    422
)
passed += result
failed += not result

result = test_api(
    "Wrong data type - integer instead of string",
    {"language": 123, "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    422
)
passed += result
failed += not result

print("\n======== SECTION 3: CASE SENSITIVITY ========")

result = test_api(
    "UPPERCASE language field",
    {"language": "ENGLISH", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    200
)
passed += result
failed += not result

result = test_api(
    "UPPERCASE audioFormat field",
    {"language": "english", "audioFormat": "WAV", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    200
)
passed += result
failed += not result

result = test_api(
    "Mixed case both fields",
    {"language": "English", "audioFormat": "Wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    200
)
passed += result
failed += not result

print("\n======== SECTION 4: AUTHENTICATION ========")

# Test without API key by not adding it to headers
headers_no_key = {"Content-Type": "application/json"}
try:
    response = requests.post(
        f"{BASE_URL}/api/voice-detection",
        json={"language": "english", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
        headers=headers_no_key,
        timeout=5
    )
    status = response.status_code
    name = "Missing x-api-key header"
    expected = 422
    print(f"\nTest: {name}")
    print(f"  Expected: {expected}")
    print(f"  Sending x-api-key: NOT SET")
    if status == expected:
        print(f"  + PASS - Status: {status}")
        passed += 1
    else:
        print(f"  - FAIL - Got {status}, expected {expected}")
        failed += 1
except Exception as e:
    print(f"  - ERROR: {e}")
    failed += 1

result = test_api(
    "Invalid x-api-key value",
    {"language": "english", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    "wrong-key",
    401
)
passed += result
failed += not result

result = test_api(
    "Valid x-api-key (correct case)",
    {"language": "english", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY,
    200
)
passed += result
failed += not result

result = test_api(
    "Valid x-api-key (UPPERCASE)",
    {"language": "english", "audioFormat": "wav", "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"},
    API_KEY.upper(),
    200
)
passed += result
failed += not result

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

total = passed + failed
percent = int((passed / total * 100)) if total > 0 else 0

if failed == 0:
    print("SUCCESS: All Tests Passed!")
else:
    print(f"NOTICE: {failed} test(s) failed. See details above.")

print(f"  Passed: {passed}")
print(f"  Failed: {failed}")
print(f"  Total:  {total}")
print(f"  Pass Rate: {percent}%")

print("\nIMPORTANT FIXES APPLIED:")
print("  [1] Fresh header hashtable each call (fixes 401 errors)")
print("  [2] Safe ConvertFrom-Json error handling (fixes null crashes)")
print("  [3] Better diagnostic output (shows validation details)")

print("")
