#!/usr/bin/env python
"""
Test script to verify FastAPI validation and authentication
"""
import requests
import json
import base64

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "your-secret-api-key"

print("\n" + "="*70)
print("Testing FastAPI Validation and Authentication")
print("="*70)

# Test 1: Missing required fields
print("\n[Test 1] Missing all required fields:")
try:
    response = requests.post(
        f"{BASE_URL}/api/voice-detection",
        json={},
        headers={"x-api-key": API_KEY}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 2: Missing audioBase64
print("\n[Test 2] Missing audioBase64 field:")
try:
    response = requests.post(
        f"{BASE_URL}/api/voice-detection",
        json={
            "language": "english",
            "audioFormat": "wav"
        },
        headers={"x-api-key": API_KEY}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 3: Invalid API Key (should get 401)
print("\n[Test 3] Invalid API Key (case-insensitive test):")
try:
    response = requests.post(
        f"{BASE_URL}/api/voice-detection",
        json={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": "dGVzdA=="  # "test" in base64
        },
        headers={"x-api-key": "wrong-key"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 4: Valid API Key with complete request (will fail on model validation but auth passes)
print("\n[Test 4] Valid API Key with complete request (correct case):")
try:
    response = requests.post(
        f"{BASE_URL}/api/voice-detection",
        json={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": "dGVzdA=="  # "test" in base64
        },
        headers={"x-api-key": API_KEY}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 5: Valid API Key with uppercase (case-insensitive)
print("\n[Test 5] Valid API Key in uppercase (case-insensitive):")
try:
    response = requests.post(
        f"{BASE_URL}/api/voice-detection",
        json={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": "dGVzdA=="  # "test" in base64
        },
        headers={"x-api-key": API_KEY.upper()}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

# Test 6: Health check (no auth required)
print("\n[Test 6] Health Check (no auth required):")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*70)
print("Testing complete. Check Swagger at: http://127.0.0.1:8000/docs")
print("="*70 + "\n")
