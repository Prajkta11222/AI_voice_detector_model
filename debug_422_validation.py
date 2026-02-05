#!/usr/bin/env python3
"""
COMPREHENSIVE FASTAPI 422 DEBUGGING GUIDE
Voice Detection API - Schema Validation & Request Testing

This script helps you systematically debug 422 Unprocessable Entity errors
by testing various request combinations and showing exact validation errors.
"""

import requests
import json
import base64
from typing import Dict, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_URL = "http://127.0.0.1:8000"
API_KEY = "your-secret-api-key"

# Sample valid base64 audio (minimal test data)
VALID_BASE64_SAMPLE = "//NExAAiYAIAJQAqACoAoAoAKgDgCgCgB9AAAB"

# ============================================================================
# SCHEMA DOCUMENTATION
# ============================================================================

SCHEMA_INFO = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    REQUEST SCHEMA SPECIFICATION                          ║
╚══════════════════════════════════════════════════════════════════════════╝

ENDPOINT:        POST /api/voice-detection
CONTENT-TYPE:    application/json
AUTHENTICATION:  x-api-key: <your-api-key> (case-insensitive)

REQUIRED FIELDS:
┌─────────────────────────────────────────────────────────────────────────┐
│ Field Name:   language                                                  │
├─────────────────────────────────────────────────────────────────────────┤
│ Type:         string                                                    │
│ Required:     YES (Pydantic: ...)                                      │
│ Min Length:   1 (empty strings rejected)                               │
│ Valid Values: 'english', 'hindi', 'malayalam', 'telugu', 'tamil'      │
│ Processing:   Automatically converted to lowercase                     │
│ Example:      "english" or "ENGLISH" or "English"                      │
│ Error if:     Missing, empty, null, or whitespace-only                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Field Name:   audioFormat                                               │
├─────────────────────────────────────────────────────────────────────────┤
│ Type:         string                                                    │
│ Required:     YES (Pydantic: ...)                                      │
│ Min Length:   1 (empty strings rejected)                               │
│ Valid Values: 'wav', 'mp3', 'flac', 'ogg'                              │
│ Processing:   Automatically converted to lowercase                     │
│ Example:      "wav" or "WAV" or "Wav"                                  │
│ Error if:     Missing, empty, null, or whitespace-only                │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Field Name:   audioBase64                                               │
├─────────────────────────────────────────────────────────────────────────┤
│ Type:         string                                                    │
│ Required:     YES (Pydantic: ...)                                      │
│ Min Length:   1 (empty strings rejected)                               │
│ Format:       Valid base64 encoding                                    │
│ Processing:   Decoded at endpoint, not at validation level            │
│ Example:      "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB" (shortened)      │
│ Error if:     Missing, empty, null, or whitespace-only                │
│ Note:         Invalid base64 causes error AFTER 422 validation        │
└─────────────────────────────────────────────────────────────────────────┘

OPTIONAL FIELDS:  None - all fields are required

PYDANTIC VALIDATION RULES:
  1. ... (ellipsis) = field is required
  2. min_length=1 = cannot be empty string
  3. @validator methods check for whitespace-only strings
  4. Fields are stripped of whitespace before use
"""

# ============================================================================
# COMMON 422 ERROR SCENARIOS
# ============================================================================

COMMON_ERRORS = """
╔══════════════════════════════════════════════════════════════════════════╗
║                    COMMON 422 ERROR CAUSES                               ║
╚══════════════════════════════════════════════════════════════════════════╝

❌ MISSING ENTIRE REQUEST BODY
   Request:  POST /api/voice-detection
   Body:     (empty or missing)
   Error:    422 - No JSON object could be decoded
   Fix:      Send JSON body with all 3 required fields

❌ MISSING REQUIRED FIELD
   Request:  {"language": "english", "audioFormat": "wav"}
   Missing:  audioBase64
   Error:    422 - field required (audioBase64)
   Fix:      Add "audioBase64" field with valid base64 string

❌ EMPTY STRING VALUE
   Request:  {"language": "", "audioFormat": "wav", "audioBase64": "..."}
   Error:    422 - ensure this value has at least 1 characters
   Fix:      Provide non-empty string: "english" instead of ""

❌ NULL/NULL VALUE
   Request:  {"language": null, "audioFormat": "wav", "audioBase64": "..."}
   Error:    422 - str type expected (type=type_error.str)
   Fix:      Use string value, not null: "english" instead of null

❌ WHITESPACE-ONLY STRING
   Request:  {"language": "   ", "audioFormat": "wav", "audioBase64": "..."}
   Error:    422 - value_error.text validation error (custom validator)
   Fix:      Use actual value: "english" instead of "   "

❌ WRONG DATA TYPE
   Request:  {"language": 123, "audioFormat": "wav", "audioBase64": "..."}
   Error:    422 - str type expected (type=type_error.str)
   Fix:      Use string: "english" instead of 123

❌ EXTRA UNKNOWN FIELDS
   Request:  {"language": "english", "audioFormat": "wav", 
              "audioBase64": "...", "extraField": "value"}
   Result:   May be silently ignored (depending on model config)
   Note:     Not a 422 error with default Pydantic config

❌ INVALID HEADER FORMAT
   Header:   missing x-api-key header entirely
   Error:    422 - "x-api-key" header not found
   Fix:      Add header: -H "x-api-key: your-secret-api-key"

❌ MISSING CONTENT-TYPE HEADER
   Request:  Missing Content-Type header
   Error:    Depends on client library
   Fix:      Set header: -H "Content-Type: application/json"
"""

# ============================================================================
# TEST CASES
# ============================================================================

class RequestValidator:
    """Test request combinations and show validation errors"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        self.results = []
    
    def test(self, name: str, json_body: Dict[str, Any] = None, 
             headers: Dict[str, str] = None, show_response: bool = True) -> None:
        """
        Send test request and display results
        
        Args:
            name: Test name for display
            json_body: Request JSON body (None = empty body)
            headers: Custom headers (x-api-key added if not present)
            show_response: Print full response if True
        """
        # Prepare headers
        if headers is None:
            headers = {}
        
        if "x-api-key" not in headers:
            headers["x-api-key"] = self.api_key
        
        # Color codes
        BOLD = "\033[1m"
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        RESET = "\033[0m"
        
        print(f"\n{BOLD}{'='*75}{RESET}")
        print(f"{BOLD}Test: {name}{RESET}")
        print(f"{BOLD}{'='*75}{RESET}")
        
        # Display request details
        print(f"\n{BLUE}Request Details:{RESET}")
        print(f"  Method:      POST /api/voice-detection")
        print(f"  Headers:     {json.dumps(headers, indent=16)}")
        print(f"  Body:        {json.dumps(json_body, indent=16) if json_body else '(empty)'}")
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/voice-detection",
                json=json_body,
                headers=headers,
                timeout=5
            )
            
            status_code = response.status_code
            response_json = response.json()
            
            # Color code status
            if status_code == 200:
                status_color = GREEN
                status_label = "✓ SUCCESS"
            elif status_code == 401:
                status_color = RED
                status_label = "✗ UNAUTHORIZED"
            elif status_code == 422:
                status_color = YELLOW
                status_label = "✗ VALIDATION ERROR"
            else:
                status_color = YELLOW
                status_label = f"? STATUS {status_code}"
            
            print(f"\n{status_color}{BOLD}Response: {status_label}{RESET}")
            print(f"  Status Code: {status_code}")
            
            if show_response:
                print(f"  Body:\n{json.dumps(response_json, indent=4)}")
            
            # Extract error details
            if status_code == 422:
                if isinstance(response_json, list):
                    print(f"\n{RED}Validation Errors:{RESET}")
                    for error in response_json:
                        print(f"  • Field: {error.get('loc', ['unknown'])[-1]}")
                        print(f"    Type:  {error.get('type', 'unknown')}")
                        print(f"    Msg:   {error.get('msg', 'no message')}")
                elif isinstance(response_json, dict) and "detail" in response_json:
                    if isinstance(response_json["detail"], list):
                        print(f"\n{RED}Validation Errors:{RESET}")
                        for error in response_json["detail"]:
                            field = error.get('loc', ['unknown'])[-1]
                            msg = error.get('msg', 'unknown error')
                            print(f"  • {field}: {msg}")
            
            self.results.append({
                'test': name,
                'status': status_code,
                'success': status_code == 200
            })
        
        except requests.exceptions.ConnectionError:
            print(f"\n{RED}✗ CONNECTION ERROR{RESET}")
            print("  Server not running or unreachable at", self.base_url)
        except requests.exceptions.Timeout:
            print(f"\n{RED}✗ TIMEOUT ERROR{RESET}")
            print("  Request exceeded 5 second timeout")
        except Exception as e:
            print(f"\n{RED}✗ ERROR{RESET}")
            print(f"  {type(e).__name__}: {str(e)}")
    
    def print_summary(self) -> None:
        """Print test results summary"""
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"
        BOLD = "\033[1m"
        
        print(f"\n\n{BOLD}{'='*75}{RESET}")
        print(f"{BOLD}TEST SUMMARY{RESET}")
        print(f"{BOLD}{'='*75}{RESET}\n")
        
        passed = sum(1 for r in self.results if r['success'])
        total = len(self.results)
        
        for result in self.results:
            status_icon = f"{GREEN}✓{RESET}" if result['success'] else f"{RED}✗{RESET}"
            print(f"  {status_icon} {result['test']:<50} [{result['status']}]")
        
        print(f"\n{BOLD}Total: {passed}/{total} passed{RESET}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run comprehensive validation tests"""
    
    print(SCHEMA_INFO)
    print(COMMON_ERRORS)
    
    validator = RequestValidator(BASE_URL, API_KEY)
    
    # ========================================================================
    # TEST SUITE 1: SCHEMA VALIDATION
    # ========================================================================
    
    print("\n\n" + "="*75)
    print("SECTION 1: SCHEMA VALIDATION TESTS")
    print("="*75)
    
    # Valid request
    validator.test(
        "✓ Valid Request - All Required Fields",
        json_body={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    # Missing fields
    validator.test(
        "✗ Missing audioBase64 Field",
        json_body={
            "language": "english",
            "audioFormat": "wav"
        }
    )
    
    validator.test(
        "✗ Missing audioFormat Field",
        json_body={
            "language": "english",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    validator.test(
        "✗ Missing language Field",
        json_body={
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    validator.test(
        "✗ Empty Request Body",
        json_body={}
    )
    
    # ========================================================================
    # TEST SUITE 2: FIELD VALUE VALIDATION
    # ========================================================================
    
    print("\n\n" + "="*75)
    print("SECTION 2: FIELD VALUE VALIDATION TESTS")
    print("="*75)
    
    # Empty strings
    validator.test(
        "✗ Empty String - language",
        json_body={
            "language": "",
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    validator.test(
        "✗ Empty String - audioFormat",
        json_body={
            "language": "english",
            "audioFormat": "",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    validator.test(
        "✗ Empty String - audioBase64",
        json_body={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": ""
        }
    )
    
    # Whitespace only
    validator.test(
        "✗ Whitespace Only - language",
        json_body={
            "language": "   ",
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    # NULL values
    validator.test(
        "✗ NULL Value - language",
        json_body={
            "language": None,
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    # Wrong data types
    validator.test(
        "✗ Wrong Type - language (integer)",
        json_body={
            "language": 123,
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    validator.test(
        "✗ Wrong Type - audioBase64 (array)",
        json_body={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": ["not", "a", "string"]
        }
    )
    
    # ========================================================================
    # TEST SUITE 3: CASE SENSITIVITY
    # ========================================================================
    
    print("\n\n" + "="*75)
    print("SECTION 3: CASE SENSITIVITY TESTS")
    print("="*75)
    
    validator.test(
        "✓ Case Insensitive - UPPERCASE language",
        json_body={
            "language": "ENGLISH",
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    validator.test(
        "✓ Case Insensitive - MixedCase audioFormat",
        json_body={
            "language": "english",
            "audioFormat": "WAV",
            "audioBase64": VALID_BASE64_SAMPLE
        }
    )
    
    # ========================================================================
    # TEST SUITE 4: AUTHENTICATION
    # ========================================================================
    
    print("\n\n" + "="*75)
    print("SECTION 4: AUTHENTICATION TESTS")
    print("="*75)
    
    validator.test(
        "✗ Missing x-api-key Header",
        json_body={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        },
        headers={}  # No x-api-key
    )
    
    validator.test(
        "✗ Invalid x-api-key Value",
        json_body={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        },
        headers={"x-api-key": "wrong-key"}
    )
    
    validator.test(
        "✓ Valid x-api-key (case-insensitive)",
        json_body={
            "language": "english",
            "audioFormat": "wav",
            "audioBase64": VALID_BASE64_SAMPLE
        },
        headers={"x-api-key": API_KEY.upper()}
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    validator.print_summary()
    
    print("\n" + "="*75)
    print("DEBUGGING TIPS")
    print("="*75)
    print("""
1. SWAGGER TESTING:
   • Visit: http://127.0.0.1:8000/docs
   • Click "Try it out" on the /api/voice-detection endpoint
   • Fill in all required fields
   • Add x-api-key header with value: your-secret-api-key
   • Click "Execute" to see exact error details

2. CURL TESTING:
   curl -X POST "http://127.0.0.1:8000/api/voice-detection" \
     -H "Content-Type: application/json" \
     -H "x-api-key: your-secret-api-key" \
     -d '{
       "language": "english",
       "audioFormat": "wav",
       "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
     }'

3. PYTHON REQUESTS:
   import requests
   response = requests.post(
     "http://127.0.0.1:8000/api/voice-detection",
     json={
       "language": "english",
       "audioFormat": "wav",
       "audioBase64": "//NExAAiYAIAJQAqACoAoAoAKgDgCgB9AAAB"
     },
     headers={"x-api-key": "your-secret-api-key"}
   )
   print(response.json())

4. ENABLING VERBOSE SERVER LOGGING:
   python -m uvicorn app:app --reload --log-level debug

5. CHECKING REQUEST IN FASTAPI:
   Add to your endpoint:
   print(f"Received: {request}")
   print(f"Raw body: {await request.body()}")
""")
    
    print("\n" + "="*75 + "\n")


if __name__ == "__main__":
    main()
