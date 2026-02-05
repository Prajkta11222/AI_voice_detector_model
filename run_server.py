#!/usr/bin/env python3
"""
Standalone server runner - prevents CLI logic from running
"""
import os
import sys

# Set no arguments so app.py doesn't trigger CLI
sys.argv = ['app.py']

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
