#!/usr/bin/env python3
"""
Convert audio file to base64 for API testing
Usage: python convert_audio_to_base64.py <audio_file_path>
"""
import base64
import sys

def audio_to_base64(file_path):
    """Convert audio file to base64 string"""
    try:
        with open(file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            base64_encoded = base64.b64encode(audio_data).decode('utf-8')
            return base64_encoded
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_audio_to_base64.py <audio_file_path>")
        print("\nExample:")
        print("python convert_audio_to_base64.py dataset/real/English/human_0000.mp3")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    print(f"Converting: {audio_file}")
    
    base64_string = audio_to_base64(audio_file)
    
    if base64_string:
        print(f"\n✓ Conversion successful!")
        print(f"Length: {len(base64_string)} characters")
        print(f"\nFirst 100 characters:")
        print(base64_string[:100])
        print("\n...")
        print(f"\nFull base64 string saved to: audio_base64.txt")
        
        # Save to file
        with open("audio_base64.txt", "w") as f:
            f.write(base64_string)
        
        print("\n✓ You can copy the base64 string from audio_base64.txt")
        print("  and paste it into the Swagger UI!")
