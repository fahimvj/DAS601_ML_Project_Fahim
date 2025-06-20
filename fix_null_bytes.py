"""
Small utility to check and fix null bytes in Python files
"""
import os

def check_and_fix_null_bytes(directory):
    """Check for null bytes in Python files and remove them"""
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'rb') as f:
                        content = f.read()
                    
                    if b'\x00' in content:
                        print(f"Found null bytes in {filepath}")
                        # Remove null bytes
                        content = content.replace(b'\x00', b'')
                        with open(filepath, 'wb') as f:
                            f.write(content)
                        print(f"Fixed null bytes in {filepath}")
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    check_and_fix_null_bytes('src')
    print("Completed checking src directory")
    check_and_fix_null_bytes('.')
    print("Completed checking root directory")
