"""
Test importing local modules
"""

import os
import sys

print("Testing local imports...")

# Print working directory and sys.path
print(f"Current working directory: {os.getcwd()}")
print(f"sys.path: {sys.path}")

try:
    print("\nImporting local modules...")
    import src
    from src import data_processing
    print("Local module imports successful")
except Exception as e:
    print(f"Local module import error: {e}")
