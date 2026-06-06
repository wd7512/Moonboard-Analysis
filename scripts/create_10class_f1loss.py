"""F1-loss with 10 classes (removing empty 6A, 6A+, 6B)."""
import sys
sys.path.insert(0, '.')

# Copy the f1loss submission and modify GRADE_ORDER
import shutil
import os

os.makedirs('submissions/coral-engineered-f1loss-10class', exist_ok=True)

with open('submissions/coral-engineered-f1loss/main.py', 'r') as f:
    code = f.read()

# Override GRADE_ORDER at the top
old = 'from moonboard_analysis.config import GRADE_ORDER'
new = '''# OVERRIDE: Only 10 classes have data in 2016
GRADE_ORDER = ["6B+", "6C", "6C+", "7A", "7A+", "7B", "7B+", "7C", "7C+", "8A"]
# Import other needed items
import sys
sys.path.insert(0, '.')
from moonboard_analysis.config import GRADE_ORDER as _GO
# Verify our override is used
assert len(GRADE_ORDER) == 10, f"GRADE_ORDER has {len(GRADE_ORDER)} classes"'''

code = code.replace(old, new, 1)

with open('submissions/coral-engineered-f1loss-10class/main.py', 'w') as f:
    f.write(code)

print("Created 10-class F1-loss submission")
