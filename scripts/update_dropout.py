"""F1-loss with higher dropout for 2017."""
import sys
sys.path.insert(0, '.')

# Just change the dropout in the f1loss submission
import shutil
shutil.copy('submissions/coral-engineered-f1loss/main.py', 'submissions/coral-engineered-f1loss/main.py.bak')

with open('submissions/coral-engineered-f1loss/main.py', 'r') as f:
    code = f.read()

# Increase dropout from 0.3 to 0.5
code = code.replace('nn.Dropout(drop)', 'nn.Dropout(0.5)')
code = code.replace('CORALModel(HOLD_VECTOR_DIM, NUM_CLASSES, dropout=0.3)', 'CORALModel(HOLD_VECTOR_DIM, NUM_CLASSES, dropout=0.5)')

with open('submissions/coral-engineered-f1loss/main.py', 'w') as f:
    f.write(code)

print("Updated dropout to 0.5")
