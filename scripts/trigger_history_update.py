import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src import history

print("Updating history actuals...")
history.update_actuals()
print("Done.")
