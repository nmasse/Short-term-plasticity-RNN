import subprocess
import time

while True:
    subprocess.call('nvidia-smi')
    time.sleep(1)
    subprocess.call('clear')
