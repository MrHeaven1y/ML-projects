import subprocess
import sys

process = subprocess.Popen(
    ["python", "./backend/server.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    encoding='utf-8'
)

while True:
    output = process.stdout.readline()
    error = process.stderr.readline()
    
    if not output and not error and process.poll() is not None:
        break
    
    if output:
        print(output.strip())
    if error:
        # Skip TensorFlow oneDNN/cpu_feature_guard messages
        if "oneDNN" not in error and "cpu_feature_guard" not in error:
            print("ERROR:", error.strip())

print("Server exited with code:", process.poll())