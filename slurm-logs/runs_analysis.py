import os
import re

runs = [run for run in os.listdir() if run.endswith("out")]

reruns = {}
for run in runs:
    with open(run, "r") as f:
        logs = f.read()
        pattern = r"Exit code:  (\d+)"
        try:
            code = int(re.search(pattern, logs).group(1))
        except:
            code = 1

        prefix, _ = run.split("-")
        # always count success if possible
        if code == 0 or prefix not in reruns:
            reruns[prefix] = code


print("Failure, rerun:")
failures = 0
for prefix, code in reruns.items():
    if code:
        print(f"sbatch {prefix}.sh")
        failures += 1
print(f"{failures} failed runs")
