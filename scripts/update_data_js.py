import csv
import json
import re
import os

csv_path = '../data/measurements.csv'
js_path = '../web/data.js'

# Read measurements.csv
measurements = {}
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        corpus = row['corpus']
        model = row['model']
        if corpus not in measurements:
            measurements[corpus] = {}
        if model not in measurements[corpus]:
            measurements[corpus][model] = []
        measurements[corpus][model].append({
            'family': row['family'],
            'config': row['config'],
            'kl': float(row['kl']),
            'flip': float(row['flip'])
        })

# Check original kappas in data.js
with open(js_path, 'r') as f:
    content = f.read()

json_str = re.sub(r"^const GLOD_DATA = ", "", content).strip().rstrip(";")
data = json.loads(json_str)

kappas = data.get('kappas', {})

# We need to make sure kappas exist for all combinations. If they don't, we can try to compute them or see if they exist in numbers.json
# Actually, let's just write all measurements back.
data['measurements'] = measurements

new_content = "const GLOD_DATA = " + json.dumps(data, indent=2, separators=(',', ': ')) + ";\n"
with open(js_path, 'w') as f:
    f.write(new_content)

print(f"Updated data.js. Total configurations: {sum(len(v) for c in measurements.values() for v in c.values())}")
