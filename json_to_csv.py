import pandas as pd
import json

#drop this file in the /DFDC/test directory and run it to create the labels.csv file
# Load the original DFDC metadata
with open('metadata.json') as f:
    data = json.load(f)

# Convert to CSV format expected by the bench
df = pd.DataFrame.from_dict(data, orient='index')
df.to_csv('labels.csv')