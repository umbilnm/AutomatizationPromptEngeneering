from collections import OrderedDict
import json
import numpy as np
with open(f'./artifacts/json_mappings/spam_map.json') as f:
    mapping = json.load(f)
a = np.array(['ham', 'spam', 'ham'])
a  = np.vectorize(mapping.get)(a)
print(a)