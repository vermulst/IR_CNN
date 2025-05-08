import sys
#print(sys.executable)
from SpectraSample import SpectraSample
import preprocessing as pp
import json


# Load the JSON metadata
with open("data/meta_data.json", "r", encoding="utf-8") as f:
    compounds = json.load(f)

# fetch a list of files
filenames = [
    attachment["identifier"].split("/", 1)[1]
    for compound in compounds
    for dataset in compound.get("datasets", [])
    for attachment in dataset.get("attacments", [])
]


def load_sample(identifier):
    path = f"data/samples/{identifier}"
    try:
        return SpectraSample(path)
    except Exception as e:
        print(f"Error in {identifier}: {e}")
        return None

# go through every file
samples = [s for s in (load_sample(fid) for fid in filenames) if s is not None]
print(f"Loaded {len(samples)} valid samples.")
