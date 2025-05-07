from SpectraSample import SpectraSample
import preprocessing as pp
import json


# Load the JSON metadata
with open("data/meta_data.json", "r", encoding="utf-8") as f:
    compounds = json.load(f)

# fetch a list of files
filenames = []
for compound in compounds:
    for dataset in compound.get("datasets", []):
        for attachment in dataset.get("attacments", []):
            # split at the first slash and pick the right part
            filename = attachment["identifier"].split("/", 1)[1] 
            filenames.append(filename)


samples = [SpectraSample(f"data/samples/{identifier}") for identifier in filenames]
print(len(samples))
