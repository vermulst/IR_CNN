from SpectraSample import SpectraSample
import preprocessing as pp
import json

# fetch a list of files
# open the meta_data.json file
filenames = []
with open("data/meta_data.json", "r", encoding="utf-8") as f:
    compounds = json.load(f)
    for compound in compounds:
        for dataset in compound.get("datasets", []):
            for attachment in dataset.get("attacments", []):
                filename = attachment["identifier"]
                filenames.append(filename[2:]) # scrap the 8/


samples = [SpectraSample(f"data/samples/{identifier}") for identifier in filenames]
print(samples.len)
