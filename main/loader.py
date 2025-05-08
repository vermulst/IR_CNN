import orjson
from SpectraSample import SpectraSample

# load a sample from a file name (JCAMP file)
def load_sample(datafolder, identifier):
    path = f"{datafolder}/samples/{identifier}"
    try:
        return SpectraSample(path)
    except Exception as e:
        print(f"Error in {identifier}: {e}")
        return None
    
def load_samples(datafolder):
    # Load the JSON metadata
    with open(f"{datafolder}/meta_data.json", "rb") as f:
        compounds = orjson.loads(f.read())
    
    # fetch a list of files
    filenames = [
        attachment["identifier"].split("/", 1)[1]
        for compound in compounds
        for dataset in compound.get("datasets", [])
        for attachment in dataset.get("attacments", [])
    ]

    samples = [s for s in (load_sample(datafolder, fid) for fid in filenames) if s is not None]
    return samples