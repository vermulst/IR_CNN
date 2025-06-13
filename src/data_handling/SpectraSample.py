import numpy as np
from dataclasses import dataclass, field
from jcamp import jcamp_readfile
from typing import Optional

# Data class to represent a spectrum sample with metadata and spectral data
@dataclass(slots=True)
class SpectraSample:
    x: Optional[np.ndarray] = field(default=None, init=False)
    y: Optional[np.ndarray] = field(default=None, init=False)
    skip: bool = field(default=True, init=False)
    path: str = field(default="", init=False)
    labels: Optional[list] = field(default=None, init=False)
    weight: float = field(default=1.0, init=False) 

    @classmethod
    def from_file(cls, path: str) -> 'SpectraSample':
        sample = cls()              # Create an empty SpectraSample
        sample.path = path          # Store file path

        try:
            data = jcamp_readfile(path)  # Attempt to read the JCAMP-DX file
        except Exception as e:
            print(f"[ERROR] Could not read file '{path}': {e}")
            return sample  # Return with skip=True if reading fails

        # Determine spectrum type (handle case-insensitive field names)
        data_type = (data.get("datatype") or data.get("data type", "")).upper()

        # Handle LINK-type JCAMP files with multiple children spectra
        if data_type == "LINK":
            children = data.get("children", [])
            data = next((c for c in children if c.get("data type", "").upper() == "INFRARED SPECTRUM"), None)
            if data is None:
                print(f"[WARN] No IR spectrum in LINK at '{path}'")
                return sample  # Still skip=True
        elif data_type != "INFRARED SPECTRUM":
            print(f"[WARN] Unsupported spectrum type '{data_type}' in '{path}'")
            return sample

        # Attempt to extract x/y data from the file
        if sample._extract(data):
            sample.skip = False  # Set skip to False if extraction succeeds

        return sample

    def _extract(self, data: dict) -> bool:
        # Extract raw x and y values
        x = data.get("x")
        y = data.get("y")

        # Check for presence and non-emptiness of data arrays
        if x is None or y is None:
            print(f"[WARN] Missing x or y data in '{self.path}'")
            return False
        if len(x) == 0 or len(y) == 0:
        #   print(f"[WARN] Empty arrays in '{self.path}'")
            return False
        if len(x) != len(y):
        #   print(f"[WARN] Length mismatch ({len(x)} vs {len(y)}) in '{self.path}'")
            return False

        # Apply scaling factors and store as numpy arrays
        self.x = np.asarray(x, dtype=np.float64) * data.get("xfactor", 1.0)
        self.y = np.asarray(y, dtype=np.float64) * data.get("yfactor", 1.0)
        return True
