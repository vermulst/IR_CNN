import numpy as np
from jcamp import jcamp_readfile

#500 - 400cm = fingerprint region

class SpectraSample():
    __slots__ = ('x', 'y', 'skip', 'path', 'labels')
    def __init__(self, path, params=False):
        # Value that keeps track of whether there was an error with reading the sample so it can be skipped in later steps
        self.skip = True 

        try:
            data = jcamp_readfile(path)
        except Exception as e:
            return

        # Assign data type
        data_type = (data.get("datatype") or data.get("data type", "")).upper()
        
        # Extra logic for LINK type
        if data_type == "LINK":
            children = data.get("children", [])
            data = next((c for c in children if c.get("data type", "").upper() == "INFRARED SPECTRUM"), None)
            if data is None:
                return
        elif data_type != "INFRARED SPECTRUM":
            return
        
        self._extract(data)

    def _extract(self, data):
        """Extract spectrum data from INFRARED SPECTRUM format"""
        x = data.get("x")
        y = data.get("y")

        if x is None or y is None:
            print(f"\nEmpty data in {self.path}")
            return
        if len(x) == 0 or len(y) == 0:
            print(f"\nEmpty array in {self.path}")
            return
        if len(x) != len(y):
            print(f"\nLength mismatch ({len(x)} vs {len(y)}) in {self.path}")
            return

        self.x = np.asarray(x, dtype=np.float64) * data.get("xfactor", 1.0)
        self.y = np.asarray(y, dtype=np.float64) * data.get("yfactor", 1.0)
        self.skip = False

        