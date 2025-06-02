import numpy as np
from jcamp import jcamp_readfile

#500 - 400cm = fingerprint region

class SpectraSample():
    def __init__(self, path, params=False):

        # Value that keeps track of whether there was an error with reading the sample so it can be skipped in later steps
        self.skip = True 

        try:
            self.data = jcamp_readfile(path)
        except Exception as e:
            return

        # Assign data type
        self.data_type = (self.data.get("datatype") or self.data.get("data type", "")).upper()
        
        # Extra logic for LINK type
        if self.data_type == "LINK":
            children = self.data.get("children", [])
            self.data = next((c for c in children if c.get("data type", "").upper() == "INFRARED SPECTRUM"), None)
            if self.data is None:
                return
        elif self.data_type != "INFRARED SPECTRUM":
            return
        
        fail_reason, xlen, ylen = self._extract()
        if (self.skip):
            if (fail_reason == "Empty"):
                print(f"\nEmpty initial x or y data and manual parsing not possible for {path}")
            elif (fail_reason == "Mismatch"):
                print(f"\nLength mismatch: ({xlen} vs {ylen}) in {path}")

    def _extract(self):
        """Extract spectrum data from INFRARED SPECTRUM format"""
        x = self.data.get("x")
        y = self.data.get("y")

        if x is None or y is None or len(x) == 0 or len(y) == 0:
            return "Empty", 0, 0
        if (len(x) != len(y)):
            return "Mismatch", len(x), len(y)

        self.x = np.array(x, dtype=np.float64) * self.data.get("xfactor", 1.0)
        self.y = np.array(y, dtype=np.float64) * self.data.get("yfactor", 1.0)
        
        self.skip = False
        return None, len(x), len(y)

        