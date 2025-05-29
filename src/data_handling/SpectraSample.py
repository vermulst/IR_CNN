import numpy as np
from jcamp import jcamp_readfile

#1500 - 400cm = fingerprint region

class SpectraSample():
    def __init__(self, path, params=False):

        # Value that keeps track of whether there was an error with reading the sample so it can be skipped in later steps
        self.skip = False 

        try:
            self.data = jcamp_readfile(path)
        except Exception as e:
            print(f"Error reading JCAMP file {path}: {e}")
            self.skip = True
            return

        # Assign data type
        self.data_type = self.data.get("data type", "").upper()
        if self.data.get("datatype"):
            self.data_type = self.data.get("datatype", "").upper()
        
        # Exctract data based on data type
        if self.data_type == "LINK":
            # Extract spectrum data from children
            spectrum_data = next((child for child in self.data.get("children", []) 
                                if child.get("data type", "").upper() == "INFRARED SPECTRUM"), None)
            if spectrum_data:
                self._extract_spectrum_data(spectrum_data, path)
            else:
                print(f"No INFRARED SPECTRUM block found in LINK file {path}")
                self.skip = True
                return
        elif self.data_type == "INFRARED SPECTRUM":
            self._extract_spectrum_data(self.data, path)
        else:
            print(f"Unsupported data type '{self.data_type}' in {path}")
            self.skip = True
            return

    def _extract_spectrum_data(self, spectrum_data, path):
        """Extract spectrum data from INFRARED SPECTRUM format"""
        self.dx = spectrum_data.get("jcamp-dx")
        self.first_x = spectrum_data.get("firstx")
        self.last_x = spectrum_data.get("lastx")
        self.delta_x = spectrum_data.get("delta") or spectrum_data.get("deltax")
        self.max_y = spectrum_data.get("maxy")
        self.min_y = spectrum_data.get("miny")
        self.x_factor = spectrum_data.get("xfactor", 1.0)
        self.y_factor = spectrum_data.get("yfactor", 1.0)
        self.npoints = spectrum_data.get("npoints")
        # self.first_y = spectrum_data.get("firsty")
        # self.xy_data = spectrum_data.get("xydata")
        self.labels = None

        # Try to extract x and y
        x = spectrum_data.get("x")
        y = spectrum_data.get("y")
        
        if x is not None and len(x != 0) and y is not None and len(y == 0):
            try:
                self.x = np.array(x, dtype=np.float64)
                self.y = np.array(y, dtype=np.float64)
            except (ValueError, TypeError) as e:
                print(f"\nError converting x or y to numerical arrays: {e}")
                self.skip = True

        elif x is None or y is None or len(x) == 0 or len(y) == 0:
            print(f"\nEmpty initial x or y data and manual parsing not possible for {path}")
            self.skip = True

        elif len(x) != len(y):
            print(f"Length mismatch: ({len(x)} vs {len(y)}) in {path}")
            self.skip = True