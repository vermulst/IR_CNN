from jcamp import jcamp_readfile

#1500 - 400cm = fingerprint region

class SpectraSample():
    def __init__(self, path, params=False):
        self.data = jcamp_readfile(path)

        self.dx = self.data.get("jcamp-dx")
        self.first_x = self.data.get("firstx")
        self.last_x = self.data.get("lastx")
        self.delta_x = self.data.get("delta") or self.data.get("deltax")
        self.max_y = self.data.get("maxy")
        self.min_y = self.data.get("miny")
        self.x_factor = self.data.get("xfactor")
        self.y_factor = self.data.get("yfactor")
        self.npoints = self.data.get("npoints")
    #    self.first_y = self.data.get("firsty")
    #    self.xy_data = self.data.get("xydata")
        self.x = self.data.get("x")
        self.y = self.data.get("y")
        self.labels = None

        # Optional metadata (uncomment if needed)
        # self.title = self.data.get("title")
        # self.data_type = self.data.get("data type")
        # self.date = self.data.get("date")
        # self.procedure = self.data.get("sampling procedure")
        # self.origin = self.data.get("origin")
        # self.x_units = self.data.get("xunits")
        # self.y_units = self.data.get("yunits")
        # self.resolution = self.data.get("resolution")