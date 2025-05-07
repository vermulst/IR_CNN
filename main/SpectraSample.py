from jcamp import jcamp_readfile

#1500 - 400cm = fingerprint region

class SpectraSample():
    def __init__(self, path, params=False):
        self.data = jcamp_readfile(path)
        #commented out probably don't matter

        #self.title = self.data["title"]
        self.dx = self.data["jcamp-dx"]
        #self.data_type = self.data["data type"]
        #self.date = self.data["date"]
        #self.procedure = self.data["sampling procedure"]
        #self.origin = self.data["origin"]
        #self.x_units = self.data["xunits"]
        #self.y_units = self.data["yunits"]
        #self.resolution = self.data["resolution"]
        self.first_x = self.data["firstx"]
        self.last_x = self.data["lastx"]
        self.delta_x = self.data["deltax"]
        self.max_y = self.data["maxy"]
        self.min_y = self.data["miny"]
        self.x_factor = self.data["xfactor"]
        self.y_factor = self.data["yfactor"]
        self.npoints = self.data["npoints"]
        self.first_y = self.data["firsty"]
        self.xy_data = self.data["xydata"]

        # array of x values (wavenumbers)
        self.x = self.data["x"] 
        # array of corresponding y values ()
        self.y = self.data["y"]