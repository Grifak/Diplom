from Patient_Abnorm import Patient_Abnorm
import numpy as np


class Abnorm(Patient_Abnorm):
    def __init__(self, id, pat, eye, scan_dim, centered, grade, scan,
                 mask, center_of_mass, coords, diameter):
        super(Abnorm, self).__init__(pat=pat, eye=eye, scan_dim=scan_dim, centered=centered)
        self.id = id
        self.grade = grade
        self.scan = scan
        self.mask = mask
        self.scan_dim = scan_dim
        self.center_of_mass = center_of_mass  # BHA coords
        self.coords = coords
        self.diameter = diameter  # BHA coords
        self.volume = np.sum(self.mask)

    def __repr__(self):
        return str(self)

    def get_full_mask(self):
        m = np.zeros(self.scan_dim)
        m[self.coords] = self.mask
        return m

    def get_mask_projection(self, projection, mask='full'):
        mask3d = self.get_full_mask() if mask == 'full' else self.mask
        return self.OCT.get_2D_mask(section=projection, mask=mask3d)

    def get_area(self, projection, mask=None):
        return np.sum(self.get_mask_projection(projection, mask=mask))

    def to_dict(self, norm_vol=False):
        vol = self.volume
        if norm_vol:
            vol /= np.product(self.scan_dim)
        d = {'id': self.id, 'grade': self.grade, 'volume': vol}
        for name, ax in self.slice_axis.items():
            d['cof_{}'.format(name)] = self.center_of_mass[ax]
            d['diameter_{}'.format(name)] = self.diameter[ax]
        #             d_area = self.get_area(projection=name)
        #             if norm_vol:
        #                 d_area /= np.product(self.projection_shape(name))
        #             d['area_{}'.format(name)] = d_area.astype(int)
        return d