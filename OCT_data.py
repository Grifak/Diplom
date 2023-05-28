import os
import numpy as np


class OCT_data(object):
    """Raw scanner OCT data.
    Arguments:
    - convert_to_log: default=True, take log of the data values."""

    def __init__(self, initials='', eye='', convert_to_log=True, comment='', centered=None):
        super(OCT_data, self).__init__()
        self.initials = initials
        self.eye = eye
        self.filename = None
        # !!!!
        self.n_hscans = 400  # horizontal
        self.n_vscans = None  # vertical (when looking onto the scan from the top)
        self.n_tscans = 400  # depth of a top slice

        self.mask = None
        self.pred_mask = None
        self.comment = comment
        self.centered = centered

    def load_scan(self, path, convert_to_log=True, n_tscans=None, n_hscans=None, n_vscans=None):
        """Load OCT array from file"""
        self.filename = os.path.basename(path)
        print(f"Filename = {self.filename}")
        if path.endswith('.npy'):
            self.scan = np.load(path)
        else:
            # fix
            self.scan = self.load_from_oct(path, n_tscans=n_tscans, n_hscans=n_hscans, n_vscans=n_vscans,
                                           convert_to_log=convert_to_log)

    def load_from_oct(self, filename, n_tscans, n_hscans, n_vscans, convert_to_log=True):
        print("Check loading from scan!")

        scan_raw = np.fromfile(filename)
        print(f"ScanRaw size = {scan_raw.shape}")
        scan = scan_raw.reshape(self.n_hscans, self.n_tscans, -1)
        scan = np.rot90(scan, 1, axes=(1, 2))
        if convert_to_log:
            scan = np.log(scan + 1e-18)
        return scan

    def load_mask(self, mask):
        if self.scan is not None:
            print(f"scanShape = {self.scan.shape} maskShape = {mask.shape}")
            assert self.scan.shape == mask.shape
        self.mask = mask

    def _check_section(self, s):
        if s.lower() == 'top':
            return 'top'
        if s.lower() == 'vertical':
            return 'vertical'
        if s.lower() == 'horizontal':
            return 'horizontal'
        raise ValueError("Cross-section argument should be either 'horizontal', 'vertical' or 'top'.")

    def section_axis(self, section):
        sl = self._check_section(section)
        if sl == 'horizontal':
            return 0
        elif sl == 'vertical':
            return 2
        else:  # sl == 'H'
            return 1

    def section_ndim(self, section):
        return self.scan.shape[self.section_axis(section)]

    def adjust_2D(self, image, section):
        if section == 'vertical':
            return np.flipud(np.rot90(image))  # np.hstack(X[::-1, :, n:n+1]) old version
        else:
            return image

    def get_slice(self, n, section='top', data=None):
        """Get slice over certain dimension"""
        sl = self._check_section(section)
        data = self.scan if data is None else data
        if sl == 'vertical':
            return self.adjust_2D(np.squeeze(data[..., n:n + 1]), sl)
        elif sl == 'horizontal':
            return self.adjust_2D(data[n, :, :], sl)
        else:
            return self.adjust_2D(np.squeeze(data[:, n:n + 1, :]), sl)

    def get_2D_mask(self, section='top', mask=None, verbose=False, grade=None):
        sl = self._check_section(section)
        m = self.mask if mask is None else mask
        if grade is not None:
            m = (m == grade).astype(int)
        m = m.max(axis=self.section_axis(sl))
        return self.adjust_2D(m, sl)
