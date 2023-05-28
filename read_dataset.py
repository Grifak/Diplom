from static_func import *
import matplotlib.pyplot as plt
from OCT_data import OCT_data
import glob


def read_dataset(anno_fld):
    background_color = '0'
    label_color = ['grade1:1', 'grade2:2', 'grade3:3']
    mask_bitness = 8

    dim = mask_bitness // 8
    background = to_scalar(background_color, dim)

    color_map = {}
    for item in label_color:
        label, color = item.split(':')
        color_map[label] = to_scalar(color, dim)


    SCANS = []
    scan_files = glob.glob('/Users/notremembering/Desktop/диплом/OCT_scans/train/*.OCT')
    # scan_files = glob.glob('/content/OCT_SCANS/*.OCT')
    not_centered = ['CEA OD', 'GKA OS', 'GMS OS', 'HVA OS', 'MVE OD', 'SYA OS', 'BDU OD']

    # for every patient get a subset of df
    for n, scan_path in enumerate(scan_files):
        pat, eye = os.path.basename(scan_path).split('_')[:2]
        if pat == 'MIN':
            continue
        print('{}/{}'.format(n, len(scan_files)), pat, eye)
        # Load scan and set mask
        centered = '{} {}'.format(pat, eye) not in not_centered
        scan = OCT_data(initials=pat, eye=eye, centered=centered)
        scan.load_scan(scan_path)
        mask_filename = os.path.join(anno_fld, match_oct_to_dumps(scan.filename))
        print(f"Mask filename = {mask_filename}")
        mask = anno_to_mask3d(mask_filename, color_map, background)
        scan.load_mask(mask)
        SCANS.append(scan)
        if n == 2:
            break

    # plt.imshow([s for s in SCANS if s.initials == 'AAA'][0].get_2D_mask('vertical'))
    # plt.show()
    #
    # for s in SCANS:
    #   if(s.initials == 'AAA'):
    #     plt.title("Slice")
    #     plt.imshow(s.get_slice(n=201, section='vertical'))
    #     plt.title(s.initials + ' ' + s.eye)
    #     plt.show()

    return SCANS