from scipy import ndimage
from Patient_Abnorm import Patient_Abnorm
from Abnorm import Abnorm
from static_func import *
import pandas as pd
import os


def analize_abnorm(SCANS, anno_fld, not_centered):
    ab_id = 0
    ABNORMS = []
    PAT_ABNORMS = []
    for n, s in enumerate(SCANS):
        pat, eye = s.initials, s.eye
        centered = False if '{} {}'.format(pat, eye) in not_centered else True

        fname = os.path.join(anno_fld, match_oct_to_dumps(s.filename))
        print('{}/{} {}'.format(n, len(SCANS), fname))

        # parse anno into separate masks per each grade
        md = anno_to_mask3d_per_grade(fname)
        mshape = next(iter(md.values())).shape

        ab_pat = Patient_Abnorm(pat=pat, eye=eye, scan_dim=mshape, centered=centered, scan_mask=s.mask)

        # for each grade find separate entities
        md_objects = {}
        for grade, gmask in md.items():
            labeled_mask, num_objects = ndimage.label(gmask)
            obj_coords = ndimage.find_objects(labeled_mask)
            obj_centers = ndimage.center_of_mass(gmask, labeled_mask, index=range(1, num_objects + 1))
            obj_centers = [np.array(i).astype(int) for i in obj_centers]
            for coords, center in zip(obj_coords, obj_centers):
                obj_mask = gmask[coords]
                obj_scan = s.scan[coords]
                if obj_mask.size < 9:  # remove tiny areas
                    continue
                ab = Abnorm(id=ab_id, pat=pat, eye=eye, scan=obj_scan, scan_dim=mshape, centered=centered,
                            grade=grade, mask=obj_mask, center_of_mass=center, coords=coords,
                            diameter=[i.stop - i.start for i in coords])
                ABNORMS.append(ab)
                ab_pat.add_abnorm(ab)
                ab_id += 1
        PAT_ABNORMS.append(ab_pat)


    grade1_abnorms = filter(lambda x: x.grade == 'grade1', ABNORMS)
    grade2_abnorms = filter(lambda x: x.grade == 'grade2', ABNORMS)
    grade3_abnorms = filter(lambda x: x.grade == 'grade3', ABNORMS)

    print(f"Grade1 count = {len(list(grade1_abnorms))}")
    print(f"Grade2 count = {len(list(grade2_abnorms))}")
    print(f"Grade3 count = {len(list(grade3_abnorms))}")

    df = []
    for n, p in enumerate(PAT_ABNORMS):
        print('{}/{} {}'.format(n, len(PAT_ABNORMS), p.pat))
        df.append(p.export_grades_ref(grade_reference='grade1', norm=True))
    df = pd.concat(df)
    df['name'] = df['pat'] + '_' + df['eye']

    df.to_csv(r'/Users/notremembering/Desktop/диплом/analiz_dataset.txt', index=None, sep='\t', mode='a')