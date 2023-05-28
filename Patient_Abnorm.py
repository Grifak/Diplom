from OCT_data import OCT_data
import numpy as np
import pandas as pd
from collections import defaultdict


class Patient_Abnorm(object):
    def __init__(self, pat, eye, scan_dim, centered, scan_mask=None):
        self.pat = pat
        self.eye = eye
        self.scan_dim = scan_dim
        self.scan_mask = scan_mask
        self.centered = centered
        self.abnorms = []
        self.grades = []
        self.OCT = OCT_data(initials='dummy')
        self.slice_axis = {'top': 1, 'vertical': 2, 'horizontal': 0}

    def projection_shape(self, projection):
        if projection == 'vertical':
            return self.scan_dim[1:]
        elif projection == 'horizontal':
            return self.scan_dim[:-1]
        else:
            return [self.scan_dim[0], self.scan_dim[2]]

    def grade_to_num(self, grade):
        if isinstance(grade, str):
            return int(''.join([x for x in grade if x.isdigit()]))
        elif isinstance(grade, int):
            return grade
        else:
            raise ValueError('Cannot convert grade', grade)

    def get_abnorm_by_id(self, id):
        return [i for i in self.abnorms if i.id == id][0]

    def add_abnorm(self, abnorm):
        g = abnorm.grade_to_num(abnorm.grade)
        if g not in self.grades:
            self.grades.append(g)
        self.abnorms.append(abnorm)

    def graded_abnorms(self):
        graded = defaultdict(list)
        for ab in self.abnorms:
            graded[ab.grade].append(ab)
        return graded

    def export_abnorm(self, abnorm, norm=False):
        abd = abnorm.to_dict(norm_vol=norm)
        abd.pop('grade')
        abd = dict(zip(['{}_{}'.format(abnorm.grade, k) for k in abd.keys()], list(abd.values())))
        return abd

    def overlap_percentage(self, container_mask, check_mask):
        overlap = np.logical_and(container_mask, check_mask)
        overlap = np.sum(overlap)
        if overlap > 0:
            overlap /= np.sum(check_mask)
        return overlap

    def export_grades_ref(self, grade_reference, norm=False):
        ref_data = self.graded_abnorms().get(grade_reference, None)
        result = []
        if ref_data is None:
            print('{} {} has no {}'.format(self.pat, self.eye, grade_reference))
        else:

            grade_masks = defaultdict(dict)
            grade_overlaps = {}
            grade_areas = {}

            for g in self.grades:
                for projection in self.slice_axis.keys():
                    grade_mask = self.OCT.get_2D_mask(section=projection, grade=g, mask=self.scan_mask)
                    area = np.sum(grade_mask) / np.product(grade_mask.shape) if norm else np.sum(grade_mask)
                    grade_areas['grade{}_areas_{}'.format(g, projection)] = area
                    grade_masks[g][projection] = grade_mask
            for g in self.grades:
                if g != grade_reference:
                    grade_overlaps[g] = self.overlap_percentage(
                        container_mask=grade_masks[self.grade_to_num(grade_reference)]['top'],
                        check_mask=grade_masks[g]['top'])
            for ref_ab in ref_data:
                ref_dict = self.export_abnorm(ref_ab, norm=norm)
                for ab in self.abnorms:
                    if ab.grade == grade_reference:
                        continue
                    ab_dict = self.export_abnorm(ab, norm=norm)
                    ab_dict['inside_{}'.format(grade_reference)] = grade_overlaps[ab.grade_to_num(ab.grade)]
                    ab_dict['pat'] = self.pat
                    ab_dict['eye'] = self.eye
                    ab_dict['centered'] = self.centered
                    ab_dict['target_grade'] = ab.grade
                    for k, v in grade_areas.items():
                        ab_dict[k] = v if ab.grade != grade_reference else None
                    for name, ax in self.slice_axis.items():
                        ab_dict['{}'.format(name)] = self.scan_dim[ax]
                    ab_dict.update(ref_dict)
                    result.append(ab_dict)
        return pd.DataFrame(result)