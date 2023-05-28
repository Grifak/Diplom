import os
import numpy as np
np.random.seed(100)
import cv2
from lxml import etree
from tqdm import tqdm
import warnings


def parse_anno_file(root):
    anno = []
    for image_tag in root.iter('track'):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            image['shapes'].append(polygon)

        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)

    return anno


def parse_meta(root):
    trivial_props = ['id', 'name', 'size', 'mode', 'overlap', 'flipped', 'created', 'updated']
    task = root.iter('meta').__next__().iter('task').__next__()
    props = {c.tag: c.text for c in task.getchildren() if c.tag in trivial_props}
    props['labels'] = [l.iter('name').__next__().text for l in task.iter('labels').__next__().getchildren()]

    props['original_size'] = {s.tag: int(s.text) \
                              for s in task.iter('original_size').__next__().getchildren()}

    return props


def parse_xml(cvat_xml):
    root = etree.parse(cvat_xml).getroot()
    anno = parse_anno_file(root)
    meta = parse_meta(root)
    return anno, meta


def to_scalar(str, dim):
    scalar = list(map(int, str.split(',')))
    if len(scalar) < dim:
        scalar.extend([scalar[-1]] * dim)
    return tuple(scalar[0:dim])


def sort_by_label_id(frames_masks):
    sorted_frames_masks = {}
    for frame_id, shapes in frames_masks.items():
        sorted_frames_masks[frame_id] = sorted(shapes, key=lambda i: i['label_id'])
    return sorted_frames_masks


def get_label_id(label):
    id_ = ''.join(i for i in label if i.isdigit())
    if id_ != '':
        return int(id_)
    else:
        print('Could not parse label:', label)
        return None


def tracks_to_frames(anno):
    frames_masks = {}

    for track in anno:
        label = track['label']
        if label.lower() in ['annotate', 'grade']:
            label = 'Grade1'
        label_id = get_label_id(label)
        if label_id is None:
            print('Skipping label')
            continue
        for shape in track['shapes']:
            if shape['outside'] == 1:
                continue  # skip outside = 1, probably means the end of tracking for the object

            frame_shape = {'label': label.lower(),
                           'label_id': label_id,
                           'points': shape['points'],
                           'type': shape['type']}
            frame_id = int(shape['frame'])
            if frame_id in frames_masks:
                if frame_shape not in frames_masks[frame_id]:  # avoid duplicates
                    frames_masks[frame_id].append(frame_shape)
            else:
                frames_masks[frame_id] = [frame_shape]
    frames_masks = sort_by_label_id(frames_masks)
    return frames_masks


def draw_shape(mask, shape_points, color):
    points = [tuple(map(float, p.split(','))) for p in shape_points.split(';')]
    points = np.array([(int(p[0]), int(p[1])) for p in points])
    mask = cv2.fillPoly(mask, [points], color=color)
    return mask


def save_masks_as_png(cvat_xml, outfld, mask_bitness, color_map, background):
    anno, meta = parse_xml(cvat_xml)
    width, height = meta['original_size']['width'], meta['original_size']['height']
    anno_frames = tracks_to_frames(anno)

    for frame_id, frame_shapes in tqdm(anno_frames.items(), desc='Generate masks'):
        mask = np.zeros((height, width, mask_bitness // 8), dtype=np.uint8)
        mask_dir = os.path.join(outfld, meta['name'])
        if mask_dir:
            os.makedirs(mask_dir, exist_ok=True)

        for shape in frame_shapes:
            color = color_map.get(shape['label'], background)
            mask = draw_shape(mask, shape['points'], color)

        im_name = '{}_{}.png'.format(meta['name'], frame_id)
        mask_path = os.path.join(mask_dir, im_name)

        mask = np.squeeze(mask)
        cv2.imwrite(mask_path, mask)


def anno_to_mask3d(cvat_xml, color_map, background):
    anno, meta = parse_xml(cvat_xml)
    width, height = meta['original_size']['width'], meta['original_size']['height']
    anno_frames = tracks_to_frames(anno)
    n_frames = int(meta['size'])
    mask_3d = np.zeros((n_frames, height, width))

    for frame_id, frame_shapes in anno_frames.items():
        mask = np.zeros((height, width))

        for shape in frame_shapes:
            color = color_map.get(shape['label'], background)
            mask = draw_shape(mask, shape['points'], color)
        mask = np.squeeze(mask)
        if frame_id < 0 or frame_id >= n_frames:
            warnings.warn('In file {} check the frame #{}.\nThis mask will be ignored.'.format(
                cvat_xml, frame_id))
        else:
            mask_3d[frame_id] = mask

    return mask_3d


def anno_to_mask3d_per_grade(cvat_xml):
    anno, meta = parse_xml(cvat_xml)
    width, height = meta['original_size']['width'], meta['original_size']['height']
    anno_frames = tracks_to_frames(anno)
    n_frames = int(meta['size'])
    masks_dict = {}

    for frame_id, frame_shapes in anno_frames.items():
        for shape in frame_shapes:
            mask = draw_shape(np.zeros((height, width)), shape['points'], color=1).squeeze()
            if frame_id < 0 or frame_id >= n_frames:
                warnings.warn('In file {} check the frame #{}.\nThis mask will be ignored.'.format(
                    cvat_xml, frame_id))
            else:
                if shape['label'] not in masks_dict.keys():
                    masks_dict[shape['label']] = np.zeros((n_frames, height, width))
                masks_dict[shape['label']][frame_id] = mask

    return masks_dict


def match_oct_to_dumps(filename):
    pat, eye = os.path.basename(filename).split('_')[:2]
    pat_dict = {'SOV': {'OS': 'SOV OS', 'OD': 'SOV OD'},
                'BGN': {'OS': 'BGN L', 'OD': 'BGN'},
                'GMS': {'OS': 'GMS OS', 'OD': 'GMS OD'},
                'MIN': {'OS': 'MIN OS', 'OD': 'MIN OD'}
               }
    if pat in pat_dict.keys():
        dname = pat_dict[pat][eye]
    else:
        dname = pat
    return '{}.dump'.format(dname)


