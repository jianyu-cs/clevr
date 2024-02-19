#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : analyze_shades.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 02/06/2022
#
# This file is part of Pragmatics-Dataset-Gen.
# Distributed under terms of the MIT license.

import os
import os.path as osp
import matplotlib; matplotlib.use('Agg')

import jacinle
import jacinle.io as io
import jaclearn.vision.coco.mask_utils as mask_utils
import jaclearn.visualize.box as bbox_vis_utils
import numpy as np
from PIL import Image

parser = jacinle.JacArgumentParser()
parser.add_argument('--data-dir', required=True)
args = parser.parse_args()

def main():
    for image_file in sorted(os.listdir(osp.join(args.data_dir, 'images'))):
        if not image_file.endswith('.png'):
            continue

        basename = image_file.replace('.png', '')
        if '.' in basename:
            continue

        png_file = osp.join(args.data_dir, 'images', basename + '.png')
        shadeless_file = osp.join(args.data_dir, 'images', basename + '.shadeless.png')
        bbox_file = osp.join(args.data_dir, 'images', basename + '.bbox.png')
        rjson_file = osp.join(args.data_dir, 'render_jsons', basename + '.render.json')

        if not (osp.isfile(png_file) and osp.isfile(shadeless_file) and osp.isfile(rjson_file)):
            print(png_filen, osp.isfile(png_file))
            print(shadeless_file, osp.isfile(shadeless_file))
            print(rjson_file, osp.isfile(rjson_file))
            print(f'Skip: {png_file}')
            continue

        shade_image = np.array(Image.open(shadeless_file))
        shade_image = shade_image[:, :, 0:1]
        all_values = np.unique(shade_image)
        all_values = sorted(all_values[all_values > 128])

        rspec = io.load(rjson_file)
        image = Image.open(png_file)

        assert len(all_values) == len(rspec['spec']['objects'])
        nr_objects = len(all_values)

        masks = list()
        for value in all_values:
            masks.append((shade_image == value).astype('uint8'))

        rspec['objects'] = list()
        boxes = list()
        for i in range(nr_objects):
            encoded_mask = mask_utils.encode(np.asfortranarray(masks[i]))[0]
            encoded_mask['counts'] = encoded_mask['counts'].decode('utf8')
            rspec['objects'].append(encoded_mask)
            boxes.append(mask_utils.toBbox(encoded_mask))

        boxes = np.array(boxes)
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]

        fig, ax = bbox_vis_utils.vis_bboxes(image, boxes, class_name=[f'#{i}' for i in range(1, nr_objects + 1)], fontsize=20)

        io.dump(rjson_file, rspec)
        fig.savefig(bbox_file)

        print(f'Saved: rjson="{rjson_file}"; bbox_file="{bbox_file}".')


if __name__ == '__main__':
    main()
