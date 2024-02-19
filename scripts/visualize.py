#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visualize.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/04/2022
#
# This file is part of Pragmatics-Dataset-Gen.
# Distributed under terms of the MIT license.

import os
import os.path as osp
from PIL import Image
from tabulate import tabulate

import jacinle
from jaclearn.visualize.html_table import HTMLTableColumnDesc, HTMLTableVisualizer

parser = jacinle.JacArgumentParser()
parser.add_argument('--data-dir', required=True)
args = parser.parse_args()

args.images_dir = osp.join(args.data_dir, 'images')
args.jsons_dir = osp.join(args.data_dir, 'jsons')
args.output_dir = osp.join(args.data_dir, 'visualize')


def main():
    vis = HTMLTableVisualizer(args.output_dir, args.data_dir)

    with vis.html(), vis.table('Scenes', [
        HTMLTableColumnDesc('id', 'Index', 'text', {'width': '40px'}),
        HTMLTableColumnDesc('image', 'Image', 'image'),
        HTMLTableColumnDesc('scene_spec', 'Scene Spec', 'code'),
        HTMLTableColumnDesc('utterances', 'Utterances', 'code'),
    ]):
        vis.row(id='-', image=osp.join(args.output_dir, 'convention.png'), scene_spec='Object Placement Convention', utterances='')

        for json_file in sorted(os.listdir(osp.join(args.data_dir, 'jsons'))):
            if not json_file.endswith('.json'):
                continue
            basename = json_file.replace('.json', '')
            json_file = osp.join(args.data_dir, 'jsons', basename + '.json')
            png_file = osp.join(args.data_dir, 'images', basename + '.png')
            bbox_file = osp.join(args.data_dir, 'images', basename + '.bbox.png')

            print('Loading "{}".'.format(json_file))

            image = bbox_file if osp.isfile(bbox_file) else png_file
            scene = jacinle.io.load(json_file)

            table = list()
            for i, obj in enumerate(scene['scene']['objects']):
                table.append((i+1, size_to_string(obj['size']), obj['shape']))
            scene_spec_str = tabulate(table, headers=['Index', 'Size', 'Shape'])

            table = list()
            for sent, dist in scene['utterance'].items():
                table.append((sent, 'Obj #{}'.format(
                    max(range(len(dist)), key=dist.__getitem__) + 1
                )))
            utterances_str = tabulate(table, headers=['Sentence', 'Referred Obj'])

            vis.row(id=basename.split('_')[1], image=image, scene_spec=scene_spec_str, utterances=utterances_str)


def size_to_string(s):
    if s == 1:
        return 'Small'
    elif s == 2:
        return 'Medium1'
    elif s == 3:
        return 'Medium2'
    elif s == 4:
        return 'Large'
    else:
        assert False


if __name__ == '__main__':
    main()
