# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile, itertools
import os.path as osp
from datetime import datetime as dt
from collections import Counter
import numpy as np
import copy

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    import mathutils
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False

import sys
sys.path.insert(0, '../pdgen/scene/clevr')
import clevr_blender_utils as utils
from render import render_scene

parser = argparse.ArgumentParser()
# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=320, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=240, type=int,
    help="The height (in pixels) for the rendered images")
parser.add_argument('--key_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the key light position.")
parser.add_argument('--fill_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the fill light position.")
parser.add_argument('--back_light_jitter', default=1.0, type=float,
    help="The magnitude of random jitter to add to the back light position.")
parser.add_argument('--camera_jitter', default=0.5, type=float,
    help="The magnitude of random jitter to add to the camera position")
parser.add_argument('--render_num_samples', default=512, type=int,
    help="The number of samples to use when rendering. Larger values will " +
         "result in nicer images but will cause rendering to take longer.")
parser.add_argument('--render_min_bounces', default=8, type=int,
    help="The minimum number of bounces to use for rendering.")
parser.add_argument('--render_max_bounces', default=8, type=int,
    help="The maximum number of bounces to use for rendering.")
parser.add_argument('--render_tile_size', default=256, type=int,
    help="The tile size to use for rendering. This should not affect the " +
         "quality of the rendered image but may affect the speed; CPU-based " +
         "rendering may achieve better performance using smaller tile sizes " +
         "while larger tile sizes may be optimal for GPU-based rendering.")

# 2x8x3x2 = 96 elements
# utterances: <Size> <Color> <Material> <Shape> in the state of <x>

INTRINSIC_PRIMITIVES = {"size": ['small', 'large'], "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"], "shape": ["cube", "sphere", "cylinder"], "material": ["rubber", "metal"]}

EXTRINSIC_PRIMITIVES = {"Relation": ["left", "right", "behind", "front"]}

with open('../question_generation/synonyms.json', 'r') as f:
    import json
    SYNONYMS = json.load(f)

for color in INTRINSIC_PRIMITIVES['color']:
    SYNONYMS[color] = [color]

def sample_attr_object_under_discussion(candidates_: list=['size', 'color', 'material', 'shape'], 
                                           secondary: bool=False):
    # <Size> <Color> <Material> <Shape>
    masks = np.random.choice(2,len(candidates_)).tolist()
    
    while 1 not in masks:
        if secondary:
            masks = np.random.choice(2,len(candidates_)).tolist()
        else:
            return random.sample(SYNONYMS['thing'], 1)[0]+"s", masks, None, candidates_
    
    candidates = copy.deepcopy(candidates_)
    for i, mask in enumerate(masks):
        if not mask:
            candidates.pop(i)
    
    composed_name= []
    attribute_values = []
    for candidate in candidates:
        attr = random.sample(INTRINSIC_PRIMITIVES[candidate], 1)[0]
        composed_name.append(random.sample(SYNONYMS[attr], 1)[0])
        attribute_values.append(attr) # different instantiations of our synonyms
        
    composed_name = ' '.join(composed_name)
    
    if 'shape' in candidates:
        return composed_name+'s', masks, attribute_values, candidates
    else:
        return composed_name+' '+random.sample(SYNONYMS['thing'], 1)[0]+"s", masks, attribute_values, candidates

# other distractor: literal but not enriched, and incorrect predicate (co-occur)?
# Some > all vs Some < all
def template_on_DSI_some_implicature(state: str="I"):
    """
    Return the <template: str> 
    <primary_object_name: str> <primary_object_candidate_categories: List> <primary_object_name_no_synonyms: List> 
    <second_object_name: str> <second_object_candidate_categories: List> <second_object_name_no_synonyms: List>
    """
    primary_object_name, masks, attrs, pri_candidates = sample_attr_object_under_discussion()
    affix = random.sample([("Some ", " are "), ("There are some ", " which are "), ("It contains some ", " which are ")], 1)
    # primary_object_name = affix[0] + primary_object_name
    
    if state == "I":
        candidates = ['size', 'color', 'material', 'shape']
        for i, mask in enumerate(masks):
            if mask:
                candidates.pop(i)
        while len(candidates) == 0:
            primary_object_name, masks, attrs, pri_candidates = sample_attr_object_under_discussion()
            candidates = ['size', 'color', 'material', 'shape']
            for i, mask in enumerate(masks):
                if mask:
                    candidates.pop(i)
            
        # sample among the candidates, could be 1~len(candidates) numbers of objects
        second_attrs_name, second_masks, second_attrs, sec_candidates = sample_attr_object_under_discussion(candidates, True)
        if 'shape' in candidates:
            return [affix[0] + primary_object_name + affix[1] + second_attrs_name], primary_object_name, pri_candidates, attrs, \
                    second_attrs_name, sec_candidates, second_attrs
        else:
            second_attrs_name = ' '.join(second_attrs_name.split(' ')[:-1])
            return [affix[0] + primary_object_name + affix[1] + second_attrs_name], primary_object_name, pri_candidates, attrs, \
                    second_attrs_name, sec_candidates, second_attrs
    elif state == "E":
        # one or two utterances each game round.
        # For primary attributes, **Not** the subset of the auxiliary attributes 
        candidates = ['size', 'color', 'material', 'shape']
        flag = True
        overlap_attr = [] 
        while flag:
            second_attrs_name, second_masks, second_attrs, sec_candidates = sample_attr_object_under_discussion(candidates, True)
            for i, attr in enumerate(second_attrs):
                if attr not in attrs: # red objects vs red cubes. 
                    flag = False
                else:
                    overlap_attr.append((i, attr))
        
        # Could exist several "compare" utterances simulatenously when there are overlaps
        # "Same color as" Utterances
        # Let us sample 1, it could be "the same color and the same shape as"
        utterances = []
        overlap_attr_count = len(overlap_attr)
        if overlap_attr_count > 0:
            
            affix[1] = affix[1].replace("are", "have")
            num_of_overlapped_attributes = int(np.random.choice(overlap_attr_count, 1))
            # copy the name
            compare_primary_name = copy.deepcopy(primary_object_name)
            compare_second_name = copy.deepcopy(second_attributes_name)
            # Set the predicate templates, such as "Same color as"
            sampled_attributes = random.sample(overlap_attr, num_of_overlapped_attributes)    
            temp_predicate_parameter_value = sec_candidates[overlap_attr.index(sampled_attributes[0])]
            temp_predicate_template = f"the same {temp_predicate_parameter_value}"
            synonym_alternatives = SYNONYMS[temp_predicate_parameter_value]
            for synonym in synonym_alternatives:
                if predicate_parameter_value != "shape":
                    compare_primary_name.replace(" "+synonym, "")
                    compare_second_name.replace(" "+synonym, "")
                else:
                    compare_primary_name.replace(" "+synonym, random.sample(['objects', 'things'], 1)[0])
                    compare_second_name.replace(" "+synonym, random.sample(['objects', 'things'], 1)[0])
            
            
            while len(sampled_attributes) != 0:
                sampled_attributes.pop(0)
                if len(sampled_attribute) != 0:
                    temp_predicate_parameter_value = sec_candidates[overlap_attr.index(sampled_attributes[0])]
                    temp_predicate_template += "and the same {temp_predicate_parameter_value}"
                    synonym_alternatives = SYNONYMS[temp_predicate_parameter_value]
                    for synonym in synonym_alternatives:
                        if predicate_parameter_value != "shape":
                            compare_primary_name.replace(" "+synonym, "")
                            compare_second_name.replace(" "+synonym, "")
                        else:
                            compare_primary_name.replace(" "+synonym, random.sample(['objects', 'things'], 1)[0])
                            compare_second_name.replace(" "+synonym, random.sample(['objects', 'things'], 1)[0])
            temp_predicate_template += "as the"
            utterances.append(affix[0]+compare_primary_name+affix[1]+temp_predicate_template+compare_second_name)
                
        # Start at the spatial relationships
        # TODO
def position_enable_partial_occlusion(args, positions: list[tuple], blender_objects: list, intended_object_size: str = "small"):
    # measure one object per run
    with open('../image_generation/data/properties.json', 'r') as f:
        properties = json.load(f)
        size_mapping = list(properties['sizes'].items())
    r = size_mapping[intended_object_size]
    
    positions = copy.deepcopy(positions)
    
    num_tries = 0
    while True:
        # If we try and fail to place an object too many times, then delete all
        # the objects in the scene and start over.
        num_tries += 1
        if num_tries > args.max_retries:
            for i, obj in enumerate(blender_objects):
                # it fails, let do all again
                return None 
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        # Check to make sure the new object is further than min_dist from all
        # other objects, and further than margin along the four cardinal directions
        dists_good = True
        margins_good = True
        for (xx, yy, rr) in positions:
            dx, dy = x - xx, y - yy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist - r - rr < args.min_dist:
                dists_good = False
                break
            for direction_name in ['left', 'right', 'front', 'behind']:
                direction_vec = scene_struct['directions'][direction_name]
                assert direction_vec[2] == 0
                margin = dx * direction_vec[0] + dy * direction_vec[1]
                if 0 < margin < args.margin:
                    print(margin, args.margin, direction_name)
                    print('BROKEN MARGIN!')
                    margins_good = False
                    break
            if not margins_good:
                break
        if dists_good and margins_good:
            break
    
    # For cube, adjust the size a bit
    if obj_name == 'Cube':
        r /= math.sqrt(2)
    
    positions.append((x, y, r))
    # for checking visibility, we left in the other funcs
    # same as the full None return case, we do need to redo the process
    return positions
                        
                
def add_pre_objects(args, attr_values: list, attr_types: list, objects_list: list[dict], count: int=3, no_more_flag: bool=False, excluded_random_attr_values: list[list]=None, excluded_random_attr_types: list=None, group: str='easy'):
    '''
    Generate {count} objects all satisfying the attr_values
    '''
    assert len(attr_values) == len(attr_types)
    unset_attrs = list(set(['size', 'color', 'material', 'shape'])-set(attr_types))
    for i, attr in attr_values:
        if attr_types[i] == 'size':
            intended_object_size = attr
    
    positions = []
    generated_count = 0
    
    objects_list_copy = copy.deepcopy(objects_list)
    
    num_main = int(np.random.choice(count, 1)) + 1
    
    if not no_more_flag:
        # Otherwise, adding some random objects
        num_random = count - num_main
        included_intrinsic_primitives = copy.deepcopy(INTRINSIC_PRIMITIVES)
        for i, exclude_attr_type in excluded_random_attr_types:
            included_intrinsic_primitives[exclude_attr_type].pop(included_intrinsic_primitives[exclude_attr_type].index(excluded_random_attr_values[i]))
            
            
    while generated_count != num_main:
        while len(positions) == 0:
            added_object = {}
            for i, attr_type in enumerate(attr_types):
                added_object[attr_type] = attr_values[i]

            for attr_type in unset_attrs:
                added_object[attr_type] = random.sample(INTRINSIC_PRIMITIVES[attr_type], 1)[0]
                if attr_type == 'size': 
                    intended_object_size = added_object[attr_type]

            positions = position_enable_partial_occlusion(args, positions, blender_objects, intended_object_size)
            if not positions: # check visibility out of this function
                generated_count = 0
                objects_list = objects_list_copy
                continue
            generated_count += 1
            added_object['pos'] = (positions[-1][0], positions[-1][1])
            objects_list.append(added_object)
            
    if no_more_flag:
        return objects_list
    #
    while generated_count != count:
        
        
        
    
    
    
        
    
            

def add_random_objects():
    

         
        
        
def images_on_DSI_some_implicature(state: str="I", cancel: bool=False, group: str="easy"):
    # easy group: 3~4, hard group: 5~10 
    assert group in ['easy', 'hard']
    # state: I
    ## Example: "Some balls are red and big"/ "Some balls are red big objects".
    # State E
    ## Example: "Some balls are the same color and same size as the cubes"/ ""
    # NO needs to create IE examples, such as "Some balls are on the left of the cubes and they are red"...
    ## Not our main focuses
    
    assert state in ['I', "E"]
    
    if state == "I":   
        templates, primary_object_name, pri_candidates, attrs, second_attrs_name, sec_candidates, second_attrs = template_on_DSI_some_implicature(state: str="I")
        template = templates[0]
        # distractor by literal grounding: all <obj> are <state1>
        if not attrs:
            # template: all objects are xxx, attrs: second_attrs
            total_count = random.sample([3, 4], 1)[0]
            scene = add_pre_objects(second_attrs, sec_candidates, [], total_count, no_more_flag, group)
            
        
        # referent for implicatures: some but not all <obj> are <state1>
        
        
        
        
    
    

def gen_scene1():
    rv = []

    dis = 0
    k = 0
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': 'small', #RSA_SCENE_SPEC_SIZE[2],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    k = 2
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': 'small', #RSA_SCENE_SPEC_SIZE[2],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    k = 7
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    return {'objects': rv}

def gen_scene2():
    rv = []

    dis = 0
    k = 0
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    k = 0.5
    rv.append({
        'color': 'green',
        'shape': 'sphere',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[2],
        'pos': (-2.5, 1)#-5 + 4.5 * (k % 3) - dis) #(-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis) # x: -3 left corner, y: k=2 up right corner
    })
    k = 7
    dis = 1
    rv.append({
        'color': 'red',
        'shape': 'sphere',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3), -5 + 4.5 * (k % 3) - dis)
    })
    rv.append({
        'color': 'blue',
        'shape': 'cube',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3), -5 + 4.5 * (k % 3) + dis)
    })
    return {'objects': rv}


def gen_scene3():
    rv = []

    dis = 0
    k = 0
    rv.append({
        'color': 'red',
        'shape': 'cube',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    k = 2
    rv.append({
        'color': 'red',
        'shape': 'cube',
        'material': 'metal',
        'size': 'small', #RSA_SCENE_SPEC_SIZE[2],
        'pos': (-3 + 3 * (k // 3) - dis, -5 + 4.5 * (k % 3) - dis)
    })
    k = 7
    dis = 1
    rv.append({
        'color': 'red',
        'shape': 'cube',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3), -5 + 4.5 * (k % 3) - dis)
    })
    rv.append({
        'color': 'blue',
        'shape': 'cube',
        'material': 'metal',
        'size': 'large', #RSA_SCENE_SPEC_SIZE[4],
        'pos': (-3 + 3 * (k // 3), -5 + 4.5 * (k % 3) + dis)
    })
    return {'objects': rv}


def main(args):
    import os
    os.makedirs('./data/v3-examples/images', exist_ok=True)
    os.makedirs('./data/v3-examples/render_jsons', exist_ok=True)

    spec = gen_scene1()
    print(spec)
    render_scene(args,
         spec,
         output_image='./data/v3-examples/images/scene_example1.png',
         output_json='./data/v3-examples/render_jsons/scene_example1.render.json',
         # output_blendfile='./dumps/rsa_vagueness/scene_example1.blend',
         output_shadeless='./data/v3-examples/images/scene_example1.shadeless.png'
    )
    spec = gen_scene2()
    print(spec)
    render_scene(args,
        spec,
        output_image='./data/v3-examples/images/scene_example2.png',
        output_json='./data/v3-examples/render_jsons/scene_example2.render.json',
        # output_blendfile='./dumps/rsa_vagueness/scene_example2.blend',
        output_shadeless='./data/v3-examples/images/scene_example2.shadeless.png'
    )
    spec = gen_scene3()
    print(spec)
    render_scene(args,
        spec,
        output_image='./data/v3-examples/images/scene_example3.png',
        output_json='./data/v3-examples/render_jsons/scene_example3.render.json',
        # output_blendfile='./dumps/rsa_vagueness/scene_example3.blend',
        output_shadeless='./data/v3-examples/images/scene_example3.shadeless.png'
    )


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')

