#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : render.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/19/2021
#
# This file is part of NSCL-PyTorch.
# Distributed under terms of the MIT license.

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import sys
import random
import json
import os.path as osp
import numpy as np

INSIDE_BLENDER = True
try:
    import bpy
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    dirname = osp.dirname(__file__)
    sys.path.insert(0, dirname)
    import clevr_blender_utils as utils
    sys.path = sys.path[1:]

BASE_DIR = osp.dirname(__file__)

INTRINSIC_PRIMITIVES = {"size": ['small', 'large'], "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"], "shape": ["cube", "sphere", "cylinder"], "material": ["rubber", "metal"]}


class SceneBuilder(object):
    # specification: formal utterance, num_objects_w_groups, tag_I/E,  
    # compute_all_relationships incorporated, calculated in the main settings
    # all objects are red - easy
    # some objects are red - adjusted in main objects, but still easy
    # all objects are on the left of xxx - ensuring one non-empty/not-low 
    def __init__(self, args, properties_json, shape_dir, metadata_json):
        super().__init__()
        self.args = args
        self.properties_json = properties_json
        self.shape_dir = shape_dir
        self.metadata_json = metadata_json
        
        self.intrinsic_attribute_value2key = {} 

        with open(self.properties_json) as f:
            properties = json.load(f)
        with open(self.metadata_json) as f:
            metadata = json.load(f)
            
        for color in metadata['Color']:
            self.intrinsic_attribute_value2key[color] = 'color'
        for shape in metadata['Shape']:
            self.intrinsic_attribute_value2key[shape] = 'shape'
        for size in metadata['Size']:
            self.intrinsic_attribute_value2key[size] = 'size'
        for material in metadata['Material']:
            self.intrinsic_attribute_value2key[material] = 'material'

        self.colors = dict()
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            self.colors[name] = rgba
        self.shapes = properties['shapes']
        self.materials = properties['materials']
        self.sizes = properties['sizes']

        self.objects = None
        self.blender_objects = None
        
        self.scene_struct = self.render_initial_scene(args)

    def render_initial_scene(self, args):
        # Load the main blendfile
        bpy.ops.wm.open_mainfile(filepath=osp.join(BASE_DIR, './data/base_scene.blend'))

        # Load materials
        utils.load_materials(osp.join(BASE_DIR, './data/materials'))

        # Set render arguments so we can get pixel coordinates later.
        # We use functionality specific to the CYCLES renderer so BLENDER_RENDER cannot be used.
        render_args = bpy.context.scene.render
        render_args.engine = "CYCLES"
        render_args.filepath = output_image
        render_args.resolution_x = args.width
        render_args.resolution_y = args.height
        render_args.resolution_percentage = 100
        render_args.tile_x = args.render_tile_size
        render_args.tile_y = args.render_tile_size
        if args.use_gpu == 1:
            # Blender changed the API for enabling CUDA at some point
            if bpy.app.version < (2, 78, 0):
                bpy.context.user_preferences.system.compute_device_type = 'CUDA'
                bpy.context.user_preferences.system.compute_device = 'CUDA_0'
            else:
                cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
                cycles_prefs.compute_device_type = 'CUDA'

        # Some CYCLES-specific stuff
        bpy.data.worlds['World'].cycles.sample_as_light = True
        bpy.context.scene.cycles.blur_glossy = 2.0
        bpy.context.scene.cycles.samples = args.render_num_samples
        bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
        bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
        if args.use_gpu == 1:
            bpy.context.scene.cycles.device = 'GPU'

        # This will give ground-truth information about the scene and its objects
        scene_struct = {'directions': {}}

        # Put a plane on the ground so we can compute cardinal directions
        bpy.ops.mesh.primitive_plane_add(radius=5)
        plane = bpy.context.object

        camera = bpy.data.objects['Camera']
        print("camera locations:", camera.location[0])
        print("camera locations:", camera.location[1])
        print("camera locations:", camera.location[2])

        def rand(L):
            return 2.0 * L * (random.random() - 0.5)

        # Add random jitter to camera position
        '''
        if args.camera_jitter > 0:
            for i in range(3):
              bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)
        '''
        # Figure out the left, up, and behind directions along the plane and record
        # them in the scene structure
        camera = bpy.data.objects['Camera']
        plane_normal = plane.data.vertices[0].normal
        cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
        cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
        cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
        plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
        plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
        plane_up = cam_up.project(plane_normal).normalized()

        # Delete the plane; we only used it for normals anyway. The base scene file
        # contains the actual ground plane.
        utils.delete_object(plane)
        # Save all six axis-aligned directions in the scene struct
        scene_struct['directions']['behind'] = tuple(plane_behind)
        scene_struct['directions']['front'] = tuple(-plane_behind)
        scene_struct['directions']['left'] = tuple(plane_left)
        scene_struct['directions']['right'] = tuple(-plane_left)
        scene_struct['directions']['above'] = tuple(plane_up)
        scene_struct['directions']['below'] = tuple(-plane_up)

        return scene_struct
    
    def judge_direction(object1_coords: tuple, object2_coords: tuple, tag='left'):
        diff = [object1_coords[k] - object2_coords[k] for k in [0, 1, 2]]
        direction_vec = self.scene_struct[tag]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
            return True
        return False
        
        
    def render_positions_good_margin(self, args, current_object_positions: list[tuple], intended_size: str = "small"):
        assert intended_size in ['small', 'large']
        size_mapping = list(self.sizes.items())
        r = size_mapping[intended_object_size]
        
        current_object_pos = copy.deepcopy(current_object_positions)
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > args.max_retries:
                return None # Fail, do it again
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for (xx, yy, rr) in current_object_pos: 
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < args.min_dist:
                    dists_good = False
                    break
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = self.scene_struct['directions'][direction_name]
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
        return (x,y,r)
    
    def assign_num_members(self, some_tag: bool, max_count: int, len_extrinsic: int, no_random_tag: bool):
        # num_main, num_extrinsic, num_some_other_states, num_random
        assert group_tag in ['Intrinsic', "Extrinsic"]
        assert len_extrinsic < 3
        assert max_count in [4, 10]
        
        num_main = 0
        num_extrinsic = 0 
        num_some_other = 0
        num_random = 0
        num_extrinsic = [1 for _ in range(len_extrinsic)]
        '''
        if max_count == 4:
            if len_extrinsic==2:
                if some_tag: 
                    return [1, [1,1], 1, 0]
                else:
                    return random.sample([[1, [1,1], 0, 0], [1, [1,1], 0, 1], \
                                          [1, [2,1], 0, 0], [1, [1,2], 0, 0]], 1)
            elif len_extrinsic == 1:
                if some_tag:
                    return random.sample([[1, [1], 1, 1], [1, [1], 1, 0], \
                                          [1, [1], 2, 0], [2, [1], 1, 0], \
                                          [1, [2], 1, 0]], 1)
                else:
                    # all objects are on the left of the <>.
                    num_main = np.random.choice(2,1)[0]+1 # 1~3
                    num_extrinsic = random.sample(range(1, max_count-num_main+1), 1)[0]
                    num_random = random.sample(range(0, max_count - num_main - num_extrinsic), 1)[0]+1 if max_count - num_main - num_extrinsic else 0 
                
            else:
                # x, 0, x, y
                num_main = random.sample(range(1,max_count), 1)[0]
                num_some_other = random.sample(range(1, max_count-num_main+1), 1)[0]
                num_random = random.sample(range(0, max_count - num_main - num_some_other), 1)[0]+1 if max_count - num_main - num_some_other else 0
        
        elif max_count == 10:
        '''
        # first, we need to narrow down possible values of main object number upper bounds
        # at least len_extrinsic should be retained
        # then some vs all
        # when max_count == 10, some adjustments
        if max_count == 10:
            max_count = random.sample(range(5, 10), 1)[0]
        
        if some_tag:
            num_main = random.sample(range(1, max_count - len_extrinsic), 1)[0]# 10 - 3, 6
            num_some_other = random.sample(range(1, max_count - len_extrinsic - num_main + 1), 1)[0]
            left_number = random.sample(range(0, max_count - num_main - num_some_other - len_extrinsic+1), 1)[0] if max_count - num_main - num_some_other - len_extrinsic else 0
            if left_number:
                random_flag = random.sample(range(2),1)[0] 
                if random_flag or len_extrinsic==0:
                    num_random += left_number
                
                else:
                    if len_extrinsic == 1:
                        num_extrinsic[0] += left_number
                    elif len_extrinsic == 2:
                        num_extrinsic[0] += random.sample(range(left_number), 1)[0]
                        num_extrinsic[1] += left_number - num_extrinsic[0]
        else:
            num_main = random.sample(range(1, max_count - len_extrinsic), 1)[0]
            left_number = random.sample(range(0, max_count - num_main - len_extrinsic + 1), 1)[0] if max_count - num_main - len_extrinsic else 0
            if left_number:
                random_flag = random.sample(range(2),1)[0] 
                if random_flag or len_extrinsic==0: # "All objects are red"
                    num_random += left_number
                else:
                    pass
                    '''
                    if len_extrinsic == 1:
                        num_extrinsic[0] += left_number
                    elif len_extrinsic == 2:
                        temp = random.sample(range(left_number+1), 1)[0]
                        num_extrinsic[0] += temp
                        
                        num_extrinsic[1] += left_number - temp
                    '''
        
        if not no_random_tag:                            
            return num_main, num_extrinsic, num_some_other, num_random
        else:
            total_count = num_main + sum(num_extrinsic) + num_some_other
            if max_count == 4 and total_count < 3:
                num_main += 3 - total_count
            elif max_count == 10 and total_count < 5:
                num_main += 5 - total_count
                
            return num_main, num_extrinsic, num_some_other, 0
                
    def judge_attributes_partial_overlapping_failure(self, added_objects, excluded_attributes_list: list[list]):
        # specific instance: some big balls are red. So not "xxx big balls", should differ in at least one attribute
        added_object_attr_set = set()
        for attr_type in ['size', 'color', 'material', 'shape']:
            added_object_attr_set.add(added_objects[attr_type])
        for i in range(len(excluded_attributes_list)):
            if set(excluded_attributes_list[i]).issubset(added_object_attr_set):
                return True
        return False
        
    def parse_specified_formal_language(self, specified_formal_language: str):
        intrinsic_addition_predicates = [] # List, ["big", "blue"] 
        extrinsic_predicates = [] # List[Tuple] (on the left of, blue cubes), (on the right of, blue)
        
        intrinsic_addition_types = [] # List, ["size", "color"]
        extrinsic_predicate_intrinsic_types = [] # List[List], List[["color", "size"]]
        
        addition_primary_types = []
        addition_primary_predicates = [] 
        
        
        if "some" in specified_formal_language.lower() or "all" in specified_formal_language.lower():
            primary_object = specified_formal_language.split(' are')[0]
            if 'some' in specified_formal_language.lower():
                addition_primary_predicates = primary_object.split(" ")[1:] # ["blue", "balls"]
                addition_primary_types = [self.intrinsic_attribute_value2key[attr_value] for attr_value in addition_primary_predicates]
            
            predicates = specified_formal_language.split(' are ')[1].split(" and ")
            for predicate in predicates:
                if 'left' in predicate or 'right' in predicate or 'behind' in predicate or 'front' in predicate or 'same' in predicate:
                    extrinsic_predicates.append(("the ".join(predicate.split('the ')[:-1]).strip(' '), predicate.split('the ')[-1])) # (the same color with, blue balls)
                    current_extrinsic_predicate_intrinsic_types = []
                    for modifier_unit in predicate.split('the ')[-1].split(" "):
                        if 'object' in modifier_unit:
                            continue
                        else:
                            current_extrinsic_predicate_intrinsic_types.append(self.intrinsic_attribute_value2key[modifier])
                    extrinsic_predicate_intrinsic_types.append(current_extrinsic_predicate_intrinsic_types)
                              
                else:
                    # blue
                    temps = predicate.split(" ")
                    for temp in temps:
                        intrinsic_addition_predicates.append(temp)
                        intrinsic_addition_types.append(self.intrinsic_attribute_value2key[temp])
        else:
            # disconnective implicatures
            
            
                    
        return primary_object, extrinsic_predicates, extrinsic_predicate_intrinsic_types, intrinsic_addition_predicates, intrinsic_addition_types, addition_primary_predicates, addition_primary_types
            
            
        
    def build(self, args, specified_formal_language: str, group: str = "small", group_tag: str = "Intrinsic"):
        assert self.blender_objects is None, 'Build can only be called once.'
        assert group in ['small', 'large']
        assert group_tag in ['Intrinsic', "Extrinsic"], "It could only be three choices, one is Intrinsic, the other is Extrinsic."
        # redundant variables: addition_primary_predicates and primary_intrinsic_types
        # Demo:
        # Some/ All (Not all) black balls are big [and] (No objects)
        # Some/ All balls are <on the left of>/ <the same color with> the cubes [and].
        # Some/ All balls are big [and] <on the left of> the black objects. 
        
        # Some/ All balls are <on the left of> some/ all cubes.
        # No balls are <on the left of> some/all cubes. 
        # Exactly one ball is <on the left of> some cubes.
        
        # Demo:
        # Red balls are big or Red cubes are small.    
       
        camera = bpy.data.objects['Camera']
        # parsing the formal natural language
        primary_object, extrinsic_predicates, extrinsic_predicate_intrinsic_types, intrinsic_addition_predicates, intrinsic_addition_types, addition_primary_predicates, addition_primary_types = self.parse_specified_formal_language(specified_formal_language)
        
        assert len(intrinsic_addition_predicates) == len(intrinsic_addition_types)
        
        quantifier=True if 'some' in specified_formal_language.lower() or "all" in specified_formal_language.lower() else False
        
        primary_predicates = []
        primary_intrinsic_types = []
        auxiliary_predicates_list = [] # list[list]
        auxiliary_intrinsic_types_list = [] # list[list]
        
        extrinsic_relationships = [] # list
        no_random_tag = False
        
        if quantifier: 
            if primary_object.split(" ")[1] == "objects":
                if len(intrinsic_addition_predicates) == 0:
                    no_random_tag = True
                else:
                    # some modifier exists in the intrinsic addition attributes
                    temp_modifiers = primary_object.split(" ")[1:]
                    if 'object' in temp_modifiers[-1]:
                        temp_modifiers = temp_modifiers[:-1]
                    for temp_modifier in temp_modifiers:
                        primary_predicates.append(temp_modifier)
                        primary_intrinsic_types.append(self.intrinsic_attribute_value2key[temp_modifier])
            else:
                for temp_modifier in primary_object.split(" ")[1:-1]:
                    primary_predicates.append(temp_modifier)
                    primary_intrinsic_types.append(self.intrinsic_attribute_value2key[temp_modifier])
            
            if group_tag in ["Extrinsic"]:
                for i in range(len(extrinsic_predicate_intrinsic_types)):
                    # current object and respective interactions
                    auxiliary_predicates = []
                    auxiliary_intrinsic_types = []
                    interaction, attr_values = extrinsic_predicates[i][0], extrinsic_predicates[i][1]
                    if 'left' in interaction:
                        extrinsic_relationships.append('left')
                    elif 'right' in interaction:
                        extrinsic_relationships.append('right')
                    elif 'above' in interaction:
                        extrinsic_relationships.append('above')
                    elif 'below' in interaction:
                        extrinsic_relationships.append('below')
                    elif 'same color' in interaction:
                        extrinsic_relationships.append('same color')
                    elif 'same shape' in interaction:
                        extrinsic_relationships.append('same shape')
                    elif 'same size' in interaction:
                        extrinsic_relationships.append('same size')
                    elif 'same material' in interaction:
                        extrinsic_relationships.append('same material')
                    
                    for j, attr_type in enumerate(extrinsic_predicate_intrinsic_types[i]):
                            
                        auxiliary_predicates.append(attr_values.split(" ")[i])
                        auxiliary_intrinsic_types.append(attr_type)
                        
                    auxiliary_predicates_list.append(auxiliary_predicates)
                    auxiliary_intrinsic_types_list.append(auxiliary_intrinsic_types)      
                
            
        assert len(auxiliary_predicates_list) <= 2    
            
        objects = list()
        blender_objects = list()
        full_objects = list()
        positions = list()
        
        # adjusted to incorporate the ""
        if group == 'small':
            # count = random.sample([3,4],1)[0]
            max_count = 4
            if len(auxiliary_predicates_list) != 0:
                # num_main, num_extrinsic, num_some_other, num_random
                num_main, num_extrinsic, num_some_other, num_random = self.assign_num_members(True, group_tag, max_count, len(extrinsic_predicates), no_random_tag)# I: at most count -1 (some), E: at most count - len(extrinsic_objects) -1 (some)
            
        else:
            max_count = 10
            num_main, num_extrinsic, num_some_other, num_random = self.assign_num_members(True, group_tag, max_count, len(extrinsic_predicates), no_random_tag)
        
        total_count = num_main + sum(num_extrinsic) + num_some_other + num_random
           
        # TODO: judge attributes whether in objects-under-discussion
        thetas = [360.0 * random.random() for _ in range(total_count)]
        # add attributes
        all_visible = False
        num_tries = 0
        
        while not all_visible:
            # Three stages
            num_tries += 1
            while generated_count != total_count:
                # generating main objects
                if group_tag == "Intrinsic":
                    while generated_count < num_main:
                        # first generate the auxiliary objects, auxiliary_intrinsic_types
                        # then generate the primary objects with extrinsic constraints
                        position = None
  
                        temp_primary_intrinsic_types = []
                        added_object = {}
                        for i, attr_type in enumerate(primary_intrinsic_types):
                            added_object[attr_type] = primary_predicates[i]
                            temp_primary_intrinsic_types.append(attr_type)

                        # additional attributes, intrinsic_addition_predicates, intrinsic_addition_types
                        for i, attr_type in enumerate(intrinsic_addition_types):
                            added_object[attr_type] = intrinsic_addition_predicates[i]
                            temp_primary_intrinsic_types.append(attr_type)

                        unset_primary_attrs = list(set(['size', 'color', 'material', 'shape'])-set(temp_primary_intrinsic_types))

                        for attr_type in unset_primary_attrs:
                            added_object[attr_type] = random.sample(INTRINSIC_PRIMITIVES[attr_type], 1)[0]
                        # Key loop        
                        while not position: # succeed
                            position = self.render_positions_good_margin(positions, intended_size=added_object['size'])
                            if not position: # check visibility out of this function
                                generated_count = 0
                                objects = list()
                                continue

                            positions.append(position)
                            generated_count += 1
                            added_object['pos'] = (positions[-1][0], positions[-1][1])
                            objects.append(added_object)
                            
                    while generated_count < num_some_other + num_main and generated_count >= num_main:
                        # process the addition_predicate_types
                        # 1. obtain the shared attributes, and the excluded attributes
                        
                        position = None
                        # intrinsic_addition_predicates, intrinsic_addition_types
                        temp_other_intrinsic_types = []
                        added_object = {}
                        temp_intrinsic_primitives = copy.deepcopy(INTRINSIC_PRIMITIVES)

                        for i, attr_type in enumerate(primary_intrinsic_types):
                            added_object[attr_type] = primary_predicates[i]
                            temp_other_intrinsic_types.append(attr_type)

                        # additional attributes, intrinsic_addition_predicates, intrinsic_addition_types
                        for i, attr_type in enumerate(intrinsic_addition_types):
                            temp_intrinsic_primitives[attr_type].pop(temp_intrinsic_primitives[attr_type].index(intrinsic_addition_predicates[i]))
                            added_object[attr_type] = random.sample(temp_intrinsic_primitives[attr_type], 1)[0]
                            temp_other_intrinsic_types.append(attr_type)

                        unset_primary_attrs = list(set(['size', 'color', 'material', 'shape'])-set(temp_other_intrinsic_types))
                        for attr_type in unset_primary_attrs:
                            added_object[attr_type] = random.sample(INTRINSIC_PRIMITIVES[attr_type], 1)[0]
                            
                        while not position:
                            position = self.render_positions_good_margin(positions, intended_size=added_object['size'])
                            if not position:
                                generated_count = 0
                                objects = list()
                                continue
                            
                            positions.append(position)
                            generated_count += 1
                            added_object['pos'] = (positions[-1][0], positions[-1][1])
                            objects.append(added_object)
                                      
                    while generated_count >= num_main + num_some_other and generated_count < total_count:
                        # generating random objects
                        # Func_judge() -- some set differences
                        added_object = {}
                        excluded_failure = True
                        while excluded_failure:
                            for attr_type in ['size', 'color', 'material', 'shape']:
                                added_object[attr_type] = random.sample(INTRINSIC_PRIMITIVES[attr_type], 1)[0]
                             
                            excluded_failure = self.judge_attributes_partial_overlapping_failure(added_object, excluded_attributes_list=[primary_predicates])

                        position = None
                        while not position:
                            position = self.render_positions_good_margin(positions, intended_size=added_object['size'])
                            if not position:
                                generated_count = 0
                                objects = list()
                                continue

                            positions.append(position)
                            generated_count += 1
                            added_object['pos'] = (positions[-1][0], positions[-1][1])
                            objects.append(added_object)
                

                elif group_tag == "Extrinsic":
                    while generated_count < num_main + num_some_other + sum(num_extrinsic):
                        total_pri_positions = []
                        total_other_positions = []
                        total_aux_positions = []
                        
                        pri_aux_constraints = []
                        
                        total_pri_objects = []
                        total_other_objects = []
                        total_aux_objects = []
                        
                        added_pri_object_general = {}
                        added_other_object_general = {}
                      
                        
                        for i, attr_type in enumerate(primary_intrinsic_types): 
                            added_pri_object_general[attr_type] = primary_predicates[i]
                            if num_some_other:
                                added_other_object_general[attr_type] = primary_predicates[i]
                            
                            # temp_primary_intrinsic_types.append(attr_type)
                        
                        for i, attr_types_list in enumerate(extrinsic_predicate_intrinsic_types): # extrinsic_predicate_intrinsic_types
                            total_aux_objects.append(dict())
                            for j, attr_type in enumerate(attr_types_list):
                                total_aux_objects[i][attr_type] = extrinsic_predicates[i][j] # extrinsic_predicates
                        
                        # first fixing the auxiliary objects (on the left of the xxx)
                        for i, aux_object in enumerate(total_aux_objects):
                            unset_auxiliary_attrs = list(set(['size', 'color', 'material', 'shape'])-set(aux_object.keys()))
                            for unset_attr_type in unset_auxiliary_attrs:
                                total_aux_objects[i][unset_attr_type] = random.sample(INTRINSIC_PRIMITIVES[unset_attr_type], 1)[0]
                        
                        # in case for the attribute comparison class
                        for i, (extrinsic_predicate, _) in enumerate(extrinsic_predicates):
                            if "same" in extrinsic_predicate:
                                attr_under_discussion = extrinsic_predicate.split(" same ")[1].split(" ")[0]
                                added_pri_object_general[attr_under_discussion] = total_aux_objects[i][attr_under_discussion]
                                
                                attribute_under_discussion = copy.deepcopy(INTRINSIC_PRIMITIVES[attr_under_discussion])
                                attribute_under_discussion.pop(attribute_under_discussion.index(total_aux_objects[i][attr_under_discussion]))
                                
                                added_other_object_general[attr_under_discussion] = random.sample(attribute_under_discussion, 1)[0]
                                pri_aux_constraints.append("same " + attr_under_discussion)
                                
                            elif "left" in extrinsic_predicate:
                                pri_aux_constraints.append("left")
                            elif "right" in extrinsic_predicate:
                                pri_aux_constraints.append("right")
                            elif "front" in extrinsic_predicate:
                                pri_aux_constraints.append("front")
                            elif "behind" in extrinsic_predicate:
                                pri_aux_constraints.append("behind")
                                
                        
                        unset_primary_attrs = list(set(['size', 'color', 'material', 'shape'])-set(added_pri_object_general.keys()))
                        unset_other_attrs = list(set(['size', 'color', 'material', 'shape'])-set(added_other_object_general.keys()))
                        # instantiations
                        # generating the objects until the number surpasses certain numbers
                        ## 1. generating the auxiliary objects
                        for i, object_spec in enumerate(total_aux_objects):
                            # ignore any occlusions
                            position = None
                            while not position:
                                position = self.render_positions_good_margin(positions, intended_size=object_spec['size'])

                                if not position:
                                    generated_count = 0
                                    objects = list()
                                    continue

                                positions.append(position)
                                total_aux_positions.append(position)
                            
                                generated_count += 1
                                total_aux_objects[i]['pos'] = (positions[-1][0], positions[-1][1])
                                objects.append(total_aux_objects[i])   
                            
                            
                            utils.add_object(args.shape_dir, added_object['shape'], position[2], (position[0], position[1]), theta=thetas[i])
                            utils.add_material(self.materials[added_object['material']], Color=self.colors[added_object['color']])
                            obj = bpy.context.object
                            blender_objects.append(obj)
                        ## 2. generating the main, and some other objects
                        ## extrinsic interaction-centric research
                        for i in range(num_main):
                            current_obj = copy.deepcopy(added_pri_object_general)
                            for attr_type in unset_primary_attrs:
                                current_obj[attr_type] = random.sample(INTRINSIC_PRIMITIVES[unset_attr_type], 1)[0]
                    
                            total_pri_objects.append(current_obj)
                        
                        for i in range(num_some_other):
                            current_obj = copy.deepcopy(added_other_object_general)
                            for attr_type in unset_other_attrs:
                                current_obj[attr_type] = random.sample(INTRINSIC_PRIMITIVES[unset_attr_type], 1)[0]
                            
                            total_other_objects.append(current_obj)
                       
                        all_same_flag=True
                        for constraint in pri_aux_constraints:
                            if 'same' not in constraint:
                                all_same_flag = False
                        # Let's only consider images with or without "attribute comparisons"
                        if all_same_flag:
                            # not caring about the position too much
                            for i, object_spec in enumerate(total_pri_objects):
                                position = None
                                while not position:
                                    position = self.render_positions_good_margin(positions, intended_size=object_spec['size'])
                                    if not position:
                                        generated_count = 0
                                        objects = list()
                                        continue

                                    positions.append(position)
                                    total_pri_positions.append(position)

                                    generated_count += 1
                                    total_pri_objects[i]['pos'] = (positions[-1][0], positions[-1][1])
                                    objects.append(total_pri_objects[i])
                                    
                            for i, object_spec in enumerate(total_other_objects):
                                position = None
                                while not position:
                                    position = self.render_positions_good_margin(positions, intended_size=object_spec['size'])
                                    if not position:
                                        generated_count = 0
                                        objects = list()
                                        continue
                                    positions.append(position)
                                    total_other_positions.append(position)
                                    
                                    generated_count += 1
                                    total_other_objects[i]['pos'] = (positions[-1][0], positions[-1][1])
                                    objects.append(total_other_objects[i])
                        else:
                            # constraints for all
                            for i, object_spec in enumerate(total_pri_objects):
                                position = None
                                num_constrain_tries = 0
                                while not position:
                                    num_constrain_tries += 1
                                    position = self.render_positions_good_margin(positions, intended_size=object_spec['size'])
                                    if not position:
                                        generated_count = 0
                                        objects = list()
                                        continue
                                        
                                    total_pri_objects[i]['pos'] = (positions[-1][0], positions[-1][1])    
                                    for j, (constraint, auxiliary_object) in enumerate(zip(pri_aux_constraints, total_aux_objects)):
                                        utils.add_object(args.shape_dir, auxiliary_object['shape'], positions[-1][2], total_pri_objects[i]['pos'], theta=thetas[j+sum(num_extrinsic)])
                                        obj = bpy.context.object
                                        judgement = self.judge_direction(obj.location, blender_objects[j].location, tag=constraint)
                                        utils.delete_object(obj)
                                        if not judgement:
                                            position = None
                                            break
                                      
                                    if not judgement:
                                        continue
                                    positions.append(position)
                                    total_other_positions.append(position)
                                    
                                    generated_count += 1
                                    objects.append(total_pri_objects[i])
                            
                            # opposite_constraints = {"left": "right", "right": "left", "behind": "front", "front": "behind"}
                            for i, object_spec in enumerate(total_other_objects):
                                position = None
                                num_constrain_tries = 0
                                while not position:
                                    position = self.render_positions_good_margin(positions, intended_size=object_spec['size'])
                                    if not position:
                                        generated_count = 0
                                        objects = list()
                                        continue
                                    
                                    total_other_objects[i]['pos'] = (positions[-1][0], positions[-1][1])  
                                    for j, (constraint, auxiliary_object) in enumerate(zip(pri_aux_constraints, total_other_objects)):
                                        
                                        utils.add_object(args.shape_dir, auxiliary_object['shape'], positions[-1][2], total_other_objects[i]['pos'], theta=thetas[j+sum(num_extrinsic)+num_main])
                                        obj = bpy.context.object
                                        judgement = self.judge_direction(obj.location, blender_objects[j].location, tag=constraint)
                                        utils.delete_object(obj)
                                        if not judgement:
                                            position = None
                                            break
                                    
                                    if not judgement:
                                        continue
                                        
                                    positions.append(position)
                                    total_other_positions.append(position)
                                    
                                    generated_count += 1
                                    objects.append(total_other_objects[i])
                                    
                    while generated_count >= num_main + num_some_other + sum(num_extrinsic) and generated_count < total_count:
                        # random objects
                        # generating random objects
                        # Func_judge() -- some set differences
                        added_object = {}
                        excluded_failure = True
                        while excluded_failure:
                            for attr_type in ['size', 'color', 'material', 'shape']:
                                added_object[attr_type] = random.sample(INTRINSIC_PRIMITIVES[attr_type], 1)[0]
                                
                            excluded_failure = self.judge_attributes_partial_overlapping_failure(added_object, excluded_attributes_list=[primary_predicates] + auxiliary_predicates_list)
                            # the main reason to use primary_predicates (blue balls): not to add additional attributes "big"

                        position = None
                        while not position:
                            position = self.render_positions_good_margin(positions, intended_size=added_object['size'])
                            if not position:
                                generated_count = 0
                                objects = list()
                                continue

                            positions.append(position)
                            generated_count += 1
                            added_object['pos'] = (positions[-1][0], positions[-1][1])
                            objects.append(added_object)
                                          
            
            for i, added_object in enumerate(objects[sum(num_extrinsic):]):
                # check the visibility using blender built-in functions
                utils.add_object(args.shape_dir, added_object['shape'], positions[i+sum(num_extrinsic)][-1], added_object['pos'], thetas[i+sum(num_extrinsic)])
                utils.add_material(self.materials[added_object['material']], Color=self.colors[added_object['color']]) # list, combing and working with the add_object function.
                # add material?
                # add camera_coods?
                bobj = bpy.context.object
                blender_objects.append(bobj)
                pixel_coords = utils.get_camera_coords(camera, bobj.location)
                full_objects.append({
                    'shape': added_object['shape'],
                    'size': added_object['size'],
                    'color': added_object['color'],
                    'material': added_object['material'],
                    '3d_coords': tuple(bobj.location),
                    'rotation': thetas[i+sum(num_extrinsic)],
                    'pixel_coords': pixel_coords
                })         
            
            all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
            if not all_visible:
                for obj in blender_objects:
                    utils.delete_object(obj)
                generated_count = 0
                objects = list()
                blender_objects = list()
                full_objects = list()
                # TODO

        self.objects = full_objects
        self.blender_objects = blender_objects


def render_scene(args, spec, output_image='render.png', output_json='render_json', output_blendfile=None, output_shadeless=None):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=osp.join(BASE_DIR, './data/base_scene.blend'))

    # Load materials
    utils.load_materials(osp.join(BASE_DIR, './data/materials'))

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = output_image
    render_args.resolution_x = args.width
    render_args.resolution_y = args.height
    render_args.resolution_percentage = 100
    render_args.tile_x = args.render_tile_size
    render_args.tile_y = args.render_tile_size
    if args.use_gpu == 1:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = 'CUDA'
            bpy.context.user_preferences.system.compute_device = 'CUDA_0'
        else:
            cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
            cycles_prefs.compute_device_type = 'CUDA'

    # Some CYCLES-specific stuff
    bpy.data.worlds['World'].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = args.render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
    if args.use_gpu == 1:
        bpy.context.scene.cycles.device = 'GPU'

    # This will give ground-truth information about the scene and its objects
    scene_struct = { 'directions': {}, 'spec': spec }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    camera = bpy.data.objects['Camera']
    print("camera locations:", camera.location[0])
    print("camera locations:", camera.location[1])
    print("camera locations:", camera.location[2])

    def rand(L):
        return 2.0 * L * (random.random() - 0.5)

    # Add random jitter to camera position
    '''
    if args.camera_jitter > 0:
        for i in range(3):
          bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)
    '''
    # Fixed camera location, birds-view
    #camera.location[0] = 0 #-5
    #camera.location[1] = 0#-20
    #camera.location[2] = 15#7 #10#15
    #bpy.data.cameras[0].lens = 38

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure.
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)  # remove the key light

    # Save all six axis-aligned directions in the scene struct
    scene_struct['directions']['behind'] = tuple(plane_behind)
    scene_struct['directions']['front'] = tuple(-plane_behind)
    scene_struct['directions']['left'] = tuple(plane_left)
    scene_struct['directions']['right'] = tuple(-plane_left)
    scene_struct['directions']['above'] = tuple(plane_up)
    scene_struct['directions']['below'] = tuple(-plane_up)

    print('initialization done')

    # Add random jitter to lamp positions
    # if args.key_light_jitter > 0:
    #     for i in range(3):
    #         bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
    # if args.back_light_jitter > 0:
    #     for i in range(3):
    #         bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
    # if args.fill_light_jitter > 0:
    #     for i in range(3):
    #         bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

    # Now make some random objects
    builder = SceneBuilder(osp.join(BASE_DIR, './data/properties.json'), osp.join(BASE_DIR, './data/shapes'))
    builder.build(spec)
    # build(self, args, specified_formal_language: str, group: str = "small", group_tag: str = "Intrinsic"):

    print('scene builder done')

    # Render the scene and dump the scene data structure
    # scene_struct['objects'] = objects
    # scene_struct['relationships'] = compute_all_relationships(scene_struct)

    while True:
        try:
            bpy.ops.render.render(write_still=True)
            break
        except Exception as e:
            print(e)

    while output_shadeless is not None:
        try:
            render_shadeless(builder, output_shadeless)
            break
        except Exception as e:
            print(e)

    if output_json is not None:
        with open(output_json, 'w') as f:
            json.dump(scene_struct, f, indent=2)

    if output_blendfile is not None:
        bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)


def render_shadeless(builder, output_path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = output_path
    render_args.engine = 'BLENDER_RENDER'
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
    utils.set_layer(bpy.data.objects['Ground'], 2)

    # Add random shadeless materials to all objects
    old_materials = []

    assert len(builder.blender_objects) <= 24
    for i, obj in enumerate(builder.blender_objects):
        old_materials.append(obj.data.materials[0])
        bpy.ops.material.new()
        mat = bpy.data.materials['Material']
        mat.name = 'Material_%d' % i
        # r, g, b = 0, (i + 1) * 10 / 255, (i + 1) * 10 / 255
        r, g, b = (i * 5 + 128) / 255, (i * 5 + 128) / 255, (i * 5 + 128) / 255
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, builder.blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects['Lamp_Key'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Fill'], 0)
    utils.set_layer(bpy.data.objects['Lamp_Back'], 0)
    utils.set_layer(bpy.data.objects['Ground'], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

def main(args):
    
    
if __name__ == '__main__':
    if INSIDE_BLENDER:
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

