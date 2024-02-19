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


class ObjectBuilder(object):
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

    def render_positions_good_margin(self, current_object_positions: list[tuple], intended_size: str = "small"):
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
        return (x,y,z)
    
    def parse_specified_formal_language(self, specified_formal_language: str):
        intrinsic_addition_predicates = [] # List, "big", "blue" 
        extrinsic_predicates = [] # List[Tuple] (on the left of, blue cubes)
        
        intrinsic_addition_types = [] # List, "size", "color"
        extrinsic_predicate_intrinsic_types = [] # List[List], List[["color", "size"]]
        
        if "some" in specified_formal_language.lower() or "all" in specified_formal_language.lower():
            primary_object = specified_formal_language.split(' are')[0]
            predicates = specified_formal_language.split(' are ')[1].split(" and ")
            for predicate in predicates:
                if 'cube' in predicate or 'sphere' in predicate or 'cylinder' in predicate or 'object' in predicate:
                    extrinsic_predicates.append(("the ".join(predicate.split('the ')[:-1]).strip(' '), predicate.split('the ')[-1]))
                    current_extrinsic_predicate_intrinsic_types = []
                    for modifier_unit in predicate.split('the ')[-1].split(" "):
                        if 'object' in modifier_unit:
                            continue
                        else:
                            current_extrinsic_predicate_intrinsic_types.append(self.intrinsic_attribute_value2key[modifier])
                    extrinsic_predicate_intrinsic_types.append(current_extrinsic_predicate_intrinsic_types)
                else:
                    # blue
                    intrinsic_addition_predicates.append(predicate)
                    intrinsic_addition_types.append(self.intrinsic_attribute_value2key[predicate])
        else:
            # disconnective implicatures
            
            
                    
        return primary_object, extrinsic_predicates, extrinsic_predicate_intrinsic_types, intrinsic_addition_predicates, intrinsic_addition_types 
            
            
        
    def build(self, specified_formal_language: str, group: str = "small", group_tag: str = "Intrinsic"):
        assert self.blender_objects is None, 'Build can only be called once.'
        assert group in ['small', 'large']
        assert group_tag in ['Intrinsic', "Extrinsic", "Integration"], "It could only be three choices, one is Intrinsic, the other is Extrinsic, while another one is the Integration."
        
        # Demo:
        # Some/ All (Not all) black balls are big [and]
        # Some/ All balls are <on the left of>/ <the same color with> the cubes [and].
        # Some/ All balls are big [and] <on the left of> the black objects. 
        # Some/ All objects are big and on the left of the cubes
        # Some objects are big blue and on the left of the cubes. 
        
        # Some/ All balls are <on the left of> some/ all cubes.
        # No balls are <on the left of> some/all cubes. 
        # Exactly one ball is <on the left of> some cubes.
        
        # Demo:
        # Red balls are big or Red cubes are small.    
       
        camera = bpy.data.objects['Camera']
        # parsing the formal natural language
        primary_object, extrinsic_predicates, extrinsic_predicate_intrinsic_types, intrinsic_addition_predicates, intrinsic_addition_types = self.parse_specified_formal_language(specified_formal_language)
        
        assert len(intrinsic_addition_predicates) == len(intrinsic_addition_types)
        
        quantifier=True if 'some' in specified_formal_language.lower() or "all" in specified_formal_language.lower() else False
        primary_predicates = []
        primary_intrinsic_types = []
        if quantifier: 
            if primary_object.split(" ")[1] == "objects":
                pass
            else:
                for modifier in primary_object.split(" ")[1:-1]:
                    primary_predicates.append(modifier)
                    primary_intrinsic_types.append(self.intrinsic_attribute_value2key[modifier])          
        
        objects = list()
        blender_objects = list()
        positions = list()
        
        if group == 'small':
            count = random.sample([3,4],1)[0]
            num_main = int(np.random.choice(count-1, 1)) + 1 #
        else:
            count = random.sample([5,6,7,8,9,10],1)[0]
            num_main = int(np.random.choice(count-1, 1)) + 1 #
        # random introduced objects
        num_random = count - num_main
        random_included_intrinsic_primitives = copy.deepcopy(INTRINSIC_PRIMITIVES)
        for i, exclude_attr_type in excluded_random_attr_types:
        
            
            
        
        
        
        # use our own object specifications
        for obj in spec['objects']:
            x, y = obj['pos']
            r = self.sizes[obj['size']] #/ 2
            theta = 360.0 * random.random()
            if obj['shape'] == 'cube':
                r /= math.sqrt(2)

            utils.add_object(self.shape_dir, self.shapes[obj['shape']], r, (x, y), theta=theta)
            utils.add_material(self.materials[obj['material']], Color=self.colors[obj['color']])

            bobj = bpy.context.object
            pixel_coords = utils.get_camera_coords(camera, bobj.location)

            objects.append({
                'shape': obj['shape'],
                'size': obj['size'],
                'color': obj['color'],
                'material': obj['material'],
                '3d_coords': tuple(bobj.location),
                'rotation': theta,
                'pixel_coords': pixel_coords
            })
            blender_objects.append(bobj)
            positions.append((x, y, r))

        self.objects = objects
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

