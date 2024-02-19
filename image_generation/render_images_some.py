# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
import copy
import itertools
import numpy as np
import shutil
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
  from mathutils import Vector
except ImportError as e:
  INSIDE_BLENDER = False
if INSIDE_BLENDER:
  try:
    import utils
  except ImportError as e:
    print("\nERROR")
    print("Running render_images.py from Blender and cannot import utils.py.") 
    print("You may need to add a .pth file to the site-packages of Blender's")
    print("bundled python with a command like this:\n")
    print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
    print("\nWhere $BLENDER is the directory where Blender is installed, and")
    print("$VERSION is your Blender version (such as 2.78).")
    sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument('--base_scene_blendfile', default='data/base_scene.blend',
    help="Base blender file on which all scenes are based; includes " +
          "ground plane, lights, and camera.")
parser.add_argument('--properties_json', default='data/properties.json',
    help="JSON file defining objects, materials, sizes, and colors. " +
         "The \"colors\" field maps from CLEVR color names to RGB values; " +
         "The \"sizes\" field maps from CLEVR size names to scalars used to " +
         "rescale object models; the \"materials\" and \"shapes\" fields map " +
         "from CLEVR material and shape names to .blend files in the " +
         "--object_material_dir and --shape_dir directories respectively.")
parser.add_argument('--shape_dir', default='data/shapes',
    help="Directory where .blend files for object models are stored")
parser.add_argument('--material_dir', default='data/materials',
    help="Directory where .blend files for materials are stored")
parser.add_argument('--shape_color_combos_json', default=None,
    help="Optional path to a JSON file mapping shape names to a list of " +
         "allowed color names for that shape. This allows rendering images " +
         "for CLEVR-CoGenT.")
parser.add_argument('--object_ud', default="objects")
parser.add_argument('--modifier', default="red")
parser.add_argument('--implicature_type', default="direct")
# Settings for objects
parser.add_argument('--min_objects', default=3, type=int,
    help="The minimum number of objects to place in each scene")
parser.add_argument('--max_objects', default=10, type=int,
    help="The maximum number of objects to place in each scene")
parser.add_argument('--min_dist', default=0.25, type=float,
    help="The minimum allowed distance between object centers")
parser.add_argument('--margin', default=0.4, type=float,
    help="Along all cardinal directions (left, right, front, back), all " +
         "objects will be at least this distance apart. This makes resolving " +
         "spatial relationships slightly less ambiguous.")
parser.add_argument('--min_pixels_per_object', default=200, type=int,
    help="All objects will have at least this many visible pixels in the " +
         "final rendered images; this ensures that no objects are fully " +
         "occluded by other objects.")
parser.add_argument('--max_retries', default=50, type=int,
    help="The number of times to try placing an object before giving up and " +
         "re-placing all objects in the scene.")

# Output settings
parser.add_argument('--start_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--file_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--num_images', default=5, type=int,
    help="The number of images to render")
parser.add_argument('--filename_prefix', default='CLEVR',
    help="This prefix will be prepended to the rendered images and JSON scenes")
parser.add_argument('--split', default='new',
    help="Name of the split for which we are rendering. This will be added to " +
         "the names of rendered images, and will also be stored in the JSON " +
         "scene structure for each image.")
parser.add_argument('--start_game_idx', default=0, type=int,
    help="The index at which to start for numbering rendered images. Setting " +
         "this to non-zero values allows you to distribute rendering across " +
         "multiple machines and recombine the results later.")
parser.add_argument('--output_image_dir', default='../output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='../output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='../output/CLEVR_scenes.json',
    help="Path to write a single JSON file containing all scene information")
parser.add_argument('--output_blend_dir', default='output/blendfiles',
    help="The directory where blender scene files will be stored, if the " +
         "user requested that these files be saved using the " +
         "--save_blendfiles flag; in this case it will be created if it does " +
         "not already exist.")
parser.add_argument('--save_blendfiles', type=int, default=0,
    help="Setting --save_blendfiles 1 will cause the blender scene file for " +
         "each generated image to be stored in the directory specified by " +
         "the --output_blend_dir flag. These files are not saved by default " +
         "because they take up ~5-10MB each.")
parser.add_argument('--version', default='1.0',
    help="String to store in the \"version\" field of the generated JSON file")
parser.add_argument('--image_template', type=int, default=0,
    help="Image Template String Name")
parser.add_argument('--json_template', default='ad_hoc_0.json',
    help="Json Template String Name")
parser.add_argument('--license',
    default="Creative Commons Attribution (CC-BY 4.0)",
    help="String to store in the \"license\" field of the generated JSON file")
parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
    help="String to store in the \"date\" field of the generated JSON file; " +
         "defaults to today's date")

# Rendering options
parser.add_argument('--use_gpu', default=0, type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. " +
         "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
         "to work.")
parser.add_argument('--width', default=480, type=int,
    help="The width (in pixels) for the rendered images")
parser.add_argument('--height', default=320, type=int,
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
parser.add_argument('--quantifier_implicture', default=True, type=bool)


INTRINSIC_PRIMITIVES = {"size": ['small', 'large'], "color": ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"], "material": ["rubber", "metal"], "shape": ["cube", "sphere", "cylinder"]}

EXTRINSIC_PRIMITIVES = {"Relation": ["left", "right", "behind", "front"]}

INVERSE_INTRINSIC_PRIMITIVES = {}
for key in INTRINSIC_PRIMITIVES.keys():
  for ele in INTRINSIC_PRIMITIVES[key]:
    INVERSE_INTRINSIC_PRIMITIVES[ele] = key

def parse_scene(scene_dict, img_template: str):
  # return the parsed sentences from the dict
  objects = scene_dict["objects"]
  relationships = scene_dict["relationships"]
  len_objects = len(objects)
  # object attributed descriptions
  objd_lists = []
  objds = []
  for obj in objects:
    objd = [obj['size'], obj['color'], obj['material'], obj['shape']]
    objd_lists.append(objd)
    objds.append(" ".join(objd))
  # 1. image-level description
  image_some2ids = {}
  image_all = []
  image_2ids = {}

  
  for i, objd_list in enumerate(objd_lists):
    # per object processing
    ## single item
    for attr in objd_list:
      count = 0
      obj_inds = []
      # Traverse 
      for j, objd in enumerate(objds):
        if attr in objd:
          count += 1
          obj_inds.append(j)
      if count != len_objects:
        image_some2ids[attr] = obj_inds
      else:
        image_all.append(attr)
      image_2ids[attr] = obj_inds
    ## double items
    comb1d_lists = []
    comb1ds = []
    comb2d_lists = []
    comb2ds = []
    comb3d_lists = []
    comb3ds = []

    for comb_i, comb_j in itertools.combinations(range(len(objd_list)+1), 2):
      if len(objd_list[comb_i:comb_j]) == 1:
        comb1d_lists.append(set(objd_list[comb_i:comb_j]))
        comb1ds.append(" ".join(objd_list[comb_i:comb_j]))
      elif len(objd_list[comb_i:comb_j]) == 2:
        comb2d_lists.append(set(objd_list[comb_i:comb_j]))
        comb2ds.append(" ".join(objd_list[comb_i:comb_j]))
      elif len(objd_list[comb_i:comb_j]) == 3:
        comb3d_lists.append(set(objd_list[comb_i:comb_j]))
        comb3ds.append(" ".join(objd_list[comb_i:comb_j]))
    #comb2d_lists = [set(_) for _ in itertools.combinations(objd_list, 2)]
    #comb2ds = [" ".join(list(_)) for _ in comb2d_lists]
    for k, comb2d_set in enumerate(comb2d_lists):
      # (red, ball)
      count = 0
      obj_inds = []
      for j, new_objd_list in enumerate(objd_lists):
        if len(set(new_objd_list) - comb2d_set) == 2:
          count += 1
          obj_inds.append(j)
          
      if count != len_objects:
        image_some2ids[comb2ds[k]] = obj_inds
      else:
        image_all.append(comb2ds[k])
      image_2ids[comb2ds[k]] = obj_inds
    ## triple items
    #comb3d_lists = [set(_) for _ in itertools.combinations(objd_list, 3)]
    #comb3ds = [" ".join(list(_)) for _ in comb3d_lists]
    for k, comb3d_set in enumerate(comb3d_lists):
      count = 0
      obj_inds = []
      for j, new_objd_list in enumerate(objd_lists):
        if len(set(new_objd_list) - comb3d_set) == 1:
          count += 1
          obj_inds.append(j)

      if count != len_objects:
        image_some2ids[comb3ds[k]] = obj_inds
      else:
        image_all.append(comb3ds[k])
      image_2ids[comb3ds[k]] = obj_inds
    ## Full items
    count = 0
    obj_inds = []
    for j, new_objd_list in enumerate(objd_lists):
      if new_objd_list == objd_list:
        count += 1
        obj_inds.append(j)

    if count != len_objects:
      image_some2ids[objds[i]] = obj_inds
    else:
      image_all.append(objds[i])
    image_2ids[objds[i]] = obj_inds
  print(image_2ids.keys())
  # 2. extrinsic, object-level
  ## tree, node structure
  objects_structs = []
  for i, objd in enumerate(objds):
    # "big red rubber ball"
    ## Full items
    objd_list = objd_lists[i]
    full_current_objd = {"obj":objd, "id": i, "left_names": [], "left_child": [], "right_child": [], "behind_child": [], "front_child": []}
    ### left_child: ["some balls"]
    available_primary_names = full_current_objd['left_names']
    available_primary_names.append("some objects")
    # first, all some just by object-level
    # some => all, left
    # 1. right, some => all
    # 2. left, some ==> all
    # generating all possible names, where objd is 4D
    comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objd_list, 1)]
    comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objd_list, 2)]
    comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objd_list, 3)]
    
    for comb_i, comb_j in itertools.combinations(range(len(objd_list)+1), 2):
      if len(objd_list[comb_i:comb_j]) == 1:
        comb1d_lists.append(" ".join(objd_list[comb_i:comb_j]))
        #comb1ds.append(" ".join(objd_list[comb_i:comb_j]))
      elif len(objd_list[comb_i:comb_j]) == 2:
        comb2d_lists.append(" ".join(objd_list[comb_i:comb_j]))#set(stuff[i:j]))
        #comb2ds.append(" ".join(objd_list[comb_i:comb_j]))
      
      elif len(objd_list[comb_i:comb_j]) == 3:
        comb3d_lists.append(" ".join(objd_list[comb_i:comb_j]))#set(stuff[i:j]))
        #comb3ds.append(" ".join(objd_list[comb_i:comb_j]))
    ## 1. first round
    '''
    if objd in image_some2ids:
      if len(image_some2ids[objd]) == 1:
        # "all"
        available_primary_names.append("Exactly one all "+objd)
      else:
        available_primary_names.append("some "+objd)
    else:
      available_primary_names.append("all "+objd)
    '''
    for comb1d in comb1d_lists:
      if comb1d in image_some2ids:
        if len(image_some2ids[comb1d]) == 1:
          available_primary_names.append("Exactly one all "+comb1d)
        else:
          available_primary_names.append("some "+comb1d)
      else:
        available_primary_names.append("all "+comb1d)
    '''
    for comb2d in comb2d_lists:
      if comb2d in image_some2ids:
        if len(image_some2ids[comb2d]) == 1:
          available_primary_names.append("Exactly one all "+comb2d)
        else:
          available_primary_names.append("some "+comb2d)
      else:
        available_primary_names.append("all "+comb2d)
    
    for comb3d in comb3d_lists:
      if comb3d in image_some2ids:
        if len(image_some2ids[comb3d]) == 1:
          available_primary_names.append("Exactly one all "+comb3d)
        else:
          available_primary_names.append("some "+comb3d)
      else:
        available_primary_names.append("all "+comb3d)
    '''
    ## 2. map the relationships from ids to full names
    left_rel = relationships['left'][i]
    right_rel = relationships['right'][i]
    front_rel = relationships['front'][i]
    behind_rel = relationships['behind'][i]
    # per relation processing
    for rel_obj_ind in left_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['left_child'].append(rel_objd_list)

    for rel_obj_ind in right_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['right_child'].append(rel_objd_list)

    for rel_obj_ind in front_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['front_child'].append(rel_objd_list)

    for rel_obj_ind in behind_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['behind_child'].append(rel_objd_list)

    objects_structs.append(full_current_objd)
  ## adjust the right names
  for i, object_struct in enumerate(objects_structs):
    # object-centric per relation processing
    object_struct['left'] = []
    object_struct['right'] = []
    object_struct['front'] = []
    object_struct['behind'] = []
    
    left_objects = object_struct["left_child"]
    left_object_ids = relationships['left'][i]

    right_objects = object_struct["right_child"]
    right_object_ids = relationships['right'][i]

    front_objects = object_struct["front_child"]
    front_object_ids = relationships['front'][i]

    behind_objects = object_struct["behind_child"]
    behind_object_ids = relationships['behind'][i]
    # process left
    for objectd_list in left_objects:
      # "big red rubber ball"
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)+1), 2):
        if len(objectd_list[comb_i:comb_j]) == 1:
          comb1d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 2:
          comb2d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 3:
          comb3d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 4:
          comb4d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
      '''
      # 4d
      for comb4d in comb4d_lists:
        # comb
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(left_object_ids)) > 0:
          #
          object_struct['left'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['left'].append("all "+comb4d)
      # 3d
      for comb3d in comb3d_lists:
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(left_object_ids)) > 0:
          object_struct['left'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb3d)
        else:
          object_struct['left'].append("all "+comb3d)
      
      # 2d
      for comb2d in comb2d_lists:
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(left_object_ids)) > 0:
          object_struct['left'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb2d)
        else:
          object_struct['left'].append("all "+comb2d)
      '''
      # 1d
      for comb1d in comb1d_lists:
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(left_object_ids)) > 0:
          object_struct['left'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb1d)
        else:
          object_struct['left'].append("all "+comb1d)
    # process right
    for objectd_list in right_objects:
      #comb1d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      #comb2d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      #comb3d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      #comb4d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)+1), 2):
        if len(objectd_list[comb_i:comb_j]) == 1:
          comb1d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 2:
          comb2d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 3:
          comb3d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 4:
          comb4d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
      '''
      # 4d
      for comb4d in comb4d_lists:
        # comb
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(right_object_ids)) > 0:
          #
          object_struct['right'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['right'].append("all "+comb4d)
      # 3d
      for comb3d in comb3d_lists:
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(right_object_ids)) > 0:
          object_struct['right'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb3d)
        else:
          object_struct['right'].append("all "+comb3d)
      
      # 2d
      for comb2d in comb2d_lists:
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(right_object_ids)) > 0:
          object_struct['right'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb2d)
        else:
          object_struct['right'].append("all "+comb2d)
      '''
      # 1d
      for comb1d in comb1d_lists:
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(right_object_ids)) > 0:
          object_struct['right'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb1d)
        else:
          object_struct['right'].append("all "+comb1d)
    # process front
    for objectd_list in front_objects:
      #comb1d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      #comb2d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      #comb3d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      #comb4d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)+1), 2):
        if len(objectd_list[comb_i:comb_j]) == 1:
          comb1d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 2:
          comb2d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 3:
          comb3d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 4:
          comb4d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
      '''
      # 4d
      for comb4d in comb4d_lists:
        # comb
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(front_object_ids)) > 0:
          #
          object_struct['front'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['front'].append("all "+comb4d)
      
      # 3d
      for comb3d in comb3d_lists:
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(front_object_ids)) > 0:
          object_struct['front'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb3d)
        else:
          object_struct['front'].append("all "+comb3d)
      
      # 2d
      for comb2d in comb2d_lists:
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(front_object_ids)) > 0:
          object_struct['front'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb2d)
        else:
          object_struct['front'].append("all "+comb2d)
      '''
      # 1d
      for comb1d in comb1d_lists:
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(front_object_ids)) > 0:
          object_struct['front'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb1d)
        else:
          object_struct['front'].append("all "+comb1d)
          
    # process behind
    for objectd_list in behind_objects:
      #comb1d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      #comb2d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      #comb3d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      #comb4d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)+1), 2):
        if len(objectd_list[comb_i:comb_j]) == 1:
          comb1d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 2:
          comb2d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 3:
          comb3d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
        elif len(objectd_list[comb_i:comb_j]) == 4:
          comb4d_lists.append(" ".join(objectd_list[comb_i:comb_j]))
      '''
      # 4d
      for comb4d in comb4d_lists:
        # comb
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          #
          object_struct['behind'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['behind'].append("all "+comb4d)
      
      # 3d
      for comb3d in comb3d_lists:
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          object_struct['behind'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb3d)
        else:
          object_struct['behind'].append("all "+comb3d)
    
      # 2d
      for comb2d in comb2d_lists:
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          object_struct['behind'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb2d)
        else:
          object_struct['behind'].append("all "+comb2d)
      '''
      # 1d
      for comb1d in comb1d_lists:
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          object_struct['behind'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb1d)
        else:
          object_struct['behind'].append("all "+comb1d)
  # adjust the primary names
  utterances = []
  for i, object_struct in enumerate(objects_structs):
    # object-centric
    primary_names = object_struct["left_names"]
    for name in primary_names:
      if "some" in name:
        name_ = copy.deepcopy(name)
        name_ = name_.replace("some ", "")
        if name_ == "objects":
            continue
        primary_object_ids = set(image_2ids[name_]) # all
        
        # per relation
        ## left
        for descrip in object_struct['left']: # on the right of
          count_ids = []
          some_flag = False
          #print(descrip)
          
          if "all"  in descrip.split(" "):
            object_under_discuss_name = descrip.split("all ")[-1]
            #print(descrip)
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['right'][second_object_id])) > 0:
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['right'][second_object_id])):
                  count_ids.append(temp)
              
          elif "some" in descrip:
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['left'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)
                
          if some_flag:
            utterances.append("some "+name_+" on the right of "+descrip)
          else:
            utterances.append("all "+name_+" on the right of "+descrip)
            print("****", utterances[-1])
            
          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
            
        ## right
        for descrip in object_struct['right']:
          count_ids = []
          some_flag = False
          if "all" in descrip.split(" "):
            object_under_discuss_name = descrip.split("all ")[-1]
            #print(descrip)
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['left'][second_object_id])) > 0:
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['left'][second_object_id])):
                  count_ids.append(temp)
            
          elif "some" in descrip:
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['right'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)
                
          if some_flag:
            utterances.append("some "+name_+" on the left of "+descrip)
          else:
            utterances.append("all "+name_+" on the left of "+descrip)
            #print("****", utterances[-1])

          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
            
        ## front
        for descrip in object_struct['front']:
          count_ids = []
          some_flag = False
          if "all" in descrip.split(" "):
            object_under_discuss_name = descrip.split("all ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['behind'][second_object_id])) > 0:
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['behind'][second_object_id])):
                  count_ids.append(temp)
                  
              
          elif "some" in descrip:
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['front'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)

          if some_flag:
            utterances.append("some "+name_+" behind "+descrip)
          else:
            utterances.append("all "+name_+" behind "+descrip)
            #print("****", utterances[-1])
          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
                
        ## behind
        for descrip in object_struct['behind']:
          some_flag = False
          count_ids = []
          if "all" in descrip.split(" "):
            object_under_discuss_name = descrip.split("all ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['front'][second_object_id])) > 0:
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['front'][second_object_id])):
                  count_ids.append(temp)
              #elif not some_flag:
              #  print("???????", descrip)
              #  print(some_flag, name_)
              #  print(primary_object_ids)
              #  print(second_object_ids)
                
          elif "some" in descrip:
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['behind'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)

          if some_flag:
            utterances.append("some "+name_+" in front of "+descrip)
          else:
            utterances.append("all "+name_+" in front of "+descrip)
            #print("****", utterances[-1])
          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
      else: # all
        name_ = copy.deepcopy(name)
        name_ = name_.replace("all ", "")
        if name_ == "objects":
            continue

        primary_object_ids = set(image_2ids[name_]) # all
        ## left
        for descrip_ in object_struct['left']:
          # All balls are on the right of xxx.
          descrip = descrip_.replace("some ", "").replace("Exactly one all ", "").replace("all ", "")
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append("some " + name_ + "on the right of" + descrip)

        ## right
        for descrip_ in object_struct['right']:
          # All balls are on the left of xxx.
          descrip = descrip_.replace("some ", "").replace("Exactly one all ", "").replace("all ", "")
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append("some " + name_ + "on the left of" + descrip)

        ## front
        for descrip_ in object_struct['front']:
          # All balls are behind xxx.
          descrip = descrip_.replace("some ", "").replace("Exactly one all ", "").replace("all ", "")
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append("some " + name_ + "behind" + descrip)
          
        ## behind
        for descrip_ in object_struct['behind']:
          # All balls are on the front of xxx.
          descrip = descrip_.replace("some ", "").replace("Exactly one all ", "").replace("all ", "")
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append("some " + name_ + "on the front of" + descrip)
              
                
  # intrinsic: image_some2ids.keys()/ image_all
  some = image_some2ids.keys()
  intri_dict = {"some": list(some), "all": list(image_all)}
  with open("../output/"+img_template.split(".")[0]+"_intrinsic.json", "w") as f:
    json.dump(intri_dict, f)
  # extrinsic: utterances
  print("The number of utterances:", len(utterances))
  for utterance in list(set(utterances)):
    if not os.path.exists("../output/utterances/"):
        os.mkdir("../output/utterances/")
    with open("../output/utterances/"+utterance+".txt", "a") as f:
      f.write(img_template)
  
def notmain(args):
  num_digits = 1
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
  img_template = args.image_template#'%s%%0%dd.png' % (prefix, num_digits)
 
  scene_template = args.json_template#'%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  all_scene_paths = []
  for i in range(5):
    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = random.randint(args.min_objects, args.max_objects)
    render_scene(args,
      num_objects=10,#num_objects,
      output_index=i,#(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
    )
    
  # After rendering all images, combine the JSON files for each scene into a
  # single JSON file.
  all_scenes = []
  for scene_path in all_scene_paths:
    with open(scene_path, 'r') as f:
      all_scenes.append(json.load(f))

  for i, scene_dict in enumerate(all_scenes):
    # scene
    parse_scene(scene_dict, img_template % (i + args.start_idx))
    
    
  output = {
    'info': {
      'date': args.date,
      'version': args.version,
      'split': args.split,
      'license': args.license,
    },
    'scenes': all_scenes
  }
  with open(args.output_scene_file, 'w') as f:
    json.dump(output, f)

def main(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
    
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist1_img_templare = '%s%%0%dd_dist1.png' % (prefix, num_digits)
  dist1_scene_template = '%s%%0%dd_dist1.json' % (prefix, num_digits)
  dist2_img_templare = '%s%%0%dd_dist2.png' % (prefix, num_digits)
  dist2_scene_template = '%s%%0%dd_dist2.json' % (prefix, num_digits)
  

  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  img_template = os.path.join(args.output_image_dir, img_template)
  scene_template = os.path.join(args.output_scene_dir, scene_template)
    
  dist1_img_template = os.path.join(args.output_image_dir, dist1_img_templare)
  dist1_scene_template = os.path.join(args.output_scene_dir, dist1_scene_template)
    
  dist2_img_template = os.path.join(args.output_image_dir, dist2_img_templare)
  dist2_scene_template = os.path.join(args.output_scene_dir, dist2_scene_template)
    
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  
  all_scene_paths = []
  for i in range(args.num_images):
    img_path = img_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    dist1_img_path = dist1_img_template % (i + args.start_idx)
    dist1_scene_path = dist1_scene_template % (i + args.start_idx)
    dist2_img_path = dist2_img_template % (i + args.start_idx)
    dist2_scene_path = dist2_scene_template % (i + args.start_idx)
    
    all_scene_paths.append(scene_path)
    #blend_path = None
    #if args.save_blendfiles == 1:
    blend_path = None#blend_template % (i + args.start_idx)
    num_objects = random.randint(4,8)
    obj_ind, utterance = render_some_intrinsic_scene(args,
      num_objects=num_objects,#num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
      object_ud="yellow objects",
      modifier="metal",
      indirect=True,
    )
    num_objects = random.randint(4,8)
    obj_ind, utterance = render_all_intrinsic_scene(args,
      num_objects=num_objects,#num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=dist2_img_path,
      output_scene=dist2_scene_path,
      output_blendfile=blend_path,
      object_ud="yellow objects",
      modifier="metal"
    )
    num_objects = random.randint(4,8)
    obj_ind, utterance = render_no_intrinsic_scene(args,
      num_objects=num_objects,#num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=dist1_img_path,
      output_scene=dist1_scene_path,
      output_blendfile=blend_path,
      object_ud="yellow objects",
      modifier="metal"
    )
    
def propose_name():
    object_ud_shape = random.sample(INTRINSIC_PRIMITIVES["shape"], 1)[0]
    
    another_object_ud_shape = random.sample(INTRINSIC_PRIMITIVES["shape"], 1)[0]
    if another_object_ud_shape == object_ud_shape:
        size_flag = random.sample([0, 1],1)[0]
        color_flag = random.sample([0, 1],1)[0]
        material_flag = random.sample([0, 1],1)[0]

        #another_size_flag = random.sample([0, 1],1)[0]
        #another_color_flag = random.sample([0, 1],1)[0]
        #another_material_flag = random.sample([0, 1],1)[0]

        obj_modifer = []
        another_obj_modifer = []
        if size_flag:
            obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["size"], 1)[0])
            temp = random.sample(INTRINSIC_PRIMITIVES["size"], 1)[0]
            while temp == obj_modifer[-1]:
                temp = random.sample(INTRINSIC_PRIMITIVES["size"], 1)[0]
            another_obj_modifer.append(temp)

        if color_flag:
            obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["color"], 1)[0])
            temp = random.sample(INTRINSIC_PRIMITIVES["color"], 1)[0]
            while temp == obj_modifer[-1]:
                temp = random.sample(INTRINSIC_PRIMITIVES["color"], 1)[0]
            another_obj_modifer.append(temp)

        if material_flag:
            obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["material"], 1)[0])
            temp = random.sample(INTRINSIC_PRIMITIVES["material"], 1)[0]
            while temp == obj_modifer[-1]:
                temp = random.sample(INTRINSIC_PRIMITIVES["material"], 1)[0]
            another_obj_modifer.append(temp)

    else:
        size_flag = random.sample([0, 1],1)[0]
        color_flag = random.sample([0, 1],1)[0]
        material_flag = random.sample([0, 1],1)[0]
        another_size_flag = random.sample([0, 1],1)[0]
        another_color_flag = random.sample([0, 1],1)[0]
        another_material_flag = random.sample([0, 1],1)[0]

        obj_modifer = []
        another_obj_modifer = []
        if size_flag:
            obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["size"], 1)[0])
        if color_flag:
            obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["color"], 1)[0])
        if material_flag:
            obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["material"], 1)[0])
      
        if another_size_flag:
            another_obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["size"], 1)[0])
        if another_color_flag:
            another_obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["color"], 1)[0])
        if another_material_flag:
            another_obj_modifer.append(random.sample(INTRINSIC_PRIMITIVES["material"], 1)[0])

    main_obj = " ".join(obj_modifer)+" "+object_ud_shape
    aux_obj = " ".join(another_obj_modifer)+" "+another_object_ud_shape
      
        
    if len(obj_modifer) == 0:
        main_obj = object_ud_shape
    if len(another_obj_modifer) == 0:
        aux_obj = another_object_ud_shape
        
    while main_obj == aux_obj:
        main_obj, aux_obj = propose_name()

    return main_obj, aux_obj   
    
def main_connective_intrinsic(args):
  num_digits = 6
  if args.implicature_type == 'direct':
    indirect = False
  else:
    indirect = True
  prefix = '%s_' % (args.filename_prefix)
    
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  jsonl_path = '%s%%0%dd.jsonl' % (prefix, num_digits)

  mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    
  dist1_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  dist1_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist1_mask_template = '%s%%0%dd.png' % (prefix, num_digits)

    
  dist2_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  dist2_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist2_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    
  another_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  another_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  another_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    
  dist3_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  dist3_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist3_mask_template = '%s%%0%dd.png' % (prefix, num_digits)

  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)

  if not os.path.exists(os.path.join(args.output_image_dir, "connectives")):
      os.mkdir(os.path.join(args.output_image_dir, "connectives"))
  if not os.path.exists(os.path.join(args.output_scene_dir, "connectives")):
      os.mkdir(os.path.join(args.output_scene_dir, "connectives"))
  if not os.path.exists(os.path.join(args.output_image_dir, "connectives", "images")):
      os.mkdir(os.path.join(args.output_image_dir, "connectives", "images"))
  if not os.path.exists(os.path.join(args.output_image_dir, "connectives", "masks")):
      os.mkdir(os.path.join(args.output_image_dir, "connectives", "masks"))
  if not os.path.exists(os.path.join(args.output_scene_dir, "connectives", "meta_data")):
      os.mkdir(os.path.join(args.output_scene_dir, "connectives", "meta_data"))
  if not os.path.exists(os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons")):
      os.mkdir(os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons"))

  img_template = os.path.join(args.output_image_dir, "connectives", "images", img_template)
  scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", scene_template)
  jsonl_path= os.path.join(args.output_scene_dir, "connectives", "meta_data", jsonl_path)
  mask_template = os.path.join(args.output_image_dir, "connectives", "masks", mask_template)
    
  another_img_template = os.path.join(args.output_image_dir, "connectives", "images", another_img_template)
  another_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", another_scene_template)
  another_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", another_mask_template)
    
  dist1_img_template = os.path.join(args.output_image_dir, "connectives", "images", dist1_img_template)
  dist1_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", dist1_scene_template)
  dist1_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", dist1_mask_template)
    
  dist2_img_template = os.path.join(args.output_image_dir, "connectives", "images", dist2_img_template)
  dist2_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", dist2_scene_template)
  dist2_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", dist2_mask_template)
  
  dist3_img_template = os.path.join(args.output_image_dir, "connectives", "images", dist3_img_template)
  dist3_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", dist3_scene_template)
  dist3_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", dist3_mask_template)
  
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  all_scene_paths = []

  for i in range(args.num_images):
    count = 2 
    main_obj, aux_obj = propose_name()
    print(main_obj, aux_obj)
    
    img_path = img_template % (5 * i + 5 * args.start_idx)
    scene_path = scene_template % (5 * i + 5 * args.start_idx)
    jsonl_path_ = jsonl_path % (i + args.start_idx)
    
    img_path_base = img_path.split('/')[-1]
    
    mask_path = mask_template % (5 * i + 5 * args.start_idx)
    
    another_img_path = another_img_template % (5 * i + 5 * args.start_idx + 1)
    another_scene_path = another_scene_template % (5 * i + 5 * args.start_idx + 1)
    another_mask_path = another_mask_template % (5 * i + 5 * args.start_idx + 1)
    
    another_img_path_base = another_img_path.split('/')[-1]
    
    dist1_img_path = dist1_img_template % (5 * i + 5 * args.start_idx + 2)
    dist1_scene_path = dist1_scene_template % (5 * i + 5 * args.start_idx + 2)
    dist1_mask_path = dist1_mask_template % (5 * i + 5 * args.start_idx + 2)
    
    dist1_img_path_base = dist1_img_path.split('/')[-1]
    
    dist2_img_path = dist2_img_template % (5 * i + 5 * args.start_idx + 3)
    dist2_scene_path = dist2_scene_template % (5 * i + 5 * args.start_idx + 3)
    dist2_mask_path = dist2_mask_template % (5 * i + 5 * args.start_idx + 3)
    
    dist2_img_path_base = dist2_img_path.split('/')[-1]
    
    dist3_img_path = dist3_img_template % (5 * i + 5 * args.start_idx + 4)
    dist3_scene_path = dist3_scene_template % (5 * i + 5 * args.start_idx + 4)
    dist3_mask_path = dist3_mask_template % (5 * i + 5 * args.start_idx + 4)
    
    dist3_img_path_base = dist3_img_path.split('/')[-1]
    
    all_scene_paths.append(scene_path)
    blend_path = None
    #if args.save_blendfiles == 1:
    blend_path = None#blend_template % (i + args.start_idx)
    num_objects = random.randint(4,10)
    if os.path.exists(jsonl_path_):
      os.remove(jsonl_path_)
        
    obj_ind, utterance = render_or_connective_scene(args,
      num_objects=num_objects,#num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path,
      output_mask=mask_path, 
      object_ud=main_obj, #"metal cube",
      another_obj_ud=aux_obj, #"cylinder",
      indirect=indirect,                                   
      jsonl_path=jsonl_path_,
      flag=True,                                            
    )

    num_objects = random.randint(4,10)
    obj_ind, utterance = render_or_connective_scene(args,
      num_objects=num_objects,#num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=another_img_path,
      output_scene=another_scene_path,
      output_blendfile=blend_path,
      output_mask=another_mask_path,
      object_ud=aux_obj, #"cylinder",
      another_obj_ud=main_obj, #"metal cube",
      indirect=indirect,
      jsonl_path=jsonl_path_
      
    )
    num_all = random.sample(range(1, 3), 1)[0]
    for _ in range(num_all):
      count += 1
      temp_img_path = None
      temp_scene_path = None
      if count == 3:
        temp_img_path = dist1_img_path
        temp_scene_path = dist1_scene_path
        temp_mask_path = dist1_mask_path
      elif count == 4:
        temp_img_path = dist2_img_path
        temp_scene_path = dist2_scene_path
        temp_mask_path = dist2_mask_path
      elif count == 5:
        temp_img_path = dist3_img_path
        temp_scene_path = dist3_scene_path
        temp_mask_path = dist3_mask_path
        
      num_objects = random.randint(4,10) 
      
      obj_ind, utterance = render_and_connective_scene(args,
        num_objects=num_objects,#num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image=temp_img_path,
        output_scene=temp_scene_path,
        output_blendfile=blend_path,
        output_mask=temp_mask_path,
        object_ud=aux_obj, #"cylinder",
        another_obj_ud=main_obj, #"metal cube"
        jsonl_path=jsonl_path_
      )
    remain_num = 5 - count
    
    for _ in range(remain_num):
      count += 1
      temp_img_path = None
      temp_scene_path = None
      if count == 4:
        temp_img_path = dist2_img_path
        temp_scene_path = dist2_scene_path
        temp_mask_path = dist2_mask_path
      elif count == 5:
        temp_img_path = dist3_img_path
        temp_scene_path = dist3_scene_path
        temp_mask_path = dist3_mask_path
        
      num_objects = random.randint(4,10) 
      obj_ind, utterance = render_no_connective_scene(args,
        num_objects=num_objects,#num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image=temp_img_path,
        output_scene=temp_scene_path,
        output_blendfile=blend_path,
        output_mask=temp_mask_path,
        object_ud=aux_obj, #"cylinder",
        another_obj_ud=main_obj, #"metal cube"
        jsonl_path=jsonl_path_
      )
      
    with open(jsonl_path_, "r") as f:
      ans = list(f)
    ans = [json.loads(_) for _ in ans]
    ans[0]['referents'] = [img_path_base, another_img_path_base]
    
    with open(jsonl_path_, 'w') as f:
      for _ in ans:
        f.write(json.dumps(_))
        f.write('\n')

def main_connective_cancel_intrinsic(args):
  num_digits = 6
  if args.implicature_type == 'direct_cancel':
    indirect = False
  elif args.implicature_type == 'indirect_cancel':
    indirect = True
    
  prefix = '%s_' % (args.filename_prefix)
    
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  jsonl_path= '%s%%0%dd.jsonl' % (prefix, num_digits)
  mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    
  dist1_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  dist1_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist1_mask_template = '%s%%0%dd.png' % (prefix, num_digits)

    
  dist2_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  dist2_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist2_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    
  another_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  another_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  another_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    
  dist3_img_template = '%s%%0%dd.png' % (prefix, num_digits)
  dist3_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist3_mask_template = '%s%%0%dd.png' % (prefix, num_digits)

  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)

  if not os.path.exists(os.path.join(args.output_image_dir, "connectives")):
      os.mkdir(os.path.join(args.output_image_dir, "connectives"))
  if not os.path.exists(os.path.join(args.output_scene_dir, "connectives")):
      os.mkdir(os.path.join(args.output_scene_dir, "connectives"))
  if not os.path.exists(os.path.join(args.output_image_dir, "connectives", "images")):
      os.mkdir(os.path.join(args.output_image_dir, "connectives", "images"))
  if not os.path.exists(os.path.join(args.output_image_dir, "connectives", "masks")):
      os.mkdir(os.path.join(args.output_image_dir, "connectives", "masks"))
  if not os.path.exists(os.path.join(args.output_scene_dir, "connectives", "meta_data")):
      os.mkdir(os.path.join(args.output_scene_dir, "connectives", "meta_data"))
  if not os.path.exists(os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons")):
      os.mkdir(os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons"))

  img_template = os.path.join(args.output_image_dir, "connectives", "images", img_template)
  scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", scene_template)
  jsonl_path= os.path.join(args.output_scene_dir, "connectives", "meta_data", jsonl_path)
  mask_template = os.path.join(args.output_image_dir, "connectives", "masks", mask_template)
    
  another_img_template = os.path.join(args.output_image_dir, "connectives", "images", another_img_template)
  another_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", another_scene_template)
  another_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", another_mask_template)
    
  dist1_img_template = os.path.join(args.output_image_dir, "connectives", "images", dist1_img_template)
  dist1_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", dist1_scene_template)
  dist1_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", dist1_mask_template)
    
  dist2_img_template = os.path.join(args.output_image_dir, "connectives", "images", dist2_img_template)
  dist2_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", dist2_scene_template)
  dist2_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", dist2_mask_template)
  
  dist3_img_template = os.path.join(args.output_image_dir, "connectives", "images", dist3_img_template)
  dist3_scene_template = os.path.join(args.output_scene_dir, "connectives", "meta_data_jsons", dist3_scene_template)
  dist3_mask_template = os.path.join(args.output_image_dir, "connectives", "masks", dist3_mask_template)
    
  blend_template = os.path.join(args.output_blend_dir, blend_template)

  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)

  all_scene_paths = []

  for i in range(args.num_images):
    count = 2 
    main_obj, aux_obj = propose_name()
    print(main_obj, aux_obj)
    
    img_path = img_template % (5 * i + 5 * args.start_idx)
    scene_path = scene_template % (5 * i + 5 * args.start_idx)
    jsonl_path_ = jsonl_path % (i + args.start_idx)
    
    mask_path = mask_template % (5 * i + 5 * args.start_idx)
    
    img_path_base = img_path.split('/')[-1]
    
    another_img_path = another_img_template % (5 * i + 5 * args.start_idx + 1)
    another_scene_path = another_scene_template % (5 * i + 5 * args.start_idx + 1)
    another_mask_path = another_mask_template % (5 * i + 5 * args.start_idx + 1)
    
    another_img_path_base = another_img_path.split('/')[-1]
    
    dist1_img_path = dist1_img_template % (5 * i + 5 * args.start_idx + 2)
    dist1_scene_path = dist1_scene_template % (5 * i + 5 * args.start_idx + 2)
    dist1_mask_path = dist1_mask_template % (5 * i + 5 * args.start_idx + 2)
    
    dist2_img_path = dist2_img_template % (5 * i + 5 * args.start_idx + 3)
    dist2_scene_path = dist2_scene_template % (5 * i + 5 * args.start_idx + 3)
    dist2_mask_path = dist2_mask_template % (5 * i + 5 * args.start_idx + 3)
    
    dist3_img_path = dist3_img_template % (5 * i + 5 * args.start_idx + 4)
    dist3_scene_path = dist3_scene_template % (5 * i + 5 * args.start_idx + 4)
    dist3_mask_path = dist3_mask_template % (5 * i + 5 * args.start_idx + 4)
    
    all_scene_paths.append(scene_path)
    blend_path = None
    #if args.save_blendfiles == 1:
    blend_path = None#blend_template % (i + args.start_idx)
    if os.path.exists(jsonl_path_):
      os.remove(jsonl_path_)
    
    seed = random.sample([0,1,2], 1)[0]
    if not indirect:
      if seed == 0:  
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_or_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path,
          output_scene=scene_path,
          output_blendfile=blend_path,
          output_mask=mask_path, 
          object_ud=main_obj, #"metal cube",
          another_obj_ud=aux_obj, #"cylinder",
          indirect=indirect,                                   
          jsonl_path=jsonl_path_,
          flag=True,
          cancel=True,                                            
        )
        #with open(referent_path, "w") as f:
        #    f.write(main_obj+" or "+aux_obj)

        num_objects = random.randint(4,10)
        obj_ind, utterance = render_and_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=another_img_path,
          output_scene=another_scene_path,
          output_blendfile=blend_path,
          output_mask=another_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_
        )
      elif seed == 1:
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_or_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path,
          output_scene=scene_path,
          output_blendfile=blend_path,
          output_mask=mask_path, 
          object_ud=aux_obj, #"metal cube",
          another_obj_ud=main_obj, #"cylinder",
          indirect=indirect,                                   
          jsonl_path=jsonl_path_,
          flag=True,
          cancel=True,                                          
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_and_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=another_img_path,
          output_scene=another_scene_path,
          output_blendfile=blend_path,
          output_mask=another_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_
        )
      else:
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_and_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path,
          output_scene=scene_path,
          output_blendfile=blend_path,
          output_mask=mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_,
          first_cancel=True,
      
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_and_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=another_img_path,
          output_scene=another_scene_path,
          output_blendfile=blend_path,
          output_mask=another_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_
        )

      for _ in range(3):
        if _ == 0:
          temp_img_path = dist1_img_path
          temp_scene_path = dist1_scene_path
          temp_mask_path = dist1_mask_path
        elif _ == 1:
          temp_img_path = dist2_img_path
          temp_scene_path = dist2_scene_path
          temp_mask_path = dist2_mask_path
        else:
          temp_img_path = dist3_img_path
          temp_scene_path = dist3_scene_path
          temp_mask_path = dist3_mask_path
        
        num_objects = random.randint(4,10) 
        obj_ind, utterance = render_no_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=temp_img_path,
          output_scene=temp_scene_path,
          output_blendfile=blend_path,
          output_mask=temp_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube"
          jsonl_path=jsonl_path_
        )
    else:
      if seed == 0:  
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_or_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path,
          output_scene=scene_path,
          output_blendfile=blend_path,
          output_mask=mask_path, 
          object_ud=main_obj, #"metal cube",
          another_obj_ud=aux_obj, #"cylinder",
          indirect=indirect,                                   
          jsonl_path=jsonl_path_,
          flag=True,
          cancel=True,                                            
        )
        #with open(referent_path, "w") as f:
        #    f.write(main_obj+" or "+aux_obj)

        num_objects = random.randint(4,10)
        obj_ind, utterance = render_no_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=another_img_path,
          output_scene=another_scene_path,
          output_blendfile=blend_path,
          output_mask=another_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_
        )
      elif seed == 1:
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_or_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path,
          output_scene=scene_path,
          output_blendfile=blend_path,
          output_mask=mask_path, 
          object_ud=aux_obj, #"metal cube",
          another_obj_ud=main_obj, #"cylinder",
          indirect=indirect,                                   
          jsonl_path=jsonl_path_,
          flag=True,
          cancel=True,                                          
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_no_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=another_img_path,
          output_scene=another_scene_path,
          output_blendfile=blend_path,
          output_mask=another_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_
        )
      else:
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_no_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path,
          output_scene=scene_path,
          output_blendfile=blend_path,
          output_mask=mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_,
      
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_no_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=another_img_path,
          output_scene=another_scene_path,
          output_blendfile=blend_path,
          output_mask=another_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube",
          indirect=indirect,
          jsonl_path=jsonl_path_
        )

      for _ in range(3):
        if _ == 0:
          temp_img_path = dist1_img_path
          temp_scene_path = dist1_scene_path
          temp_mask_path = dist1_mask_path
        elif _ == 1:
          temp_img_path = dist2_img_path
          temp_scene_path = dist2_scene_path
          temp_mask_path = dist2_mask_path
        else:
          temp_img_path = dist3_img_path
          temp_scene_path = dist3_scene_path
          temp_mask_path = dist3_mask_path
        
        num_objects = random.randint(4,10) 
        obj_ind, utterance = render_and_connective_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=temp_img_path,
          output_scene=temp_scene_path,
          output_blendfile=blend_path,
          output_mask=temp_mask_path,
          object_ud=aux_obj, #"cylinder",
          another_obj_ud=main_obj, #"metal cube"
          jsonl_path=jsonl_path_
        )


        
    with open(jsonl_path_, "r") as f:
      ans = list(f)
    ans = [json.loads(_) for _ in ans]
    ans[0]['referents'] = [img_path_base, another_img_path_base]
    #with open(json_path_, "w") as f:
    with open(jsonl_path_, 'w') as f:
      for _ in ans:
        f.write(json.dumps(_))
        f.write('\n')
        
def main_some_sample_extrinsic(args):
  num_digits = 6
  prefix = '%s_%s_' % (args.filename_prefix, args.split)
    
  img_template = 'extrinsic_%s%%0%dd.png' % (prefix, num_digits)
  scene_template = 'extrinsic_%s%%0%dd.json' % (prefix, num_digits)
  
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)  
  # generated images and languages, symmetric languages
  selected_num_attr = random.sample([0,1,2,3], 1)[0]
  selected_two = [
              ["size", "color", "material"][i] for i in sorted(random.sample(range(len(["size", "color", "material"])), selected_num_attr))
              ]
  #selected_num_attr = random.sample([0,1,2,3], 1)[0]
  #selected_two = random.sample(range(3), selected_num_attr)
  #selected_two = ["size", "color", "material"][[0, 1, 2, 3][i] for i in sorted(selected_two)]
  #selected_two.sort(key = ["size", "color", "material"])
  main_obj_attr = []
  aux_obj_attr = []
  shapes = random.sample(INTRINSIC_PRIMITIVES["shape"], 2)

  for attr in selected_two:
    main_obj_attr.append(random.sample(INTRINSIC_PRIMITIVES[attr], 1)[0])
    aux_obj_attr.append(random.sample(INTRINSIC_PRIMITIVES[attr], 1)[0])
  #INTRINSIC_PRIMITIVES["color"]
  #INTRINSIC_PRIMITIVES["material"]
  #INTRINSIC_PRIMITIVES["shape"]
  main_obj_attr.append(shapes[0])
  aux_obj_attr.append(shapes[1])
  main_name = " ".join(main_obj_attr) #"small rubber cube"
  aux_name = " ".join(aux_obj_attr)  #"large metal sphere"
  main_obj = "_".join(main_name.split(" "))
  aux_obj = "_".join(aux_name.split(" "))
  # <size> <color> <material> <shape> (must have shape)
  if not os.path.isdir(args.output_image_dir):
    os.makedirs(args.output_image_dir)
  if not os.path.isdir(args.output_scene_dir):
    os.makedirs(args.output_scene_dir)
  if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
    os.makedirs(args.output_blend_dir)
  
  if os.path.exists(os.path.join(args.output_scene_dir, "%s_and_%s" % (main_obj, aux_obj))):
    return None
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj))):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj)))
    
  if not os.path.exists(os.path.join(args.output_scene_dir, "%s_and_%s" % (main_obj, aux_obj))):
    os.mkdir(os.path.join(args.output_scene_dir, "%s_and_%s" % (main_obj, aux_obj)))
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_some")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_some"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_some", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_some", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_some", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_some", "right"))
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_all")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_all"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_all", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_all", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_all", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "some_all", "right"))
    
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_all")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_all"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_all", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_all", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_all", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_all", "right"))
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_some")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_some"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_some", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_some", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_some", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "all_some", "right"))
    
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_all")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_all"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_all", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_all", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_all", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_all", "right"))
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_some")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_some"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_some", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_some", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_some", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_some_some", "right"))
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_all")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_all"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_all", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_all", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_all", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_all", "right"))
  if not os.path.exists(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_some")):
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_some"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_some", "behind"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_some", "front"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_some", "left"))
    os.mkdir(os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj), "one_all_some", "right"))
    
  img_prefix = os.path.join(args.output_image_dir, "%s_and_%s" % (main_obj, aux_obj))
  scene_prefix = os.path.join(args.output_scene_dir, "%s_and_%s" % (main_obj, aux_obj))
    
  blend_template = os.path.join(args.output_blend_dir, blend_template)
  
  for i in range(100):
    
    img_suffix_path = img_template % (i + 20 + args.start_idx)
    scene_suffix_path = scene_template % (i + 20 + args.start_idx)
    
    blend_path = None
    num_objects = random.randint(4,8)
    obj_ind, utterances = render_random_extrinsic_scene(args,
        num_objects=num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image_prefix=img_prefix, 
        output_image=os.path.join(img_prefix, img_suffix_path),
        output_scene_prefix=scene_prefix,
        output_scene=os.path.join(scene_prefix, scene_suffix_path),
        output_blendfile=None,
        main_name=main_name,
        aux_name=aux_name, # single
        one_flag = False,
      )
    for utterance in utterances:
      if "behind" in utterance:
        insert_key = "behind"
      elif "front" in utterance:
        insert_key = "front"
      elif "left" in utterance:
        insert_key = "left"
      elif "right" in utterance:
        insert_key = "right"
        
      if "Exactly one all" in utterance:
        if " some" in utterance.lower():
          file_path = os.path.join(img_prefix, "one_all_some", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "one_all_some", img_suffix_path))
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "one_some", img_suffix_path))
        else: # if " all" in utterance.lower():
          file_path = os.path.join(img_prefix, "one_all_all", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "one_all_all", img_suffix_path))
          #shutil.copy(img_suffix_path, os.path.join(img_prefix, "one_all", img_suffix_path))
            
      elif "Exactly one some" in utterance:
        if " all" in utterance.lower():
          file_path = os.path.join(img_prefix, "one_some_all", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "one_some_all", img_suffix_path))
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "one_some", img_suffix_path))
        else: # if " all" in utterance.lower():
          file_path = os.path.join(img_prefix, "one_some_some", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
          # shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "one_some_some", img_suffix_path))
        
      elif "some" in utterance.split(" ")[0].lower():
        if " all" in utterance.lower():
          file_path = os.path.join(img_prefix, "some_all", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
            
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "some_all", img_suffix_path))
          #shutil.copy(img_suffix_path, os.path.join(img_prefix, "some_all", img_suffix_path))
        elif " some" in utterance.lower():
          file_path = os.path.join(img_prefix, "some_some", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "some_some", img_suffix_path))
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "some_some", img_suffix_path))
      elif "all" in utterance.split(" ")[0].lower():
        if " all" in utterance.lower():
          file_path = os.path.join(img_prefix, "all_all", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "all_all", img_suffix_path))
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "all_all", img_suffix_path))
        elif " some" in utterance.lower():
          file_path = os.path.join(img_prefix, "all_some", insert_key, "image.txt") 
          with open(file_path, "a+") as f:
            f.write(os.path.join(img_prefix, img_suffix_path)+"\n")
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "all_some", img_suffix_path))
          #shutil.copy(os.path.join(img_prefix, img_suffix_path), os.path.join(img_prefix, "all_some", img_suffix_path))
        
    
def main_some_intrinsic(args):
  num_digits = 6
  if args.implicature_type == "direct":
    impli_type = "direct implicature"
    indirect = False
    
  elif args.implicature_type == "indirect":
    impli_type = "indirect implicature"
    indirect = True
    
  elif args.implicature_type == "direct_cancel":
    impli_type = "direct contextual cancellation"
    indirect = False
    
  elif args.implicature_type == "indirect_cancel":
    impli_type = "indirect contextual cancellation"
    indirect = True
    
  if 'cancel' in impli_type:
    cancel = True
  else:
    cancel = False
    
  prefix = '%s_' % (args.filename_prefix)
    
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  mask_template = '%s%%0%dd.png' % (prefix, num_digits) #os.path.join(args.output_image_dir, "connectives", "masks", dist1_mask_template)

  dist1_img_templare = '%s%%0%dd.png' % (prefix, num_digits)
  dist1_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist1_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
    
  dist2_img_templare = '%s%%0%dd.png' % (prefix, num_digits)
  dist2_scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  dist2_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
  
  jsonl_scene_template = '%s%%0%dd.jsonl' % (prefix, num_digits)

  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  
  if not os.path.exists(os.path.join(args.output_image_dir, 'clevr_some_intrinsic')):
    os.mkdir(os.path.join(args.output_image_dir, 'clevr_some_intrinsic'))
  #if not os.path.exists(os.path.join(args.output_scene_dir, 'clever_some_intrinsic')):
  #  os.mkdir(os.path.join(args.output_scene_dir, 'clever_some_intrinsic'))
    
  if not os.path.exists(os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'images')):
    os.mkdir(os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'images'))
  if not os.path.exists(os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'masks')):
    os.mkdir(os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'masks'))
   
  if not os.path.exists(os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'meta_data')):
    os.mkdir(os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'meta_data'))
    
  img_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'images', img_template)
  scene_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'meta_data', scene_template)
  mask_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'masks', mask_template)

  dist1_img_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'images', dist1_img_templare)
  dist1_scene_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'meta_data', dist1_scene_template)
  dist1_mask_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'masks', dist1_mask_template)
    
  dist2_img_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'images', dist2_img_templare)
  dist2_scene_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'meta_data', dist2_scene_template)
  dist2_mask_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'masks', dist2_mask_template)

  blend_template = os.path.join(args.output_blend_dir, blend_template)
  jsonl_scene_template = os.path.join(args.output_image_dir, 'clevr_some_intrinsic', 'meta_data', jsonl_scene_template)
  #if not os.path.isdir(args.output_image_dir):
  #  os.makedirs(args.output_image_dir)
  #if not os.path.isdir(args.output_scene_dir):
  #  os.makedirs(args.output_scene_dir)
  #if args.save_blendfiles == 1 and not os.path.isdir(args.output_blend_dir):
  #  os.makedirs(args.output_blend_dir)
  # sample object-centri
    
        
  
  all_scene_paths = []
  for i in range(args.num_images):
    num_main_attr = random.sample([0,1,2,3], 1)[0]
    if impli_type == 'indirect contextual cancellation':
      num_main_attr = random.sample([1,2,3], 1)[0]
    if num_main_attr == 0:
      object_centric = "objects"
      attr_centric = random.sample(['size', 'color', 'material', 'shape'], 1)[0]
      aux_attr = random.sample(INTRINSIC_PRIMITIVES[attr_centric], 1)[0]

    else:
      attr_list = []
      size = None
      color = None
      material = None
      shape = None
      shape_flag = False
      attr_centric = random.sample(['size', 'color', 'material', 'shape'], num_main_attr)
      aux_attr_type = set(['size', 'color', 'material', 'shape']) - set(attr_centric)
      aux_attr_type = random.sample(aux_attr_type, 1)[0]
      aux_attr = random.sample(INTRINSIC_PRIMITIVES[aux_attr_type], 1)[0]

      for attr_type in attr_centric:
        if attr_type == 'size':
          size = random.sample(INTRINSIC_PRIMITIVES["size"], 1)[0]
        elif attr_type == 'color':
          color = random.sample(INTRINSIC_PRIMITIVES["color"], 1)[0]
        elif attr_type == 'material':
          material = random.sample(INTRINSIC_PRIMITIVES["material"], 1)[0]
        else:
          shape = random.sample(INTRINSIC_PRIMITIVES["shape"], 1)[0]
          shape_flag = True

      for _ in [size, color, material, shape]:
        if _ != None:
          attr_list.append(_)
      object_centric = " ".join(attr_list)
      if shape_flag:
        object_centric += "s"
      else:
        object_centric += " objects"
    
    img_path = img_template % (3 * i + args.start_idx)
    scene_path = scene_template % (3 * i + args.start_idx)
    mask_path = mask_template % (3 * i + args.start_idx)
    
    dist1_img_path = dist1_img_template % (3 * i + args.start_idx + 1)
    dist1_scene_path = dist1_scene_template % (3 * i + args.start_idx + 1)
    dist1_mask_path = dist1_mask_template % (3 * i + args.start_idx + 1)
    
    dist2_img_path = dist2_img_template % (3 * i + args.start_idx + 2)
    dist2_scene_path = dist2_scene_template % (3 * i + args.start_idx + 2)
    dist2_mask_path = dist2_mask_template % (3 * i + args.start_idx + 2)
    
    jsonl_scene = jsonl_scene_template % (i + args.start_game_idx)
    
    all_scene_paths.append(scene_path)
    #blend_path = None
    #if args.save_blendfiles == 1:
    blend_path = None#blend_template % (i + args.start_idx)
    if not cancel:
      num_objects = random.randint(4,10)
      
      obj_ind, utterance = render_some_intrinsic_scene(args,
        num_objects=num_objects,#num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image=img_path,
        output_mask=mask_path, 
        output_scene=scene_path,
        output_blendfile=blend_path,
        object_ud=object_centric, #object_centric,
        modifier=aux_attr, #aux_attr,
        jsonl_path=jsonl_scene,
        flag = True,
        key = impli_type,
        indirect=indirect,
      )
      
      num_objects = random.randint(4,10)
      obj_ind, utterance = render_all_intrinsic_scene(args,
        num_objects=num_objects,#num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image=dist2_img_path,
        output_mask=dist2_mask_path, 
        output_scene=dist2_scene_path,
        output_blendfile=blend_path,
        object_ud=object_centric,# object_centric,
        modifier=aux_attr, #aux_attr,
        jsonl_path=jsonl_scene,
        first_cancel = False,
        key = impli_type,
        indirect=indirect,
      )
      
      num_objects = random.randint(4,10)
      obj_ind, utterance = render_no_intrinsic_scene(args,
        num_objects=num_objects,#num_objects,
        output_index=(i + args.start_idx),
        output_split=args.split,
        output_image=dist1_img_path,
        output_mask=dist1_mask_path, 
        output_scene=dist1_scene_path,
        output_blendfile=blend_path,
        object_ud=object_centric,
        modifier=aux_attr,
        jsonl_path=jsonl_scene,
      )
      
    else:
      if impli_type == 'direct contextual cancellation':
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_all_intrinsic_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path,
          output_mask=mask_path, 
          output_scene=scene_path,
                                                        
          output_blendfile=blend_path,
          object_ud=object_centric,
          modifier=aux_attr,
          jsonl_path=jsonl_scene,
          first_cancel = True,
          key = impli_type,
          indirect=indirect,
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_no_intrinsic_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=dist1_img_path,
          output_mask=dist1_mask_path, 
          output_scene=dist1_scene_path,
          output_blendfile=blend_path,
          object_ud=object_centric,
          modifier=aux_attr,
          jsonl_path=jsonl_scene,
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_no_intrinsic_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=dist2_img_path, # dist2_img_path
          output_mask=dist2_mask_path, # dist2_mask_path
          output_scene=dist2_scene_path, # dist2_scene_path
          output_blendfile=blend_path,
          object_ud=object_centric,
          modifier=aux_attr,
          jsonl_path=jsonl_scene,
        )
      elif impli_type == 'indirect contextual cancellation':
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_no_intrinsic_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=img_path, # img_path
          output_mask=mask_path, # mask_path
          output_scene=scene_path, # scene_path
          output_blendfile=blend_path,
          object_ud=object_centric,
          modifier=aux_attr,
          jsonl_path=jsonl_scene,
          flag = True,
                           
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_all_intrinsic_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=dist2_img_path,
          output_mask=dist2_mask_path,
          output_scene=dist2_scene_path,
          output_blendfile=blend_path,
          object_ud=object_centric,
          modifier=aux_attr,
          jsonl_path=jsonl_scene,
          first_cancel = False,
          key = impli_type,
          indirect=indirect,
        )
        num_objects = random.randint(4,10)
        obj_ind, utterance = render_all_intrinsic_scene(args,
          num_objects=num_objects,#num_objects,
          output_index=(i + args.start_idx),
          output_split=args.split,
          output_image=dist1_img_path,
          output_mask=dist1_mask_path,
          output_scene=dist1_scene_path,
          output_blendfile=blend_path,
          object_ud=object_centric,
          modifier=aux_attr,
          jsonl_path=jsonl_scene,
          first_cancel = False,
          key = impli_type,
          indirect=indirect,
        )
      
    
def parse_scene_struct(scene_dict, main_name, aux_name):
  # return the parsed sentences from the dict
  objects = scene_dict["objects"]
  relationships = scene_dict["relationships"]
  len_objects = len(objects)
  # object attributed descriptions
  objd_lists = []
  objds = []
  for obj in objects:
    objd = [obj['size'], obj['color'], obj['material'], obj['shape']]
    objd_lists.append(objd)
    objds.append(" ".join(objd))
  # 1. image-level description
  image_some2ids = {}
  image_all = []
  image_2ids = {}

  
  for i, objd_list in enumerate(objd_lists):
    # per object processing
    ## single item
    for attr in objd_list:
      count = 0
      obj_inds = []
      # Traverse 
      for j, objd in enumerate(objds):
        if attr in objd:
          count += 1
          obj_inds.append(j)
      if count != len_objects:
        image_some2ids[attr] = obj_inds
      else:
        image_all.append(attr)
      image_2ids[attr] = obj_inds
    ## double items
    comb1d_lists = []
    comb1ds = []
    comb2d_lists = []
    comb2ds = []
    comb3d_lists = []
    comb3ds = []
    comb4d_lists = []
    comb4ds = []
    for comb_i in range(len(objd_list)):
      comb1d_lists.append(objd_list[comb_i])
      comb1ds.append(objd_list[comb_i])

    for comb_i, comb_j in itertools.combinations(range(len(objd_list)), 2):
      print(comb_i, comb_j)
      temp_1, temp_2 = objd_list[comb_i], objd_list[comb_j]
      comb2d_lists.append([temp_1, temp_2])
      comb2ds.append(" ".join([temp_1, temp_2]))
    
    for comb_i, comb_j, comb_k in itertools.combinations(range(len(objd_list)), 3):
      temp_1, temp_2, temp_3 = objd_list[comb_i], objd_list[comb_j], objd_list[comb_k]
      comb3d_lists.append([temp_1, temp_2, temp_3])
      comb3ds.append(" ".join([temp_1, temp_2, temp_3]))
 
     
      
    comb4d_lists.append(objd_list)
    comb4ds.append(" ".join(objd_list))
    
    #print(comb2ds)
    for k, comb2d_set in enumerate(comb2d_lists):
      # (red, ball)
      count = 0
      obj_inds = []
      for j, new_objd_list in enumerate(objd_lists):
        if len(set(new_objd_list) - set(comb2d_set)) == 2:
          count += 1
          obj_inds.append(j)
          
      if count != len_objects:
        image_some2ids[comb2ds[k]] = obj_inds
      else:
        image_all.append(comb2ds[k])
      image_2ids[comb2ds[k]] = obj_inds
 
    for k, comb3d_set in enumerate(comb3d_lists):
      count = 0
      obj_inds = []
      for j, new_objd_list in enumerate(objd_lists):
        if len(set(new_objd_list) - set(comb3d_set)) == 1:
          count += 1
          obj_inds.append(j)

      if count != len_objects:
        image_some2ids[comb3ds[k]] = obj_inds
      else:
        image_all.append(comb3ds[k])
      image_2ids[comb3ds[k]] = obj_inds
    
    for k, comb4d_set in enumerate(comb4d_lists):
      count = 0
      obj_inds = []
      for j, new_objd_list in enumerate(objd_lists):
        if len(set(new_objd_list) - set(comb4d_set)) == 0:
          count += 1
          obj_inds.append(j)

      if count != len_objects:
        image_some2ids[comb4ds[k]] = obj_inds
      else:
        image_all.append(comb4ds[k])
      image_2ids[comb4ds[k]] = obj_inds
    
    ## Full items
    count = 0
    obj_inds = []
    for j, new_objd_list in enumerate(objd_lists):
      if new_objd_list == objd_list:
        count += 1
        obj_inds.append(j)

    if count != len_objects:
      image_some2ids[objds[i]] = obj_inds
    else:
      image_all.append(objds[i])
    
    image_2ids[objds[i]] = obj_inds
    
  print(image_2ids.keys())

  # 2. extrinsic, object-level
  ## tree, node structure
  objects_structs = []
  for i, objd in enumerate(objds):
    # "big red rubber ball"
    ## Full items
    objd_list = objd_lists[i]
    full_current_objd = {"obj":objd, "id": i, "left_names": [], "left_child": [], "right_child": [], "behind_child": [], "front_child": []}
    ### left_child: ["some balls"]
    available_primary_names = full_current_objd['left_names']
    available_primary_names.append("some objects")
    # first, all some just by object-level
    # some => all, left
    # 1. right, some => all
    # 2. left, some ==> all
    # generating all possible names, where objd is 4D
    comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objd_list, 1)]
    comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objd_list, 2)]
    comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objd_list, 3)]
    comb4d_lists = []
    
    for comb_i in range(len(objd_list)):
      comb1d_lists.append(objd_list[comb_i])
      

    for comb_i, comb_j in itertools.combinations(range(len(objd_list)), 2):
      temp_1, temp_2 = objd_list[comb_i], objd_list[comb_j]
      comb2d_lists.append([temp_1, temp_2])
      
    
    for comb_i, comb_j, comb_k in itertools.combinations(range(len(objd_list)), 3):
      temp_1, temp_2, temp_3 = objd_list[comb_i], objd_list[comb_j], objd_list[comb_k]
      comb3d_lists.append([temp_1, temp_2, temp_3])
      
    comb4d_lists.append(objd_list)
        
    ## 1. first round
    print(comb2d_lists)
    for comb1d in comb1d_lists:
      
      if comb1d in image_some2ids:
        if len(image_some2ids[comb1d]) == 1:
          available_primary_names.append("Exactly one all "+comb1d)
        else:
          available_primary_names.append("some "+comb1d)
      else:
        available_primary_names.append("all "+comb1d)
    
    for comb2d in comb2d_lists:
      comb2d = " ".join(comb2d)
      if comb2d in image_some2ids:

        if len(image_some2ids[comb2d]) == 1:
          available_primary_names.append("Exactly one all "+comb2d)
        else:
          available_primary_names.append("some "+comb2d)
      else:
        available_primary_names.append("all "+comb2d)
    
    for comb3d in comb3d_lists:
      comb3d = " ".join(comb3d)
      if comb3d in image_some2ids:
        if len(image_some2ids[comb3d]) == 1:
          available_primary_names.append("Exactly one all "+comb3d)
        else:
          available_primary_names.append("some "+comb3d)
      else:
        available_primary_names.append("all "+comb3d)
    
    for comb4d in comb4d_lists:
      comb4d = " ".join(comb4d)
      if comb4d in image_some2ids:
        if len(image_some2ids[comb4d]) == 1:
          available_primary_names.append("Exactly one all "+comb4d)
        else:
          available_primary_names.append("some "+comb4d)
      else:
        available_primary_names.append("all "+comb4d)
    
    ## 2. map the relationships from ids to full names
    left_rel = relationships['left'][i]
    right_rel = relationships['right'][i]
    front_rel = relationships['front'][i]
    behind_rel = relationships['behind'][i]
    # per relation processing
    for rel_obj_ind in left_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['left_child'].append(rel_objd_list)

    for rel_obj_ind in right_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['right_child'].append(rel_objd_list)

    for rel_obj_ind in front_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['front_child'].append(rel_objd_list)

    for rel_obj_ind in behind_rel:
      rel_objd_list = objd_lists[rel_obj_ind]
      full_current_objd['behind_child'].append(rel_objd_list)

    objects_structs.append(full_current_objd)
  ## adjust the right names
  for i, object_struct in enumerate(objects_structs):
    # object-centric per relation processing
    object_struct['left'] = []
    object_struct['right'] = []
    object_struct['front'] = []
    object_struct['behind'] = []
    
    left_objects = object_struct["left_child"]
    left_object_ids = relationships['left'][i]

    right_objects = object_struct["right_child"]
    right_object_ids = relationships['right'][i]

    front_objects = object_struct["front_child"]
    front_object_ids = relationships['front'][i]

    behind_objects = object_struct["behind_child"]
    behind_object_ids = relationships['behind'][i]
    # process left
    for objectd_list in left_objects:
      # "big red rubber ball"
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
    
      for comb_i in range(len(objectd_list)):
        comb1d_lists.append(objectd_list[comb_i])
      
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)), 2):
        temp_1, temp_2 = objectd_list[comb_i], objectd_list[comb_j]
        comb2d_lists.append([temp_1, temp_2])
        
      for comb_i, comb_j, comb_k in itertools.combinations(range(len(objectd_list)), 3):
        temp_1, temp_2, temp_3 = objectd_list[comb_i], objectd_list[comb_j], objectd_list[comb_k]
        comb3d_lists.append([temp_1, temp_2, temp_3])
        
      comb4d_lists.append(objectd_list)
      
      
      # 4d
      for comb4d in comb4d_lists:
        # comb
        comb4d = " ".join(comb4d)
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(left_object_ids)) > 0:
          #
          object_struct['left'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['left'].append("all "+comb4d)
      # 3d
      for comb3d in comb3d_lists:
        comb3d = " ".join(comb3d)
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(left_object_ids)) > 0:
          object_struct['left'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb3d)
        else:
          object_struct['left'].append("all "+comb3d)
      
      # 2d
      for comb2d in comb2d_lists:
        comb2d = " ".join(comb2d)
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(left_object_ids)) > 0:
          object_struct['left'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb2d)
        else:
          object_struct['left'].append("all "+comb2d)
      
      # 1d
      for comb1d in comb1d_lists:
        
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(left_object_ids)) > 0:
          object_struct['left'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["left"].append("Exactly one all "+comb1d)
        else:
          object_struct['left'].append("all "+comb1d)
    # process right
    for objectd_list in right_objects:
      #comb1d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      #comb2d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      #comb3d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      #comb4d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      
      for comb_i in range(len(objectd_list)):
        comb1d_lists.append(objectd_list[comb_i])
        
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)), 2):
        temp_1, temp_2 = objectd_list[comb_i], objectd_list[comb_j]
        comb2d_lists.append([temp_1, temp_2])
    
      for comb_i, comb_j, comb_k in itertools.combinations(range(len(objectd_list)), 3):
        temp_1, temp_2, temp_3 = objectd_list[comb_i], objectd_list[comb_j], objectd_list[comb_k]
        comb3d_lists.append([temp_1, temp_2, temp_3])
        
      comb4d_lists.append(objectd_list)
    
    
      
      # 4d
      for comb4d in comb4d_lists:
        # comb
        comb4d = " ".join(comb4d)
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(right_object_ids)) > 0:
          #
          object_struct['right'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['right'].append("all "+comb4d)
      # 3d
      for comb3d in comb3d_lists:
        comb3d = " ".join(comb3d)
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(right_object_ids)) > 0:
          object_struct['right'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb3d)
        else:
          object_struct['right'].append("all "+comb3d)
      
      # 2d
      for comb2d in comb2d_lists:
        comb2d = " ".join(comb2d)
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(right_object_ids)) > 0:
          object_struct['right'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb2d)
        else:
          object_struct['right'].append("all "+comb2d)
      
      # 1d
      for comb1d in comb1d_lists:
       
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(right_object_ids)) > 0:
          object_struct['right'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["right"].append("Exactly one all "+comb1d)
        else:
          object_struct['right'].append("all "+comb1d)
    # process front
    for objectd_list in front_objects:
      #comb1d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      #comb2d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      #comb3d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      #comb4d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      for comb_i in range(len(objectd_list)):
        comb1d_lists.append(objectd_list[comb_i])
        
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)), 2):
        temp_1, temp_2 = objectd_list[comb_i], objectd_list[comb_j]
        comb2d_lists.append([temp_1, temp_2])  
    
      for comb_i, comb_j, comb_k in itertools.combinations(range(len(objectd_list)), 3):
        temp_1, temp_2, temp_3 = objectd_list[comb_i], objectd_list[comb_j], objectd_list[comb_k]
        comb3d_lists.append([temp_1, temp_2, temp_3])
        
      comb4d_lists.append(objectd_list)
      
      # 4d
      for comb4d in comb4d_lists:
        # comb
        comb4d = " ".join(comb4d)
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(front_object_ids)) > 0:
          #
          object_struct['front'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['front'].append("all "+comb4d)
      
      # 3d
      for comb3d in comb3d_lists:
        comb3d = " ".join(comb3d)
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(front_object_ids)) > 0:
          object_struct['front'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb3d)
        else:
          object_struct['front'].append("all "+comb3d)
      
      # 2d
      for comb2d in comb2d_lists:
        comb2d = " ".join(comb2d)
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(front_object_ids)) > 0:
          object_struct['front'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb2d)
        else:
          object_struct['front'].append("all "+comb2d)
      
      # 1d
      for comb1d in comb1d_lists:
    
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(front_object_ids)) > 0:
          object_struct['front'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["front"].append("Exactly one all "+comb1d)
        else:
          object_struct['front'].append("all "+comb1d)
          
    # process behind
    for objectd_list in behind_objects:
      #comb1d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      #comb2d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      #comb3d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      #comb4d_lists = [" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      comb1d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 1)]
      comb2d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 2)]
      comb3d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 3)]
      comb4d_lists = []#[" ".join(list(_)) for _ in itertools.combinations(objectd_list, 4)]
      for comb_i in range(len(objectd_list)):
        comb1d_lists.append(objectd_list[comb_i])
        
      for comb_i, comb_j in itertools.combinations(range(len(objectd_list)), 2):
        temp_1, temp_2 = objectd_list[comb_i], objectd_list[comb_j]
        comb2d_lists.append([temp_1, temp_2])
    
      for comb_i, comb_j, comb_k in itertools.combinations(range(len(objectd_list)), 3):
        temp_1, temp_2, temp_3 = objectd_list[comb_i], objectd_list[comb_j], objectd_list[comb_k]
        comb3d_lists.append([temp_1, temp_2, temp_3])
        
      comb4d_lists.append(objectd_list)
      
      
      # 4d
      for comb4d in comb4d_lists:
        # comb
        comb4d = " ".join(comb4d)
        all_ids = image_2ids[comb4d]
        
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          #
          object_struct['behind'].append("some "+comb4d)
          
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb4d)
          
        else:
          object_struct['behind'].append("all "+comb4d)
      
      # 3d
      for comb3d in comb3d_lists:
        comb3d = " ".join(comb3d)
        all_ids = image_2ids[comb3d]
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          object_struct['behind'].append("some "+comb3d)
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb3d)
        else:
          object_struct['behind'].append("all "+comb3d)
    
      # 2d
      for comb2d in comb2d_lists:
        comb2d = " ".join(comb2d)
        all_ids = image_2ids[comb2d]
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          object_struct['behind'].append("some "+comb2d)
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb2d)
        else:
          object_struct['behind'].append("all "+comb2d)
     
      # 1d
      for comb1d in comb1d_lists:
        
        all_ids = image_2ids[comb1d]
        if len(set(all_ids)-set(behind_object_ids)) > 0:
          object_struct['behind'].append("some "+comb1d)
        elif len(all_ids) == 1:
          object_struct["behind"].append("Exactly one all "+comb1d)
        else:
          object_struct['behind'].append("all "+comb1d)
      
  # adjust the primary names
  utterances = []
  for i, object_struct in enumerate(objects_structs):
    # object-centric
    print(main_name, object_struct['obj'])
    print(type(main_name), type(object_struct['obj']))
    if len(set(main_name.split(" ")) - set(object_struct['obj'].split(" "))) > 0:
      continue
    primary_names = object_struct["left_names"]
    for name in primary_names: # some, all? 
      if "some" in name:
        name_ = copy.deepcopy(name)
        name_ = name_.replace("some ", "")
        name_ = name_.replace("Exactly one ", "")
        if name_ == "objects":
            continue
        primary_object_ids = set(image_2ids[name_]) # all
        
        # per relation
        ## left
        for descrip in object_struct['left']: # on the right of
          count_ids = []
          some_flag = False
          exist_flag = False
          #print(descrip)
          
          if "all"  in descrip.split(" "):
            object_under_discuss_name = descrip.replace("small", "yeah").split("all ")[-1].replace("yeah", "small")
            #print(descrip)
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['right'][second_object_id])) > 0:
                print(primary_object_ids)
                print(set(relationships['right'][second_object_id]))
                print("****right")
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['right'][second_object_id])):
                  count_ids.append(temp)
              
          elif "some" in descrip:
            
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['left'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)
              if len(set(second_object_ids) - set(relationships['left'][primary_object_id])) > 0:
                exist_flag = True
                
          if some_flag:
            utterances.append("some "+name_+" on the right of "+descrip)
          else:
            utterances.append("all "+name_+" on the right of "+descrip)
          if exist_flag:
            utterances.append("some "+name_+" on the right of "+descrip)
            
          print("****", utterances[-1])
            
          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
          print(utterances[-1])
            
        ## right
        for descrip in object_struct['right']:
          count_ids = []
          some_flag = False
          exist_flag = False
          if "all" in descrip.split(" "):
            object_under_discuss_name = descrip.replace("small", "yeah").split("all ")[-1].replace("yeah", "small")
            #print(descrip)
            #print(descrip)
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['left'][second_object_id])) > 0:
                print(primary_object_ids)
                print(set(relationships['left'][second_object_id]))
                print("****left")
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['left'][second_object_id])):
                  count_ids.append(temp)
            
          elif "some" in descrip:
            
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['right'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)
              if len(set(second_object_ids) - set(relationships['right'][primary_object_id])) > 0:
                exist_flag = True
                
          if some_flag:
            utterances.append("some "+name_+" on the left of "+descrip)
          else:
            utterances.append("all "+name_+" on the left of "+descrip)
          if exist_flag:
            utterances.append("some "+name_+" on the left of "+descrip)
            #print("****", utterances[-1])

          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
          print(utterances[-1])
        
        ## front
        for descrip in object_struct['front']:
          count_ids = []
          some_flag = False
          exist_flag = False
          if "all" in descrip.split(" "):
            object_under_discuss_name = descrip.replace("small", "yeah").split("all ")[-1].replace("yeah", "small")
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['behind'][second_object_id])) > 0:
                print(primary_object_ids)
                print(set(relationships['behind'][second_object_id]))
                print("****front")
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['behind'][second_object_id])):
                  count_ids.append(temp)
                  
              
          elif "some" in descrip:
            
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['front'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)
              if len(set(second_object_ids) - set(relationships['front'][primary_object_id])) > 0:
                exist_flag = True

          if some_flag:
            utterances.append("some "+name_+" behind "+descrip)
          else:
            utterances.append("all "+name_+" behind "+descrip)
          if exist_flag:
            utterances.append("some "+name_+" behind "+descrip)
            #print("****", utterances[-1])
          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
          print(utterances[-1])    
        ## behind
        for descrip in object_struct['behind']:
          some_flag = False
          exist_flag = False
          count_ids = []
          if "all" in descrip.split(" "):
            #print(descrip)
            object_under_discuss_name = descrip.replace("small", "yeah").split("all ")[-1].replace("yeah", "small")
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for second_object_id in second_object_ids:
              #
              if len(primary_object_ids - set(relationships['front'][second_object_id])) > 0:
                print(primary_object_ids)
                print(set(relationships['front'][second_object_id]))
                print("****hihihi behind")
                some_flag=True if some_flag == False else True
                for temp in primary_object_ids-(primary_object_ids - set(relationships['front'][second_object_id])):
                  count_ids.append(temp)
              #elif not some_flag:
              #  print("???????", descrip)
              #  print(some_flag, name_)
              #  print(primary_object_ids)
              #  print(second_object_ids)
                
          elif "some" in descrip:
            
            object_under_discuss_name = descrip.split("some ")[-1]
            second_object_ids = list(image_2ids[object_under_discuss_name]) # green balls
            for primary_object_id in primary_object_ids:
              if not set(relationships['behind'][primary_object_id]).intersection(set(second_object_ids)):
                some_flag=True if some_flag == False else True
              else:
                count_ids.append(primary_object_id)
              if len(set(second_object_ids) - set(relationships['behind'][primary_object_id])) > 0:
                exist_flag = True

          if some_flag:
            utterances.append("some "+name_+" on the front of "+descrip)
          else:
            utterances.append("all "+name_+" on the front of "+descrip)
          if exist_flag:
            utterances.append("some "+name_+" on the front of "+descrip)
            #print("****", utterances[-1])
          count_ids = set(count_ids)  
          if len(count_ids) == 1:
            utterances[-1] = "Exactly one " + utterances[-1]
          print(utterances[-1])
      else: # all
        name_ = copy.deepcopy(name)
        
        name_ = name_.replace("Exactly one ", "")
        if 'all' in name_.split(" ")[0]:
          name_ = " ".join(name_.split(" ")[1:]) #name_.replace("all ", "")
        
        if name_ == "objects":
            continue
        #print(name, name_)
        primary_object_ids = set(image_2ids[name_]) # all
        
        ## left
        for descrip_ in object_struct['left']:
          # All balls are on the right of xxx.
          #print(descrip_)
          descrip = descrip_.replace("Exactly one all ", "all ")
          #print(descrip_)
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append(name + " on the right of " + descrip)

        ## right
        for descrip_ in object_struct['right']:
          # All balls are on the left of xxx.
          #print(descrip_)
          descrip = descrip_.replace("Exactly one all ", "all ")
          #print(descrip_)
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append(name + " on the left of " + descrip)

        ## front
        for descrip_ in object_struct['front']:
          # All balls are behind xxx.
         
          descrip = descrip_.replace("Exactly one all ", "all ")
      
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append(name + " behind " + descrip)
          
        ## behind
        for descrip_ in object_struct['behind']:
          # All balls are on the front of xxx.
          #print(descrip_)
          descrip = descrip_.replace("Exactly one all ", "all ")
          #print(descrip_)
          ## avoid trivial expressions
          if descrip == name_:
            continue
          utterances.append(name + " on the front of " + descrip)
              
                
  # intrinsic: image_some2ids.keys()/ image_all
  final_utterances = []
  main_utter = []
  aux_utter = []
  for utterance in list(set(utterances)):
    if "left" in utterance or "right" in utterance:
      temp = utterance.split(" on ")
    elif "behind" in utterance:
      temp = utterance.split(" behind ")
    elif "front" in utterance:
      temp = utterance.split(" front ")
    
    if main_name in temp[0]:
      main_utter.append(utterance)
    if aux_name in temp[1]:
      aux_utter.append(utterance)
    
    
    
    if main_name in temp[0] and aux_name in temp[1]:
      first_sec = copy.deepcopy(temp[0])
      last_sec = copy.deepcopy(temp[1])
      print(first_sec, last_sec)
      
      flag = True
      
      for ele in first_sec.replace(main_name, "").split(" "):
        if ele in INVERSE_INTRINSIC_PRIMITIVES:
          flag = False
          break
            
      for ele in last_sec.replace(aux_name, "").split(" "):
        if ele in INVERSE_INTRINSIC_PRIMITIVES:
          flag = False
          break
      
      if flag:      
        final_utterances.append(utterance)
      #print(temp)
  print(aux_utter)
  return final_utterances
      
        
  '''
  some = image_some2ids.keys()
  intri_dict = {"some": list(some), "all": list(image_all)}
  with open("../output/"+img_template.split(".")[0]+"_intrinsic.json", "w") as f:
    json.dump(intri_dict, f)
  # extrinsic: utterances
  print("The number of utterances:", len(utterances))
  for utterance in list(set(utterances)):
    if not os.path.exists("../output/utterances/"):
        os.mkdir("../output/utterances/")
    with open("../output/utterances/"+utterance+".txt", "a") as f:
      f.write(img_template)
  '''  
def render_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_objects(scene_struct, num_objects, args, camera)

  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

def render_ad_hoc_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)

  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects, utterance_obj_pairs, full_matrix = add_ad_hoc_random_objects(scene_struct, num_objects, args, camera)
  # Pick the utterance, and the object
  obj_uttr_pair = random.sample(utterance_obj_pairs, 1)[0]
  print(obj_uttr_pair)
  obj_ind = obj_uttr_pair[0]
  utterance = obj_uttr_pair[1]
  # ensure the uniqueness
  count = 0
  for ind, utter in utterance_obj_pairs:
    if utter == utterance:
      count+=1
  if count > 1:
    print("break")
    return None
    
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  scene_struct['object_ind'] = objects[obj_ind]
  scene_struct['utterance'] = utterance
  scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (obj_ind, utterance)

def render_and_connective_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    output_mask='mask.png',
    object_ud: str='ball',
    another_obj_ud: str='green cylinder',
    #modifier: str='green', # single
    indirect: bool=False,
    jsonl_path=None,
    first_cancel=False,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)
  
  # parse object_ud
  # <Size> <Color> <Material> <Shape>
  # obj_split = object_ud.split(" ")
  
  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
      bpy.context.user_preferences.system.compute_device = 'CUDA_1'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_and_connective_random_objects(scene_struct, num_objects, args, camera, object_ud, another_obj_ud)
  # Pick the utterance, and the object
  #if "cube" in object_ud or "sphere" in object_ud or "cylinder" in object_ud:
  if indirect == False:
    utterance = "It contains " + object_ud+"s or "+another_obj_ud + "s." #random.sample(utterance_obj_pairs, 1)[0]
    key = "direct implicature"
    if first_cancel:
      key="direct contextual cancellation"
  else:
    utterance = "Not "+object_ud+" and "+another_obj_ud
    key = "indirect implicature"
    if first_cancel:
      key="indirect contextual cancellation"
  # print(obj_uttr_pair)
  #obj_ind = obj_uttr_pair[0]
  #utterance = obj_uttr_pair[1]
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  #scene_struct['object_ind'] = objects[object_ud]
  #scene_struct['utterance'] = utterance
  #scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)
    
  render_mask_shadeless(blender_objects, output_mask)
  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)
  
  #if jsonl_path:
  with open(jsonl_path, 'a+') as f:
    if first_cancel:
      f.write(json.dumps({"utterance": utterance, "type": key}))
      f.write('\n')
    f.write(json.dumps(scene_struct))
    f.write('\n')
    
  

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (objects, None)

def render_or_connective_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    output_mask='mask.png',
    object_ud: str='ball',
    another_obj_ud: str='green cylinder',
    #modifier: str='green', # single
    indirect: bool=False,
    jsonl_path=None,
    flag=False,
    cancel=False,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)
  
  # parse object_ud
  # <Size> <Color> <Material> <Shape>
  # obj_split = object_ud.split(" ")
  
  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
      bpy.context.user_preferences.system.compute_device = 'CUDA_1'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_or_connective_random_objects(scene_struct, num_objects, args, camera, object_ud, another_obj_ud)
  # Pick the utterance, and the object
  #if "cube" in object_ud or "sphere" in object_ud or "cylinder" in object_ud:
  object_ud += "s"
  another_obj_ud += "s"
    
  if indirect == False:
    utterance = "It contains" + " " + object_ud+" or "+another_obj_ud + "." #random.sample(utterance_obj_pairs, 1)[0]
    key = "direct implicature"
    if cancel:
      key="direct contextual cancellation"
  else:
    utterance = "Not "+object_ud+" and "+another_obj_ud
    key = "indirect implicature"
    if cancel:
      key="indirect contextual cancellation"
    
  # print(obj_uttr_pair)
  #obj_ind = obj_uttr_pair[0]
  #utterance = obj_uttr_pair[1]
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  #scene_struct['object_ind'] = objects[object_ud]
  # scene_struct['utterance'] = utterance
  #scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)
    
  render_mask_shadeless(blender_objects, output_mask)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)
  
  #if jsonl_path:
  
    
  with open(jsonl_path, 'a') as f:
    if flag:
      f.write(json.dumps({"utterance": utterance, "type": key}))
      f.write('\n')
      
    f.write(json.dumps(scene_struct))
    f.write('\n')

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (objects, None)

def render_no_connective_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    output_mask='mask.png',
    object_ud: str='ball',
    another_obj_ud: str='green cylinder',
    #modifier: str='green', # single
    indirect: bool=False,
    jsonl_path=None
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)
  
  # parse object_ud
  # <Size> <Color> <Material> <Shape>
  # obj_split = object_ud.split(" ")
  
  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
      bpy.context.user_preferences.system.compute_device = 'CUDA_1'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_no_connective_random_objects(scene_struct, num_objects, args, camera, object_ud, another_obj_ud)
  # Pick the utterance, and the object
  #if "cube" in object_ud or "sphere" in object_ud or "cylinder" in object_ud:
    
  # print(obj_uttr_pair)
  #obj_ind = obj_uttr_pair[0]
  #utterance = obj_uttr_pair[1]
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  #scene_struct['object_ind'] = objects[object_ud]
  #scene_struct['utterance'] = utterance
  #scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  render_mask_shadeless(blender_objects, output_mask)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)
  
  with open(jsonl_path, 'a') as f:
    f.write(json.dumps(scene_struct))
    f.write('\n')

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (objects, None)

def render_random_extrinsic_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image_prefix="temp", 
    output_image='render.png',
    output_scene_prefix="temp_scene",
    output_scene='render_json',
    output_blendfile=None,
    main_name: str='red ball',
    aux_name: str='green cube', # single
    one_flag = True,
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)
  
  # parse object_ud
  # <Size> <Color> <Material> <Shape>
  main_obj_split = main_name.split(" ")
  aux_obj_split = aux_name.split(" ")
  
  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
      bpy.context.user_preferences.system.compute_device = 'CUDA_2'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag)
  # Pick the utterance, and the object
  #if "cube" in object_ud or "sphere" in object_ud or "cylinder" in object_ud:
  #  object_ud += "s"
  #if indirect == False:
  #  utterance = "some "+object_ud+" are "+modifier #random.sample(utterance_obj_pairs, 1)[0]
  #else:
  #  utterance = "Not all "+object_ud+" are "+modifier
  # print(obj_uttr_pair)
  #obj_ind = obj_uttr_pair[0]
  #utterance = obj_uttr_pair[1]
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  #scene_struct['object_ind'] = objects[object_ud]
  #if modifier in ["cube", "sphere", "cylinder"]:
  #  utterance = utterance + "s"
  #scene_struct['utterance'] = utterance
  #scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  utterances = parse_scene_struct(scene_struct, main_name, aux_name)
  scene_struct['utterances'] = utterances
  
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (objects, utterances)

def render_some_intrinsic_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_mask='render.png',
    output_blendfile=None,
    object_ud: str='objects',
    modifier: str='green', # single
    jsonl_path: str=None,
    flag: bool=False,
    key: str="direct implicature",
    indirect: bool=False,
    
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)
  
  # parse object_ud
  # <Size> <Color> <Material> <Shape>
  obj_split = object_ud.split(" ")
  
  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
      bpy.context.user_preferences.system.compute_device = 'CUDA_1'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
  # Pick the utterance, and the object
  #if "cube" in object_ud or "sphere" in object_ud or "cylinder" in object_ud:
  #  object_ud += "s"
    
  if indirect == False:
    utterance = "Some "+object_ud+" are "+modifier+"." #random.sample(utterance_obj_pairs, 1)[0]
  else:
    utterance = "Not all "+object_ud+" are "+modifier+"."
  # print(obj_uttr_pair)
  #obj_ind = obj_uttr_pair[0]
  #utterance = obj_uttr_pair[1]
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  #scene_struct['object_ind'] = objects[object_ud]
  #if modifier in ["cube", "sphere", "cylinder"]:
  #  utterance = utterance + "s"
  #scene_struct['utterance'] = utterance
  #scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)
    
  render_mask_shadeless(blender_objects, output_mask)

  with open(jsonl_path, "a") as f:
    if flag:
      f.write(json.dumps({"utterance": utterance, "type": key}))
      f.write('\n')
    f.write(json.dumps(scene_struct))
    f.write('\n')

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (objects, utterance)
    
def render_all_intrinsic_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_mask='render.png',
    output_scene='render_json',                
    output_blendfile=None,
    object_ud: str='objects',
    modifier: str='green', # single
    key: str="direct implicature",
    jsonl_path=None,
    first_cancel=False,
    indirect = False
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)
  
  # parse object_ud
  # <Size> <Color> <Material> <Shape>
  obj_split = object_ud.split(" ")
  
  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
      bpy.context.user_preferences.system.compute_device = 'CUDA_1'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_all_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
  # Pick the utterance, and the object
  if indirect:
    utterance = "Not all "+object_ud+" are "+modifier+"."
  else:
    utterance = "Some "+object_ud+" are "+modifier+"." #random.sample(utterance_obj_pairs, 1)[0]
  # print(obj_uttr_pair)
  #obj_ind = obj_uttr_pair[0]
  #utterance = obj_uttr_pair[1]
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  #scene_struct['object_ind'] = objects[object_ud]
  #if modifier in ["cube", "sphere", "cylinder"]:
  #  utterance = utterance + "s"
  #scene_struct['utterance'] = utterance
  #scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)
    
  render_mask_shadeless(blender_objects, output_mask)
  
  with open(jsonl_path, 'a+') as f:
    if first_cancel:
      f.write(json.dumps({"utterance": utterance, "type": key}))
      f.write('\n')
    f.write(json.dumps(scene_struct))
    f.write('\n')

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (objects, utterance)

def render_no_intrinsic_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_mask='render.png', 
    output_scene='render_json',
    output_blendfile=None,
    object_ud: str='objects',
    modifier: str='green', # single
    jsonl_path=None,
    flag=False,
    type_key: str="indirect contextual cancellation",
  ):

  # Load the main blendfile
  bpy.ops.wm.open_mainfile(filepath=args.base_scene_blendfile)

  # Load materials
  utils.load_materials(args.material_dir)
  
  # parse object_ud
  # <Size> <Color> <Material> <Shape>
  obj_split = object_ud.split(" ")
  
  # Set render arguments so we can get pixel coordinates later.
  # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
  # cannot be used.
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
      bpy.context.user_preferences.system.compute_device = 'CUDA_1'
    else:
      cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
      cycles_prefs.compute_device_type = 'CUDA'

  # Some CYCLES-specific objd_list
  bpy.data.worlds['World'].cycles.sample_as_light = True
  bpy.context.scene.cycles.blur_glossy = 2.0
  bpy.context.scene.cycles.samples = args.render_num_samples
  bpy.context.scene.cycles.transparent_min_bounces = args.render_min_bounces
  bpy.context.scene.cycles.transparent_max_bounces = args.render_max_bounces
  if args.use_gpu == 1:
    bpy.context.scene.cycles.device = 'GPU'

  # This will give ground-truth information about the scene and its objects
  scene_struct = {
      'split': output_split,
      'image_index': output_index,
      'image_filename': os.path.basename(output_image),
      'objects': [],
      'directions': {},
  }

  # Put a plane on the ground so we can compute cardinal directions
  bpy.ops.mesh.primitive_plane_add(radius=5)
  plane = bpy.context.object

  def rand(L):
    return 2.0 * L * (random.random() - 0.5)

  # Add random jitter to camera position
  if args.camera_jitter > 0:
    for i in range(3):
      bpy.data.objects['Camera'].location[i] += rand(args.camera_jitter)

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

  # Add random jitter to lamp positions
  if args.key_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Key'].location[i] += rand(args.key_light_jitter)
  if args.back_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Back'].location[i] += rand(args.back_light_jitter)
  if args.fill_light_jitter > 0:
    for i in range(3):
      bpy.data.objects['Lamp_Fill'].location[i] += rand(args.fill_light_jitter)

  # Now make some random objects
  objects, blender_objects = add_no_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
  # Pick the utterance, and the object
  if flag:
    utterance = "Not all "+object_ud+" are "+modifier+"." #random.sample(utterance_obj_pairs, 1)[0]
  # print(obj_uttr_pair)
  #obj_ind = obj_uttr_pair[0]
  #utterance = obj_uttr_pair[1]
  # Render the scene and dump the scene data structure
  scene_struct['objects'] = objects
  #scene_struct['object_ind'] = objects[object_ud]
  #if modifier in ["cube", "sphere", "cylinder"]:
  #  utterance = utterance + "s"
  #scene_struct['utterance'] = utterance
  #scene_struct['referent'] = objects[obj_ind]['size'] + " " + objects[obj_ind]['color'] + " " + objects[obj_ind]['material'] + " " + objects[obj_ind]['shape'] 
  scene_struct['relationships'] = compute_all_relationships(scene_struct)
  while True:
    try:
      bpy.ops.render.render(write_still=True)
      break
    except Exception as e:
      print(e)
    
  render_mask_shadeless(blender_objects, output_mask)

  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f)
    
  with open(jsonl_path, 'a+') as f:
    if flag:
      f.write(json.dumps({"utterance": utterance, "type": type_key}))
      f.write('\n')
    f.write(json.dumps(scene_struct))
    f.write('\n')
 

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (objects, None)
    
    
def add_random_objects(scene_struct, num_objects, args, camera):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  for i in range(num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, args, camera)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_objects(scene_struct, num_objects, args, camera)

  return objects, blender_objects

def calculate_ad_hoc_matrix(objects):
  utterance_lists = {}
  # measure existence
  objects_whole_string = {}
  for i, obj in enumerate(objects):
    string = obj['shape']+obj['size']+obj['material']+obj['color']
    if string not in objects_whole_string:
      objects_whole_string[string] = 1
    else:
      objects_whole_string[string] += 1
    if obj['shape'] not in utterance_lists:
      utterance_lists[obj['shape']] = np.zeros(len(objects))
      utterance_lists[obj['shape']][i] = 1
    else:
      utterance_lists[obj['shape']][i] += 1
      
    if obj['size'] not in utterance_lists:
      utterance_lists[obj['size']] = np.zeros(len(objects))
      utterance_lists[obj['size']][i] = 1
    else:
      utterance_lists[obj['size']][i] += 1
    
    if obj['material'] not in utterance_lists:
      utterance_lists[obj['material']] = np.zeros(len(objects))
      utterance_lists[obj['material']][i] = 1
    else:
      utterance_lists[obj['material']][i] += 1
    
    if obj['color'] not in utterance_lists:
      utterance_lists[obj['color']] = np.zeros(len(objects))
      utterance_lists[obj['color']][i] = 1
    else:
      utterance_lists[obj['color']][i] += 1
  

  # process the column
  full_matrix = np.zeros((len(utterance_lists.keys()), len(objects)))
  keys = list(utterance_lists.keys())
  for i, key in enumerate(keys):
    utterance_lists[key] = utterance_lists[key]/(np.sum(utterance_lists[key])+1e-3)
    full_matrix[i] = utterance_lists[key]
  # transpose the matrix
  utterance_obj_pairs = []
  repeat_obj_pairs = []
  for i, obj_name in enumerate(objects):
    # current ind max
    string = obj_name['shape']+obj_name['size']+obj_name['material']+obj_name['color']
    if objects_whole_string[string] > 1:
      continue
    #print(full_matrix[:,i])
    if len(np.where(full_matrix[:,i]==full_matrix[:,i].max())[0]) == 1 and full_matrix[:,i].max() < 0.9:
      print(full_matrix[:,i])
      print(keys)
      print(np.where(full_matrix[:,i]==full_matrix[:,i].max()))
      print(obj_name)
      utterance_obj_pairs.append([i, keys[np.argmax(full_matrix[:,i])]])
    elif len(np.where(full_matrix[:,i]==full_matrix[:,i].max())[0]) != 1:
      temp = np.where(full_matrix[:,i]==full_matrix[:,i].max())[0]
      temp_keys = []
      for tmp in temp:
        temp_keys.append(keys[tmp])
      #temp_key = keys[]
      repeat_obj_pairs = repeat_obj_pairs+temp_keys
    
  # exclude repeat utterance
  utterance_count = {}
  for j, (temp, happy) in enumerate(utterance_obj_pairs):
    if happy not in utterance_count:
      utterance_count[happy] = [j]
    else:
      utterance_count[happy].append(j)
  print(utterance_count)
  print(repeat_obj_pairs)
    
  updated_utterance_obj_pairs = []
  for key in utterance_count.keys():
     if len(utterance_count[key]) == 1 and key not in repeat_obj_pairs:
       updated_utterance_obj_pairs.append(utterance_obj_pairs[utterance_count[key][0]])
  print(updated_utterance_obj_pairs)
    
  if len(updated_utterance_obj_pairs) == 0:
    return None, None
  else:
    return updated_utterance_obj_pairs, full_matrix
    
def add_ad_hoc_random_objects(scene_struct, num_objects, args, camera):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  for i in range(num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_ad_hoc_random_objects(scene_struct, num_objects, args, camera)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  utterance_obj_pairs, full_matrix = calculate_ad_hoc_matrix(objects)
  if utterance_obj_pairs == None:
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_ad_hoc_random_objects(scene_struct, num_objects, args, camera)

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_ad_hoc_random_objects(scene_struct, num_objects, args, camera)

  return objects, blender_objects, utterance_obj_pairs, full_matrix

def add_no_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier):
  """
  Add random objects to the current blender scene
  """
  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    material_inv_mapping = properties['materials'] # [(k, v) for k, v in properties['materials'].items()]
    
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    object_inv_mapping = properties['shapes'] # [(k, v) for k, v in properties['shapes'].items()]
    
    size_mapping = list(properties['sizes'].items())
    
  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())
    
  positions = []
  objects = []
  blender_objects = []
  # process the modifier
  modifier_key = 0
  obj_attr_types = {}
  
  for spl in obj_split:
    if spl == 'objects':
      continue
    if spl[-1] == 's':
      obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl[:-1]]] = spl[:-1]
    else:
      obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  for key in list(INTRINSIC_PRIMITIVES.keys()):
    if modifier in INTRINSIC_PRIMITIVES[key]:
      modifier_key = key
      break
  for i in range(num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_no_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    mat_name, mat_name_out = random.choice(material_mapping)
    # process objects to avoid repetition
    value_set = [obj_name_out, size_name, mat_name_out, color_name]
    
    obj_ud = " ".join(obj_split)
    
    count = 0
    split_count = len(obj_split)
    
    for value in value_set:
      if value in obj_ud:
        count+=1
        
    if "objects" in obj_ud:
        for value in value_set:
            if value == modifier:
                count += 1
                
    while count == split_count:
      size_name, r = random.choice(size_mapping)
      num_tries = 0
      while True:
        num_tries += 1
        if num_tries > args.max_retries:
          for obj in blender_objects:
            utils.delete_object(obj)
          return add_no_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
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
      # Choose random color and shape
      if shape_color_combos is None:
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      else:
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
      # For cube, adjust the size a bit
      if obj_name == 'Cube':
        r /= math.sqrt(2)
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      mat_name, mat_name_out = random.choice(material_mapping)
      # process objects to avoid repetition
      value_set = [obj_name_out, size_name, mat_name_out, color_name]
     
      count = 0
      for value in value_set:
        if value in obj_ud:
          count+=1
      if "objects" in obj_ud:
        for value in value_set:
            if value == modifier:
                count += 1
        
      
    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })


  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_no_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)

  return objects, blender_objects
   
def add_no_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    material_inv_mapping = properties['materials'] # [(k, v) for k, v in properties['materials'].items()]
    
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    object_inv_mapping = properties['shapes'] # [(k, v) for k, v in properties['shapes'].items()]
    
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  # parsing obj_ud, and another_obj_ud
  obj_split = obj_ud.split(" ")
  another_obj_split = another_obj_ud.split(" ")
    
  obj_attr_types = {}
  for spl in obj_split:
    if spl == 'objects':
      continue
    obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
      
  another_obj_attr_types = {}
  for spl in another_obj_split:
    if spl == 'objects':
      continue
    another_obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
    
  for i in range(num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_no_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    mat_name, mat_name_out = random.choice(material_mapping)
    # process objects to avoid repetition
    value_set = [obj_name_out, size_name, mat_name_out, color_name]
    
    print(value_set)
    
    
    count = 0
    another_count = 0
    split_count = len(obj_split)
    another_split_count = len(another_obj_split)
    
    
    for value in value_set:
      if value in obj_ud:
        count+=1
      if value in another_obj_ud:
        another_count+=1
    
    while count == split_count or another_count == another_split_count:
      size_name, r = random.choice(size_mapping)
      num_tries = 0
      while True:
        num_tries += 1
        if num_tries > args.max_retries:
          for obj in blender_objects:
            utils.delete_object(obj)
          return add_no_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
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
      # Choose random color and shape
      if shape_color_combos is None:
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      else:
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
      # For cube, adjust the size a bit
      if obj_name == 'Cube':
        r /= math.sqrt(2)
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      mat_name, mat_name_out = random.choice(material_mapping)
      # process objects to avoid repetition
      value_set = [obj_name_out, size_name, mat_name_out, color_name]
      
     
      count = 0
      another_count = 0
      for value in value_set:
        if value in obj_ud:
          count+=1
        if value in another_obj_ud:
          another_count+=1
      
    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
   
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })


  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_no_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)

  return objects, blender_objects 

def add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag=False):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    material_inv_mapping = properties['materials'] # [(k, v) for k, v in properties['materials'].items()]
    
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    object_inv_mapping = properties['shapes'] # [(k, v) for k, v in properties['shapes'].items()]
    
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  # parsing obj_ud, and another_obj_ud
  obj_split = main_obj_split
  another_obj_split = aux_obj_split
    
  obj_attr_types = {}
  for spl in obj_split:
    obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  another_obj_attr_types = {}
  for spl in another_obj_split:
    another_obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  if one_flag:
    num_main_obj = 1 #random.sample(range(1, num_objects), 1)[0]
    num_aux_obj = random.sample(range(1, num_objects), 1)[0]
  else:
    num_main_obj = random.sample(range(2, num_objects), 1)[0]
    num_aux_obj = random.sample(range(1, num_objects-num_main_obj+1), 1)[0]
    
  for i in range(num_main_obj):
    if True:
      # process main object
      if "small" in obj_split or "large" in obj_split:
        # additional process size
        if "small" in obj_split:
          size_name = "small"
          r = 0.35
        elif "large" in obj_split:
          size_name = "large"
          r = 0.7
        num_tries = 0
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag)
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

        # Choose random color and shape
        if 'color' in obj_attr_types: 
          color_name = obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in obj_attr_types:
          obj_name_out = obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in obj_attr_types:
          mat_name_out = obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })   
        
      else:
        
        num_tries = 0
        
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag)
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

        # Choose random color and shape
        if 'color' in obj_attr_types: 
          color_name = obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in obj_attr_types:
          obj_name_out = obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in obj_attr_types:
          mat_name_out = obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
        
  for i in range(num_aux_obj):
    if True:
      # process main object
      if "small" in another_obj_split or "large" in another_obj_split:
        # additional process size
        if "small" in another_obj_split:
          size_name = "small"
          r = 0.35
        elif "large" in another_obj_split:
          size_name = "large"
          r = 0.7
        num_tries = 0
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag)
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

        # Choose random color and shape
        if 'color' in another_obj_attr_types: 
          color_name = another_obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in another_obj_attr_types:
          obj_name_out = another_obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in another_obj_attr_types:
          mat_name_out = another_obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })   
        
      else:
        
        num_tries = 0
        
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag)
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

        # Choose random color and shape
        if 'color' in another_obj_attr_types: 
          color_name = another_obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in another_obj_attr_types:
          obj_name_out = another_obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in another_obj_attr_types:
          mat_name_out = another_obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })      
      
  for i in range(num_aux_obj+num_main_obj, num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    mat_name, mat_name_out = random.choice(material_mapping)
 
    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })


  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_random_extrinsic_objects(scene_struct, num_objects, args, camera, main_obj_split, aux_obj_split, one_flag)

  return objects, blender_objects  
  
def add_or_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    material_inv_mapping = properties['materials'] # [(k, v) for k, v in properties['materials'].items()]
    
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    object_inv_mapping = properties['shapes'] # [(k, v) for k, v in properties['shapes'].items()]
    
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  # parsing obj_ud, and another_obj_ud
  obj_split = obj_ud.split(" ")
  another_obj_split = another_obj_ud.split(" ")
    
  obj_attr_types = {}
  for spl in obj_split:
    if spl == 'objects':
      continue
    obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  another_obj_attr_types = {}
  for spl in another_obj_split:
    if spl == 'objects':
      continue
    another_obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  num_main_obj = random.sample(range(1, num_objects), 1)[0]
    
  for i in range(num_main_obj):
    if True:
      # process main object
      if "small" in obj_split or "large" in obj_split:
        # additional process size
        if "small" in obj_split:
          size_name = "small"
          r = 0.35
        elif "large" in obj_split:
          size_name = "large"
          r = 0.7
        num_tries = 0
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_or_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

        # Choose random color and shape
        if 'color' in obj_attr_types: 
          color_name = obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in obj_attr_types:
          obj_name_out = obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in obj_attr_types:
          mat_name_out = obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })   
        
      else:
        
        num_tries = 0
        
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_or_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

        # Choose random color and shape
        if 'color' in obj_attr_types: 
          color_name = obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in obj_attr_types:
          obj_name_out = obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in obj_attr_types:
          mat_name_out = obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
        
        
      
  for i in range(num_main_obj, num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_or_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    mat_name, mat_name_out = random.choice(material_mapping)
    # process objects to avoid repetition
    value_set = [obj_name_out, size_name, mat_name_out, color_name]
    
    print(value_set)
    
    obj_ud = " ".join(obj_split)
    
    count = 0
    another_count = 0
    split_count = len(obj_split)
    another_split_count = len(another_obj_split)
    
    for value in value_set:
      if value in obj_ud:
        count+=1
      if value in another_obj_ud:
        another_count+=1
    
    while count == split_count or another_count == another_split_count:
      size_name, r = random.choice(size_mapping)
      num_tries = 0
      while True:
        num_tries += 1
        if num_tries > args.max_retries:
          for obj in blender_objects:
            utils.delete_object(obj)
          return add_or_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
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
      # Choose random color and shape
      if shape_color_combos is None:
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      else:
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
      # For cube, adjust the size a bit
      if obj_name == 'Cube':
        r /= math.sqrt(2)
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      mat_name, mat_name_out = random.choice(material_mapping)
      # process objects to avoid repetition
      value_set = [obj_name_out, size_name, mat_name_out, color_name]
      
     
      count = 0
      another_count = 0
      for value in value_set:
        if value in obj_ud:
          count+=1
        if value in another_obj_ud:
          another_count+=1
      
    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
   
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })


  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_or_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)

  return objects, blender_objects

def add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    material_inv_mapping = properties['materials'] # [(k, v) for k, v in properties['materials'].items()]
    
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    object_inv_mapping = properties['shapes'] # [(k, v) for k, v in properties['shapes'].items()]
    
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  # parsing obj_ud, and another_obj_ud
  obj_split = obj_ud.split(" ")
  another_obj_split = another_obj_ud.split(" ")
    
  obj_attr_types = {}
  for spl in obj_split:
    if spl == 'objects':
      continue
    obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
   
  another_obj_attr_types = {}
  for spl in another_obj_split:
    if spl == 'objects':
      continue
    another_obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  num_main_obj = random.sample(range(1, num_objects-1), 1)[0]
  sec_main_obj = random.sample(range(1, num_objects-num_main_obj+1), 1)[0]
    
  for i in range(num_main_obj):
    if True:
      # process main object
      if "small" in obj_split or "large" in obj_split:
        # additional process size
        if "small" in obj_split:
          size_name = "small"
          r = 0.35
        elif "large" in obj_split:
          size_name = "large"
          r = 0.7
        num_tries = 0
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

        # Choose random color and shape
        if 'color' in obj_attr_types: 
          color_name = obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in obj_attr_types:
          obj_name_out = obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in obj_attr_types:
          mat_name_out = obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
      else:
        num_tries = 0
        
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

        # Choose random color and shape
        if 'color' in obj_attr_types: 
          color_name = obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in obj_attr_types:
          obj_name_out = obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in obj_attr_types:
          mat_name_out = obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
        
        
  for i in range(sec_main_obj):
    if True:
      # process main object
      if "small" in another_obj_split or "large" in another_obj_split:
        # additional process size
        if "small" in another_obj_split:
          size_name = "small"
          r = 0.35
        elif "large" in another_obj_split:
          size_name = "large"
          r = 0.7
        num_tries = 0
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

        # Choose random color and shape
        if 'color' in another_obj_attr_types: 
          color_name = another_obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in another_obj_attr_types:
          obj_name_out = another_obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in another_obj_attr_types:
          mat_name_out = another_obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
      else:
        num_tries = 0
        #print(another_obj_split, another_obj_split, another_obj_split)
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

        # Choose random color and shape
        if 'color' in another_obj_attr_types: 
          color_name = another_obj_attr_types['color']
          rgba = color_name_to_rgba[color_name]
         
        else:
          color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
        if 'shape' in another_obj_attr_types:
          obj_name_out = another_obj_attr_types['shape']
          obj_name = object_inv_mapping[obj_name_out]
        
        else:
          obj_name, obj_name_out = random.choice(object_mapping)
        if obj_name == 'Cube':
          r /= math.sqrt(2)
        
        # Choose random orientation for the object.
        theta = 360.0 * random.random()
        if 'material' in another_obj_attr_types:
          mat_name_out = another_obj_attr_types['material']
          mat_name = material_inv_mapping[mat_name_out]
        else:
          mat_name, mat_name_out = random.choice(material_mapping)
     
        # Actually add the object to the scene
        utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))
        # Attach a random material
        utils.add_material(mat_name, Color=rgba)
        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
        
  for i in range(num_main_obj+sec_main_obj, num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    mat_name, mat_name_out = random.choice(material_mapping)
    # process objects to avoid repetition
    value_set = [obj_name_out, size_name, mat_name_out, color_name]
    
    print(value_set)
    
    obj_ud = " ".join(obj_split)
    
    count = 0
    another_count = 0
    split_count = len(obj_split)
    
    for value in value_set:
      if value in obj_ud:
        count+=1
      if value in another_obj_ud:
        another_count+=1
    
    while count == split_count or another_count == split_count:
      size_name, r = random.choice(size_mapping)
      num_tries = 0
      while True:
        num_tries += 1
        if num_tries > args.max_retries:
          for obj in blender_objects:
            utils.delete_object(obj)
          return add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
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
      # Choose random color and shape
      if shape_color_combos is None:
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      else:
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
      # For cube, adjust the size a bit
      if obj_name == 'Cube':
        r /= math.sqrt(2)
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      mat_name, mat_name_out = random.choice(material_mapping)
      # process objects to avoid repetition
      value_set = [obj_name_out, size_name, mat_name_out, color_name]
      
     
      count = 0
      another_count = 0
      for value in value_set:
        if value in obj_ud:
          count+=1
        if value in another_obj_ud:
          another_count+=1
      
    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
   
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })

  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_and_connective_random_objects(scene_struct, num_objects, args, camera, obj_ud, another_obj_ud)

  return objects, blender_objects  
  

def add_all_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier):
  """
  Add random objects to the current blender scene
  """

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    material_inv_mapping = properties['materials'] # [(k, v) for k, v in properties['materials'].items()]
    
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    object_inv_mapping = properties['shapes'] # [(k, v) for k, v in properties['shapes'].items()]
    
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  # process the modifier
  modifier_key = 0
  obj_attr_types = {}
  #for spl in obj_split:
  #  if spl == 'objects':
  #    continue
  #  obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  for spl in obj_split:
    if spl == 'objects':
      continue
    if spl[-1] == 's':
      obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl[:-1]]] = spl[:-1]
    else:
      obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  
  for key in list(INTRINSIC_PRIMITIVES.keys()):
    if modifier in INTRINSIC_PRIMITIVES[key]:
      modifier_key = key
      break
  
  num_main_obj = random.sample(range(1, num_objects), 1)[0]
  if len(obj_split) == 1 and obj_split[0] == "objects":
    num_main_obj = num_objects
  print(num_main_obj, num_objects)
    
  for i in range(num_main_obj):
    if True:
      # process main object
      if ("small" in obj_split or modifier == "small") or ("large" in obj_split or modifier == "large"):
        # additional process size
        if modifier == "small" or "small" in obj_split:
          size_name = "small"
          r = 0.35
        elif modifier == "large" or "large" in obj_split:
          size_name = "large"
          r = 0.7
        num_tries = 0
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_all_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
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
            
      else:
        num_tries = 0
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
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

      # Choose random color and shape
      if 'color' in obj_attr_types: 
        color_name = obj_attr_types['color']
        rgba = color_name_to_rgba[color_name]
            
      elif modifier_key == 'color':
        color_name = modifier
        rgba = color_name_to_rgba[color_name]
         
      else:
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
        
      if 'shape' in obj_attr_types:
        obj_name_out = obj_attr_types['shape']
        obj_name = object_inv_mapping[obj_name_out]
          
      elif modifier_key == 'shape':
        obj_name_out = modifier
        obj_name = object_inv_mapping[obj_name_out]
        
      else:
        obj_name, obj_name_out = random.choice(object_mapping)
      if obj_name == 'Cube':
        r /= math.sqrt(2)
        
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      if 'material' in obj_attr_types:
        mat_name_out = obj_attr_types['material']
        mat_name = material_inv_mapping[mat_name_out]
      elif modifier_key == 'material':
        mat_name_out = modifier
        mat_name = material_inv_mapping[mat_name_out]
      else:
        mat_name, mat_name_out = random.choice(material_mapping)
      # Actually add the object to the scene
      utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
      obj = bpy.context.object
      blender_objects.append(obj)
      positions.append((x, y, r))
      # Attach a random material
      utils.add_material(mat_name, Color=rgba)
      # Record data about the object in the scene data structure
      pixel_coords = utils.get_camera_coords(camera, obj.location)
      objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
      })  
      #print(objects[-1])
        
      
  for i in range(num_main_obj, num_objects):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_all_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()
    mat_name, mat_name_out = random.choice(material_mapping)
    # process objects to avoid repetition
    value_set = [obj_name_out, size_name, mat_name_out, color_name]
    print(value_set)
    
    obj_ud = " ".join(obj_split)
    
    count = 0
    split_count = len(obj_split)
    
    for value in value_set:
      if value in obj_ud:
        count+=1
    if "objects" in obj_ud:
        for value in value_set:
            if value == modifier:
                count += 1
                
    while count == split_count:
      size_name, r = random.choice(size_mapping)
      num_tries = 0
      while True:
        num_tries += 1
        if num_tries > args.max_retries:
          for obj in blender_objects:
            utils.delete_object(obj)
          return add_all_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
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
      # Choose random color and shape
      if shape_color_combos is None:
        obj_name, obj_name_out = random.choice(object_mapping)
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))
      else:
        obj_name_out, color_choices = random.choice(shape_color_combos)
        color_name = random.choice(color_choices)
        obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
        rgba = color_name_to_rgba[color_name]
      # For cube, adjust the size a bit
      if obj_name == 'Cube':
        r /= math.sqrt(2)
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      mat_name, mat_name_out = random.choice(material_mapping)
      # process objects to avoid repetition
      value_set = [obj_name_out, size_name, mat_name_out, color_name]
      
     
      count = 0
      for value in value_set:
        if value in obj_ud:
          count+=1
      if "objects" in obj_ud:
        for value in value_set:
            if value == modifier:
                count += 1
      
    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
   
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })


  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_all_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)

  return objects, blender_objects

def add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier):
  """
  Add random objects to the current blender scene
  """
  

  # Load the property file
  with open(args.properties_json, 'r') as f:
    properties = json.load(f)
    color_name_to_rgba = {}
    for name, rgb in properties['colors'].items():
      rgba = [float(c) / 255.0 for c in rgb] + [1.0]
      color_name_to_rgba[name] = rgba
    material_mapping = [(v, k) for k, v in properties['materials'].items()]
    material_inv_mapping = properties['materials'] # [(k, v) for k, v in properties['materials'].items()]
    
    object_mapping = [(v, k) for k, v in properties['shapes'].items()]
    object_inv_mapping = properties['shapes'] # [(k, v) for k, v in properties['shapes'].items()]
    
    size_mapping = list(properties['sizes'].items())

  shape_color_combos = None
  if args.shape_color_combos_json is not None:
    with open(args.shape_color_combos_json, 'r') as f:
      shape_color_combos = list(json.load(f).items())

  positions = []
  objects = []
  blender_objects = []
  # process the modifier
  modifier_key = 0
  another_modifier = ""
  obj_attr_types = {}
  for spl in obj_split:
    if spl == 'objects':
      continue
    if spl[-1] == 's':
      obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl[:-1]]] = spl[:-1]
    else:
      obj_attr_types[INVERSE_INTRINSIC_PRIMITIVES[spl]] = spl
  #print(obj_attr_types,obj_attr_types,obj_attr_types,obj_attr_types)
  for key in list(INTRINSIC_PRIMITIVES.keys()):
    if modifier in INTRINSIC_PRIMITIVES[key]:
      another_modifier = random.sample(INTRINSIC_PRIMITIVES[key], 1)[0]
      while another_modifier == modifier:
        another_modifier = random.sample(INTRINSIC_PRIMITIVES[key], 1)[0]
      modifier_key = key
      break
        
  for i in range(2):
    if i == 0:
      num_tries = 0
      # process main object
      if ("small" in obj_split or modifier == "small") or ("large" in obj_split or modifier == "large"):
        # additional process size
        if modifier == "small" or "small" in obj_split:
          size_name = "small"
          r = 0.35
        elif modifier == "large" or "large" in obj_split:
          size_name = "large"
          r = 0.7
        
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
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
      else:
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
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
          

      # Choose random color and shape
        
      if 'color' in obj_attr_types: 
        color_name = obj_attr_types['color']
        rgba = color_name_to_rgba[color_name]
            
      elif modifier_key == 'color':
        color_name = modifier
        rgba = color_name_to_rgba[color_name]
         
      else:
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
      print(color_name, color_name,color_name,color_name,color_name,color_name,color_name)
        
      if 'shape' in obj_attr_types:
        obj_name_out = obj_attr_types['shape']
        obj_name = object_inv_mapping[obj_name_out]
          
      elif modifier_key == 'shape':
        obj_name_out = modifier
        obj_name = object_inv_mapping[obj_name_out]
        
      else:
        obj_name, obj_name_out = random.choice(object_mapping)
      if obj_name == 'Cube':
        r /= math.sqrt(2)
        
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      # Attach a random material
      if 'material' in obj_attr_types:
        mat_name_out = obj_attr_types['material']
        mat_name = material_inv_mapping[mat_name_out]
      elif modifier_key == 'material':
        mat_name_out = modifier
        mat_name = material_inv_mapping[mat_name_out]
      else:
        mat_name, mat_name_out = random.choice(material_mapping)
      # Actually add the object to the scene
      utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
      obj = bpy.context.object
      blender_objects.append(obj)
      positions.append((x, y, r))
      utils.add_material(mat_name, Color=rgba)
      # Record data about the object in the scene data structure
      pixel_coords = utils.get_camera_coords(camera, obj.location)
      objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
        
        
    else:
      num_tries = 0
      # process main object
      if ("small" in obj_split or another_modifier == "small") or ("large" in obj_split or another_modifier == "large"):
        # additional process size
        if another_modifier == "small" or "small" in obj_split:
          size_name = "small"
          r = 0.35
        elif another_modifier == "large" or "large" in obj_split:
          size_name = "large"
          r = 0.7
        
        
        while True:
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, another_modifier)
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
      else:
        while True:
          size_name, r = random.choice(size_mapping)
          # If we try and fail to place an object too many times, then delete all
          # the objects in the scene and start over.
          num_tries += 1
          if num_tries > args.max_retries:
            for obj in blender_objects:
              utils.delete_object(obj)
            return add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, another_modifier)
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
          

      # Choose random color and shape
        
      if 'color' in obj_attr_types: 
        color_name = obj_attr_types['color']
        rgba = color_name_to_rgba[color_name]
            
      elif modifier_key == 'color':
        color_name = another_modifier
        rgba = color_name_to_rgba[color_name]
         
      else:
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))  
      #print(color_name, color_name,color_name,color_name,color_name,color_name,color_name)
        
      if 'shape' in obj_attr_types:
        obj_name_out = obj_attr_types['shape']
        obj_name = object_inv_mapping[obj_name_out]
          
      elif modifier_key == 'shape':
        obj_name_out = another_modifier
        obj_name = object_inv_mapping[obj_name_out]
        
      else:
        obj_name, obj_name_out = random.choice(object_mapping)
      if obj_name == 'Cube':
        r /= math.sqrt(2)
        
      # Choose random orientation for the object.
      theta = 360.0 * random.random()
      # Attach a random material
      if 'material' in obj_attr_types:
        mat_name_out = obj_attr_types['material']
        mat_name = material_inv_mapping[mat_name_out]
      elif modifier_key == 'material':
        mat_name_out = another_modifier
        mat_name = material_inv_mapping[mat_name_out]
      else:
        mat_name, mat_name_out = random.choice(material_mapping)
      # Actually add the object to the scene
      utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
      obj = bpy.context.object
      blender_objects.append(obj)
      positions.append((x, y, r))
      utils.add_material(mat_name, Color=rgba)
      # Record data about the object in the scene data structure
      pixel_coords = utils.get_camera_coords(camera, obj.location)
      objects.append({
          'shape': obj_name_out,
          'size': size_name,
          'material': mat_name_out,
          '3d_coords': tuple(obj.location),
          'rotation': theta,
          'pixel_coords': pixel_coords,
          'color': color_name,
        })
      
     
      
  for i in range(num_objects - 2):
    # Choose a random size
    size_name, r = random.choice(size_mapping)

    # Try to place the object, ensuring that we don't intersect any existing
    # objects and that we are more than the desired margin away from all existing
    # objects along all cardinal directions.
    num_tries = 0
    while True:
      # If we try and fail to place an object too many times, then delete all
      # the objects in the scene and start over.
      num_tries += 1
      if num_tries > args.max_retries:
        for obj in blender_objects:
          utils.delete_object(obj)
        return add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)
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

    # Choose random color and shape
    if shape_color_combos is None:
      obj_name, obj_name_out = random.choice(object_mapping)
      color_name, rgba = random.choice(list(color_name_to_rgba.items()))
    else:
      obj_name_out, color_choices = random.choice(shape_color_combos)
      color_name = random.choice(color_choices)
      obj_name = [k for k, v in object_mapping if v == obj_name_out][0]
      rgba = color_name_to_rgba[color_name]

    # For cube, adjust the size a bit
    if obj_name == 'Cube':
      r /= math.sqrt(2)

    # Choose random orientation for the object.
    theta = 360.0 * random.random()

    # Actually add the object to the scene
    utils.add_object(args.shape_dir, obj_name, r, (x, y), theta=theta)
    obj = bpy.context.object
    blender_objects.append(obj)
    positions.append((x, y, r))

    # Attach a random material
    mat_name, mat_name_out = random.choice(material_mapping)
    utils.add_material(mat_name, Color=rgba)

    # Record data about the object in the scene data structure
    pixel_coords = utils.get_camera_coords(camera, obj.location)
    objects.append({
      'shape': obj_name_out,
      'size': size_name,
      'material': mat_name_out,
      '3d_coords': tuple(obj.location),
      'rotation': theta,
      'pixel_coords': pixel_coords,
      'color': color_name,
    })


  # Check that all objects are at least partially visible in the rendered image
  all_visible = check_visibility(blender_objects, args.min_pixels_per_object)
  if not all_visible:
    # If any of the objects are fully occluded then start over; delete all
    # objects from the scene and place them all again.
    print('Some objects are occluded; replacing objects')
    for obj in blender_objects:
      utils.delete_object(obj)
    return add_some_intrinsic_random_objects(scene_struct, num_objects, args, camera, obj_split, modifier)

  return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
  """
  Computes relationships between all pairs of objects in the scene.
  
  Returns a dictionary mapping string relationship names to lists of lists of
  integers, where output[rel][i] gives a list of object indices that have the
  relationship rel with object i. For example if j is in output['left'][i] then
  object j is left of object i.
  """
  all_relationships = {}
  for name, direction_vec in scene_struct['directions'].items():
    if name == 'above' or name == 'below': continue
    all_relationships[name] = []
    for i, obj1 in enumerate(scene_struct['objects']):
      coords1 = obj1['3d_coords']
      related = set()
      for j, obj2 in enumerate(scene_struct['objects']):
        if obj1 == obj2: continue
        coords2 = obj2['3d_coords']
        diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
        dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
        if dot > eps:
          related.add(j)
      all_relationships[name].append(sorted(list(related)))
  return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
  """
  Check whether all objects in the scene have some minimum number of visible
  pixels; to accomplish this we assign random (but distinct) colors to all
  objects, and render using no lighting or shading or antialiasing; this
  ensures that each object is just a solid uniform color. We can then count
  the number of pixels of each color in the output image to check the visibility
  of each object.

  Returns True if all objects are visible and False otherwise.
  """
  f, path = tempfile.mkstemp(suffix='.png')
  object_colors = render_shadeless(blender_objects, path=path)
  img = bpy.data.images.load(path)
  p = list(img.pixels)
  color_count = Counter((p[i], p[i+1], p[i+2], p[i+3])
                        for i in range(0, len(p), 4))
  os.remove(path)
  if len(color_count) != len(blender_objects) + 1:
    return False
  for _, count in color_count.most_common():
    if count < min_pixels_per_object:
      return False
  return True

def render_mask_shadeless(blender_objects, path='flat.png'):
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
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    #while True:
    r, g, b = (i * 5 + 128) / 255, (i * 5 + 128) / 255, (i * 5 + 128) / 255 # [random.random() for _ in range(3)]
    #if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
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

  # return object_colors


def render_shadeless(blender_objects, path='flat.png'):
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
  render_args.filepath = path
  render_args.engine = 'BLENDER_RENDER'
  render_args.use_antialiasing = False

  # Move the lights and ground to layer 2 so they don't render
  utils.set_layer(bpy.data.objects['Lamp_Key'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Fill'], 2)
  utils.set_layer(bpy.data.objects['Lamp_Back'], 2)
  utils.set_layer(bpy.data.objects['Ground'], 2)

  # Add random shadeless materials to all objects
  object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    while True:
      r, g, b = [random.random() for _ in range(3)]
      if (r, g, b) not in object_colors: break
    object_colors.add((r, g, b))
    mat.diffuse_color = [r, g, b]
    mat.use_shadeless = True
    obj.data.materials[0] = mat

  # Render the scene
  bpy.ops.render.render(write_still=True)

  # Undo the above; first restore the materials to objects
  for mat, obj in zip(old_materials, blender_objects):
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

  return object_colors


if __name__ == '__main__':
  if INSIDE_BLENDER:
    # Run normally
    argv = utils.extract_args()
    args = parser.parse_args(argv)
    
    # main_connective_cancel_intrinsic(args)
    main_some_intrinsic(args)

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

