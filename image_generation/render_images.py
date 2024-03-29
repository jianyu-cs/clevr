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
  #import cv2
  #import pycocotools
  #from pycocotools.mask import encode
  prefix = '%s_' % (args.filename_prefix)
  img_template = '%s%%0%dd.png' % (prefix, num_digits)
  mask_template = '%s%%0%dd.png' % (prefix, num_digits)
  scene_template = '%s%%0%dd.json' % (prefix, num_digits)
  blend_template = '%s%%0%dd.blend' % (prefix, num_digits)
  type_txt_template = '%sgame_difficulty_level.txt' % (prefix)

  if not os.path.exists(os.path.join(args.output_image_dir, 'ad_hoc')):
    os.mkdir(os.path.join(args.output_image_dir, 'ad_hoc'))
    
  if not os.path.exists(os.path.join(args.output_scene_dir, 'ad_hoc')):
    os.mkdir(os.path.join(args.output_scene_dir, 'ad_hoc'))
    
  if not os.path.exists(os.path.join(args.output_image_dir, 'ad_hoc', 'images')):
    os.mkdir(os.path.join(args.output_image_dir, 'ad_hoc', 'images'))
     
  if not os.path.exists(os.path.join(args.output_image_dir, 'ad_hoc', 'masks')):
    os.mkdir(os.path.join(args.output_image_dir, 'ad_hoc', 'masks'))
    
  if not os.path.exists(os.path.join(args.output_scene_dir, 'ad_hoc', 'meta_data')):
    os.mkdir(os.path.join(args.output_scene_dir, 'ad_hoc', 'meta_data'))
        
  img_template = os.path.join(args.output_image_dir, 'ad_hoc', 'images', img_template)
  mask_template = os.path.join(args.output_image_dir, 'ad_hoc', 'masks', mask_template)
  
  scene_template = os.path.join(args.output_scene_dir, 'ad_hoc', 'meta_data', scene_template)
  type_txt_template = os.path.join(args.output_scene_dir, 'ad_hoc', 'meta_data', type_txt_template)
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
    mask_path = mask_template % (i + args.start_idx)
    scene_path = scene_template % (i + args.start_idx)
    all_scene_paths.append(scene_path)
    blend_path = None
    if args.save_blendfiles == 1:
      blend_path = blend_template % (i + args.start_idx)
    num_objects = random.randint(4, 10)
    obj_ind, utterance = render_ad_hoc_scene(args,
      num_objects=num_objects,
      output_index=(i + args.start_idx),
      output_split=args.split,
      output_image=img_path,
      output_scene=scene_path,
      output_blendfile=blend_path, #os.path.join(args.output_blend_dir, '%sad_hoc_%d.blend' % (prefix, i))
      output_mask_image=mask_path,
      output_type_txt=type_txt_template,
    )
    while obj_ind==None:
      num_objects = random.randint(4, 10)
      obj_ind, utterance = render_ad_hoc_scene(args,
              num_objects=num_objects,
              output_index=(i + args.start_idx),
              output_split=args.split,
              output_image=img_path,
              output_scene=scene_path,
              output_blendfile=blend_path,
              output_mask_image=mask_path,
              output_type_txt=type_txt_template
              )
      
  '''  
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
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)

def render_ad_hoc_scene(args,
    num_objects=5,
    output_index=0,
    output_split='none',
    output_image='render.png',
    output_scene='render_json',
    output_blendfile=None,
    output_mask_image='mask.png',
    output_type_txt='easy.txt',
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
  objects, blender_objects, utterance_obj_pairs, full_matrix, raw_utterance_obj_pairs = add_ad_hoc_random_objects(scene_struct, num_objects, args, camera)
  print("****", objects, "****")
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
  
  utterance_split = utterance.split(" ")
  other_utterances = []
  other_utterances_set = set()
  for i, obj in enumerate(objects):
    string = obj['size'] + " " + obj['color'] + " " + obj['material']+ " " + obj['shape']
    temp_flag = True
    for u_split in utterance_split:
        if u_split not in string:
            temp_flag = False
    if temp_flag:
        if obj_ind != i:
            print(i, raw_utterance_obj_pairs)
            other_utterances.append(raw_utterance_obj_pairs[i][1][0])
            other_utterances_set.add(raw_utterance_obj_pairs[i][1][0])
            
  with open(output_type_txt, "a") as f:
    if len(other_utterances) == len(other_utterances_set):
        f.write(output_scene + "\tEasy"+"\n")
    else:
        f.write(output_scene + "\tDifficult"+"\n")
   
  for i, id_utter_list in enumerate(raw_utterance_obj_pairs):
    utter = id_utter_list[1]
    if utter == utterance:
        temp_obj_id = id_utter_list[0]

  if count > 1:
    print("break")
    return None, None

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

  render_mask_shadeless(blender_objects, output_mask_image)
  #mask = cv2.imread(output_mask_image, cv2.IMREAD_GRAYSCALE)
  #mask_list = mask.tolist()
  #value_set = []
  #for value in mask_list:
  #  for element in value:
  #      if element!=64 and element!=255:
  #          value_set.append(element)
  #value_set = list(set(value_set))
  #value_set.sort()
  #for i, obj in enumerate(scene_struct['objects']):
  #  temp_mask = (mask == value_set[i])
  #  scene_struct['objects'][i]['mask'] = encode(np.asfortranarray(temp_mask))
  
  with open(output_scene, 'w') as f:
    json.dump(scene_struct, f, indent=2)

  if output_blendfile is not None:
    bpy.ops.wm.save_as_mainfile(filepath=output_blendfile)
    
  return (obj_ind, utterance)
    
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
  full_utterance_obj_pairs = []
  for i, obj_name in enumerate(objects):
    # current ind max
    string = obj_name['shape']+obj_name['size']+obj_name['material']+obj_name['color']
    #if objects_whole_string[string] > 1:
    #  continue
    #print(full_matrix[:,i])
    if len(np.where(full_matrix[:,i]==full_matrix[:,i].max())[0]) == 1 and full_matrix[:,i].max() < 0.9:
      print(full_matrix[:,i])
      print(keys)
      print(np.where(full_matrix[:,i]==full_matrix[:,i].max()))
      print(obj_name)
      utterance_obj_pairs.append([i, keys[np.argmax(full_matrix[:,i])]])
      full_utterance_obj_pairs.append([i, [keys[np.argmax(full_matrix[:,i])]]])
        
    elif len(np.where(full_matrix[:,i]==full_matrix[:,i].max())[0]) != 1:
      temp = np.where(full_matrix[:,i]==full_matrix[:,i].max())[0]
      temp_keys = []
      for tmp in temp:
        temp_keys.append(keys[tmp])
      #temp_key = keys[]
      full_utterance_obj_pairs.append([i, temp_keys])
      repeat_obj_pairs = repeat_obj_pairs+temp_keys
    elif full_matrix[:,i].max() >= 0.9:
      full_utterance_obj_pairs.append([i, keys[np.argmax(full_matrix[:,i])]])
        
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
    print("()()()()()()()()()()()()()()()")
    return None, None, None
  else:
    return full_utterance_obj_pairs, updated_utterance_obj_pairs, full_matrix 
    
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

  raw_utterance_obj_pairs, utterance_obj_pairs, full_matrix = calculate_ad_hoc_matrix(objects)

  print("dsjggfhdgfhdghfghdgfhdvcdfd")
  print(raw_utterance_obj_pairs)
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

  return objects, blender_objects, utterance_obj_pairs, full_matrix, raw_utterance_obj_pairs

def add_connective_random_objects(scene_struct, num_objects, args, camera, language = "blue balls and green cubes"):
  """
  Add random objects to the current blender scene
  Input: <A and B> or <A> or <CA and CB>, or <CA and DA> or <CA>, where A is the shape. 
  <size> <color> <material> <shape>
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
  # parse the language
  and_idx = 2 if "and" in language else 1
  involved_objects = language.split(" and ")
    
  SIZES = list(properties['sizes'].keys())
  SHAPES = list(properties['shapes'].keys())
  MATERIALS = list(properties['materials'].keys())
  COLORS = list(properties['colors'].keys())
    
  # first object
  for i in range(and_idx):
    # parsing
    current_obj = involved_objects[i]
    #current_size = 
    
    
  
    
  # additional objects
  for i in range(num_objects-and_idx):
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
  #object_colors = set()
  old_materials = []
  for i, obj in enumerate(blender_objects):
    old_materials.append(obj.data.materials[0])
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.name = 'Material_%d' % i
    #while True:
    r, g, b = (i * 5 + 128) / 255, (i * 5 + 128) / 255, (i * 5 + 128) / 255 #[random.random() for _ in range(3)]
    #if (r, g, b) not in object_colors: break
    #object_colors.add((r, g, b))
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

