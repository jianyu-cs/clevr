import cv2
import pycocotools
from pycocotools.mask import encode
import argparse
import os
import json
import pandas as pd
import copy
import numpy as np
import shutil

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
parser.add_argument('--output_image_dir', default='./output/images/',
    help="The directory where output images will be stored. It will be " +
         "created if it does not exist.")
parser.add_argument('--output_scene_dir', default='./output/scenes/',
    help="The directory where output JSON scene structures will be stored. " +
         "It will be created if it does not exist.")
parser.add_argument('--output_scene_file', default='./output/CLEVR_scenes.json',
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
parser.add_argument('--type', default="direct_cancel", type=str,
    help="The minimum number of bounces to use for rendering.")


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)
    
def process_mask(mask_path, scene_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask_list = mask.tolist()
    value_set = []
    for value in mask_list:
        for element in value:
            if element!=64 and element!=255:
                value_set.append(element)
    print(mask_path)
    value_set = list(set(value_set))
    value_set.sort()
    # load the scene struct
    with open(scene_path, "r") as f:
        scene_struct = json.load(f)

    for i, obj in enumerate(scene_struct['objects']):
        temp_mask = (mask == value_set[i])
        scene_struct['objects'][i]['mask'] = encode(np.asfortranarray(temp_mask))
        print(scene_struct['objects'][i]['mask'])

    f = open(scene_path, "w", encoding="utf-8")
    f.write(json.dumps(scene_struct, cls=MyEncoder))
    f.close()
    
def main(args):
    num_digits=6
    prefix = '%s_' % (args.filename_prefix)
    if 'direct' == args.type: 
   
      mask_template = '%s%%0%dd.png' % (prefix, 6)
      dist1_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist2_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      another_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist3_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      mask_template_ = os.path.join(args.output_image_dir, 'connect_implicature_mask', mask_template)
      another_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", another_mask_template)
      dist1_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", dist1_mask_template)
      dist2_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", dist2_mask_template)
      dist3_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", dist3_mask_template)
    
      scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist1_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist2_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist3_scene_template = '%s%%0%dd.json' % (prefix, 6)
      another_scene_template = '%s%%0%dd.json' % (prefix, 6)
        
      args.implicature_type = 'direct_implicature'
      
      img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist1_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist2_img_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      another_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist3_img_template = '%s%%0%dd.png' % (prefix, num_digits)

    elif 'direct_cancel' == args.type: 
    
      mask_template = '%s%%0%dd.png' % (prefix, 6)
      dist1_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist2_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      another_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist3_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      mask_template_ = os.path.join(args.output_image_dir, 'cancel_connect_implicature_mask', mask_template)
      another_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", another_mask_template)
      dist1_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", dist1_mask_template)
      dist2_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", dist2_mask_template)
      dist3_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", dist3_mask_template)
      
      args.implicature_type = 'cancel_direct_implicature'
    
      scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist1_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist2_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist3_scene_template = '%s%%0%dd.json' % (prefix, 6)
      another_scene_template = '%s%%0%dd.json' % (prefix, 6)
      
      img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist1_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist2_img_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      another_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist3_img_template = '%s%%0%dd.png' % (prefix, num_digits)
        
    if 'indirect' == args.type: 
    
      mask_template = '%s%%0%dd.png' % (prefix, 6)
      dist1_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist2_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      another_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist3_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      mask_template_ = os.path.join(args.output_image_dir, 'connect_implicature_mask', mask_template)
      another_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", another_mask_template)
      dist1_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", dist1_mask_template)
      dist2_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", dist2_mask_template)
      dist3_mask_template_ = os.path.join(args.output_image_dir, "connect_implicature_mask", dist3_mask_template)
    
      scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist1_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist2_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist3_scene_template = '%s%%0%dd.json' % (prefix, 6)
      another_scene_template = '%s%%0%dd.json' % (prefix, 6)
        
      args.implicature_type = 'indirect_implicature'
      
      img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist1_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist2_img_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      another_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist3_img_template = '%s%%0%dd.png' % (prefix, num_digits)
        
    elif 'indirect_cancel' == args.type: 
    
      mask_template = '%s%%0%dd.png' % (prefix, 6)
      dist1_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist2_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      another_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      dist3_mask_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      
      mask_template_ = os.path.join(args.output_image_dir, 'cancel_connect_implicature_mask', mask_template)
      another_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", another_mask_template)
      dist1_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", dist1_mask_template)
      dist2_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", dist2_mask_template)
      dist3_mask_template_ = os.path.join(args.output_image_dir, "cancel_connect_implicature_mask", dist3_mask_template)
    
      scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist1_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist2_scene_template = '%s%%0%dd.json' % (prefix, 6)
      dist3_scene_template = '%s%%0%dd.json' % (prefix, 6)
      another_scene_template = '%s%%0%dd.json' % (prefix, 6)
      
      args.implicature_type = 'cancel_indirect_implicature'
      
      img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist1_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist2_img_template = '%s%%0%dd.png' % (prefix, num_digits)
      
      another_img_template = '%s%%0%dd.png' % (prefix, num_digits)
    
      dist3_img_template = '%s%%0%dd.png' % (prefix, num_digits)
     
    if 'cancel' in args.type:
      scene_template_ = os.path.join(args.output_scene_dir, "cancel_connect_implicature", scene_template)
      another_scene_template_ = os.path.join(args.output_scene_dir, "cancel_connect_implicature", another_scene_template)
      dist1_scene_template_ = os.path.join(args.output_scene_dir, "cancel_connect_implicature", dist1_scene_template)
      dist2_scene_template_ = os.path.join(args.output_scene_dir, "cancel_connect_implicature", dist2_scene_template)
      dist3_scene_template_ = os.path.join(args.output_scene_dir, "cancel_connect_implicature", dist3_scene_template)

    else:
      scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", scene_template)
      another_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", another_scene_template)
      dist1_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", dist1_scene_template)
      dist2_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", dist2_scene_template)
      dist3_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", dist3_scene_template)

    jsonl_path_ = '%s%%0%dd.jsonl' % (prefix, num_digits)
    if 'cancel' in args.type:
      jsonl_path= copy.deepcopy(os.path.join(args.output_image_dir, "cancel_connect_implicature", jsonl_path_))
    else:
      jsonl_path= copy.deepcopy(os.path.join(args.output_image_dir, "connect_implicature", jsonl_path_))
    print(jsonl_path)
    print(dist3_img_template)
    #with open(utterance_template, "r") as f:
    #    utterance = f.readlines()[0]
    #scene_template = os.path.join(args.output_scene_dir, "connect_implicature", scene_template)
    #another_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", another_scene_template)
    #dist1_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", dist1_scene_template)
    #dist2_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", dist2_scene_template)
    #dist3_scene_template_ = os.path.join(args.output_scene_dir, "connect_implicature", dist3_scene_template)
    
    if not os.path.exists(os.path.join(args.output_image_dir, 'connect_implicature', "snippets")):
      os.mkdir(os.path.join(args.output_image_dir, 'connect_implicature', "snippets"))
    '''
    final_mask_template = os.path.join(args.output_image_dir, 'connect_implicature', "snippets", mask_template)
    final_another_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", another_mask_template)
    final_dist1_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", dist1_mask_template)
    final_dist2_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", dist2_mask_template)
    final_dist3_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", dist3_mask_template)
    
    final_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", scene_template)
    final_another_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", another_scene_template)
    final_dist1_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", dist1_scene_template)
    final_dist2_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", dist2_scene_template)
    final_dist3_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", dist3_scene_template)
   
    final_jsonl_path = os.path.join(args.output_image_dir, "connect_implicature", "snippets", jsonl_path_)
    '''
    
    class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, bytes):
                return str(obj, encoding='utf-8')
            return json.JSONEncoder.default(self, obj)
        
    for i in range(args.num_images):
        if not os.path.exists(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type)):
          os.mkdir(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type))
        if not os.path.exists(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i))):
          os.mkdir(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i)))
        if not os.path.exists(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "images")):
          os.mkdir(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "images"))
        if not os.path.exists(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "masks")):
          os.mkdir(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "masks"))
        if not os.path.exists(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "SoMs")):
          os.mkdir(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "SoMs"))
        if not os.path.exists(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "jsons")):
          os.mkdir(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "jsons"))
    
        final_mask_template = os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "masks", mask_template % (5 * i + args.start_idx))
        final_another_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "masks", another_mask_template % (5 * i + args.start_idx + 1))
        final_dist1_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "masks", dist1_mask_template % (5 * i + args.start_idx + 2))
        final_dist2_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "masks", dist2_mask_template % (5 * i + args.start_idx + 3))
        final_dist3_mask_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "masks", dist3_mask_template % (5 * i + args.start_idx + 4))
    
        final_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "SoMs", scene_template)
        final_another_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "SoMs", another_scene_template)
        final_dist1_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "SoMs", dist1_scene_template)
        final_dist2_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "SoMs", dist2_scene_template)
        final_dist3_scene_template = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "SoMs", dist3_scene_template)
    
        final_jsonl_path = os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "jsons", jsonl_path_)
        
        scene_path = scene_template_ % (5 * i + args.start_idx)
        another_scene_path = another_scene_template_ % (5 * i + args.start_idx + 1)
        
        dist1_scene_path = dist1_scene_template_ % (5 * i + args.start_idx + 2)
        dist2_scene_path = dist2_scene_template_ % (5 * i + args.start_idx + 3)
        dist3_scene_path = dist3_scene_template_ % (5 * i + args.start_idx + 4)
        
        if 'cancel' in args.implicature_type:
          temp_dir = "cancel_connect_implicature"
          temp_dir_mask = "cancel_connect_implicature_mask"
        else:
          temp_dir = "connect_implicature"
          temp_dir_mask = "connect_implicature_mask"
        
        mask_path = os.path.join(args.output_image_dir, temp_dir_mask, mask_template % (5 * i + args.start_idx))
        another_mask_path = os.path.join(args.output_image_dir, temp_dir_mask, another_mask_template % (5 * i + args.start_idx + 1))
        dist1_mask_path = os.path.join(args.output_image_dir, temp_dir_mask, dist1_mask_template % (5 * i + args.start_idx + 2))
        dist2_mask_path = os.path.join(args.output_image_dir, temp_dir_mask,dist2_mask_template % (5 * i + args.start_idx + 3))
        dist3_mask_path = os.path.join(args.output_image_dir, temp_dir_mask,dist3_mask_template % (5 * i + args.start_idx + 4))
        
        final_scene_path = final_scene_template % (5 * i + args.start_idx)
        final_another_scene_path = final_another_scene_template % (5 * i + args.start_idx + 1)
        
        final_dist1_scene_path = final_dist1_scene_template % (5 * i + args.start_idx + 2)
        final_dist2_scene_path = final_dist2_scene_template % (5 * i + args.start_idx + 3)
        final_dist3_scene_path = final_dist3_scene_template % (5 * i + args.start_idx + 4)

        final_mask_path = final_mask_template #% (i + args.start_idx)
        final_another_mask_path = final_another_mask_template #% (i + args.start_idx)
        final_dist1_mask_path = final_dist1_mask_template #% (i + args.start_idx)
        final_dist2_mask_path = final_dist2_mask_template #% (i + args.start_idx)
        final_dist3_mask_path = final_dist3_mask_template #% (i + args.start_idx)
        print(img_template, jsonl_path)
        img_path = img_template % (5 * i + args.start_idx)
        print(img_path, jsonl_path)
        
        jsonl_pathh = jsonl_path % (i + args.start_idx)
        jsonl_pathh_ = jsonl_path_ % (i + args.start_idx)
   
        another_img_path = another_img_template % (5 * i + args.start_idx + 1)
        
        dist1_img_path = dist1_img_template % (5 * i + args.start_idx + 2)
       
        dist2_img_path = dist2_img_template % (5 * i + args.start_idx + 3)
        
        dist3_img_path = dist3_img_template % (5 * i + args.start_idx + 4)
        
        shutil.copy(os.path.join(args.output_image_dir, temp_dir, img_path), os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i), "images", img_path))
        shutil.copy(os.path.join(args.output_image_dir, temp_dir, another_img_path), os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i),"images",another_img_path))
        shutil.copy(os.path.join(args.output_image_dir, temp_dir, dist1_img_path), os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i),"images",dist1_img_path))
        shutil.copy(os.path.join(args.output_image_dir, temp_dir, dist2_img_path), os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i),"images",dist2_img_path))
        shutil.copy(os.path.join(args.output_image_dir, temp_dir, dist3_img_path), os.path.join(args.output_image_dir, "connect_implicature", "snippets", args.implicature_type, str(i),"images",dist3_img_path))
         
        
        process_mask(mask_path, scene_path)
        process_mask(another_mask_path, another_scene_path)
        process_mask(dist1_mask_path, dist1_scene_path)
        process_mask(dist2_mask_path, dist2_scene_path)
        process_mask(dist3_mask_path, dist3_scene_path)
        
        shutil.copy(scene_path, final_scene_path)
        shutil.copy(another_scene_path, final_another_scene_path)
        shutil.copy(dist1_scene_path, final_dist1_scene_path)
        shutil.copy(dist2_scene_path, final_dist2_scene_path)
        shutil.copy(dist3_scene_path, final_dist3_scene_path)
        
        shutil.copy(mask_path, final_mask_template)
        print(mask_path, final_mask_template)
        shutil.copy(another_mask_path, final_another_mask_template)
        shutil.copy(dist1_mask_path, final_dist1_mask_template)
        shutil.copy(dist2_mask_path, final_dist2_mask_template)
        shutil.copy(dist3_mask_path, final_dist3_mask_template)
        
        '''
        new_jsonl = []
        with open(jsonl_pathh, 'r') as f:
            result = list(f)
        new_jsonl.append(json.dumps(result[0]))
        
        with open(final_scene_path, 'r') as f:
            new_jsonl.append(json.load(f))
            
        with open(final_another_scene_path, 'r') as f:
            new_jsonl.append(json.load(f))
        
        with open(final_dist1_scene_path, 'r') as f:
            new_jsonl.append(json.load(f))
        
        with open(final_dist2_scene_path, 'r') as f:
            new_jsonl.append(json.load(f))
        
        with open(final_dist3_scene_path, 'r') as f:
            new_jsonl.append(json.load(f))
            
        with open(jsonl_pathh, 'w') as f:
            for entry in new_jsonl:
                json.dump(entry, f)
                f.write('\n')
        '''
                
        shutil.copy(jsonl_pathh, os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "jsons", jsonl_pathh_))
        ans = []
        with open(final_scene_path, "r") as f:
          ans.append(json.load(f))
        with open(final_another_scene_path, "r") as f:
          ans.append(json.load(f))
        with open(final_dist1_scene_path, "r") as f:
          ans.append(json.load(f))
        with open(final_dist2_scene_path, "r") as f:
          ans.append(json.load(f))
        with open(final_dist3_scene_path, "r") as f:
          ans.append(json.load(f))
          
        with open(jsonl_pathh, 'r') as f:
          temp_ans = list(f)
        temp_ans = [json.loads(_) for _ in temp_ans]
        with open(os.path.join(args.output_image_dir, 'connect_implicature', "snippets", args.implicature_type, str(i), "jsons", jsonl_pathh_), "w") as f:
          f.write(json.dumps(temp_ans[0]))
          f.write('\n')
          for _ in ans:
            f.write(json.dumps(_))
            f.write('\n')
        

if __name__ == '__main__':
    #argv = utils.extract_args()
    args = parser.parse_args()
    main(args)
