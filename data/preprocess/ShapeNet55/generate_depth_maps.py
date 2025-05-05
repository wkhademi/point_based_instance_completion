'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

"""
Modified version of Wentao Yuan's script for rendering depth maps in blender.
This version has been updated to run with newer versions of blender/bpy (3.4.1).

https://github.com/wentaoyuan/pcn/blob/master/render/render_depth.py
"""

import os
import bpy
import sys
import time
import argparse
import mathutils
import numpy as np


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def setup_blender(width, height, focal_length, data_dir):
    # camera
    camera = bpy.data.objects["Camera"]
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2

    # render layer
    scene = bpy.context.scene
    scene.render.filepath = "buffer"
    scene.render.image_settings.color_depth = "16"
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # compositor nodes
    scene.use_nodes = True
    scene.view_layers["ViewLayer"].use_pass_z = True
    tree = scene.node_tree
    rl = tree.nodes.new("CompositorNodeRLayers")
    output = tree.nodes.new("CompositorNodeOutputFile")
    output.base_path = data_dir
    output.file_slots[0].use_node_format = True
    output.format.file_format = "OPEN_EXR"
    tree.links.new(rl.outputs["Depth"], output.inputs[0])

    # remove default cube
    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.object.delete()

    return scene, camera, output


def random_pose():
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi

    # construct rotation
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    R = np.dot(Rz, np.dot(Ry, Rx))

    # set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2], 1)

    pose = np.concatenate([np.concatenate([R, t], axis=1), [[0, 0, 0, 1]]], axis=0)

    return pose


if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--num_scans", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--focal", type=float, default=100)
    parser.add_argument("--mesh_scale", type=float, default=0.65)
    args = parser.parse_args()

    # setup blender scene for rendering
    scene, camera, output = setup_blender(
        args.width, args.height, args.focal, args.data_dir
    )

    # save intrinsics
    intrinsics = np.array([
        [args.focal, 0, args.width / 2],
        [0, args.focal, args.height / 2],
        [0, 0, 1]
    ])
    np.savetxt(os.path.join(args.data_dir, "intrinsics.txt"), intrinsics, "%f")

    open("blender.log", "w+").close()

    # get list of category directories
    category_ids = [
        category_id for category_id in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, category_id))
    ]

    for category_id in category_ids:
        category_dir = os.path.join(args.data_dir, category_id)
        model_list = os.listdir(category_dir)

        for model_id in model_list:
            start = time.time()

            depth_dir = os.path.join(category_id, model_id, "depth")
            pose_dir = os.path.join(category_dir, model_id, "pose")
            os.makedirs(pose_dir, exist_ok=True)

            # redirect output to log file
            old_os_out = os.dup(1)
            os.close(1)
            os.open("blender.log", os.O_WRONLY)

            # import mesh model
            model_path = os.path.join(category_dir, model_id, "model.obj")
            bpy.ops.import_scene.obj(filepath=model_path)

            # rotate model by 90 degrees around x-axis (z-up => y-up)
            # to match ShapeNet's coordinates
            bpy.ops.transform.rotate(value=np.pi / 2, orient_axis="X")

            # scale model to be within [-0.5, 0.5] cube
            bpy.ops.transform.resize(
                value=(args.mesh_scale, args.mesh_scale, args.mesh_scale)
            )

            # render
            for i in range(args.num_scans):
                scene.frame_set(i)

                # generate random pose
                pose = random_pose()
                camera.matrix_world = mathutils.Matrix(pose)
                np.savetxt(os.path.join(pose_dir, f"{str(i)}.txt"), pose, "%f")

                # render and save depth
                output.file_slots[0].path = os.path.join(depth_dir, "#.exr")
                bpy.ops.render.render(write_still=True)

            # clean up
            bpy.ops.object.delete()
            for m in bpy.data.meshes:
                bpy.data.meshes.remove(m)
            for m in bpy.data.materials:
                m.user_clear()
                bpy.data.materials.remove(m)

            # Show time
            os.close(1)
            os.dup(old_os_out)
            os.close(old_os_out)
            print("%s %s done, time=%.4f sec" % (category_id, model_id, time.time() - start))
