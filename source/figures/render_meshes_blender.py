import os
import sys
import bpy
import site

# # run blender with elevated privileges and uncomment to following lines to install required packages
# import ensurepip
# import subprocess
# ensurepip.bootstrap()
# os.environ.pop('PIP_REQ_TRACKER', None)
# from pathlib import Path
#
# python_path = next((Path(sys.prefix)/'bin').glob('python*'))
#
# subprocess.check_output([python_path, '-m', 'pip', 'install', 'numpy'])
# subprocess.check_output([python_path, '-m', 'pip', 'install', 'scipy'])
# subprocess.check_output([python_path, '-m', 'pip', 'install', 'trimesh'])
# subprocess.check_output([python_path, '-m', 'pip', 'install', 'networkx'])

usersitepackagespath = site.getusersitepackages()
if os.path.exists(usersitepackagespath) and usersitepackagespath not in sys.path:
    sys.path.append(usersitepackagespath)

import numpy as np
import scipy.spatial
import trimesh
import bmesh


def eval_cmap(vals, cmap_colors):
    # print((np.clip(vals, a_min=0.0, a_max=1.0)*(cmap_colors.shape[0]-1)).round().astype('int32').max())
    # print((np.clip(vals, a_min=0.0, a_max=1.0)*(cmap_colors.shape[0]-1)).round().astype('int32').min())
    # print(vals.min())
    # print(np.isnan(vals).sum())
    colors = cmap_colors[(np.clip(vals, a_min=0.0, a_max=1.0) * (cmap_colors.shape[0] - 1)).round().astype('int32'), :]
    return np.concatenate([colors, np.ones(shape=[colors.shape[0], 1], dtype=colors.dtype)], axis=1)  # add alpha


def rotation_between_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)), (vec2 / np.linalg.norm(vec2, axis=1, keepdims=True))
    cos_angle = (a * b).sum(axis=1, keepdims=True)
    axis = np.cross(a, b)
    axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
    axis[cos_angle[:, 0] < -0.99999, :] = np.array([[1.0, 0.0, 0.0]])

    rot = scipy.spatial.transform.Rotation.from_rotvec(axis * np.arccos(cos_angle))
    return rot.as_matrix()


def copy_animation_data(src_obj, dst_obj):
    src_ad = src_obj.animation_data

    if dst_obj.animation_data is None:
        dst_obj.animation_data_create()
    dst_ad = dst_obj.animation_data

    for src_prop in src_ad.bl_rna.properties:
        if not src_prop.is_readonly:
            setattr(dst_ad, src_prop.identifier, getattr(src_ad, src_prop.identifier))


def render_meshes(input_dir, output_dir):
    clear = False
    fix_wires = True
    turning_animation = False
    render_wireframe = False

    # input_dir = '/home/lizeth/Downloads/for rendering/comp/abc/00013052_9084b77631834dd584b2ac93_trimesh_033/'
    # output_dir = '/home/lizeth/Downloads/for rendering/rendered/abc/00013052_9084b77631834dd584b2ac93_trimesh_033/'
    # input_dir = '/home/lizeth/Downloads/for rendering/comp/abc/00014452_55263057b8f440a0bb50b260_trimesh_017/'
    # output_dir = '/home/lizeth/Downloads/for rendering/rendered/abc/00014452_55263057b8f440a0bb50b260_trimesh_017/'
    # input_dir = '/home/lizeth/Downloads/for rendering/comp/abc/00017014_fbef9df8f24940a0a2df6ccb_trimesh_001/'
    # output_dir = '/home/lizeth/Downloads/for rendering/rendered/abc/00017014_fbef9df8f24940a0a2df6ccb_trimesh_001/'
    # input_dir = '/home/lizeth/Downloads/for rendering/comp/abc/00990573_d1914c7f68f9a6b58bed9421_trimesh_000/'
    # output_dir = '/home/lizeth/Downloads/for rendering/rendered/abc/00990573_d1914c7f68f9a6b58bed9421_trimesh_000/'


    scale_vecfield = True
    vecfield_prefix = None
    use_boundaries = True
    boundary_edges_prefix = 'boundary_edges_'
    boundary_verts_prefix = 'boundary_coordinates_'
    vcolor_max = 1.5560769862496584
    vcolor_min = -1.60086702642704
    vcolor_prefix = 'trig_size_'
    vcolor_suffix = ''
    render_wireframe = False
    # boundary_exclude_prefix = ['gt', 'baseline', 'ours']
    boundary_exclude_prefix = []
    wireframe_exclude_prefix = ['gt']
    vert_colors_exclude_prefix = ['ours']
    scale_vecfield_exclude_prefix = ['gt']
    vecfield_exclude_prefix = []
    recompute_vcolor_range = True  # only for steps figure and video
    recompute_vcolor_range_each_mesh = False
    cmap = np.load('/home/lizeth/Downloads/ppsurf/ppsurf/figures/blender_script/cmap_YlOrRd.npy')

    # shared (models in both size and curv applications)
    model_list = [
        # '75106',
        # '75660',
        '75667',
        # '78251',
        # '100349',
        # '100478',
        # '101865',
        # '103141',
        # '116066',
        # '762604',
    ]

    method_list = [
        # 'init',
        'ours',
        # 'gt',
        # 'baseline'
    ]

    # # analytic surfaces
    # model_list = [
    #     # 'catenoid_curvature_trig_size',
    #     # 'catenoid_equal_area_triangles',
    #     'enneper_equal_area_triangles',
    #     ]

    # method_list = [
    #     'init3D',
    #     'opt3D',
    #     ]

    # method_list = [
    #     'init2D',
    #     'opt2D',
    #     ]

    steps = [None]
    # steps = [2, 6, 18, 66, 254, 998] # for triangle size steps figure
    # steps = [1, 3, 9, 33, 127, 499] # for cuvature steps figure (every two steps)
    # steps = list(range(1295))
    # steps = list(range(500))
    # mesh_color = np.array([255.0, 255, 255, 255])
    mesh_color = np.array([231.0, 166, 130, 255])  # clay # clay
    vec_size = 0.025

    # test
    # axes = 'x'
    # rot = [90]
    # y_offset = 0.0
    # scale = 1.0

    # default
    # axes = ['x', 'y', 'z']
    # rot = [90, 0, 0]
    # y_offset = 0.0
    # scale = 1.0

    # happy model
    # axes = ['x', 'y', 'z']
    # rot = [10, 80, -40]
    # y_offset = 0.0
    # scale = 1.0

    import json
    camera_filename = input_dir + 'camera_params.json'

    def distances_to_vertex_colors(dist_per_vertex: np.ndarray, cut_off=0.3):

        dist_per_vertex[dist_per_vertex > cut_off] = cut_off
        dist_per_vertex /= cut_off

        # use parula colormap: dist=0 -> blue, dist=0.5 -> green, dist=1.0 -> yellow
        parulas_indices = (dist_per_vertex * (cmap.shape[0] - 1)).astype(np.int32)
        dist_greater_than_norm_target = parulas_indices >= cmap.shape[0]
        parulas_indices[dist_greater_than_norm_target] = cmap.shape[0] - 1
        dist_colors_rgb = [cmap[parula_indices] for parula_indices in parulas_indices]

        return dist_colors_rgb

    if os.path.exists(camera_filename):
        with open(camera_filename, 'r') as file:
            camera_settings = json.load(file)
            axes = camera_settings['axes']
            rot = camera_settings['rot']
            y_offset = camera_settings['y_offset']
            scale = camera_settings['scale']

    else:  # default
        axes = ['x', 'y', 'z']
        rot = [0, 0, 0]
        y_offset = 0.0
        scale = 1.0

    camera_settings = {}
    camera_settings['axes'] = axes
    camera_settings['rot'] = rot
    camera_settings['y_offset'] = y_offset
    camera_settings['scale'] = scale

    write_camera_params = False
    test = True
    use_vert_colors = True
    use_vecfield = False

    if write_camera_params:
        # Save the dictionary as a JSON file
        with open(camera_filename, 'w') as file:
            json.dump(camera_settings, file)

    # get mesh list from list of models and list of methods
    # mesh_names = []
    # for model_name in model_list:
    #     for method_name in method_list:
    #         mesh_names.append(f'{method_name}_{model_name}')
    #
    # # pre-process meshes to get vertex colors
    # print(f'getting vertex colors ...')

    #     if steps[0] is None:
    #         raise RuntimeError('vcolor_max should only be determined once and then set consistently throughout all experiments')

    def get_ply_files(directory, output_path):

        ply_files = []
        output_files = []
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.ply'):
                    file_path = os.path.join(root, filename)
                    ply_files.append(os.path.join(root, filename))
                    output_file = file_path.replace(directory, output_path, 1)
                    output_file = os.path.splitext(output_file)[0] + '.png'
                    output_files.append(output_file)
        return ply_files, output_files

    mesh_names, output_files = get_ply_files(input_dir, output_dir)
    # mesh_names = ['/home/lizeth/Downloads/ppsurf/ppsurf/figures/comp mathods/abc/ppsurf.ply']
    # vcolor_filename = '/home/lizeth/Downloads/for rendering/comp/abc/00010429_fc56088abf10474bba06f659_trimesh_004/ppsurf_merge_sum_dist.npz'
    # vert_colors = np.load(vcolor_filename)

    # save camera parameters
    # camera_config = np.array([axes, rot, y_offset, scale])

    if use_vert_colors and recompute_vcolor_range:
        all_vcolors = []
        for mesh_name in mesh_names:
            if not os.path.basename(mesh_name) == 'gt.ply' and not os.path.basename(mesh_name) == 'pc.ply':
                vcolor_filename = os.path.splitext(mesh_name)[0] + '_dist.npz'
                vert_color_vals = np.load(vcolor_filename)['distances']
                all_vcolors.append(vert_color_vals)
        vcolor_max = np.percentile(np.concatenate(all_vcolors), 95)
        vcolor_min = np.percentile(np.concatenate(all_vcolors), 5)
        print(f'vcolor_max: {vcolor_max}')
        print(f'vcolor_min: {vcolor_min}')
        # save color map values
        colormap_min_max = np.array([vcolor_min, vcolor_max])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.savetxt(output_dir + 'vcolor_min_max.txt', colormap_min_max, delimiter=' ')

    scene = bpy.context.scene

    scene.render.engine = 'CYCLES'

    # Set the device_type
    bpy.context.preferences.addons[
        'cycles'
    ].preferences.compute_device_type = 'CUDA'

    # Set the device and feature set
    scene.cycles.device = 'GPU'

    #    scene.render.tile_x = 256
    #    scene.render.tile_y = 256

    # get_devices() to let Blender detects GPU device
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    print(bpy.context.preferences.addons['cycles'].preferences.compute_device_type)
    for d in bpy.context.preferences.addons['cycles'].preferences.devices:
        if d.type == 'CPU':
            d.use = False
        else:
            d.use = True
        print(d.name, d.use)

    for mesh_ind, mesh_name in enumerate(mesh_names):
        # pre-process meshes to get bounding box for each step
        print(f'preprocessing mesh {mesh_name} ...')
        mesh_bbmin = np.array([np.inf, np.inf, np.inf])
        mesh_bbmax = np.array([-np.inf, -np.inf, -np.inf])
        # for step in steps:
        # sample_name = f'{mesh_name}_{step}' if step is not None else mesh_name
        #            mesh_filename = os.path.join(input_dir, f'{sample_name}.ply')
        mesh_filename = mesh_name
        mesh = trimesh.load(mesh_filename, process=False)  # disable processing to preserve vertex order

        # align abc var-noise to abc extra noisy
        if '/abc/' in mesh_name or '\\abc\\' in mesh_name:
            import trimesh
            import trimesh.registration
            ref_mesh_name = mesh_name.replace('/abc/', '/abc_extra_noisy/').replace('\\abc\\', '\\abc_extra_noisy\\')
            ref_mesh: trimesh.Trimesh = trimesh.load(ref_mesh_name, process=False)

            # get the transformation matrix
            align_matrix, cost = trimesh.registration.mesh_other(mesh, ref_mesh.vertices, samples=10000)
            mesh.apply_transform(align_matrix)

        # automatic view parameters for missing
        if rot == [0, 0, 0]:
            import trimesh
            import trimesh.geometry
            import trimesh.transformations as trafo
            up = [0, 0, 1]
            points_pit = mesh.bounding_box_oriented.principal_inertia_transform
            up_rotated = trimesh.transform_points([up], points_pit)[0]
            rotate_to_up = trimesh.geometry.align_vectors(up_rotated, up)
            mesh.apply_transform(rotate_to_up)

            # a little bit of rotation
            mesh.apply_transform(trafo.rotation_matrix(np.pi/4, [0, 1, 0]))

        if not os.path.basename(mesh_filename) == 'pc.ply':
            faces = np.array(mesh.faces).astype('int32')
        verts = np.array(mesh.vertices).astype('float32')
        mesh_bbmin = np.minimum(mesh_bbmin, verts.min(axis=0))
        mesh_bbmax = np.maximum(mesh_bbmax, verts.max(axis=0))
        mesh_bbcenter = 0.5 * (mesh_bbmin + mesh_bbmax)
        mesh_bbsize = (mesh_bbmax - mesh_bbmin).max()

        for step_ind, step in enumerate(steps):
            # print(f'[{step_ind + mesh_ind*len(steps) + 1} / {len(mesh_names)*len(steps)}] rendering {sample_name}')

            sample_name = f'{mesh_name}_{step}' if step is not None else mesh_name

            #            mesh_filename = os.path.join(input_dir, f'{sample_name}.ply')
            #             mesh_filename = '/home/lizeth/Downloads/ppsurf/ppsurf/figures/comp mathods/abc/ppsurf.ply'
            #            output_filename = os.path.join(output_dir, f'{mesh_name}_{step:05d}{output_suffix}.png' if step is not None else f'{mesh_name}{output_suffix}.png')
            output_filename = output_files[mesh_ind]

            # remove objects from previous iteration
            if 'object' in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects['object'], do_unlink=True)

            if 'wireframe' in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects['wireframe'], do_unlink=True)

            if 'attachments' in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects['attachments'], do_unlink=True)

            if 'vecfield' in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects['vecfield'], do_unlink=True)

            if 'boundary' in bpy.data.objects:
                bpy.data.objects.remove(bpy.data.objects['boundary'], do_unlink=True)

            if 'object' in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes['object'], do_unlink=True)

            if 'wireframe' in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes['wireframe'], do_unlink=True)

            if 'attachments' in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes['attachments'], do_unlink=True)

            if 'vecfield' in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes['vecfield'], do_unlink=True)

            if 'boundary' in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes['boundary'], do_unlink=True)

            if 'vec' in bpy.data.meshes:
                bpy.data.meshes.remove(bpy.data.meshes['vec'], do_unlink=True)

            if clear:
                break

            # create vectorfield 'arrow' mesh
            if use_vecfield and os.path.basename(mesh_filename) == 'pc.ply':
                vec_mesh = bpy.data.meshes.new('vec')
                vec_bmesh = bmesh.new()
                #    bmesh.ops.create_cone(vec_bmesh, cap_ends=True, cap_tris=True, segments=5, diameter1=vec_size*0.3, diameter2=vec_size*0.05, depth=vec_size)
                # bmesh.ops.create_cone(vec_bmesh, cap_ends=True, cap_tris=True, segments=12, diameter1=vec_size * 0.08,
                #                       diameter2=vec_size * 0.01, depth=vec_size)
                bmesh.ops.create_icosphere(vec_bmesh, subdivisions=2, radius=0.005)
                bmesh.ops.triangulate(vec_bmesh, faces=vec_bmesh.faces[:])
                #    bmesh.ops.create_cone(vec_bmesh)
                vec_bmesh.to_mesh(vec_mesh)
                vec_bmesh.free()

                vec_verts = np.array([[v.co.x, v.co.y, v.co.z] for v in vec_mesh.vertices])
                vec_faces = np.array([[p.vertices[0], p.vertices[1], p.vertices[2]] for p in
                                      vec_mesh.polygons])  # vec_mesh.loop_triangles
                vec_verts[:, 2] -= vec_verts.min(axis=0)[2]

            # load mesh of main object
            mesh = trimesh.load(mesh_filename, process=False)  # disable processing to preserve vertex order
            if not os.path.basename(mesh_filename) == 'pc.ply':
                faces = np.array(mesh.faces).astype('int32')
            verts = np.array(mesh.vertices).astype('float32')

            # move bounding box center to origin and normalize max. bounding box side length to 1
            # mesh_bbcenter = (verts.max(axis=0)+verts.min(axis=0))/2
            # mesh_bbsize = (verts.max(axis=0)-verts.min(axis=0)).max()
            verts = verts - mesh_bbcenter
            verts = verts / mesh_bbsize

            # to blender coordinates and apply any custom rotation, scaling, and translation
            coord_rot = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
            for i in range(len(rot)):
                coord_rot = np.matmul(
                    scipy.spatial.transform.Rotation.from_euler(axes[i], rot[i], degrees=True).as_matrix(), coord_rot)
            coord_rot = np.matmul(np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]]), coord_rot)
            verts = np.transpose(np.matmul(coord_rot, np.transpose(verts)))
            y_min = verts.min(axis=0)[2]
            verts[:, 2] -= y_min  # make objects 'stand' on the xz coordinate plane (y_min = 0)
            verts[:, 2] += y_offset  # apply custom translation in y direction

            mesh_scale_vecfield = scale_vecfield and not any(
                os.path.basename(sample_name).startswith(x) for x in scale_vecfield_exclude_prefix)

            if (use_vert_colors and not os.path.basename(mesh_filename) == 'gt.ply' and not os.path.basename(
                    mesh_filename) == 'pc.ply'):
                # vcolor_filename = os.path.join(input_dir, os.path.dirname(sample_name), f'{vcolor_prefix}{os.path.basename(sample_name)}{vcolor_suffix}.npy')
                vcolor_filename = os.path.splitext(mesh_filename)[0] + '_dist.npz'
                vert_color_vals = np.load(vcolor_filename)['distances']
                if np.isnan(vert_color_vals).any():
                    print('WARNING: some vertex color values are NaN! Setting them to zero.')
                    vert_color_vals[np.isnan(vert_color_vals)] = 0
                if recompute_vcolor_range_each_mesh:
                    vcolor_max = np.percentile(vert_color_vals, 95)
                    vcolor_min = np.percentile(vert_color_vals, 5)
                print(vert_color_vals.min())
                print(vert_color_vals.max())
                vert_color_vals = (vert_color_vals - vcolor_min) / (vcolor_max - vcolor_min)
                vert_colors = eval_cmap(vert_color_vals, cmap_colors=cmap)
                mix = 1.0
                vert_colors = vert_colors * mix + (mesh_color / 255.0) * (1 - mix)
            else:
                vert_colors = (np.repeat([mesh_color], verts.shape[0], axis=0) / 255).astype('float32').clip(min=0.0,
                                                                                                             max=1.0)

            if use_vecfield and os.path.basename(mesh_filename) == 'pc.ply':
                # # get rotated instances of the arrow mesh
                # rotmats = np.concatenate([
                #     rotation_between_vectors(np.array([[0, 0, 1.0]]), vecfield_dirs),
                #     rotation_between_vectors(np.array([[0, 0, 1.0]]), -vecfield_dirs)], axis=0)
                # vecfield_verts = np.matmul(np.expand_dims(rotmats, 1), np.expand_dims(vecfield_verts, 3)).squeeze(
                #     -1)
                # rotation
                vecfield_verts = (
                        np.expand_dims(vec_verts, axis=0) + np.expand_dims(verts, axis=1)).reshape(-1, 3)

                # translation
                vecfield_faces = (np.expand_dims(vec_faces, axis=0) + (
                        np.arange(verts.shape[0] * 2) * vec_verts.shape[0]).reshape(-1, 1, 1)).reshape(-1, 3)

                vecfield_verts = vecfield_verts.tolist()
                vecfield_faces = vecfield_faces.tolist()

            verts = verts.tolist()

            if not os.path.basename(mesh_filename) == 'pc.ply':
                faces = faces.tolist()

            vert_colors = vert_colors.tolist()

            # create blender mesh for cuboids
            mesh = bpy.data.meshes.new('object')

            if not os.path.basename(mesh_filename) == 'pc.ply':
                mesh.from_pydata(verts, [], faces)
            mesh.validate()

            mesh.vertex_colors.new(name='Col')  # named 'Col' by default
            mesh_vert_colors = mesh.vertex_colors['Col']

            # wireframe_mesh = mesh.copy()
            # wireframe_mesh.name = 'wireframe'

            for poly in mesh.polygons:
                for loop_index in poly.loop_indices:
                    loop_vert_index = mesh.loops[loop_index].vertex_index
                    if loop_vert_index < len(vert_colors):
                        mesh.vertex_colors['Col'].data[loop_index].color = vert_colors[loop_vert_index]

            # create blender object for cuboids
            obj = bpy.data.objects.new('object', mesh)
            if not os.path.basename(mesh_filename) == 'pc.ply':
                obj.data.materials.append(bpy.data.materials['sphere_material'])
            scene.collection.objects.link(obj)
            if turning_animation:
                copy_animation_data(src_obj=scene.objects['turntable'], dst_obj=obj)

            if use_vecfield and os.path.basename(mesh_filename) == 'pc.ply':
                vecfield_mesh = bpy.data.meshes.new('vecfield')
                vecfield_mesh.from_pydata(vecfield_verts, [], vecfield_faces)
                vecfield_mesh.validate()

                vecfield_mesh.vertex_colors.new(name='Col')  # named 'Col' by default
                mesh_vert_color = np.array((mesh_color / 255.0).tolist(), dtype=np.float32).clip(min=0.0, max=1.0)

                for poly in vecfield_mesh.polygons:
                    for loop_index in poly.loop_indices:
                        loop_vert_index = vecfield_mesh.loops[loop_index].vertex_index
                        vecfield_mesh.vertex_colors['Col'].data[loop_index].color = mesh_vert_color

                # create blender object for vector field
                vecfield_obj = bpy.data.objects.new('vecfield', vecfield_mesh)
                vecfield_obj.data.materials.append(bpy.data.materials['sphere_material'])
                scene.collection.objects.link(vecfield_obj)
                if turning_animation:
                    copy_animation_data(src_obj=scene.objects['turntable'], dst_obj=vecfield_obj)

            # render scene
            scene.render.image_settings.file_format = 'PNG'
            # print(f'rendering to {scene.render.filepath}')

            if use_vecfield == False and use_vert_colors == True and os.path.basename(mesh_filename) == 'pc.ply':
                break

            else:
                scene.render.filepath = output_files[mesh_ind]
                bpy.ops.render.render(write_still=True)

        if clear or test:
            break


if __name__ == '__main__':
    input_dir_par = '/home/lizeth/Downloads/for rendering/comp/'
    output_dir_par = '/home/lizeth/Downloads/for rendering/rendered/'
    input_dirs_datasets = [os.path.join(input_dir_par, d) for d in os.listdir(input_dir_par) if os.path.isdir(os.path.join(input_dir_par, d))]
    output_dirs_datasets = [os.path.join(output_dir_par, d) for d in os.listdir(input_dir_par) if os.path.isdir(os.path.join(input_dir_par, d))]

    # input_dir = '/home/lizeth/Downloads/for rendering/comp/abc/00014452_55263057b8f440a0bb50b260_trimesh_017/'
    # output_dir = '/home/lizeth/Downloads/for rendering/rendered/abc/00014452_55263057b8f440a0bb50b260_trimesh_017/'
    for input_dir_dataset, output_dir_dataset in zip(input_dirs_datasets, output_dirs_datasets):
        input_dirs_meshes = [os.path.join(input_dir_dataset, d) for d in os.listdir(input_dir_dataset)
                             if os.path.isdir(os.path.join(input_dir_dataset, d))]
        output_dirs_meshes = [os.path.join(output_dir_dataset, d) for d in os.listdir(input_dir_dataset)
                              if os.path.isdir(os.path.join(input_dir_dataset, d))]
        for input_dir, output_dir in zip(input_dirs_meshes, output_dirs_meshes):
            print('Rendering meshes in {} to {}'.format(input_dir, output_dir))
            render_meshes(input_dir+'/', output_dir+'/')
