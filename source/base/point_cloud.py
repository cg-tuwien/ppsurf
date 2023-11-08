import numpy as np

from source.base import fs


def load_xyz(file_path):
    data = np.loadtxt(file_path).astype('float32')
    nan_lines = np.isnan(data).any(axis=1)
    num_nan_lines = np.sum(nan_lines)
    if num_nan_lines > 0:
        data = data[~nan_lines]  # filter rows with nan values
        print('Ignored {} points containing NaN coordinates in point cloud {}'.format(num_nan_lines, file_path))
    return data


def write_ply(file_path: str, points: np.ndarray, normals=None, colors=None):
    """
    Write point cloud file as .ply.
    :param file_path:
    :param points:
    :param normals:
    :param colors:
    :return: None
    """

    import trimesh

    assert(file_path.endswith('.ply'))

    def sanitize_inputs(arr: np.ndarray):
        if arr is None:
            return arr

        # should be array
        arr = np.asarray(arr)

        # should be 2 dims
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        # convert 2d points to 3d
        if arr.shape[1] == 2:
            arr_2p5d = np.zeros((arr.shape[0], 3))
            arr_2p5d[:, :2] = arr
            arr_2p5d[:, 2] = 0.0
            arr = arr_2p5d

        # should be (n, dims)
        if arr.shape[0] == 3 and arr.shape[1] != 3:
            arr = arr.transpose([1, 0])

        return arr

    points = sanitize_inputs(points)
    colors = sanitize_inputs(colors)
    normals = sanitize_inputs(normals)

    mesh = trimesh.Trimesh(vertices=points, vertex_colors=colors, vertex_normals=normals)
    fs.make_dir_for_file(file_path)
    mesh.export(file_path)


def write_xyz(file_path, points: np.ndarray, normals=None, colors=None):
    """
    Write point cloud file.
    :param file_path:
    :param points:
    :param normals:
    :param colors:
    :return: None
    """

    fs.make_dir_for_file(file_path)

    if points.shape == (3,):
        points = np.expand_dims(points, axis=0)

    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.transpose([1, 0])

    if colors is not None and colors.shape[0] == 3 and colors.shape[1] != 3:
        colors = colors.transpose([1, 0])

    if normals is not None and normals.shape[0] == 3 and normals.shape[1] != 3:
        normals = normals.transpose([1, 0])

    with open(file_path, 'w') as fp:

        # convert 2d points to 3d
        if points.shape[1] == 2:
            vertices_2p5d = np.zeros((points.shape[0], 3))
            vertices_2p5d[:, :2] = points
            vertices_2p5d[:, 2] = 0.0
            points = vertices_2p5d

        # write points
        # meshlab doesn't like colors, only using normals. try cloud compare instead.
        for vi, v in enumerate(points):
            line_vertex = str(v[0]) + ' ' + str(v[1]) + ' ' + str(v[2]) + ' '
            if normals is not None:
                line_vertex += str(normals[vi][0]) + ' ' + str(normals[vi][1]) + ' ' + str(normals[vi][2]) + ' '
            if colors is not None:
                line_vertex += str(colors[vi][0]) + ' ' + str(colors[vi][1]) + ' ' + str(colors[vi][2]) + ' '
            fp.write(line_vertex + '\n')


def load_pcd(file_in):
    # PCD: https://pointclouds.org/documentation/tutorials/pcd_file_format.html
    # PCD RGB: http://docs.pointclouds.org/trunk/structpcl_1_1_r_g_b.html#a4ad91ab9726a3580e6dfc734ab77cd18

    def read_header(lines_header):
        header_info = dict()

        def add_line_to_header_dict(header_dict, line, expected_field):
            line_parts = line.split(sep=' ')
            assert (line_parts[0] == expected_field), \
                ('Warning: "' + expected_field + '" expected but not found in pcd header!')
            header_dict[expected_field] = (' '.join(line_parts[1:])).replace('\n', '')

        add_line_to_header_dict(header_info, lines_header[0], '#')
        add_line_to_header_dict(header_info, lines_header[1], 'VERSION')
        add_line_to_header_dict(header_info, lines_header[2], 'FIELDS')
        add_line_to_header_dict(header_info, lines_header[3], 'SIZE')
        add_line_to_header_dict(header_info, lines_header[4], 'TYPE')
        add_line_to_header_dict(header_info, lines_header[5], 'COUNT')
        add_line_to_header_dict(header_info, lines_header[6], 'WIDTH')
        add_line_to_header_dict(header_info, lines_header[7], 'HEIGHT')
        add_line_to_header_dict(header_info, lines_header[8], 'VIEWPOINT')
        add_line_to_header_dict(header_info, lines_header[9], 'POINTS')
        add_line_to_header_dict(header_info, lines_header[10], 'DATA')

        assert header_info['VERSION'] == '0.7'
        assert header_info['FIELDS'] == 'x y z rgb label'
        assert header_info['SIZE'] == '4 4 4 4 4'
        assert header_info['TYPE'] == 'F F F F U'
        assert header_info['COUNT'] == '1 1 1 1 1'
        # assert header_info['HEIGHT'] == '1'
        assert header_info['DATA'] == 'ascii'
        # assert header_info['WIDTH'] == header_info['POINTS']

        return header_info

    f = open(file_in, 'r')
    f_lines = f.readlines()
    f_lines_header = f_lines[:11]
    f_lines_points = f_lines[11:]
    header_info = read_header(f_lines_header)
    header_info['_file_'] = file_in

    num_points = int(header_info['POINTS'])
    point_data_list_str_ = [l.split(sep=' ')[:3] for l in f_lines_points]
    point_data_list = [[float(l[0]), float(l[1]), float(l[2])] for l in point_data_list_str_]

    # filter nan points that appear through the blensor kinect sensor
    point_data_list = [p for p in point_data_list if
                       (not np.isnan(p[0]) and not np.isnan(p[1]) and not np.isnan(p[2]))]

    point_data = np.array(point_data_list)

    f.close()

    return point_data, header_info


def numpy_to_ply(npy_file_in: str, ply_file_out: str):
    pts_in = np.load(npy_file_in)

    if pts_in.shape[1] >= 6:
        normals = pts_in[:, 3:6]
    else:
        normals = None

    if pts_in.shape[1] >= 9:
        colors = pts_in[:, 6:9]
    else:
        colors = None

    write_ply(file_path=ply_file_out, points=pts_in[:, :3], normals=normals, colors=colors)


def sample_mesh(mesh_file, num_samples, rejection_radius=None):
    import trimesh.sample

    try:
        mesh = trimesh.load(mesh_file)
    except:
        return np.zeros((0, 3))
    samples, face_indices = trimesh.sample.sample_surface_even(mesh, num_samples, rejection_radius)
    return samples


if __name__ == '__main__':
    # convert all datasets to ply
    import os
    from source.base import mp

    datasets = [
        'abc_train',
        # 'abc',
        # 'abc_extra_noisy',
        # 'abc_noisefree',
        # 'famous_noisefree',
        # 'famous_original',
        # 'famous_extra_noisy',
        # 'famous_sparse',
        # 'famous_dense',
        # 'thingi10k_scans_original',
        # 'thingi10k_scans_dense',
        # 'thingi10k_scans_sparse',
        # 'thingi10k_scans_extra_noisy',
        # 'thingi10k_scans_noisefree',
    ]

    # test on dir, multi-threaded
    # num_processes = 0
    # num_processes = 4
    num_processes = 15
    # num_processes = 48

    for dataset in datasets:
        in_dir = r'D:\repos\meshnet\datasets\{}\04_pts'.format(dataset)
        in_files = os.listdir(in_dir)
        in_files = [os.path.join(in_dir, f) for f in in_files if
                    os.path.isfile(os.path.join(in_dir, f)) and f.endswith('.npy')]
        out_dir = in_dir + '_vis'
        calls = []
        for fi, f in enumerate(in_files):
            file_base_name = os.path.basename(f)
            file_out = os.path.join(out_dir, file_base_name[:-4] + '.ply')
            if fs.call_necessary(f, file_out):
                calls.append([f, file_out])
        mp.start_process_pool(numpy_to_ply, calls, num_processes)

