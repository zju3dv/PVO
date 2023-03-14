from scipy.spatial.transform import Rotation as R
import numpy as np
import glob
import imageio
import os.path as osp


def get_replica_cam_list(cam_list_dir, base_dir):
    """
    direactly get cam matrix
    """
    cam_list = []
    with open(osp.join(base_dir, cam_list_dir), 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            data = line.strip('\n').split(' ')
            cam_mat = np.array([float(x) for x in data[:-4]]).reshape(3, 4)
            cam_list.append(cam_mat)
    return cam_list


def quat_to_rmat(quat):
    """
    input:
        quat 四元数
        list/numpy
    output:
        rot matrix 旋转矩阵
        numpy mat
    using numpy
    """
    r = R.from_quat(quat)
    mat = r.as_matrix()
    return mat


def rmat_to_quad(mat):
    r = R.from_matrix(mat)
    quat = r.as_quat()
    return quat


def get_trajectories_quad(cam_list, norm_val=1):
    trajectories = []
    for cam in cam_list:
        if not isinstance(cam, np.ndarray):
            np_cam = cam.detach().cpu().numpy()
        else:
            np_cam = cam
        np_cam[:3] /= norm_val
        trajectories.append(np_cam)
    return trajectories


def get_trajectories_mat(cam_list, norm_val=1):
    trajectories = []
    for cam in cam_list:
        if not isinstance(cam, np.ndarray):
            np_cam = cam.detach().cpu().numpy()
        else:
            np_cam = cam
        rot_mat = np_cam[:, 0:3]
        t_vec = np_cam[:, 3]/norm_val
        quad = rmat_to_quad(rot_mat)
        trajectories.append(np.append(t_vec, quad))
    return trajectories


def build_track_pred_file(gt_time_dir, pred_dir, trajectories, keyframe_freq=1):
    time_stamp = []
    with open(gt_time_dir, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue
            time_stamp.append(line.strip('\n').split(' ')[0])
    # time_stamp = time_stamp[::keyframe_freq]
    trajectories = trajectories[::keyframe_freq]
    cnt = min(len(trajectories), len(time_stamp))
    with open(pred_dir, 'w') as f:
        for i in range(cnt):
            data = ''
            for j in range(7):
                data += (' '+str(trajectories[i][j]))
            data = time_stamp[i]+data+'\n'
            f.write(data)


def build_track_pred_file_notime(pred_dir, trajectories, keyframe_freq=1):
    trajectories = trajectories[::keyframe_freq]
    cnt = len(trajectories)
    with open(pred_dir, 'w') as f:
        for i in range(cnt):
            s = []
            for j in range(7):
                s.append(str(trajectories[i][j]))
            data = ' '.join(s)
            data = data+'\n'
            f.write(data)


def build_timestamps(file_dir, time_num, interval=0.05):
    cur_time = 0.
    with open(file_dir, 'w') as f:
        for _ in range(time_num):
            s = str(cur_time)+'\n'
            cur_time += interval
            f.write(s)


def build_depth_npy(root_dir, scene_dir, dep_scale):
    depths = glob.glob(osp.join(root_dir, scene_dir, 'depth_left/*.png'))
    for dep in depths:
        dep_np = imageio.imread(dep)
        dep_np = dep_np.astype(np.float32)
        dep_np /= dep_scale
        dep_np[dep_np == 0] = 1
        npy_dir = dep.split('.')[0]+'.npy'
        np.save(npy_dir, dep_np)


if __name__ == '__main__':
    root_dir = 'datasets/Replica'
    scene_dir = 'office4'

    # cam_list = get_replica_cam_list('traj.txt', osp.join(root_dir, scene_dir))
    # gt_traj = get_trajectories_mat(cam_list)
    # build_track_pred_file_notime(osp.join(root_dir, scene_dir, 'pose_left.txt'),
    #                              gt_traj, 1)

    # timestamps_dir = osp.join(root_dir, scene_dir, 'timestamp.txt')
    # build_timestamps(timestamps_dir, 2000, 0.025)
    # build_track_pred_file(timestamps_dir, osp.join(root_dir, scene_dir, 'pose_left_tum.txt'),
    #                       gt_traj, 1)

    # dep_scale = 6553.5
    # build_depth_npy(root_dir, scene_dir, dep_scale)

    mat = np.array([-3.205696220032456800e-01, 4.480551946958012954e-01, -8.345547674986967257e-01, 3.452987416406734678e+00,
                    9.472249560947475500e-01, 1.516354520391845484e-01, -
                    2.824386167580063001e-01, 4.546110134135947223e-01,
                    1.078977932490423164e-16, -8.810523436158487209e-01, -
                    4.730187816662466682e-01, 5.936285447159415085e-01,
                    0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00,
                    ]).reshape(4, 4)
    mat2 = mat.copy()
    mats = np.stack([mat, mat2], axis=0)
    quads = rmat_to_quad(mats[:, 0:3, 0:3])
    print(quads.shape)
