import os
import multiprocessing as mp
import numpy as np
import plyfile
import torch
import open3d as o3d
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import nuscenes.utils.geometry_utils as geoutils
from pyquaternion import Quaternion

#nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuScenes_lidarseg', verbose=True)

NUSCENES_FULL_CLASSES = ( # 32 classes
    'noise',
    'animal',
    'human.pedestrian.adult',
    'human.pedestrian.child',
    'human.pedestrian.construction_worker',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'human.pedestrian.wheelchair',
    'movable_object.barrier',
    'movable_object.debris',
    'movable_object.pushable_pullable',
    'movable_object.trafficcone',
    'static_object.bicycle_rack',
    'vehicle.bicycle',
    'vehicle.bus.bendy',
    'vehicle.bus.rigid',
    'vehicle.car',
    'vehicle.construction',
    'vehicle.emergency.ambulance',
    'vehicle.emergency.police',
    'vehicle.motorcycle',
    'vehicle.trailer',
    'vehicle.truck',
    'flat.driveable_surface',
    'flat.other',
    'flat.sidewalk',
    'flat.terrain',
    'static.manmade',
    'static.other',
    'static.vegetation',
    'vehicle.ego',
    'unlabeled',
)

VALID_NUSCENES_CLASS_IDS = ()

NUSCENES_CLASS_REMAP = 256*np.ones(32) # map from 32 classes to 16 classes
NUSCENES_CLASS_REMAP[2] = 7 # person
NUSCENES_CLASS_REMAP[3] = 7
NUSCENES_CLASS_REMAP[4] = 7
NUSCENES_CLASS_REMAP[6] = 7
NUSCENES_CLASS_REMAP[9] = 1 # barrier
NUSCENES_CLASS_REMAP[12] = 8 # traffic cone
NUSCENES_CLASS_REMAP[14] = 2 # bicycle
NUSCENES_CLASS_REMAP[15] = 3 # bus
NUSCENES_CLASS_REMAP[16] = 3
NUSCENES_CLASS_REMAP[17] = 4 # car
NUSCENES_CLASS_REMAP[18] = 5 # construction vehicle
NUSCENES_CLASS_REMAP[21] = 6 # motorcycle
NUSCENES_CLASS_REMAP[22] = 9 # trailer ???
NUSCENES_CLASS_REMAP[23] = 10 # truck
NUSCENES_CLASS_REMAP[24] = 11 # drivable surface
NUSCENES_CLASS_REMAP[25] = 12 # other flat??
NUSCENES_CLASS_REMAP[26] = 13 # sidewalk
NUSCENES_CLASS_REMAP[27] = 14 # terrain
NUSCENES_CLASS_REMAP[28] = 15 # manmade
NUSCENES_CLASS_REMAP[30] = 16 # vegetation

def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def nusc_to_pose(rotation, translation):
    pose = np.eye(4) 
    pose[:3,:3], pose[:3,3] = quaternion_rotation_matrix(rotation), translation

    return pose

def make_o3d_pointcloud(xyz):
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    return pcd


def process_one_sequence(scene_num):
    '''process one sequence.'''

    scene = nusc.scene[scene_num]

    scene_name = scene['name']

    current_token = scene['first_sample_token']
    sample = nusc.get('sample', current_token)
    next_token = sample['next']
    coords_total = torch.zeros((0,3))
    labels_total = torch.zeros((0,))
    count = 0 

    #pcl, misc = LidarPointCloud.from_file_multisweep(nusc, sample, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=2)

    #import pdb; pdb.set_trace()
    while next_token != '':
        
        
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec_lidar = nusc.get('sample_data', lidar_token)
        pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])

        calib_data = nusc.get("calibrated_sensor", sd_rec_lidar["calibrated_sensor_token"])
        car_to_velo = geoutils.transform_matrix(calib_data["translation"], Quaternion(calib_data["rotation"]))
        pose_car = geoutils.transform_matrix(pose_record_lidar["translation"], Quaternion(pose_record_lidar["rotation"]))
        lidar_pose = np.dot(pose_car, car_to_velo)

        lidar_path = nusc.get_sample_data(lidar_token)[0]
        is_key_frame = nusc.get('sample_data', lidar_token)['is_key_frame']
        if is_key_frame:
            label_path = nusc.get('lidarseg', lidar_token)['filename']

            labels = np.fromfile(os.path.join('/dataset/nuScenes_lidarseg',label_path), dtype=np.uint8)
            remapped_labels = NUSCENES_CLASS_REMAP[labels]
            remapped_labels -= 1

        else : 
            labels = torch.ones_like(coords[:,0])*-100

        coords = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,5)[:,:4]
        lidar_h = coords
        lidar_h[:,3] = 1
        #coords[:,:3] = (np.linalg.inv(ref_pose) @ lidar_pose @ lidar_h.T).T[:,:3]
        coords[:,:3] = (lidar_pose @ lidar_h.T).T[:,:3]
        

        coords_total = torch.cat([coords_total, torch.from_numpy(coords[:,:3])],axis=0)
        labels_total = torch.cat([labels_total, torch.from_numpy(labels)],axis=0)
        
        sample = nusc.get('sample', next_token)
        current_token = next_token
        next_token = sample['next']
        count+=1

    torch.save((coords, 0, remapped_labels), os.path.join(out_dir,  scene_name, 'scene.pth'))
        
    #pcd = make_o3d_pointcloud(coords_labels[:,:3])
    #o3d.io.write_point_cloud(os.path.join(out_dir,  scene_name + '.pcd'), pcd)

    print(scene_name, ' done')


'''
def process_one_sequence(fn):
    #process one sequence.

    scene_name = fn.split('/')[-2]
    a = plyfile.PlyData().read(fn)
    v = np.array([list(x) for x in a.elements[0]])
    coords = np.ascontiguousarray(v[:, :3])
    category_id = np.ascontiguousarray(v[:, -1]).astype(int)

    if not export_all_points: # we only consider points with annotations
        dir_timestamp = fn[:-9] + 'scene-timestamps.npy'
        timestamp = np.load(dir_timestamp)
        mask = (timestamp==timestamp.max())[:, 0] # mask for points with annotations
        coords = coords[mask]
        category_id = category_id[mask]

    category_id[category_id==-1] = 0
    remapped_labels = NUSCENES_CLASS_REMAP[category_id]
    remapped_labels -= 1

    torch.save((coords, 0, remapped_labels), os.path.join(out_dir,  scene_name + '.pth'))
    print(fn)


def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines
'''

nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuScenes_lidarseg', verbose=True)
#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'v1.0_trainval' # 'train' | 'val'
data_path ='/dataset/nuScenes_lidarseg/{}'.format(split)
out_dir = '/dataset/openscene/nuscenes' # downloaded original nuscenes data
export_all_points = True # default we export all points within 0.5 sec
#scene_list = os.listdir(in_path)
################
for i in range(len(nusc.scene)):
    process_one_sequence(i)

#os.makedirs(out_dir, exist_ok=True)
#files = []
#for i in range(len(nusc.scene)):
#    files.append(os.path.join(data_path, scene, 'scene.ply'))

#for scene in scene_list:
#    files.append(os.path.join(in_path, scene, 'scene.ply'))
'''
p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_sequence, range(len(nusc.scene)))
p.close()
p.join()
'''
