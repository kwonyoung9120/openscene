import os
import math
import multiprocessing as mp
import numpy as np
import imageio
import cv2
from nuscenes.nuscenes import NuScenes

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



def process_txt(filename):
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    return lines

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(
        image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


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


def nusc_to_pose(rotation, translation):
    pose = np.eye(4) 
    pose[:3,:3], pose[:3,3] = quaternion_rotation_matrix(rotation), translation

    return pose


def make_dirs(image_num):
    '''process one sequence.'''

    sample = nusc.sample[image_num]
    lidar_token = sample["data"]["LIDAR_TOP"]

    scene_token = nusc.sample[image_num]['scene_token']

    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']

    import pdb; pdb.set_trace()

    out_dir_lidar = os.path.join(out_dir, scene_name, 'lidar')
    out_dir_label = os.path.join(out_dir, scene_name, 'label')
    out_dir_color = os.path.join(out_dir, scene_name, 'color')
    out_dir_pose = os.path.join(out_dir, scene_name, 'pose')
    out_dir_K = os.path.join(out_dir, scene_name, 'K')

    os.makedirs(out_dir_color, exist_ok=True)
    os.makedirs(out_dir_pose, exist_ok=True)
    os.makedirs(out_dir_K, exist_ok=True)

def process_one_sequence(image_num):
    '''process one sequence.'''

    sample = nusc.sample[image_num]
    lidar_token = sample["data"]["LIDAR_TOP"]

    scene_token = nusc.sample[image_num]['scene_token']

    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']

    #import pdb; pdb.set_trace()

    out_dir_lidar = os.path.join(out_dir, scene_name, 'lidar')
    out_dir_label = os.path.join(out_dir, scene_name, 'label')
    out_dir_color = os.path.join(out_dir, scene_name, 'color')
    out_dir_pose = os.path.join(out_dir, scene_name, 'pose')
    out_dir_K = os.path.join(out_dir, scene_name, 'K')

    os.makedirs(out_dir_color, exist_ok=True)
    os.makedirs(out_dir_pose, exist_ok=True)
    os.makedirs(out_dir_K, exist_ok=True)

    sd_rec_lidar = nusc.get('sample_data', sample['data']["LIDAR_TOP"])

    pose_record_lidar = nusc.get('ego_pose', sd_rec_lidar['ego_pose_token'])
    
    def return_pose(cam):
        pose_record_cam = nusc.get('ego_pose', nusc.get('sample_data', sample['data'][cam])['ego_pose_token'])
        pose = np.eye(4) 
        pose[:3,:3], pose[:3,3] = quaternion_rotation_matrix(pose_record_cam['rotation']), pose_record_cam['translation']

        return pose

    camera_token = {'CAM_FRONT' : sample["data"]["CAM_FRONT"], 
                    'CAM_FRONT_RIGHT' : sample["data"]["CAM_FRONT_RIGHT"], 
                    'CAM_BACK_RIGHT' : sample["data"]['CAM_BACK_RIGHT'],
                    'CAM_BACK' : sample["data"]["CAM_BACK"], 
                    'CAM_BACK_LEFT' : sample["data"]["CAM_BACK_LEFT"], 
                    'CAM_FRONT_LEFT' : sample["data"]["CAM_FRONT_LEFT"]}

    cam_path = {'CAM_FRONT' : nusc.get_sample_data(sample["data"]["CAM_FRONT"])[0], 
                    'CAM_FRONT_RIGHT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_RIGHT"])[0], 
                    'CAM_BACK_RIGHT' : nusc.get_sample_data(sample["data"]['CAM_BACK_RIGHT'])[0],
                    'CAM_BACK' : nusc.get_sample_data(sample["data"]["CAM_BACK"])[0], 
                    'CAM_BACK_LEFT' : nusc.get_sample_data(sample["data"]["CAM_BACK_LEFT"])[0], 
                    'CAM_FRONT_LEFT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_LEFT"])[0]}

    cam_intrinsics = {'CAM_FRONT' : nusc.get_sample_data(sample["data"]["CAM_FRONT"])[2], 
                    'CAM_FRONT_RIGHT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_RIGHT"])[2], 
                    'CAM_BACK_RIGHT' : nusc.get_sample_data(sample["data"]['CAM_BACK_RIGHT'])[2],
                    'CAM_BACK' : nusc.get_sample_data(sample["data"]["CAM_BACK"])[2], 
                    'CAM_BACK_LEFT' : nusc.get_sample_data(sample["data"]["CAM_BACK_LEFT"])[2], 
                    'CAM_FRONT_LEFT' : nusc.get_sample_data(sample["data"]["CAM_FRONT_LEFT"])[2]}

    lidar_pose = nusc_to_pose(pose_record_lidar['rotation'], pose_record_lidar['translation'])


    lidar_path = nusc.get_sample_data(lidar_token)[0]
    label_path = nusc.get('lidarseg', lidar_token)['filename']

    coords = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,5)[:,:3]
    labels = np.fromfile(os.path.join('/dataset/nuScenes_lidarseg',label_path), dtype=np.uint8)
    remapped_labels = NUSCENES_CLASS_REMAP[labels]
    remapped_labels -= 1
    torch.save((coords, 0, remapped_labels),
            os.path.join(out_dir,  scene_name + '.pth'))
    import pdb; pdb.set_trace()
    #timestamp = nuim.get('sample',im_token)['timestamp']
    
    cam_locs = ['CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
    timestamp = sample['timestamp']
    #timestamp = sorted(os.listdir(os.path.join(data_path, scene_name, 'frames')))[-1] # take only the last timestamp
    for cam in cam_locs:
        #img_name = os.path.join(data_path, scene_name, 'frames', timestamp, cam, 'color_image.jpg')
        img = imageio.v3.imread(cam_path[cam])
        img = cv2.resize(img, img_size)
        imageio.imwrite(os.path.join(out_dir_color, str(timestamp) + '_' + cam + '.jpg'), img)
        # copy the camera parameters to the folder
        #pose_dir = os.path.join(data_path, scene_name, 'frames', timestamp, cam, 'cam2scene.txt')
        #pose = np.asarray([[float(x[0]), float(x[1]), float(x[2]), float(x[3])] for x in
        #            (x.split(" ") for x in open(pose_dir).read().splitlines())])
        np.save(os.path.join(out_dir_pose, cam+'.npy'), return_pose(cam))
        # shutil.copyfile(pose_dir, os.path.join(out_dir_pose, cam+'.txt'))
        #K_dir = os.path.join(data_path, scene_name, 'frames', timestamp, cam, 'K.txt')
        #K = np.asarray([[float(x[0]), float(x[1]), float(x[2])] for x in
        #            (x.split(" ") for x in open(K_dir).read().splitlines())])
        #K = nusc.get('ego_pose',nusc.get('sample_data', camera_token[cam])['ego_pose_token'])
        K = adjust_intrinsic(cam_intrinsics[cam], intrinsic_image_dim=(1600, 900), image_dim=img_size)
        np.save(os.path.join(out_dir_K, cam+'.npy'), K)

        # shutil.copyfile(pose_dir, os.path.join(out_dir_K, cam+'.txt'))

    print(scene_name, ' done')


nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuScenes_lidarseg', verbose=True)

#! YOU NEED TO MODIFY THE FOLLOWING
#####################################
split = 'v1.0_trainval' # 'train' | 'val'
data_path ='/dataset/nuScenes_lidarseg/{}'.format(split)
out_dir = '/dataset/openscene/nuscenes' # downloaded original nuscenes data
#scene_list = [nusc.scene[i] for i in ]
#####################################

os.makedirs(out_dir, exist_ok=True)

#cam_locs = ['back', 'back_left', 'back_right', 'front', 'front_left', 'front_right']
img_size = (800, 450)

global count
global prev_scene

count = 0
prev_scene = None

process_one_sequence(100)
#process_one_sequence(101)
'''
p = mp.Pool(processes=mp.cpu_count())
p.map(make_dirs, range(len(nusc.sample)))
p.close()
p.join()

#import pdb; pdb.set_trace()

p = mp.Pool(processes=mp.cpu_count())
p.map(process_one_sequence, range(len(nusc.sample)))
p.close()
p.join()
'''