import os
import torch
import numpy as np
import cv2
import struct
import glob
from plyfile import PlyData, PlyElement
from PIL import Image
from dataclasses import dataclass



@dataclass
class CameraModel:
    model_id: int
    model_name: str
    num_params: int

@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray

# Kamera-Modelle
CAMERA_MODELS = [
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
]

CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}

# read-write utils in colmap: https://github.com/colmap/colmap/blob/main/scripts/python/read_write_model.py

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)

def read_cameras_binary(input_file):
    cameras = {}
    with open(input_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ"
            )
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(
                fid,
                num_bytes=8 * num_params,
                format_char_sequence="d" * num_params,
            )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
        assert len(cameras) == num_cameras
    return cameras

def write_intrinsics_text(cameras, output_dir):
    file = os.path.join(output_dir, "intrinsics.txt")

    with open(file, "w") as f:
        for cam_id, cam in cameras.items():
            if cam.model in ["PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"]:
                if cam.model == "PINHOLE":
                    fx, fy, cx, cy = cam.params
                elif cam.model == "SIMPLE_PINHOLE":
                    fx = cam.params[0]
                    cx = cam.params[1]
                    cy = cam.params[2]
                    fy = fx  # SIMPLE_PINHOLE hat fx = fy
                elif cam.model == "SIMPLE_RADIAL":
                    fx = cam.params[0]
                    cx = cam.params[1]
                    cy = cam.params[2]
                    distortion_k1 = cam.params[3]  # Verzerrungskoeffizient k1
                    fy = fx  # SIMPLE_RADIAL hat fx = fy
                elif cam.model == "RADIAL":
                    fx = cam.params[0]
                    cx = cam.params[1]
                    cy = cam.params[2]
                    distortion_k1 = cam.params[3]  # Verzerrungskoeffizient k1
                    distortion_k2 = cam.params[4]  # Verzerrungskoeffizient k2
                    fy = fx  # RADIAL hat fx = fy
                
                intrinsics_matrix = [
                    [fx, 0.0, cx - 0.005, 0.0], #cx & cy are the camera width & height. They are calculated as following: 1/2 * width = cx, 1/2 * heihgt = cy
                    [0.0, fy, cy - 0.005, 0.0], # -0.005 because of the boarder that is removed in PointCloudToImageMapper 
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
                
                for row in intrinsics_matrix:
                    f.write(" ".join(f"{value:.18e}" for value in row) + "\n")
                
                #if cam.model in ["SIMPLE_RADIAL", "RADIAL"]:
                #    f.write("# Distortion Parameters:\n")
                #    if cam.model == "SIMPLE_RADIAL":
                #        f.write(f"k1: {distortion_k1:.18e}\n")
                #    elif cam.model == "RADIAL":
                #        f.write(f"k1: {distortion_k1:.18e}\n")
                #        f.write(f"k2: {distortion_k2:.18e}\n")
                
                f.write("\n")
            else:
                print(f"Kamera {cam_id} hat ein nicht unterstütztes Modell: {cam.model}")
    
    if(is_debug):
        for cam_id, cam in cameras.items():
            print(f"Camera ID: {cam_id}")
            print(f"  Model: {cam.model}")
            print(f"  Width: {cam.width}, Height: {cam.height}")
            print(f"  Params: {cam.params}")

def read_depth_map_binary(input_file):
    with open(input_file, "rb") as f:
        header = b""
        ampersand_count = 0

        while True:
            char = f.read(1)
            if not char:
                break
            header += char
            if char == b"&":
                ampersand_count += 1
                if ampersand_count == 3:
                    break

        header_str = header.decode("utf-8").rstrip("&")
        header_parts = header_str.split("&")
        if len(header_parts) < 3:
            raise ValueError(f"Ungültiges Header-Format: {header_str}")

        width, height, channels = map(int, header_parts[:3])
        assert channels == 1, "Only Depth Maps with 1 channel are supported."

        depth_data = np.fromfile(f, dtype=np.float32, count=width * height)
        depth_map_npy_array = depth_data.reshape((height, width))  # Row-Major: (Height, Width)

    return depth_map_npy_array

def save_depth_as_16bit_image(depth_map_array, output_dir, scale_factor=1000.0, target_size=None):
    depth_map_array[np.isnan(depth_map_array)] = 0
    depth_map_array[np.isinf(depth_map_array)] = 0
    
    depth_resized = cv2.resize(depth_map_array, target_size, interpolation=cv2.INTER_NEAREST)

    depth_16bit = np.clip(depth_resized * scale_factor, 0, 65535).astype(np.uint16)
    output_file = os.path.join(output_dir) # We want to work with the geometric depth-data. Photometric depth-data does have artifacs
    cv2.imwrite(output_file, depth_16bit)

def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            binary_image_name = b""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                binary_image_name += current_char
                current_char = read_next_bytes(fid, 1, "c")[0]
            image_name = binary_image_name.decode("utf-8")
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            x_y_id_s = read_next_bytes(
                fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D
            )
            xys = np.column_stack(
                [tuple(map(float, x_y_id_s[0::3])), tuple(map(float, x_y_id_s[1::3]))]
            )
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images

def quaternion_to_rotation_matrix(qvec):
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
        [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 - q0 * q1)],
        [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
    ])

def convert_ply_to_pth(filename, out_dir):
    ply_cleanup_path = f'/home/moritz/openscene/demo/region_segmentations/region_{filename}.ply'
    ply_colmap_meshed_path = f'/home/moritz/data/{filename}/dense/0/meshed-poisson.ply'
    if os.path.isfile(ply_cleanup_path):
        ply_path = ply_cleanup_path
        print(f"Using cleaned up mesh: {ply_path}")
    else:
        ply_path = ply_colmap_meshed_path
        print(f"Using mesh computed by colmap - not recommended: {ply_path}")

    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data

    coords = np.column_stack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).astype(np.float32)

    if 'red' in vertex_data.dtype.names and 'green' in vertex_data.dtype.names and 'blue' in vertex_data.dtype.names:
        colors = np.column_stack((vertex_data['red'], vertex_data['green'], vertex_data['blue'])).astype(np.uint8)
        colors = (colors / 127.5) - 1.0
    else:
        colors = np.full((coords.shape[0], 3), -1, dtype=np.float32)

    if 'label' in vertex_data.dtype.names:
        labels = vertex_data['label'].astype(np.int32)
    else:
        labels = np.full((coords.shape[0],), 255, dtype=np.int32)

    torch.save((coords, colors, labels), out_dir)
    print(f"PTH-Datei gespeichert unter: {out_dir}")

def apply_transformation_to_poses(transform_matrix_file, pose_output_path):
    """
    Liest die Transformationsmatrix aus transform_matrix_file und multipliziert sie mit jeder Pose-Matrix
    in pose_output_path. Überschreibt die Pose-Dateien mit den transformierten Matrizen.
    """

    # **Transformationsmatrix einlesen**
    with open(transform_matrix_file, "r") as f:
        lines = f.readlines()
        transform_matrix = np.array([[float(value) for value in line.split()] for line in lines])
        #transform_matrix[:, 2] *= -1 # based on cloudcompares output invert z-axis

    print("Transformationsmatrix geladen:")
    print(transform_matrix)

    # **Alle Pose-Dateien finden**
    pose_files = glob.glob(os.path.join(pose_output_path, "*.txt"))

    for pose_file in pose_files:
        # **Pose-Matrix aus Datei einlesen**
        with open(pose_file, "r") as f:
            lines = f.readlines()
            pose_matrix = np.array([[float(value) for value in line.split()] for line in lines])

        # **Transformation anwenden**
        transformed_pose = np.dot(transform_matrix, pose_matrix)
        #transformed_pose[:3, 2] *= -1

        # **Alte Datei leeren und neue Matrix speichern**
        with open(pose_file, "w") as f:
            for row in transformed_pose:
                f.write(" ".join(f"{value:.18e}" for value in row) + "\n")


def convert_colmap_data_to_openscene(work_dir, name):
    work_dir = os.path.join(work_dir, name)

    colmap_dir_sparse = os.path.join(work_dir, 'sparse/0/')
    colmap_dir_dense = os.path.join(work_dir, 'dense/0/')
    colmap_dir_images = os.path.join(work_dir, 'images')

    openscene_scene01_dir = os.path.join(work_dir, 'custom_2d', 'scene01')
    openscene_dir_3d = os.path.join(work_dir, 'custom_3d')

    subdirs = ['depth', 'pose', 'color']
    for subdir in subdirs:
        os.makedirs(os.path.join(openscene_scene01_dir, subdir), exist_ok=True)
    os.makedirs(openscene_dir_3d, exist_ok=True)

    #custom_3d
    convert_ply_to_pth(name, openscene_dir_3d + "/scene01.pth")

    #custom_2d
    cameras = read_cameras_binary(os.path.join(colmap_dir_sparse, 'cameras.bin'))
    images = read_images_binary(os.path.join(colmap_dir_sparse, 'images.bin'))
    
    write_intrinsics_text(cameras, os.path.join(work_dir, 'custom_2d'))

    depth_files = sorted(glob.glob(os.path.join(colmap_dir_dense, 'stereo/depth_maps/' , '*geometric.bin')))
    image_files = sorted(glob.glob(os.path.join(colmap_dir_images, '*.jpg')) +
                         glob.glob(os.path.join(colmap_dir_images, '*.JPG')))

    depth_output_dir = os.path.join(openscene_scene01_dir, 'depth')
    color_output_dir = os.path.join(openscene_scene01_dir, 'color')
    pose_output_dir = os.path.join(openscene_scene01_dir, 'pose')

    assert len(depth_files) == len(image_files), f"Error - Images and Depthmaps are not the same amount {len(depth_files)}-{len(image_files)}"

    for idx, (depth_file, image_file) in enumerate(zip(depth_files, image_files)):
        name = os.path.basename(image_file)
        id = os.path.splitext(name)[0]
        print(f"Verarbeite Depth-Map {depth_file} und Farbbild {image_file} ... als {id}")

        # read and write depth-map
        color_image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        target_size = (color_image.shape[1], color_image.shape[0])
        depth_map = read_depth_map_binary(depth_file)
        depth_output_path = os.path.join(depth_output_dir, f"{id}.png")
        save_depth_as_16bit_image(depth_map, depth_output_path, target_size=target_size)

        # Copy and safe color images
        image = cv2.imread(image_file)
        color_output_path = os.path.join(color_output_dir, f"{id}.jpg")
        cv2.imwrite(color_output_path, image)

        # Save camera pose
        image_data = next((img for img in images.values() if img.name == os.path.basename(image_file)), None)
        if image_data:
            print(image_data.name + "<name and filename>" + image_file)
            rotation_matrix = quaternion_to_rotation_matrix(image_data.qvec)
            # colmap World -> Camera (We dont want this because we need to invert the pose for Camera -> World)
            #pose_matrix = np.eye(4)
            #pose_matrix[:3, :3] = rotation_matrix
            #pose_matrix[:3, 3] = image_data.tvec

            # as mentioned above invert: W2C -> C2W
            M_w2c = np.eye(4)
            M_w2c[:3, :3] = rotation_matrix
            M_w2c[:3, 3] = image_data.tvec
            M_c2w = np.linalg.inv(M_w2c)
            pose_matrix = M_c2w
            #print(pose_matrix)
            M_w2c = np.linalg.inv(pose_matrix)
            #print(M_w2c)

            name = image_data.name
            id = name.replace('.jpg', '')

            pose_output_path = os.path.join(pose_output_dir, f"{id}.txt")
            with open(pose_output_path, "w") as f:
                for row in pose_matrix:
                    f.write(" ".join(f"{value:.18e}" for value in row) + "\n")

    transform_file = glob.glob(os.path.join(colmap_dir_dense, "*.mat.txt"))
    if transform_file:
        transform_file = transform_file[0]
        print(f"Transforming cameraposes based on given matrix.")
        apply_transformation_to_poses(transform_file, pose_output_dir)    
    else:
        print("No camerapose transformations.")

        

            

is_debug = False

#name = 'woerthsee'
#convert_colmap_data_to_openscene('/home/moritz/data/', name)

# Numpy Version 1.24.3 is needed to convert to pth so it can be read later!!!!

