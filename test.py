import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ======================================================
# 读取 KITTI 标定文件
# ======================================================
def load_calib(calib_path):
    data = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key] = np.array([float(x) for x in value.split()])

    # R0_rect
    R0 = data["R0_rect"].reshape(3, 3)

    # Tr_velo_to_cam（3x4）
    Tr = data["Tr_velo_to_cam"].reshape(3, 4)
    Tr4 = np.eye(4)
    Tr4[:3, :] = Tr

    return R0, Tr4


# ======================================================
# Camera → LiDAR 坐标转换（核心正确公式）
# ======================================================
def camera_to_lidar_xyz(x, y, z, R0, Tr):
    cam_xyz = np.array([x, y, z, 1.0])

    # 扩展 R0 为 4x4
    R0_4 = np.eye(4)
    R0_4[:3, :3] = R0

    # 反向变换
    lidar_xyz = np.linalg.inv(Tr) @ np.linalg.inv(R0_4) @ cam_xyz
    return lidar_xyz[:3]


# ======================================================
# Camera → LiDAR 3D Box 转换（包含 yaw）
# ======================================================
def camera_to_lidar_box(box_cam, R0, Tr):
    h, w, l, x, y, z, yaw = box_cam

    # yaw 转换：Camera（绕Y）→ LiDAR（绕Z）
    yaw_lidar = -(yaw + np.pi / 2)

    # 转换中心坐标
    lx, ly, lz = camera_to_lidar_xyz(x, y, z, R0, Tr)

    return np.array([h, w, l, lx, ly, lz, yaw_lidar])


# ======================================================
# 裁剪 3D 物体点云
# ======================================================
def crop(points, box):
    h, w, l, cx, cy, cz, yaw = box

    pts = points - np.array([cx, cy, cz])

    cos_t = np.cos(-yaw)
    sin_t = np.sin(-yaw)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0, 0, 0]
    ])
    pts_rot = pts @ R.T

    mask = (
        (pts_rot[:, 0] >= -l/2) & (pts_rot[:, 0] <= l/2) &
        (pts_rot[:, 1] >= -w/2) & (pts_rot[:, 1] <= w/2) &
        (pts_rot[:, 2] >= -h)   & (pts_rot[:, 2] <= 0)
    )

    return points[mask]


# ======================================================
# 保存 PNG 可视化
# ======================================================
def save_png(points, save_path):
    if len(points) == 0:
        return

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=5, c='red')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================================
# 主函数：裁剪一个 bin 中的所有物体
# ======================================================
def process_all_objects(frame_id, root):
    bin_path = os.path.join(root, "velodyne", frame_id + ".bin")
    calib_path = os.path.join(root, "calib", frame_id + ".txt")
    label_path = os.path.join(root, "label_2", frame_id + ".txt")

    # 读取点云（N×4 → N×3）
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    print(f"点云总点数：{len(points)}")

    # 加载标定
    R0, Tr = load_calib(calib_path)

    # 加载标签
    lines = open(label_path).readlines()

    # 输出目录
    save_dir = os.path.join("output", frame_id)
    os.makedirs(save_dir, exist_ok=True)

    obj_id = 0

    for line in lines:
        obj = line.split()
        cls = obj[0]

        # 只处理 3 类目标
        if cls not in ["Car", "Pedestrian", "Cyclist"]:
            continue

        print(f"处理物体：{cls}")

        h, w, l = map(float, obj[8:11])
        x, y, z = map(float, obj[11:14])
        yaw = float(obj[14])

        box_cam = np.array([h, w, l, x, y, z, yaw])

        # 正确 Camera → LiDAR
        box_lidar = camera_to_lidar_box(box_cam, R0, Tr)

        # 裁剪
        obj_points = crop(points, box_lidar)
        print(f"→ 点数：{len(obj_points)}")

        # 保存点云
        np.save(os.path.join(save_dir, f"{cls}_{obj_id}.npy"), obj_points)

        # 保存 PNG
        save_png(obj_points, os.path.join(save_dir, f"{cls}_{obj_id}.png"))

        obj_id += 1


# ======================================================
# 运行示例：只处理一帧（000000）
# ======================================================
root = r"E:\postgraduate\Python_program\point cloud test\KITTI"
frame_id = "007480"

process_all_objects(frame_id, root)
