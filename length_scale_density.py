# -*- coding: utf-8 -*-
"""
长度标尺驱动密度场外轮廓提取算法
Length Scale Driven Density Field Method for Sparse Image Contour Extraction

基于方案文档实现的完整算法，包含：
1. 长度标尺场计算
2. 密度场建模与插值
3. 各向异性扩散平滑
4. 基于梯度的外轮廓提取

依赖: numpy, scipy, opencv-python
"""

import os
import sys
import argparse
import numpy as np
import cv2
from scipy.spatial import Delaunay, cKDTree
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata, RBFInterpolator
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 工具函数
# ============================================================================

def ensure_dir(path: str):
    """确保目录存在"""
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def imread_gray(path: str) -> np.ndarray:
    """读取灰度图像"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return img


def extract_boundary_points(img: np.ndarray, threshold: int = 126) -> np.ndarray:
    """
    从二值图像提取边界点
    返回: (N, 2) 数组，格式为 (row, col)
    """
    binary = img < threshold
    # 提取边界点
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary.astype(np.uint8), kernel, iterations=1)
    boundary = binary.astype(np.uint8) - eroded
    
    points = np.argwhere(boundary > 0)  # (row, col)
    return points


def rc_to_xy(points_rc: np.ndarray) -> np.ndarray:
    """将 (row, col) 转换为 (x, y)"""
    return points_rc[:, ::-1].astype(np.float64)


def xy_to_rc(points_xy: np.ndarray) -> np.ndarray:
    """将 (x, y) 转换为 (row, col)"""
    return points_xy[:, ::-1]


# ============================================================================
# 1. 长度标尺场计算
# ============================================================================

def compute_length_scale(points_xy: np.ndarray, 
                         tri_simplices: np.ndarray,
                         k_neighbors: int = 6) -> np.ndarray:
    """
    计算每个顶点的局部长度标尺
    
    参数:
        points_xy: (N, 2) 点坐标数组
        tri_simplices: (M, 3) Delaunay三角剖分的三角形索引
        k_neighbors: 用于计算平均距离的邻居数
    
    返回:
        length_scale: (N,) 每个点的长度标尺值
    """
    N = len(points_xy)
    
    # 构建邻接关系
    adjacency = [set() for _ in range(N)]
    for tri in tri_simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[tri[i]].add(tri[j])
    
    # 计算每个点的长度标尺
    length_scale = np.zeros(N)
    
    for i in range(N):
        neighbors = list(adjacency[i])
        if len(neighbors) == 0:
            # 孤立点，使用全局平均
            length_scale[i] = 10.0
            continue
        
        # 计算到邻居的平均距离
        neighbor_points = points_xy[neighbors]
        distances = np.linalg.norm(neighbor_points - points_xy[i], axis=1)
        mean_dist = np.mean(distances)
        
        # L(p_i) = sqrt(3)/2 * mean_distance
        length_scale[i] = (np.sqrt(3) / 2.0) * mean_dist
    
    return length_scale


# ============================================================================
# 2. 密度场构建与插值
# ============================================================================

def build_density_field(points_xy: np.ndarray,
                       length_scale: np.ndarray,
                       grid_shape: Tuple[int, int],
                       method: str = 'rbf') -> np.ndarray:
    """
    从长度标尺构建连续密度场
    
    参数:
        points_xy: (N, 2) 点坐标
        length_scale: (N,) 长度标尺值
        grid_shape: (H, W) 输出网格形状
        method: 插值方法 ('rbf', 'linear', 'cubic')
    
    返回:
        density_field: (H, W) 密度场
    """
    H, W = grid_shape
    
    # 从长度标尺计算密度: ρ = C / L^2
    # 归一化常数
    C = np.mean(length_scale ** 2)
    density_values = C / (length_scale ** 2 + 1e-6)
    
    # 创建网格
    x = np.arange(W)
    y = np.arange(H)
    grid_x, grid_y = np.meshgrid(x, y)
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    
    # 插值到网格
    if method == 'rbf':
        # 使用RBF插值（更平滑）
        # 为了性能，如果点太多则采样
        if len(points_xy) > 2000:
            indices = np.random.choice(len(points_xy), 2000, replace=False)
            points_sample = points_xy[indices]
            density_sample = density_values[indices]
        else:
            points_sample = points_xy
            density_sample = density_values
        
        # 使用thin_plate_spline核
        rbf = RBFInterpolator(points_sample, density_sample, 
                             kernel='thin_plate_spline', smoothing=0.1)
        density_grid = rbf(grid_points).reshape(H, W)
    else:
        # 使用griddata插值
        density_grid = griddata(points_xy, density_values, 
                               (grid_x, grid_y), method=method, 
                               fill_value=0.0)
    
    # 确保非负
    density_grid = np.maximum(density_grid, 0.0)
    
    return density_grid


# ============================================================================
# 3. 各向异性扩散平滑
# ============================================================================

def anisotropic_diffusion(density: np.ndarray,
                         iterations: int = 10,
                         kappa: float = 50.0,
                         gamma: float = 0.1) -> np.ndarray:
    """
    各向异性扩散平滑密度场
    
    参数:
        density: (H, W) 密度场
        iterations: 迭代次数
        kappa: 扩散阈值参数
        gamma: 时间步长
    
    返回:
        smoothed: (H, W) 平滑后的密度场
    """
    img = density.copy()
    
    for _ in range(iterations):
        # 计算梯度
        grad_y, grad_x = np.gradient(img)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # 扩散系数: D(s) = exp(-(s/kappa)^2)
        diffusion_coef = np.exp(-((grad_mag / kappa) ** 2))
        
        # 计算扩散项
        diff_y = np.gradient(diffusion_coef * grad_y, axis=0)
        diff_x = np.gradient(diffusion_coef * grad_x, axis=1)
        
        # 更新
        img = img + gamma * (diff_x + diff_y)
    
    return img


# ============================================================================
# 4. Mean-Shift梯度迭代
# ============================================================================

def mean_shift_contour(points_xy: np.ndarray,
                      density_field: np.ndarray,
                      tau: float,
                      eta: float = 0.3,
                      max_iter: int = 50,
                      epsilon: float = 1e-3) -> np.ndarray:
    """
    Mean-Shift梯度下降到等密度线
    
    参数:
        points_xy: (N, 2) 初始点坐标
        density_field: (H, W) 密度场
        tau: 目标密度阈值
        eta: 步长
        max_iter: 最大迭代次数
        epsilon: 收敛阈值
    
    返回:
        converged_points: (N, 2) 收敛后的点坐标
    """
    H, W = density_field.shape
    points = points_xy.copy()
    
    # 计算密度场梯度
    grad_y, grad_x = np.gradient(density_field)
    
    for iteration in range(max_iter):
        # 获取当前点的密度值和梯度
        x_coords = np.clip(points[:, 0], 0, W - 1).astype(int)
        y_coords = np.clip(points[:, 1], 0, H - 1).astype(int)
        
        rho_current = density_field[y_coords, x_coords]
        grad_x_current = grad_x[y_coords, x_coords]
        grad_y_current = grad_y[y_coords, x_coords]
        
        # Mean-Shift更新: x_{t+1} = x_t - eta * (rho - tau) * grad_rho
        delta_rho = rho_current - tau
        
        # 更新位置
        points[:, 0] -= eta * delta_rho * grad_x_current
        points[:, 1] -= eta * delta_rho * grad_y_current
        
        # 边界约束
        points[:, 0] = np.clip(points[:, 0], 0, W - 1)
        points[:, 1] = np.clip(points[:, 1], 0, H - 1)
        
        # 检查收敛
        if np.max(np.abs(delta_rho)) < epsilon:
            break
    
    return points


# ============================================================================
# 5. 基于梯度的外轮廓提取
# ============================================================================

def extract_gradient_descent_contours(
    density_field: np.ndarray,
    step_size: float = 1.2,
    max_steps: int = 300,
    grad_threshold: float = 1e-3,
    start_percentile: float = 60.0,
    sample_stride: int = 4,
    min_length: int = 25
) -> List[np.ndarray]:
    """
    基于密度梯度的外轮廓提取。

    通过在密度场的高梯度区域撒点并沿负梯度方向做最速下降，
    收集能够到达边界或梯度平坦区域的路径，适用于稀疏密度场。

    参数:
        density_field: (H, W) 密度场
        step_size: 梯度下降的步长
        max_steps: 每条轨迹的最大迭代步数
        grad_threshold: 梯度幅值低于该值视为收敛
        start_percentile: 作为起点的梯度幅值分位数
        sample_stride: 起点网格采样间隔，用于覆盖密度稀疏区域
        min_length: 保留的最短轨迹长度

    返回:
        contours: 梯度下降得到的外轮廓轨迹列表
    """
    H, W = density_field.shape
    grad_y, grad_x = np.gradient(density_field)
    grad_mag = np.hypot(grad_x, grad_y)

    start_thresh = np.percentile(grad_mag, start_percentile)
    high_grad_mask = grad_mag >= start_thresh

    # 在高梯度区域撒点，同时结合规则网格保证稀疏区也被探索
    seeds = []
    high_grad_points = np.argwhere(high_grad_mask)
    for idx in range(0, len(high_grad_points), max(1, sample_stride)):
        y, x = high_grad_points[idx]
        seeds.append((x + 0.5, y + 0.5))
    for y in range(0, H, sample_stride * 2):
        for x in range(0, W, sample_stride * 2):
            seeds.append((x + 0.5, y + 0.5))

    visited = np.zeros((H, W), dtype=bool)
    contours: List[np.ndarray] = []

    for sx, sy in seeds:
        xi, yi = int(np.clip(sx, 0, W - 1)), int(np.clip(sy, 0, H - 1))
        if visited[yi, xi]:
            continue

        path = []
        point = np.array([sx, sy], dtype=np.float64)

        for _ in range(max_steps):
            px = int(np.clip(point[0], 0, W - 1))
            py = int(np.clip(point[1], 0, H - 1))

            visited[py, px] = True
            gx = grad_x[py, px]
            gy = grad_y[py, px]
            gmag = grad_mag[py, px]

            path.append(point.copy())

            if gmag < grad_threshold:
                break

            direction = -np.array([gx, gy]) / (gmag + 1e-8)
            next_point = point + direction * step_size

            point = np.array([
                np.clip(next_point[0], 0, W - 1),
                np.clip(next_point[1], 0, H - 1)
            ])

            if point[0] in (0, W - 1) or point[1] in (0, H - 1):
                path.append(point.copy())
                break

        if len(path) >= min_length:
            contours.append(np.array(path))

    return contours


# ============================================================================
# 6. 能量演化与样条平滑（当前未在pipeline中启用）
# ============================================================================

def elastic_smoothing(contour: np.ndarray,
                     density_field: np.ndarray,
                     tau: float,
                     alpha: float = 0.2,
                     beta: float = 0.6,
                     gamma: float = 3.0,
                     iterations: int = 20,
                     dt: float = 0.1) -> np.ndarray:
    """
    能量演化平滑轮廓
    
    能量泛函: E = α∫|C'|²ds + β∫|C''|²ds + γ∫(ρ(C)-τ)²ds
    
    参数:
        contour: (N, 2) 轮廓点
        density_field: (H, W) 密度场
        tau: 目标密度
        alpha: 拉伸平滑权重
        beta: 弯曲平滑权重
        gamma: 密度吸附权重
        iterations: 迭代次数
        dt: 时间步长
    
    返回:
        smoothed_contour: (N, 2) 平滑后的轮廓
    """
    H, W = density_field.shape
    points = contour.copy()
    N = len(points)
    
    # 计算密度场梯度
    grad_y, grad_x = np.gradient(density_field)
    
    for _ in range(iterations):
        # 计算一阶导数（切向量）
        tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
        tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-6
        tangent = tangent / tangent_norm
        
        # 计算二阶导数（曲率）
        curvature = np.roll(points, -1, axis=0) - 2 * points + np.roll(points, 1, axis=0)
        
        # 拉伸力（一阶平滑）
        stretch_force = alpha * (np.roll(points, -1, axis=0) + np.roll(points, 1, axis=0) - 2 * points)
        
        # 弯曲力（二阶平滑）
        bending_force = beta * curvature
        
        # 密度吸附力
        x_coords = np.clip(points[:, 0].astype(int), 0, W - 1)
        y_coords = np.clip(points[:, 1].astype(int), 0, H - 1)
        
        rho_current = density_field[y_coords, x_coords]
        grad_x_current = grad_x[y_coords, x_coords]
        grad_y_current = grad_y[y_coords, x_coords]
        
        density_force = gamma * (rho_current - tau)[:, np.newaxis] * \
                       np.column_stack([grad_x_current, grad_y_current])
        
        # 总力
        total_force = stretch_force + bending_force - density_force
        
        # 更新位置
        points = points + dt * total_force
        
        # 边界约束
        points[:, 0] = np.clip(points[:, 0], 0, W - 1)
        points[:, 1] = np.clip(points[:, 1], 0, H - 1)
    
    # 最后用样条平滑
    points = spline_smooth(points, smoothness=0.5)
    
    return points


def spline_smooth(contour: np.ndarray, smoothness: float = 0.5) -> np.ndarray:
    """
    使用高斯滤波进行样条平滑
    
    参数:
        contour: (N, 2) 轮廓点
        smoothness: 平滑强度
    
    返回:
        smoothed: (N, 2) 平滑后的轮廓
    """
    if len(contour) < 3:
        return contour
    
    # 对x和y坐标分别平滑
    sigma = smoothness * len(contour) / 20.0
    
    # 周期性边界条件
    x_smooth = gaussian_filter(contour[:, 0], sigma=sigma, mode='wrap')
    y_smooth = gaussian_filter(contour[:, 1], sigma=sigma, mode='wrap')
    
    return np.column_stack([x_smooth, y_smooth])


# ============================================================================
# 7. 完整Pipeline
# ============================================================================

def length_scale_density_pipeline(img: np.ndarray,
                                 tau_percentile: float = 70.0,
                                 diffusion_iter: int = 10,
                                 gradient_steps: int = 300,
                                 sample_stride: int = 4) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    完整的长度标尺驱动密度场轮廓提取pipeline
    
    参数:
        img: 输入灰度图像
        tau_percentile: 梯度起点的密度梯度分位阈值
        diffusion_iter: 扩散迭代次数
        gradient_steps: 梯度下降的最大迭代步数
        sample_stride: 撒点步长，控制稀疏区域的采样密度
    
    返回:
        contours: 提取的轮廓列表
        density_field: 密度场
    """
    H, W = img.shape
    
    # 1. 提取边界点
    print("步骤1: 提取边界点...")
    boundary_rc = extract_boundary_points(img)
    if len(boundary_rc) < 10:
        print("警告: 边界点太少")
        return [], np.zeros((H, W))
    
    points_xy = rc_to_xy(boundary_rc)
    print(f"  提取了 {len(points_xy)} 个边界点")
    
    # 2. Delaunay三角剖分
    print("步骤2: Delaunay三角剖分...")
    try:
        tri = Delaunay(points_xy)
        tri_simplices = tri.simplices
        print(f"  生成了 {len(tri_simplices)} 个三角形")
    except Exception as e:
        print(f"三角剖分失败: {e}")
        return [], np.zeros((H, W))
    
    # 3. 计算长度标尺
    print("步骤3: 计算长度标尺场...")
    length_scale = compute_length_scale(points_xy, tri_simplices)
    print(f"  长度标尺范围: [{np.min(length_scale):.2f}, {np.max(length_scale):.2f}]")
    
    # 4. 构建密度场
    print("步骤4: 构建密度场...")
    density_field = build_density_field(points_xy, length_scale, (H, W), method='linear')
    print(f"  密度场范围: [{np.min(density_field):.4f}, {np.max(density_field):.4f}]")
    
    # 5. 各向异性扩散平滑
    print("步骤5: 各向异性扩散平滑...")
    density_smooth = anisotropic_diffusion(density_field, iterations=diffusion_iter)
    
    # 6. 基于梯度的外轮廓提取
    print("步骤6: 基于梯度的外轮廓提取...")
    gradient_contours = extract_gradient_descent_contours(
        density_smooth,
        max_steps=gradient_steps,
        start_percentile=tau_percentile,
        sample_stride=sample_stride
    )
    print(f"  梯度下降得到 {len(gradient_contours)} 条候选轮廓")

    print(f"完成! 最终得到 {len(gradient_contours)} 条轮廓")

    return gradient_contours, density_smooth


# ============================================================================
# 8. 可视化与输出
# ============================================================================

def visualize_results(img: np.ndarray,
                     contours: List[np.ndarray],
                     density_field: np.ndarray,
                     output_path: str):
    """
    可视化结果并保存
    
    参数:
        img: 原始图像
        contours: 轮廓列表
        density_field: 密度场
        output_path: 输出路径
    """
    ensure_dir(output_path)
    
    # 1. 轮廓叠加图
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for contour in contours:
        pts = contour.astype(np.int32)
        # 转换为OpenCV格式 (x, y) -> (col, row)
        pts_cv = np.column_stack([pts[:, 0], pts[:, 1]])
        cv2.polylines(overlay, [pts_cv.reshape(-1, 1, 2)], True, 
                     (0, 0, 255), 2, lineType=cv2.LINE_AA)
    
    cv2.imwrite(output_path, overlay)
    print(f"保存轮廓叠加图到: {output_path}")
    
    # 2. 密度场可视化
    density_vis = (density_field - density_field.min()) / (density_field.max() - density_field.min() + 1e-6)
    density_vis = (density_vis * 255).astype(np.uint8)
    density_colored = cv2.applyColorMap(density_vis, cv2.COLORMAP_JET)
    
    density_path = output_path.replace('.png', '_density.png')
    cv2.imwrite(density_path, density_colored)
    print(f"保存密度场可视化到: {density_path}")


# ============================================================================
# 9. 命令行接口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='长度标尺驱动密度场外轮廓提取算法'
    )
    parser.add_argument('--input', default='data/input/6.png',
                       help='输入图像路径')
    parser.add_argument('--output', default='data/output/length_scale_result.png',
                       help='输出图像路径')
    parser.add_argument('--tau-percentile', type=float, default=65.0,
                       help='梯度起点选择的分位阈值 (default: 65.0)')
    parser.add_argument('--diffusion-iter', type=int, default=10,
                       help='扩散迭代次数 (default: 10)')
    parser.add_argument('--gradient-steps', type=int, default=300,
                       help='梯度下降最大步数 (default: 300)')
    parser.add_argument('--sample-stride', type=int, default=4,
                       help='梯度撒点间隔，控制稀疏区域覆盖 (default: 4)')
    
    args = parser.parse_args()
    
    # 读取图像
    print(f"读取图像: {args.input}")
    img = imread_gray(args.input)
    print(f"图像尺寸: {img.shape}")
    
    # 运行pipeline
    print("\n" + "="*60)
    print("开始长度标尺驱动密度场轮廓提取")
    print("="*60 + "\n")
    
    contours, density_field = length_scale_density_pipeline(
        img,
        tau_percentile=args.tau_percentile,
        diffusion_iter=args.diffusion_iter,
        gradient_steps=args.gradient_steps,
        sample_stride=args.sample_stride
    )
    
    # 可视化结果
    print("\n" + "="*60)
    print("保存结果")
    print("="*60 + "\n")
    
    visualize_results(img, contours, density_field, args.output)
    
    print("\n处理完成!")


if __name__ == '__main__':
    main()