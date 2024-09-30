import cv2
import numpy as np
import gradio as gr

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    ### FILL: 基于MLS or RBF 实现 image warping
    
    # get p and q arrays
    p = target_pts[..., [1, 0]] 
    q = source_pts[..., [1, 0]]
    
    # generate v matrix and compute w
    H, W = image.shape[:2]
    m_u = np.arange(H)[:, None]  # (H, 1)
    m_v = np.arange(W)[None, :]  # (1, W)
    m_u, m_v = np.meshgrid(m_u, m_v)
    m_u = m_u[..., None]
    m_v = m_v[..., None]
    v = np.concatenate([m_u, m_v], axis=-1)  # (H, W, 2)
    w = p[:, None, None, :] - v[None, ...]
    w = np.linalg.norm(w, axis=-1, keepdims=True) ** (2 * alpha)
    w = 1 / (w + eps)

    # compute p_star, q_star, p_hat, q_hat
    p_star = np.sum(w * p[:, None, None, :], axis=0, keepdims=True) / np.sum(w, axis=0, keepdims=True) # (1, H, W, 2)
    q_star = np.sum(w * q[:, None, None, :], axis=0, keepdims=True) / np.sum(w, axis=0, keepdims=True)    
    p_hat = p[:, None, None, :] - p_star  # (n, H, W, 2)
    q_hat = q[:, None, None, :] - q_star

    # compute fa and get final image
    A_1 = v[None, ...] - p_star # (1,H,W,2)
    A_2 = np.linalg.inv(np.sum(p_hat[..., None] @ p_hat[..., None, :] * w[..., None], axis=0, keepdims=True)) # (1,H,W,2,2)
    A_3 = w * p_hat # (n,H,W,2)
    A = A_1[..., None, :] @ A_2 @ A_3[..., None] # (n,H,W,1,1)
    fa = np.sum(A[..., 0] * q_hat, axis=0) + q_star[0] # (1,H,W,2)
    fa[..., 0] = np.clip(fa[..., 0], 0, H-1)
    fa[..., 1] = np.clip(fa[..., 1], 0, W-1)
    fa = fa.astype(np.int32)
    final_image = np.zeros_like(warped_image)
    final_image[v[..., 0], v[..., 1]] = warped_image[fa[..., 0], fa[..., 1]]
    warped_image = final_image
    return warped_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()