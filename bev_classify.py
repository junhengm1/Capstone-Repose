import numpy as np

def classify_category(x_b, y_b, bev_x_min, bev_x_max, bev_y_min, bev_y_max,
                     impass_binary, pass_binary, white_binary):
    """
    返回bev区域的类别：-1=无，0=passable，1=impassable，2=white
    
    Parameters:
    -----------
    x_b, y_b : float
        BEV坐标系中的点坐标
    bev_x_min, bev_x_max, bev_y_min, bev_y_max : float
        BEV坐标范围
    impass_binary, pass_binary, white_binary : np.ndarray
        二值mask数组，形状应该一致 (H, W)
    
    Returns:
    --------
    int : 类别代码
        -1: 不在BEV范围内或不在任何mask上
        0: passable
        1: impassable (优先级最高)
        2: white
    """
    # 归一化
    u = (x_b - bev_x_min) / (bev_x_max - bev_x_min)
    v = (y_b - bev_y_min) / (bev_y_max - bev_y_min)

    # 检查点是否在bev中（使用严格边界检查）
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return -1
    
    # 获取mask尺寸
    impass_H, impass_W = impass_binary.shape
    
    # 归一化坐标映射到bev像素坐标
    # 使用clip确保索引在有效范围内，防止浮点误差导致的越界
    col = int(np.clip(round(u * (impass_W - 1)), 0, impass_W - 1))
    row = int(np.clip(round(v * (impass_H - 1)), 0, impass_H - 1))

    # 重叠区域按impass > pass > white的优先级
    if impass_binary[row, col]:
        return 1  # 落在impassable
    if pass_binary[row, col]:
        return 0  # 落在passable
    if white_binary[row, col]:
        return 2  # 落在white
    return -1  # 不在任何mask上


# 改进版本：支持批量处理
def classify_category_batch(x_b_array, y_b_array, bev_x_min, bev_x_max, 
                           bev_y_min, bev_y_max, impass_binary, pass_binary, white_binary):
    """
    批量处理点分类，更高效
    
    Parameters:
    -----------
    x_b_array, y_b_array : np.ndarray
        点坐标数组
    其他参数同 classify_category
    
    Returns:
    --------
    np.ndarray : 类别代码数组
    """
    x_b_array = np.asarray(x_b_array)
    y_b_array = np.asarray(y_b_array)
    
    # 归一化
    u = (x_b_array - bev_x_min) / (bev_x_max - bev_x_min)
    v = (y_b_array - bev_y_min) / (bev_y_max - bev_y_min)
    
    # 检查点是否在bev中
    valid_mask = (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    
    # 获取mask尺寸
    impass_H, impass_W = impass_binary.shape
    
    # 映射到像素坐标
    cols = np.clip(np.round(u * (impass_W - 1)).astype(int), 0, impass_W - 1)
    rows = np.clip(np.round(v * (impass_H - 1)).astype(int), 0, impass_H - 1)
    
    # 初始化结果数组
    result = np.full(len(x_b_array), -1, dtype=int)
    
    # 只处理有效点
    valid_rows = rows[valid_mask]
    valid_cols = cols[valid_mask]
    
    # 按优先级检查mask
    valid_idx = np.where(valid_mask)[0]
    
    # impassable优先级最高
    impass_mask = impass_binary[valid_rows, valid_cols]
    result[valid_idx[impass_mask]] = 1
    
    # passable
    remaining_mask = ~impass_mask
    pass_mask = pass_binary[valid_rows[remaining_mask], valid_cols[remaining_mask]]
    result[valid_idx[remaining_mask][pass_mask]] = 0
    
    # white
    remaining_mask2 = remaining_mask & ~pass_mask
    white_mask = white_binary[valid_rows[remaining_mask2], valid_cols[remaining_mask2]]
    result[valid_idx[remaining_mask2][white_mask]] = 2
    
    return result


# 使用示例
if __name__ == "__main__":
    # 示例：创建测试数据
    # 假设BEV范围
    bev_x_min, bev_x_max = -10.0, 10.0
    bev_y_min, bev_y_max = -10.0, 10.0
    
    # 假设mask尺寸
    H, W = 100, 100
    impass_binary = np.zeros((H, W), dtype=bool)
    pass_binary = np.zeros((H, W), dtype=bool)
    white_binary = np.zeros((H, W), dtype=bool)
    
    # 设置一些测试区域
    impass_binary[40:60, 40:60] = True
    pass_binary[20:80, 20:80] = True
    white_binary[0:50, 0:50] = True
    
    # 测试单个点
    x_test, y_test = 0.0, 0.0
    category = classify_category(x_test, y_test, bev_x_min, bev_x_max, 
                                bev_y_min, bev_y_max, impass_binary, 
                                pass_binary, white_binary)
    print(f"点 ({x_test}, {y_test}) 的类别: {category}")
    
    # 测试批量点
    x_batch = np.array([0.0, 5.0, -5.0, 15.0])
    y_batch = np.array([0.0, 5.0, -5.0, 15.0])
    categories = classify_category_batch(x_batch, y_batch, bev_x_min, bev_x_max,
                                        bev_y_min, bev_y_max, impass_binary,
                                        pass_binary, white_binary)
    print(f"批量点的类别: {categories}")
