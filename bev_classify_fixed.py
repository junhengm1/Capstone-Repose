"""
改进后的BEV区域分类函数
修复了原始代码中的潜在问题
"""

def classify_category(x_b, y_b, bev_x_min, bev_x_max, bev_y_min, bev_y_max, 
                     impass_W, impass_H, impass_binary, pass_binary, white_binary):
    """
    返回bev区域的类别：-1=无，0=passable，1=impassable，2=white
    
    参数:
        x_b, y_b: BEV坐标系中的点坐标
        bev_x_min, bev_x_max: BEV的x轴范围
        bev_y_min, bev_y_max: BEV的y轴范围
        impass_W, impass_H: impassable mask的宽度和高度
        impass_binary, pass_binary, white_binary: 二值化mask数组
    
    改进点:
        1. 添加了数组索引边界检查，防止越界
        2. 处理了边界情况（u=1.0或v=1.0时）
        3. 确保所有变量在使用前已定义
    """
    # 归一化
    u = (x_b - bev_x_min) / (bev_x_max - bev_x_min)
    v = (y_b - bev_y_min) / (bev_y_max - bev_y_min)

    # 检查点是否在bev中
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return -1
    
    # 归一化坐标映射到bev像素坐标
    # 使用 min 确保索引不会超出数组范围
    # 当 u=1.0 时，col = impass_W-1（而不是impass_W）
    col = min(int(round(u * (impass_W - 1))), impass_W - 1)
    row = min(int(round(v * (impass_H - 1))), impass_H - 1)
    
    # 确保索引非负（虽然理论上不会出现，但为了安全）
    col = max(0, col)
    row = max(0, row)

    # 重叠区域按impass > pass > white的优先级
    if impass_binary[row, col]:
        return 1  # 落在impassable
    if pass_binary[row, col]:
        return 0  # 落在passable
    if white_binary[row, col]:
        return 2  # 落在white
    return -1


# 如果使用numpy，可以使用更简洁的版本
def classify_category_numpy(x_b, y_b, bev_x_min, bev_x_max, bev_y_min, bev_y_max, 
                           impass_W, impass_H, impass_binary, pass_binary, white_binary):
    """
    使用numpy的改进版本（更简洁）
    """
    import numpy as np
    
    # 归一化
    u = (x_b - bev_x_min) / (bev_x_max - bev_x_min)
    v = (y_b - bev_y_min) / (bev_y_max - bev_y_min)

    # 检查点是否在bev中
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return -1
    
    # 使用 np.clip 确保索引在有效范围内
    col = int(np.clip(round(u * (impass_W - 1)), 0, impass_W - 1))
    row = int(np.clip(round(v * (impass_H - 1)), 0, impass_H - 1))

    # 重叠区域按impass > pass > white的优先级
    if impass_binary[row, col]:
        return 1  # 落在impassable
    if pass_binary[row, col]:
        return 0  # 落在passable
    if white_binary[row, col]:
        return 2  # 落在white
    return -1
