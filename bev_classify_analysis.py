"""
BEV区域分类代码分析
分析 classify_category 函数的正确性和潜在问题
"""

import numpy as np

# 示例：分析代码逻辑
def classify_category_original(x_b, y_b, bev_x_min, bev_x_max, bev_y_min, bev_y_max, 
                               impass_W, impass_H, impass_binary, pass_binary, white_binary):
    """
    原始代码版本 - 存在潜在问题
    返回bev区域的类别：-1=无，0=passable，1=impassable，2=white
    """
    # 归一化
    u = (x_b - bev_x_min) / (bev_x_max - bev_x_min)
    v = (y_b - bev_y_min) / (bev_y_max - bev_y_min)

    # 检查点是否在bev中
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return -1
    
    # 归一化坐标映射到bev像素坐标
    col = int(round(u * (impass_W - 1)))
    row = int(round(v * (impass_H - 1)))

    # 重叠区域按impass > pass > white的优先级
    if impass_binary[row, col]:
        return 1  # 落在impassable
    if pass_binary[row, col]:
        return 0  # 落在passable
    if white_binary[row, col]:
        return 2  # 落在white
    return -1


def classify_category_improved(x_b, y_b, bev_x_min, bev_x_max, bev_y_min, bev_y_max, 
                               impass_W, impass_H, impass_binary, pass_binary, white_binary):
    """
    改进版本 - 修复潜在问题
    返回bev区域的类别：-1=无，0=passable，1=impassable，2=white
    """
    # 归一化
    u = (x_b - bev_x_min) / (bev_x_max - bev_x_min)
    v = (y_b - bev_y_min) / (bev_y_max - bev_y_min)

    # 检查点是否在bev中（使用更严格的边界检查）
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return -1
    
    # 归一化坐标映射到bev像素坐标
    # 使用 np.clip 确保索引在有效范围内
    col = int(np.clip(round(u * (impass_W - 1)), 0, impass_W - 1))
    row = int(np.clip(round(v * (impass_H - 1)), 0, impass_H - 1))

    # 边界情况：当 u=1.0 或 v=1.0 时，确保不会越界
    # 上面的 clip 已经处理了这个问题

    # 重叠区域按impass > pass > white的优先级
    if impass_binary[row, col]:
        return 1  # 落在impassable
    if pass_binary[row, col]:
        return 0  # 落在passable
    if white_binary[row, col]:
        return 2  # 落在white
    return -1


def classify_category_alternative(x_b, y_b, bev_x_min, bev_x_max, bev_y_min, bev_y_max, 
                                 impass_W, impass_H, impass_binary, pass_binary, white_binary):
    """
    替代方案 - 使用 floor 而不是 round（更符合像素网格映射）
    返回bev区域的类别：-1=无，0=passable，1=impassable，2=white
    """
    # 归一化
    u = (x_b - bev_x_min) / (bev_x_max - bev_x_min)
    v = (y_b - bev_y_min) / (bev_y_max - bev_y_min)

    # 检查点是否在bev中
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return -1
    
    # 使用 floor 映射，确保索引在 [0, W-1] 和 [0, H-1] 范围内
    col = min(int(u * impass_W), impass_W - 1)
    row = min(int(v * impass_H), impass_H - 1)

    # 重叠区域按impass > pass > white的优先级
    if impass_binary[row, col]:
        return 1  # 落在impassable
    if pass_binary[row, col]:
        return 0  # 落在passable
    if white_binary[row, col]:
        return 2  # 落在white
    return -1


# 测试函数
def test_classify_category():
    """测试分类函数的边界情况"""
    # 创建测试数据
    impass_W, impass_H = 100, 100
    impass_binary = np.zeros((impass_H, impass_W), dtype=bool)
    pass_binary = np.zeros((impass_H, impass_W), dtype=bool)
    white_binary = np.zeros((impass_H, impass_W), dtype=bool)
    
    # 设置一些测试区域
    impass_binary[50:60, 50:60] = True
    pass_binary[30:40, 30:40] = True
    white_binary[10:20, 10:20] = True
    
    bev_x_min, bev_x_max = 0.0, 100.0
    bev_y_min, bev_y_max = 0.0, 100.0
    
    # 测试边界情况
    test_cases = [
        (0.0, 0.0, "左下角"),
        (100.0, 100.0, "右上角"),
        (50.0, 50.0, "中心点"),
        (-10.0, 50.0, "超出左边界"),
        (150.0, 50.0, "超出右边界"),
        (50.0, -10.0, "超出下边界"),
        (50.0, 150.0, "超出上边界"),
    ]
    
    print("=" * 60)
    print("测试原始版本")
    print("=" * 60)
    for x, y, desc in test_cases:
        try:
            result = classify_category_original(
                x, y, bev_x_min, bev_x_max, bev_y_min, bev_y_max,
                impass_W, impass_H, impass_binary, pass_binary, white_binary
            )
            print(f"{desc:15s} ({x:6.1f}, {y:6.1f}) -> {result}")
        except IndexError as e:
            print(f"{desc:15s} ({x:6.1f}, {y:6.1f}) -> IndexError: {e}")
    
    print("\n" + "=" * 60)
    print("测试改进版本")
    print("=" * 60)
    for x, y, desc in test_cases:
        result = classify_category_improved(
            x, y, bev_x_min, bev_x_max, bev_y_min, bev_y_max,
            impass_W, impass_H, impass_binary, pass_binary, white_binary
        )
        print(f"{desc:15s} ({x:6.1f}, {y:6.1f}) -> {result}")


if __name__ == "__main__":
    test_classify_category()
