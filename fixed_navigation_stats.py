    def _draw_navigation_stats(self, ax, batch_dict, batch_idx):
        """绘制导航导航对比结果，由navigation_stats开关控制
        
        显示导航对比：
        1. Amp Information：外部给的导航引导，从定义对标到可阅读含义可以参考RoutingVisualizer
        2. featuremap:e2e_passable：上游给的bev形式下的导航引导
        """

        ax.set_aspect("equal", "box")
        ax.set_xlim(-60, 60)
        ax.set_ylim(-60, 140)

        MAIN_ACTION = {
            0: "No primary navigation action",
            1: "Turn left",
            2: "Turn right",
            3: "Bear left ahead",
            4: "Bear right ahead",
            5: "Rear-left turn",
            6: "Rear-right turn",
            7: "Left U-turn",
            8: "Go straight",
            9: "Keep left",
            10: "Keep right",
            11: "Enter roundabout",
            12: "Exit roundabout",
            13: "Slow down",
        }

        ASSIST_ACTION = {
            0: "No assist navigation action",
            1: "Enter main road",
            2: "Enter auxiliary road",
            3: "Enter highway",
            4: "Enter ramp",
            5: "Enter tunnel",
            6: "Enter middle branch",
            7: "Enter right branch",
            8: "Enter left branch",
            9: "Enter right-turn lane",
            10: "Enter left-turn lane",
            11: "Enter middle lane",
            12: "Enter right-side lane",
            13: "Enter left-side lane",
            14: "Keep right to enter auxiliary road",
            15: "Keep left to enter auxiliary road",
            16: "Keep right to enter main road",
            17: "Keep left to enter main road",
            18: "Enter right-turn exclusive lane",
            19: "Arrive at ferry channel",
            20: "Depart from ferry",
            23: "Follow the current road",
            24: "Follow auxiliary road",
            25: "Follow main road",
            32: "Arrive at exit",
            33: "Arrive at service area",
            34: "Arrive at toll station",
            35: "Arrive at via point",
            36: "Arrive at destination",
            37: "Arrive at charging station (EV)",
            48: "Roundabout left turn",
            49: "Roundabout right turn",
            50: "Roundabout straight",
            51: "Roundabout U-turn",
            52: "Small roundabout (no exit count)",
            64: "Complex junction: take 1st right exit",
            65: "Complex junction: take 2nd right exit",
            66: "Complex junction: take 3rd right exit",
            67: "Complex junction: take 4th right exit",
            68: "Complex junction: take 5th right exit",
            69: "Complex junction: take 1st left exit",
            70: "Complex junction: take 2nd left exit",
            71: "Complex junction: take 3rd left exit",
            72: "Complex junction: take 4th left exit",
            73: "Complex junction: take 5th left exit",
            80: "Enter U-turn lane",
            90: "Cross pedestrian crossing",
            91: "Cross overpass",
            92: "Cross underpass",
            93: "Pass through square",
            94: "Pass through park",
            95: "Take escalator",
            96: "Take elevator",
            97: "Take cableway",
            98: "Pass through sky bridge",
            99: "Pass through building passage",
            100: "Pass pedestrian road",
            101: "Follow boat route",
            102: "Follow sightseeing vehicle route",
            103: "Follow slideway",
            105: "Climb stairs",
            106: "Follow ramp",
            107: "Cross bridge",
            108: "Take ferry",
            109: "Pass subway passage",
            112: "Entering building (not issued)",
            113: "Leaving building (not issued)",
            114: "Enter roundabout (bike/pedestrian)",
            115: "Exit roundabout (bike/pedestrian)",
            116: "Enter small road",
            117: "Enter internal road",
            118: "Enter 2nd left branch",
            119: "Enter 3rd left branch",
            120: "Enter 2nd right branch",
            121: "Enter 3rd right branch",
            122: "Enter gas station road",
            123: "Enter residential road",
            124: "Enter industrial park road",
            125: "Enter overpass",
            126: "Take middle ramp to overpass",
            127: "Take right ramp to overpass",
            128: "Take left ramp to overpass",
            129: "Continue straight",
            130: "Exit overpass",
            131: "Take left lane to overpass",
            132: "Take right lane to overpass",
            133: "Enter bridge",
            134: "Enter parking lot",
            135: "Enter interchange",
            136: "Enter elevated bridge",
            137: "Enter underground passage",
        }

        lane_action_map = {
            0: "Straight",
            1: "Left turn",
            2: "Straight | Left",
            3: "Right turn",
            4: "Straight | Right",
            5: "Left U-turn",
            6: "Left | Right",
            7: "Straight | Left | Right",
            8: "Right U-turn",
            9: "Straight | Left U-turn",
            10: "Straight | Right U-turn",
            11: "Left | Left U-turn",
            12: "Right | Right U-turn",
            13: "Straight | Extend",
            14: "Left | Left U-turn | Extend",
            15: "Reserved",
            16: "Straight | Left | Left U-turn",
            17: "Right | Left U-turn",
            18: "Left | Left U-turn",
            19: "Straight | Right | Right U-turn",
            20: "Left | Right U-turn",
            21: "Bus lane",
            22: "Empty lane",
            23: "Reversible lane",
            24: "Dedicated lane",
            25: "Tidal lane",
            255: "No corresponding lane",
            30: "None",
        }

        # 1. 从batch中提取跟amap相关信息
        # 提取scalar并反归一
        def _extract_scalar(batch_val, scale):
            if batch_val is None:
                return None
            arr = np.array(batch_val).reshape(-1)
            if arr.size == 0:
                return None
            return float(arr[0] * scale)

        main_action_raw = self._extract_tensor(batch_dict.get("structure:main_action"), batch_idx)
        assistant_action_raw = self._extract_tensor(batch_dict.get("structure:assistant_action"), batch_idx)
        step_main_action_raw = self._extract_tensor(batch_dict.get("structure:step_main_action"), batch_idx)
        step_assistant_action_raw = self._extract_tensor(batch_dict.get("structure:step_assistant_action"), batch_idx)
        dist_step_raw = self._extract_tensor(batch_dict.get("structure:distance_to_next_step"), batch_idx)
        dist_lane_raw = self._extract_tensor(batch_dict.get("structure:distance_to_next_lane"), batch_idx)

        main_id = _extract_scalar(main_action_raw, 12)
        assist_id = _extract_scalar(assistant_action_raw, 132)
        step_main_id = _extract_scalar(step_main_action_raw, 12)
        step_assist_id = _extract_scalar(step_assistant_action_raw, 132)
        dist_step = _extract_scalar(dist_step_raw, 2000)
        dist_lane = _extract_scalar(dist_lane_raw, 2000)

        # 生成导航文本
        nav_lines = []

        if main_id is not None:
            nav_lines.append(f"Main: {MAIN_ACTION.get(int(round(main_id)), 'Unknown')}")

        if assist_id is not None:
            nav_lines.append(f"Assist: {ASSIST_ACTION.get(int(round(assist_id)), 'Unknown')}")

        if step_main_id is not None:
            nav_lines.append(f"Next:  {MAIN_ACTION.get(int(round(step_main_id)), 'Unknown')}")

        if step_assist_id is not None:
            nav_lines.append(f"Next Assist: {ASSIST_ACTION.get(int(round(step_assist_id)), 'Unknown')}")

        if dist_step is not None:
            nav_lines.append(f"Dist to Next Step: {dist_step:.1f} m")

        if dist_lane is not None:
            nav_lines.append(f"Dist to Lane Action: {dist_lane:.1f} m")

        # 显示文本
        ax.text(
            0.02, 0.98,
            "\n".join(nav_lines),
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            ha="left", va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="#1a1a1a",
                edgecolor="white",
                alpha=0.8,
            ),
        )

        # Lane bitmaps & arrows
        left_bitmap_raw  = self._extract_tensor(batch_dict.get("structure:left_bitmap"),  batch_idx)
        right_bitmap_raw = self._extract_tensor(batch_dict.get("structure:right_bitmap"), batch_idx)
        left_arrow_raw   = self._extract_tensor(batch_dict.get("structure:left_arrow"),   batch_idx)
        right_arrow_raw  = self._extract_tensor(batch_dict.get("structure:right_arrow"),  batch_idx)

        def _to_1d_int(arr):
            """安全转换到 1D int 数组"""
            if arr is None:
                return None
            a = np.array(arr)
            if a.ndim == 0:
                return np.array([int(a)])
            return a.reshape(-1).astype(int)

        # bitmap (already 0/1 numbers)
        lb = _to_1d_int(left_bitmap_raw)
        rb = _to_1d_int(right_bitmap_raw)

        # arrow: builder 除了 30 → 还原
        la = _to_1d_int(left_arrow_raw * 30 if left_arrow_raw is not None else None)
        ra = _to_1d_int(right_arrow_raw * 30 if right_arrow_raw is not None else None)

        # ---------------- Format string ----------------
        def _fmt_bitmap(b):
            if b is None:
                return ""
            return "".join(f"[{int(x)}]" for x in b)

        def _fmt_arrow(a):
            if a is None:
                return ""
            labels = []
            for v in a:
                v = int(v)
                labels.append(lane_action_map.get(v, str(v)))
            return "".join(f"[{lab}]" for lab in labels)

        lb_str = _fmt_bitmap(lb)
        rb_str = _fmt_bitmap(rb)
        la_str = _fmt_arrow(la)
        ra_str = _fmt_arrow(ra)

        lane_lines = []

        if lb_str:
            lane_lines.append("Left Bitmap: " + lb_str)
        if rb_str:
            lane_lines.append("Right Bitmap: " + rb_str)
        if la_str:
            lane_lines.append("Left Arrow: " + la_str)
        if ra_str:
            lane_lines.append("Right Arrow: " + ra_str)

        if lane_lines:
            ax.text(
                0.02, 0.02,
                "\n".join(lane_lines),
                transform=ax.transAxes,
                fontsize=8,
                color="white",
                ha="left", va="bottom",
                bbox=dict(
                    boxstyle="round",
                    facecolor="#1a1a1a",
                    edgecolor="white",
                    alpha=0.8,
                ),
            )

        # 2. 绘制 e2e_passable（将 BEV 轨迹投影到自车坐标系）
        e2e_pass_raw = batch_dict.get("feature_map:e2e_passable_mask", None)

        if e2e_pass_raw is not None:

            mask = self._extract_tensor(e2e_pass_raw, batch_idx)
            if isinstance(mask, torch.Tensor):
                mask = mask.detach().cpu().numpy()

            # 解包各种维度格式 -> [H, W]
            if mask.ndim == 5:         # [B, C, T, H, W]
                mask_img = mask[0, 0, 0]
            elif mask.ndim == 4:       # [C, T, H, W] or [1,1,H,W]
                mask_img = mask[0, 0]
            elif mask.ndim == 3:       # [T, H, W]
                mask_img = mask[0]
            else:                      # [H, W]
                mask_img = mask

            if mask_img is not None:
                H, W = mask_img.shape

                # --- BEV 像素坐标 -> 自车坐标 (x_forward, y_lateral) ---
                def bev_uv_to_ego_xy(row_indices, col_indices):
                    """
                    将BEV图像像素坐标转换为自车坐标系
                    
                    Args:
                        row_indices: 行索引（u），对应前后方向 x_forward
                        col_indices: 列索引（v），对应左右方向 y_lateral
                    
                    Returns:
                        (x_forward, y_lateral): 自车坐标系坐标
                    """
                    row_indices = np.asarray(row_indices, dtype=np.float32)
                    col_indices = np.asarray(col_indices, dtype=np.float32)

                    # BEV图像覆盖范围（自车坐标系，单位：米）
                    x_min, x_max = -60.0, 140.0   # 前后方向（forward）
                    y_min, y_max = -60.0,  60.0   # 左右方向（lateral）

                    # 计算每个像素对应的物理尺寸（米/像素）
                    x_resolution = (x_max - x_min) / H  # 前后方向分辨率
                    y_resolution = (y_max - y_min) / W  # 左右方向分辨率

                    # 在BEV图像中，通常第一行（row=0）对应最远的前方（x_max）
                    # 最后一行（row=H-1）对应最近的前方（x_min）
                    # 所以需要反转行索引
                    x_forward = x_max - (row_indices + 0.5) * x_resolution
                    
                    # 列索引：第一列（col=0）对应最左侧（y_min），最后一列（col=W-1）对应最右侧（y_max）
                    y_lateral = y_min + (col_indices + 0.5) * y_resolution
                    
                    return x_forward, y_lateral

                row_indices_list = []
                col_indices_list = []

                # 从每一行取 "通行区域均值以上的像素"，计算横向中心点
                for row_idx in range(H):
                    row = mask_img[row_idx]
                    # 找到大于均值的像素索引（列索引）
                    col_indices = np.where(row > row.mean())[0]
                    if col_indices.size > 0:
                        # 计算该行的横向中心点（列索引的平均值）
                        col_center = col_indices.mean()
                        col_indices_list.append(col_center)
                        row_indices_list.append(row_idx)

                if len(row_indices_list) > 2:
                    # 像素坐标 -> 自车坐标
                    x_fwd, y_lat = bev_uv_to_ego_xy(row_indices_list, col_indices_list)

                    # 注意：横轴画左右 y_lat，纵轴画前后 x_fwd
                    ax.plot(
                        y_lat,
                        x_fwd,
                        linestyle="--",
                        color="#78c8ff",   # 淡蓝色
                        linewidth=2.0,
                        alpha=0.95,
                        label="e2e_passable"
                    )
                    
        else:
            print("No e2e_passable_mask found")

        ax.axvline(x=0, color='white', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(True, alpha=0.2, color='gray', linestyle='--')
        ax.tick_params(colors='white', labelsize=9)
