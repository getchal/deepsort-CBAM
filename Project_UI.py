import cv2

# ==========================================
# 全局变量：用于鼠标点击事件跨函数通信
# ==========================================
current_focus_id = -1
current_frame_boxes = []


def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调函数：实现鼠标直观点选锁定"""
    global current_focus_id, current_frame_boxes
    if event == cv2.EVENT_LBUTTONDOWN:
        # 遍历当前帧画面上的所有框，看鼠标点在了哪个框里面
        for track_id, x1, y1, x2, y2 in current_frame_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                current_focus_id = track_id
                print(f"[交互] 鼠标点选 -> 瞬间锁定 ID: {current_focus_id}")
                break  # 锁定后跳出循环


def nothing(x):
    pass


def main():
    global current_focus_id, current_frame_boxes

    print("[*] 正在加载跟踪数据...")
    tracking_data = {}

    try:
        with open("tracking_data.txt", "r") as f:
            for line in f:
                parts = line.strip().split(",")
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x1, y1, x2, y2 = map(int, map(float, parts[2:6]))

                if frame_id not in tracking_data:
                    tracking_data[frame_id] = []
                tracking_data[frame_id].append((track_id, x1, y1, x2, y2))
    except FileNotFoundError:
        print("[报错] 找不到 tracking_data.txt！请先运行 generate_data.py。")
        return

    video_path = 'E:/c98c8-main/videos/test.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[致命错误] 找不到视频文件: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25
    wait_time = max(1, int(1000 / fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    window_name = "Embodied Interactive System"
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Progress", window_name, 0, total_frames - 1, nothing)

    # [新增] 绑定鼠标点击事件监听器
    cv2.setMouseCallback(window_name, mouse_callback)

    frame_idx = 0
    paused = False
    input_buffer = ""  # [新增] 用于存储键盘输入的多位数字

    ret, last_frame = cap.read()
    if not ret: return

    print("\n[*] 具身视觉交互界面(增强版)已启动！")
    print("[*] 🖱️ 鼠标操作：直接点击画面中的人框锁定，或拖动底部滚动条。")
    print("[*] ⌨️ 键盘操作：输入任意数字按 Enter 锁定，按 C 恢复全局，按空格暂停，按 Q 退出。")

    while cap.isOpened():
        trackbar_pos = cv2.getTrackbarPos("Progress", window_name)

        is_dragging = False
        if paused and trackbar_pos != frame_idx:
            is_dragging = True
        elif not paused and abs(trackbar_pos - frame_idx) > 1:
            is_dragging = True

        if is_dragging:
            frame_idx = trackbar_pos
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret: last_frame = frame.copy()
        elif not paused:
            if frame_idx < total_frames - 1:
                ret, frame = cap.read()
                if ret:
                    last_frame = frame.copy()
                    frame_idx += 1
                    cv2.setTrackbarPos("Progress", window_name, frame_idx)
                else:
                    paused = True
            else:
                paused = True

            # 3. 更新全局框数据，供鼠标点击逻辑使用
        current_frame_boxes = tracking_data.get(frame_idx, [])
        render_frame = last_frame.copy()

        for track_id, x1, y1, x2, y2 in current_frame_boxes:
            is_focused = (track_id == current_focus_id)
            if is_focused:
                color = (0, 0, 255)
                thickness = 4
                label = f"LOCKED ID: {track_id}"
            else:
                color = (255, 128, 0)
                thickness = 2
                label = f"ID: {track_id}"

            cv2.rectangle(render_frame, (x1, y1), (x2, y2), color, thickness)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(render_frame, (x1, y1 - t_size[1] - 5), (x1 + t_size[0], y1), color, -1)
            cv2.putText(render_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 4. 绘制高级 HUD
        overlay = render_frame.copy()
        cv2.rectangle(overlay, (0, 0), (render_frame.shape[1], 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, render_frame, 0.4, 0, render_frame)

        status_text = f"FRAME: {frame_idx}/{total_frames} "
        if paused:
            status_text += "[PAUSED] | "
        else:
            status_text += "| "

        status_text += f"FOCUS: ID {current_focus_id} (LOCKED)" if current_focus_id != -1 else "FOCUS: GLOBAL"

        # [新增] 如果正在键盘输入，在屏幕上显示出来
        if input_buffer:
            status_text += f" | TYPE: {input_buffer}_"

        cv2.putText(render_frame, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(render_frame, "Mouse Click to Lock | Input ID + Enter | C (Reset) | Space (Pause) | Q (Quit)",
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(window_name, render_frame)

        # 5. [升级] 动态指令解析器
        key = cv2.waitKey(wait_time if not paused else 50) & 0xFF

        if ord('0') <= key <= ord('9'):
            # 拼装数字缓存
            input_buffer += chr(key)
        elif key == 13 or key == 10:  # 监听 Enter (回车) 键
            if input_buffer:
                current_focus_id = int(input_buffer)
                print(f"[交互] 键盘锁定 ID: {current_focus_id}")
                input_buffer = ""  # 清空缓存
        elif key == 8 or key == 127:  # <--- 修复：现在只监听真正的退格键！
            input_buffer = input_buffer[:-1]
        elif key == ord('c') or key == ord('C'):  # 取消锁定
            current_focus_id = -1
            input_buffer = ""
            print("[交互] 恢复全局监控")
        elif key == ord(' '):
            paused = not paused
        elif key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()