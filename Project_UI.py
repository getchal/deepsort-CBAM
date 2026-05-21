import cv2
import datetime
import time  # [新增] 引入高精度时间模块

# ==========================================
# 全局变量
# ==========================================
current_focus_id = -1
current_frame_boxes = []

# [新增] 用于存储指令下达时的真实物理时间戳
cmd_receive_time = 0.0
measure_latency_flag = False


def mouse_callback(event, x, y, flags, param):
    """鼠标点击回调函数"""
    global current_focus_id, current_frame_boxes, cmd_receive_time, measure_latency_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        for track_id, x1, y1, x2, y2 in current_frame_boxes:
            if x1 <= x <= x2 and y1 <= y <= y2:
                # 🌟 第一步：在捕获到物理鼠标点击的瞬间，记录纳秒级起始时间！
                cmd_receive_time = time.perf_counter()
                measure_latency_flag = True

                current_focus_id = track_id
                now_str = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"[{now_str}] [UI_Event] 捕获鼠标真实物理点选, 目标 ID: {current_focus_id}")
                break


def nothing(x):
    pass


def main():
    global current_focus_id, current_frame_boxes, cmd_receive_time, measure_latency_flag

    print("[*] 正在加载跟踪数据...")
    tracking_data = {}

    try:
        with open("tracking_data_baseline.txt", "r") as f:
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

    cv2.setMouseCallback(window_name, mouse_callback)

    frame_idx = 0
    paused = False
    input_buffer = ""

    ret, last_frame = cap.read()
    if not ret: return

    print("\n[*] 具身视觉交互界面(高精度性能测试版)已启动！")
    print("[*] 提示：每次进行锁定操作，系统将自动测算真实的端到端渲染延迟。\n" + "-" * 50)

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

        # === 以下为状态解析与渲染核心逻辑 ===
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

        overlay = render_frame.copy()
        cv2.rectangle(overlay, (0, 0), (render_frame.shape[1], 65), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, render_frame, 0.4, 0, render_frame)

        status_text = f"FRAME: {frame_idx}/{total_frames} "
        status_text += "[PAUSED] | " if paused else "| "
        status_text += f"FOCUS: ID {current_focus_id} (LOCKED)" if current_focus_id != -1 else "FOCUS: GLOBAL"

        if input_buffer:
            status_text += f" | TYPE: {input_buffer}_"

        cv2.putText(render_frame, status_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(render_frame, "Mouse Click to Lock | Input ID + Enter | C (Reset) | Space (Pause) | Q (Quit)",
                    (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 🌟 第二步：真正将像素推送到屏幕上
        cv2.imshow(window_name, render_frame)

        # 🌟 第三步：如果刚才发生了交互，计算并打印真实延迟！
        if measure_latency_flag:
            render_finish_time = time.perf_counter()
            # 真实延迟 = (渲染完成时间 - 指令接收时间) * 1000 转换为毫秒
            real_latency_ms = (render_finish_time - cmd_receive_time) * 1000

            now_str = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"[{now_str}] [FSM_Core] 意图解析与状态机跃迁完成...")
            print(f"[{now_str}] [Render_Stream] UI渲染结束. ✅ 真实物理响应延迟: {real_latency_ms:.2f} ms\n")

            measure_latency_flag = False  # 测算完毕，复位标志位

        # === 键盘监听逻辑 ===
        key = cv2.waitKey(wait_time if not paused else 50) & 0xFF

        if ord('0') <= key <= ord('9'):
            input_buffer += chr(key)
        elif key == 13 or key == 10:
            if input_buffer:
                # 🌟 第一步：在捕获到物理回车键的瞬间，记录起始时间！
                cmd_receive_time = time.perf_counter()
                measure_latency_flag = True

                current_focus_id = int(input_buffer)
                now_str = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"[{now_str}] [UI_Event] 捕获键盘真实物理按键, 目标 ID: {current_focus_id}")
                input_buffer = ""
        elif key == 8 or key == 127:
            input_buffer = input_buffer[:-1]
        elif key == ord('c') or key == ord('C'):
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