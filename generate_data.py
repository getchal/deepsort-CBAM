import cv2
import time
from AIDetector_pytorch import Detector


def main():
    det = Detector()
    func_status = {'headpose': None}

    # 填入你的原视频路径
    video_path = 'E:/c98c8-main/videos/test.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[报错] 无法打开视频文件: {video_path}")
        return

    # 打开一个文本文件，准备写入跟踪数据
    txt_file = open("tracking_data.txt", "w")
    frame_id = 0

    print("[*] 后端推理引擎已启动...")
    print("[*] 正在全力进行目标检测与特征追踪，不进行图像渲染。")
    print("[*] 请耐心等待，此过程耗时取决于您的 CPU 算力...")

    start_time = time.time()
    while True:
        ret, im = cap.read()
        if not ret or im is None:
            break

        # 核心推理：只算数据，不关心画面
        result = det.feedCap(im, func_status)

        # 提取当前帧所有的边界框数据
        bboxes = result['face_bboxes']

        # 按照格式写入 txt 文件：帧号, ID, x1, y1, x2, y2
        for box in bboxes:
            x1, y1, x2, y2, _, track_id = box
            txt_file.write(f"{frame_id},{track_id},{x1},{y1},{x2},{y2}\n")

        frame_id += 1

        if frame_id % 50 == 0:
            print(f"已处理 {frame_id} 帧...")

    txt_file.close()
    cap.release()

    total_time = time.time() - start_time
    print(f"\n🎉 离线数据生成完毕！共处理 {frame_id} 帧，耗时 {total_time:.2f} 秒。")
    print("您可以关闭此脚本，运行 Project_UI.py 进行交互演示了。")


if __name__ == '__main__':
    main()