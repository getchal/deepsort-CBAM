import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

# ---------------------------------------------------------
# 学术图表规范设置 (解决中文显示与负号问题)
# ---------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']  # 兼容不同系统的中文字体
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 设置高分辨率，保证论文打印清晰


def evaluate_tracking_data(txt_path="tracking_data.txt"):
    """
    读取标准的跟踪数据文本文件，分析轨迹寿命，并生成学术图表。
    期望的 txt 格式: frame_id, track_id, x1, y1, x2, y2
    """
    if not os.path.exists(txt_path):
        print(f"\n[致命错误] 找不到数据文件：'{txt_path}'")
        print("请确保你已经先运行了 generate_data.py 生成了此文件！\n")
        return

    track_lifespans = defaultdict(int)
    total_frames = 0

    print(f"[*] 正在深度分析跟踪数据文件: {txt_path} ...")

    # 1. 读取并统计数据
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue

            try:
                frame_id = int(parts[0])
                track_id = int(parts[1])
            except ValueError:
                continue

            # 统计每个 ID 出现的总帧数 (即生命周期)
            track_lifespans[track_id] += 1
            if frame_id > total_frames:
                total_frames = frame_id

    if not track_lifespans:
        print("[错误] 数据文件解析成功，但未能提取到有效的跟踪数据！")
        return

    # 2. 计算核心评价指标
    total_ids = len(track_lifespans)
    lifespans_list = list(track_lifespans.values())

    avg_lifespan = np.mean(lifespans_list)
    max_lifespan = np.max(lifespans_list)
    median_lifespan = np.median(lifespans_list)

    # 过滤掉存活极短的“碎片噪点”（例如小于 15 帧的通常是误检或严重遮挡导致的闪烁）
    noise_threshold = 15
    valid_lifespans = [L for L in lifespans_list if L > noise_threshold]
    valid_avg_lifespan = np.mean(valid_lifespans) if valid_lifespans else 0
    fragment_ratio = (total_ids - len(valid_lifespans)) / total_ids * 100 if total_ids > 0 else 0

    # 3. 在终端打印给写论文用的结构化数据
    print("\n" + "=" * 50)
    print(" 📊 多目标跟踪稳定性评估报告 (可用于论文表格)")
    print("=" * 50)
    print(f" 视频总帧数:\t\t {total_frames} 帧")
    print(f" 系统分配的总 ID 数:\t {total_ids} 个")
    print(f" 轨迹平均生存寿命:\t {avg_lifespan:.2f} 帧")
    print(f" 轨迹中位数寿命:\t {median_lifespan:.1f} 帧")
    print(f" 最长连续跟踪寿命:\t {max_lifespan} 帧")
    print("-" * 50)
    print(f" 有效目标(>{noise_threshold}帧)平均寿命:\t {valid_avg_lifespan:.2f} 帧")
    print(f" 极短碎片 ID 占比 (IDSW指征):\t {fragment_ratio:.2f}%")
    print("=" * 50)
    print(" 💡 论文撰写提示：")
    print(" 1. '总 ID 数' 越接近视频真实人数越好。数量庞大意味着发生了大量 IDSW。")
    print(" 2. '有效目标平均寿命' 反映了系统抗遮挡的能力，越长越好。")
    print(" 3. 建议使用'带CBAM'和'无CBAM'的数据分别跑一次，进行定量对比！\n")


    plt.figure(figsize=(10, 6.5))

    # 科学划分寿命区间 (贴合 MOT 评估习惯)
    bins = [0, 15, 45, 90, 150, 300, max(max_lifespan + 1, 301)]
    labels = [
        '0-15帧\n(严重碎片)',
        '16-45帧\n(极短跟踪)',
        '46-90帧\n(短程跟踪)',
        '91-150帧\n(中程跟踪)',
        '151-300帧\n(稳定跟踪)',
        '>300帧\n(长效锁定)'
    ]

    # 统计各个区间内的 ID 数量
    counts, _ = np.histogram(lifespans_list, bins=bins)

    # 设定渐变配色方案 (从冷色调到暖色调，象征稳定性增强)
    colors = ['#e74c3c', '#f39c12', '#f1c40f', '#3498db', '#2ecc71', '#27ae60']

    # 绘制带边缘黑线的柱状图，更具学术感
    bars = plt.bar(labels, counts, color=colors[:len(labels)], edgecolor='black', linewidth=1.2, alpha=0.85)

    # 设置标题和坐标轴 (中英双语，显得专业)
    plt.title('Distribution of Tracker ID Lifespans\n(目标跟踪轨迹生命周期分布)', fontsize=16, fontweight='bold',
              pad=20)
    plt.xlabel('Track Lifespan Intervals (Frames / 连续存活帧数区间)', fontsize=13, labelpad=10)
    plt.ylabel('Number of Unique IDs (落入该区间的 ID 总数)', fontsize=13, labelpad=10)

    # 在每根柱子上标出具体的数字
    for bar in bars:
        yval = bar.get_height()
        if yval > 0:  # 只标记非零的柱子
            plt.text(bar.get_x() + bar.get_width() / 2, yval + (max(counts) * 0.015),
                     int(yval), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加网格线辅助阅读
    plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

    # 调整布局，防止标签被切掉
    plt.tight_layout()

    # 保存图片
    chart_filename = "Ablation_Track_Lifespan.png"
    plt.savefig(chart_filename, bbox_inches='tight')
    print(f"[*] 成功生成论文配图：{chart_filename}")

    # 显示图片
    plt.show()


if __name__ == '__main__':
    evaluate_tracking_data()