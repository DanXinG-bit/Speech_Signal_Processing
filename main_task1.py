import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import functions_task1 as func

def endpoint_detection(energy, zcr, fs):
    """
    双门限前端检测算法逻辑实现
    来源: 课程设计-语音信号.pdf [11]
    """
    # 1. 阈值设定 (基于前10帧背景噪声)
    noise_energy = np.mean(energy[:10])
    noise_zcr = np.mean(zcr[:10])
    
    ITU = noise_energy * 5   # 高能量门限
    ITL = noise_energy * 2   # 低能量门限
    IZCT = noise_zcr * 3     # 过零率门限
    
    # 状态机或多级搜索逻辑
    voiced_segments = []
    in_voiced = False
    temp_start = 0
    
    # 简化逻辑示例：利用高低门限寻找 A1, A2, B1, B2 [11]
    for i in range(len(energy)):
        if not in_voiced and energy[i] > ITU:  # 达到高门限，初步判定进入有声段
            in_voiced = True
            temp_start = i
        elif in_voiced and energy[i] < ITL:    # 低于低门限，判定段结束
            in_voiced = False
            voiced_segments.append([temp_start, i])
            
    return voiced_segments, [ITU, ITL, IZCT]

def main():
    # 1. 读取语音文件 (要求: 两端静音，中间至少3段静音)
    # 请确保将给定数据中的某个文件路径填入此处 [1]
    filename = "your_speech_file.wav" 
    try:
        fs, data = wavfile.read(filename)
    except FileNotFoundError:
        print("错误: 找不到音频文件，请确认路径。")
        return

    # 归一化处理
    data = data / np.max(np.abs(data))
    
    # 2. 特征提取 [11]
    frame_len = int(0.025 * fs)   # 25ms 典型窗长 [7]
    frame_shift = int(0.01 * fs)  # 10ms 帧移
    frames = func.enframe(data, frame_len, frame_shift)
    
    energy = func.calc_short_time_energy(frames)
    zcr = func.calc_short_time_zcr(frames)
    
    # 3. 执行检测算法 [11]
    segments, thresholds = endpoint_detection(energy, zcr, fs)
    
    # 4. 可视化结果 [1]
    plt.figure(figsize=(12, 8))
    
    # 原语音波形
    plt.subplot(3, 1, 1)
    plt.plot(data)
    plt.title("Speech Waveform and Detected Endpoints")
    for start, end in segments:
        # 将帧序号转换回采样点序号进行标注
        plt.axvline(x=start*frame_shift, color='r', linestyle='--')
        plt.axvline(x=end*frame_shift, color='g', linestyle='--')
    
    # 短时能量
    plt.subplot(3, 1, 2)
    plt.plot(energy)
    plt.axhline(y=thresholds, color='r', label='ITU')
    plt.axhline(y=thresholds[1], color='orange', label='ITL')
    plt.title("Short-time Energy")
    plt.legend()
    
    # 过零率
    plt.subplot(3, 1, 3)
    plt.plot(zcr)
    plt.axhline(y=thresholds[2], color='r', label='IZCT')
    plt.title("Zero Crossing Rate")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    print(f"检测完成。共发现 {len(segments)} 个有声段。")

if __name__ == "__main__":
    main()