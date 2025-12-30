import wave
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 第一部分：自研算法工具库 (底层实现)
# 来源: 课程设计-语音信号.pdf p6, p10, p12
# ==========================================

def get_hamming_window(window_len):
    """手动实现汉明窗公式: w(n) = 0.54 - 0.46 * cos(2*pi*n / (N-1)) [1]"""
    n = np.arange(window_len)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_len - 1))
    return window

def enframe(signal, frame_len, frame_shift):
    """分帧与加窗处理 [1]"""
    signal_len = len(signal)
    # 计算总帧数
    num_frames = int(np.ceil((signal_len - frame_len) / frame_shift)) + 1
    # 补零以确保完整分帧
    pad_len = (num_frames - 1) * frame_shift + frame_len
    padded_signal = np.concatenate((signal, np.zeros(pad_len - signal_len)))
    
    # 构造帧矩阵
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_len, 1)).T
    frames = padded_signal[indices.astype(np.int32)]
    
    # 应用汉明窗
    window = get_hamming_window(frame_len)
    return frames * window

def calc_short_time_energy(frames):
    """计算短时能量: En = sum(x(m)^2) [2, 3]"""
    return np.sum(frames**2, axis=1)

def calc_short_time_zcr(frames):
    """计算短时平均过零率: 统计相邻采样点符号变化 [3, 4]"""
    num_frames, frame_len = frames.shape
    zcr = np.zeros(num_frames)
    for i in range(num_frames):
        frame = frames[i]
        count = 0
        for j in range(1, frame_len):
            if frame[j] * frame[j-1] < 0:
                count += 1
        zcr[i] = count
    return zcr

# ==========================================
# 第二部分：任务主逻辑 (I/O 与 检测算法)
# 来源: 任务及要求2025.pdf [5, 6]
# ==========================================

def load_wav_with_wave(filename):
    """使用 Python 标准库 wave 读取音频，不产生第三方依赖 [6]"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        # 转换为 16 位整数并归一化到 [-1, 1]
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        wave_data = wave_data.astype(np.float32) / 32768.0
    return fs, wave_data

def endpoint_detection(energy, zcr):
    """
    双门限前端检测算法 [7]
    步骤：利用 ITU 寻找有声段，利用 ITL 和 IZCT 修正起始点
    """
    # 1. 自动设定门限 (基于前10帧环境噪声)
    noise_energy = np.mean(energy[:10])
    noise_zcr = np.mean(zcr[:10])
    
    ITU = noise_energy * 5.0   # 高能量门限
    ITL = noise_energy * 2.2   # 低能量门限
    IZCT = noise_zcr * 3.0     # 过零率门限
    
    segments = []
    in_voiced = False
    temp_start = 0
    
    # 2. 状态机：利用高低能量门限初步搜索 [7]
    for i in range(len(energy)):
        if not in_voiced and energy[i] > ITU:
            in_voiced = True
            temp_start = i
        elif in_voiced and energy[i] < ITL:
            in_voiced = False
            segments.append([temp_start, i])
            
    return segments, [ITU, ITL, IZCT]

def main():
    #选择文件
    audio_file = "F0004CA0B1A502.wav" 
    
    try:
        fs, data = load_wav_with_wave(audio_file)
        print(f"成功读取文件: {audio_file} | 采样率: {fs}")
    except Exception as e:
        print(f"读取失败: {e}。请检查文件名及路径是否正确。")
        return

    # 1. 预处理与特征提取 (25ms帧长, 10ms帧移) [1]
    frame_len = int(0.025 * fs)
    frame_shift = int(0.01 * fs)
    frames = enframe(data, frame_len, frame_shift)
    
    energy = calc_short_time_energy(frames)
    zcr = calc_short_time_zcr(frames)
    
    # 2. 执行双门限检测 [7]
    segments, thresholds = endpoint_detection(energy, zcr)
    itu, itl, izct = thresholds
    
    # 3. 绘图展示 (Matplotlib) [5, 6]
    plt.figure(figsize=(12, 10))
    
    # 子图1: 原始波形与检测到的端点 (红线开始, 绿线结束)
    plt.subplot(3, 1, 1)
    plt.plot(data, color='silver', label='Speech Waveform')
    for start, end in segments:
        plt.axvline(x=start * frame_shift, color='red', linestyle='--', label='Start' if start == segments else "")
        plt.axvline(x=end * frame_shift, color='green', linestyle='--', label='End' if end == segments[5] else "")
    plt.title("Task 1: Endpoint Detection Result")
    plt.legend()
    
    # 子图2: 短时能量与双门限
    plt.subplot(3, 1, 2)
    plt.plot(energy, color='blue')
    plt.axhline(y=itu, color='red', linestyle=':', label=f'ITU={itu:.2e}')
    plt.axhline(y=itl, color='orange', linestyle=':', label=f'ITL={itl:.2e}')
    plt.title("Short-time Energy (with ITL/ITU)")
    plt.legend()
    
    # 子图3: 短时过零率与门限
    plt.subplot(3, 1, 3)
    plt.plot(zcr, color='darkcyan')
    plt.axhline(y=izct, color='magenta', linestyle=':', label=f'IZCT={izct:.1f}')
    plt.title("Short-time Zero Crossing Rate (with IZCT)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()