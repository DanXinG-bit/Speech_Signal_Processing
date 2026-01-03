import wave
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# ==========================================
# 第一部分：自研算法工具库 (底层实现)
# ==========================================

def load_wav(filename):
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        return fs, wave_data.astype(np.float32) / 32768.0

def simple_bandpass_filter(data, fs, low=60, high=900):
    """
    简化的带通滤波：滤除 60Hz 以下和 900Hz 以上成分 [3, 4]
    使用 FFT 频域置零法实现
    """
    n = len(data)
    # 计算FFT
    fft_data = np.fft.fft(data)
    
    # 计算频率轴
    freqs = np.zeros(n)
    for i in range(n):
        if i <= n//2:
            freqs[i] = i * fs / n
        else:
            freqs[i] = (i - n) * fs / n
    
    # 频域置零（只保留低频部分）
    for i in range(n):
        freq_abs = abs(freqs[i])
        if freq_abs < low or freq_abs > high:
            fft_data[i] = 0 + 0j
    
    # 逆变换回时域
    filtered = np.fft.ifft(fft_data)
    return filtered.real

def calc_autocorr_segment(frame, k_min, k_max):
    """计算指定延迟范围内的自相关函数 R(k) [2, 5]"""
    frame_len = len(frame)
    R = np.zeros(k_max + 1)  # 创建数组，不是单个值！
    
    # 计算所有k值的自相关（只计算需要的范围）
    for k in range(k_min, k_max + 1):
        sum_val = 0.0
        for n in range(frame_len - k):
            sum_val += frame[n] * frame[n + k]
        R[k] = sum_val
    
    return R

# ==========================================
# 第二部分：基音检测主逻辑
# ==========================================

def pitch_detection(data, fs):
    # 1. 预处理：带通滤波 (60-900Hz) [4]
    print("正在进行带通滤波...")
    filtered_data = simple_bandpass_filter(data, fs)
    
    # 2. 分帧 (典型值：30ms 帧长, 10ms 帧移) [3]
    frame_len = int(0.030 * fs)
    frame_shift = int(0.010 * fs)
    signal_len = len(filtered_data)
    num_frames = (signal_len - frame_len) // frame_shift
    print(f"总帧数: {num_frames}, 帧长: {frame_len}, 帧移: {frame_shift}")
    
    # 初始化结果数组
    pitches = np.zeros(num_frames)
    v_u_decisions = np.zeros(num_frames)  # 1 为浊音, 0 为清音/静音

    # 3. 设定基音搜索范围 (对应 60Hz - 500Hz)
    k_min = int(fs / 500)  # 对应500Hz
    k_max = int(fs / 60)   # 对应60Hz
    print(f"基音搜索范围: k_min={k_min}, k_max={k_max}")

    # 逐帧处理
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_len
        frame = filtered_data[start:end]
        
        # 4. 计算自相关并寻找最大值 Rmax [2]
        R = calc_autocorr_segment(frame, k_min, k_max)
        
        # 在k_min到k_max范围内找最大值
        max_value = R[k_min]
        max_k = k_min
        for k in range(k_min + 1, k_max + 1):
            if R[k] > max_value:
                max_value = R[k]
                max_k = k
        
        # 5. 清浊音判别逻辑 [2]
        # 计算R(0)
        R0 = 0.0
        for n in range(frame_len):
            R0 += frame[n] * frame[n]
        
        # 计算能量
        energy = 0.0
        for n in range(frame_len):
            energy += frame[n] * frame[n]
        
        # 判别条件：
        # 1. 能量足够高（不是静音）
        # 2. 自相关峰值足够显著
        energy_threshold = 0.01
        if energy > energy_threshold and R0 > 0.25 * max_value:
            pitches[i] = max_k
            v_u_decisions[i] = 1  # 浊音
        else:
            pitches[i] = 0
            v_u_decisions[i] = 0  # 清音或静音
    
    return pitches, v_u_decisions

def main():
    # 创建隐藏的Tkinter窗口
    root = tk.Tk()
    root.withdraw()
    
    # 文件选择
    current_dir = os.getcwd()
    audio_file = filedialog.askopenfilename(
        initialdir=current_dir, 
        title="任务三：选择语音文件进行基音检测",
        filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
    )
    
    if not audio_file:
        print("未选择文件，程序退出")
        return
    
    print(f"选择文件: {os.path.basename(audio_file)}")
    
    # 加载音频
    try:
        fs, data = load_wav(audio_file)
        print(f"采样率: {fs} Hz, 数据长度: {len(data)} 点")
    except Exception as e:
        print(f"加载音频失败: {e}")
        return
    
    # 进行基音检测
    print("开始基音检测...")
    pitches, v_u = pitch_detection(data, fs)
    print(f"基音检测完成，检测到 {np.sum(v_u)} 个浊音帧")
    
    # 将基音周期转换为频率（Hz）
    frequencies = np.zeros(len(pitches))
    for i in range(len(pitches)):
        if pitches[i] > 0:
            frequencies[i] = fs / pitches[i]
    
    # 绘图展示
    plt.figure(figsize=(10, 8))
    
    # 图 1: 原始波形
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(data)) / fs
    plt.plot(time_axis, data, color='silver', linewidth=0.5)
    plt.title(f"Waveform: {os.path.basename(audio_file)}")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # 图 2: 基音频率轨迹
    plt.subplot(3, 1, 2)
    frame_times = np.arange(len(pitches)) * 0.01  # 10ms帧移
    plt.plot(frame_times, frequencies, 'r.', markersize=3)
    plt.title("Pitch Frequency (Hz)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(50, 500)  # 基音频率通常范围
    plt.grid(True, alpha=0.3)
    
    # 图 3: 清浊音判别结果
    plt.subplot(3, 1, 3)
    plt.step(frame_times, v_u, where='post', color='blue', linewidth=1)
    plt.title("Voiced(1) / Unvoiced(0) Decision")
    plt.xlabel("Time (s)")
    plt.ylabel("Decision")
    plt.yticks([0, 1], ["Unvoiced/Silence", "Voiced"])
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()