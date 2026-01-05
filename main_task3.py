import wave
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import time

# ==========================================
# 第一部分：优化算法库 (使用NumPy向量化)
# ==========================================

def load_wav(filename):
    """高效读取WAV文件"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        return fs, wave_data.astype(np.float32) / 32768.0

def simple_bandpass_filter_vectorized(data, fs, low=60, high=900):
    """
    向量化带通滤波器
    使用FFT频域置零法，但用向量化操作替代循环
    """
    n = len(data)
    # 计算FFT
    fft_data = np.fft.fft(data)
    
    # 向量化计算频率轴
    freqs = np.fft.fftfreq(n, d=1/fs)
    
    # 向量化频域置零
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    fft_data[~mask] = 0
    
    # 逆变换回时域
    return np.fft.ifft(fft_data).real

def calc_autocorr_vectorized(frame, k_min, k_max):
    """
    向量化自相关计算
    使用卷积或向量化操作加速计算
    """
    frame_len = len(frame)
    
    # 方法1：使用卷积计算自相关（最快）
    # R(k) = sum(frame[n] * frame[n+k]) = 卷积(frame, frame[::-1])
    full_corr = np.correlate(frame, frame, mode='full')
    
    # 提取需要的延迟范围
    # 自相关对称，取后半部分，索引从frame_len-1开始
    start_idx = frame_len - 1 + k_min
    end_idx = frame_len - 1 + k_max + 1
    R = full_corr[start_idx:end_idx]
    
    return R

def find_autocorr_peak(R):
    """
    向量化寻找自相关峰值
    返回：峰值位置（相对于k_min的偏移），峰值大小
    """
    # 找到最大值的位置
    max_idx = np.argmax(R)
    max_value = R[max_idx]
    return max_idx, max_value

# ==========================================
# 第二部分：高效基音检测
# ==========================================

def pitch_detection_optimized(data, fs):
    """优化版基音检测算法"""
    print("正在进行带通滤波...")
    start_time = time.time()
    
    # 1. 预处理：带通滤波 (60-900Hz)
    filtered_data = simple_bandpass_filter_vectorized(data, fs)
    
    # 2. 分帧参数
    frame_len = int(0.030 * fs)      # 30ms
    frame_shift = int(0.010 * fs)    # 10ms
    signal_len = len(filtered_data)
    num_frames = (signal_len - frame_len) // frame_shift
    
    print(f"总帧数: {num_frames}, 帧长: {frame_len}, 帧移: {frame_shift}")
    
    # 3. 设定基音搜索范围 (对应 60Hz - 500Hz)
    k_min = int(fs / 500)   # 对应500Hz
    k_max = int(fs / 60)    # 对应60Hz
    search_range = k_max - k_min + 1
    print(f"基音搜索范围: {k_min}-{k_max} ({search_range}点)")
    
    # 4. 预分配结果数组
    pitches = np.zeros(num_frames)
    v_u_decisions = np.zeros(num_frames)
    
    # 5. 向量化计算帧能量（一次性计算）
    # 创建帧矩阵
    indices = np.arange(frame_len).reshape(1, -1) + \
              np.arange(0, num_frames * frame_shift, frame_shift).reshape(-1, 1)
    frames = filtered_data[indices.astype(np.int32)]
    
    # 一次性计算所有帧的R(0)和能量
    R0_values = np.sum(frames * frames, axis=1)  # R(0) = 能量
    
    # 设置能量阈值（前10%帧的中位数作为参考）
    ref_frames = min(100, num_frames // 10)
    energy_threshold = np.median(R0_values[:ref_frames]) * 5
    
    print(f"能量阈值: {energy_threshold:.6f}")
    
    # 6. 逐帧处理自相关（仍需要循环，但内部计算向量化）
    frame_times = []
    for i in range(num_frames):
        frame = frames[i]
        energy = R0_values[i]
        
        # 能量检查：跳过静音帧
        if energy < energy_threshold:
            pitches[i] = 0
            v_u_decisions[i] = 0
            continue
        
        # 计算自相关（向量化）
        R = calc_autocorr_vectorized(frame, k_min, k_max)
        
        # 在搜索范围内找峰值（向量化）
        max_idx, max_value = find_autocorr_peak(R)
        
        # 清浊音判别
        # 条件：R(0) > 0.25 * Rmax 且 能量足够高
        if R0_values[i] > 0.25 * max_value:
            pitch_period = max_idx + k_min
            pitches[i] = pitch_period
            v_u_decisions[i] = 1  # 浊音
        else:
            pitches[i] = 0
            v_u_decisions[i] = 0  # 清音
    
    # 7. 后处理：简单的中值滤波平滑基音轨迹
    voiced_indices = v_u_decisions == 1
    if np.any(voiced_indices):
        # 对浊音帧的基音周期进行中值滤波（窗口大小3）
        smoothed_pitches = np.copy(pitches)
        for i in range(1, num_frames-1):
            if voiced_indices[i]:
                window = pitches[max(0, i-1):min(num_frames, i+2)]
                window = window[window > 0]  # 只考虑非零值
                if len(window) > 0:
                    smoothed_pitches[i] = np.median(window)
        pitches = smoothed_pitches
    
    total_time = time.time() - start_time
    print(f"基音检测完成，耗时: {total_time:.2f}秒")
    print(f"检测到 {np.sum(v_u_decisions)} 个浊音帧 ({np.sum(v_u_decisions)/num_frames*100:.1f}%)")
    
    return pitches, v_u_decisions

# ==========================================
# 第三部分：主程序
# ==========================================

def main():
    # 创建隐藏窗口
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
    
    print("=" * 60)
    print(f"文件: {os.path.basename(audio_file)}")
    
    # 加载音频
    try:
        fs, data = load_wav(audio_file)
        duration = len(data) / fs
        print(f"采样率: {fs} Hz, 时长: {duration:.2f}秒")
        print(f"数据点数: {len(data):,}")
    except Exception as e:
        print(f"加载音频失败: {e}")
        return
    
    # 进行基音检测
    print("-" * 60)
    pitches, v_u = pitch_detection_optimized(data, fs)
    
    # 将基音周期转换为频率（Hz）
    frequencies = np.zeros_like(pitches)
    valid_pitches = pitches > 0
    frequencies[valid_pitches] = fs / pitches[valid_pitches]
    
    # 计算统计信息
    voiced_frames = np.sum(v_u)
    if voiced_frames > 0:
        voiced_frequencies = frequencies[valid_pitches]
        mean_freq = np.mean(voiced_frequencies)
        std_freq = np.std(voiced_frequencies)
        min_freq = np.min(voiced_frequencies)
        max_freq = np.max(voiced_frequencies)
        
        print("\n基音频率统计:")
        print(f"  均值: {mean_freq:.1f} Hz")
        print(f"  标准差: {std_freq:.1f} Hz")
        print(f"  范围: {min_freq:.1f} - {max_freq:.1f} Hz")
    
    print("-" * 60)
    
    # 绘图展示
    plt.figure(figsize=(12, 9))
    
    # 图1: 原始波形
    ax1 = plt.subplot(4, 1, 1)
    time_axis = np.arange(len(data)) / fs
    ax1.plot(time_axis, data, color='silver', linewidth=0.5, alpha=0.8)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title(f"Waveform: {os.path.basename(audio_file)}", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, time_axis[-1])
    
    # 图2: 基音频率轨迹
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    frame_times = np.arange(len(pitches)) * 0.01  # 10ms帧移
    
    # 绘制浊音帧的频率
    voiced_mask = v_u == 1
    ax2.plot(frame_times[voiced_mask], frequencies[voiced_mask], 
             'r.', markersize=4, alpha=0.7, label='Pitch Frequency')
    
    # 可选：添加平滑曲线
    if np.sum(voiced_mask) > 10:
        # 使用移动平均平滑
        window_size = 5
        smoothed = np.convolve(frequencies[voiced_mask], 
                               np.ones(window_size)/window_size, 
                               mode='valid')
        smoothed_times = frame_times[voiced_mask][window_size-1:]
        ax2.plot(smoothed_times, smoothed, 'b-', linewidth=1.5, 
                 alpha=0.8, label='Smoothed')
    
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    ax2.set_title("Pitch Frequency Trajectory", fontsize=12, fontweight='bold')
    ax2.set_ylim(50, 500)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)
    
    # 图3: 清浊音判别
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.fill_between(frame_times, 0, v_u, 
                     where=(v_u == 1), 
                     color='green', alpha=0.3, label='Voiced')
    ax3.fill_between(frame_times, 0, v_u, 
                     where=(v_u == 0), 
                     color='gray', alpha=0.3, label='Unvoiced/Silence')
    ax3.step(frame_times, v_u, where='post', color='blue', linewidth=1)
    ax3.set_ylabel("V/U Decision", fontsize=10)
    ax3.set_title("Voiced/Unvoiced Detection", fontsize=12, fontweight='bold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Unvoiced', 'Voiced'])
    ax3.set_ylim(-0.1, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=9)
    
    # 图4: 基音周期直方图
    ax4 = plt.subplot(4, 1, 4)
    if np.sum(valid_pitches) > 0:
        pitch_periods = pitches[valid_pitches]
        hist, bins, _ = ax4.hist(pitch_periods, bins=30, 
                                 color='purple', alpha=0.7, 
                                 edgecolor='black')
        ax4.set_xlabel("Pitch Period (samples)", fontsize=10)
        ax4.set_ylabel("Count", fontsize=10)
        ax4.set_title("Pitch Period Distribution", fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 添加统计信息
        stats_text = f"Mean: {np.mean(pitch_periods):.1f} samples\n"
        stats_text += f"Std: {np.std(pitch_periods):.1f} samples"
        ax4.text(0.95, 0.95, stats_text,
                 transform=ax4.transAxes,
                 verticalalignment='top',
                 horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=9)
    else:
        ax4.text(0.5, 0.5, "No voiced frames detected",
                 horizontalalignment='center',
                 verticalalignment='center',
                 transform=ax4.transAxes,
                 fontsize=12)
        ax4.set_xlabel("Pitch Period (samples)", fontsize=10)
        ax4.set_ylabel("Count", fontsize=10)
        ax4.set_title("Pitch Period Distribution", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 保存结果选项
    save_option = input("\n是否保存结果到文件？ (y/n): ")
    if save_option.lower() == 'y':
        filename = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = f"{filename}_pitch_results.txt"
        
        with open(output_file, 'w') as f:
            f.write(f"File: {audio_file}\n")
            f.write(f"Sample Rate: {fs} Hz\n")
            f.write(f"Duration: {duration:.2f} s\n")
            f.write(f"Total Frames: {len(pitches)}\n")
            f.write(f"Voiced Frames: {voiced_frames}\n")
            f.write(f"Voiced Percentage: {voiced_frames/len(pitches)*100:.1f}%\n")
            
            if voiced_frames > 0:
                f.write(f"\nPitch Frequency Statistics:\n")
                f.write(f"  Mean: {mean_freq:.1f} Hz\n")
                f.write(f"  Std: {std_freq:.1f} Hz\n")
                f.write(f"  Min: {min_freq:.1f} Hz\n")
                f.write(f"  Max: {max_freq:.1f} Hz\n")
            
            f.write("\nFrame-by-Frame Results:\n")
            f.write("Frame, Time(s), Pitch(Hz), V/U\n")
            for i in range(len(pitches)):
                f.write(f"{i}, {i*0.01:.3f}, {frequencies[i]:.1f}, {int(v_u[i])}\n")
        
        print(f"结果已保存到: {output_file}")

if __name__ == "__main__":
    main()