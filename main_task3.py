import wave
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os
import time

# ==========================================
# 第一部分：合规算法库 (手动实现核心算法)
# ==========================================

def load_wav(filename):
    """读取WAV文件 - 合规"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        return fs, wave_data.astype(np.float32) / 32768.0

def center_clipping_filter(data, clip_ratio=0.3):
    """
    手动实现时域中心削波滤波器
    符合要求：不使用高级函数包
    """
    max_amplitude = np.max(np.abs(data))
    clip_level = clip_ratio * max_amplitude
    
    # 手动实现削波
    clipped_data = np.zeros_like(data)
    for i in range(len(data)):
        if data[i] > clip_level:
            clipped_data[i] = data[i] - clip_level
        elif data[i] < -clip_level:
            clipped_data[i] = data[i] + clip_level
        else:
            clipped_data[i] = 0
    
    return clipped_data

def calc_autocorr_manual(frame, k_min, k_max):
    """
    手动实现自相关计算
    使用基本循环和求和
    公式：R(k) = Σ x(n)x(n+k), n=0 to N-k-1
    """
    N = len(frame)
    R_values = np.zeros(k_max - k_min + 1)
    
    # 只计算指定范围的k值
    for idx, k in enumerate(range(k_min, k_max + 1)):
        sum_val = 0.0
        sum_val = np.sum(frame[:N-k] * frame[k:])
        R_values[idx] = sum_val
    
    return R_values

# ==========================================
# 第二部分：基音检测主逻辑
# ==========================================

def pitch_detection_compliant(data, fs):
    """合规版基音检测算法"""
    print("Preprocessing (center clipping)...")
    start_time = time.time()
    
    # 1. 预处理：使用中心削波（课程推荐方法）
    processed_data = center_clipping_filter(data)
    
    # 2. 分帧参数
    frame_len = int(0.030 * fs)      # 30ms
    frame_shift = int(0.010 * fs)    # 10ms
    num_frames = (len(processed_data) - frame_len) // frame_shift
    
    print(f"Total frames: {num_frames}")
    print(f"Frame length: {frame_len} samples ({frame_len/fs*1000:.1f}ms)")
    print(f"Frame shift: {frame_shift} samples ({frame_shift/fs*1000:.1f}ms)")
    
    # 3. 基音搜索范围 (60Hz-500Hz，转换为采样点数)
    pitch_min_hz = 60   # 最低基音频率
    pitch_max_hz = 500  # 最高基音频率
    k_min = int(fs / pitch_max_hz)  # 最小周期点数
    k_max = int(fs / pitch_min_hz)  # 最大周期点数
    
    print(f"Pitch period search range: {k_min}-{k_max} samples")
    print(f"Corresponding frequency range: {pitch_min_hz}-{pitch_max_hz} Hz")
    
    # 4. 结果数组
    pitch_periods = np.zeros(num_frames)  # 基音周期（采样点数）
    voiced_decisions = np.zeros(num_frames)  # 1=浊音，0=清音/静音
    
    # 5. 逐帧处理（允许使用循环，但内部计算要合规）
    for i in range(num_frames):
        start_idx = i * frame_shift
        frame = processed_data[start_idx:start_idx + frame_len]
        
        # 计算帧能量R(0)
        R0 = np.sum(frame * frame)
        
        # 能量阈值：前20帧的中位数能量
        if i < 20:
            energy_threshold = np.median([np.sum(
                processed_data[j*frame_shift:j*frame_shift+frame_len] ** 2
            ) for j in range(min(20, num_frames))]) * 0.1
        energy_threshold = max(energy_threshold, 1e-10)  # 防止过小
        
        # 跳过静音帧
        if R0 < energy_threshold:
            pitch_periods[i] = 0
            voiced_decisions[i] = 0
            continue
        
        # 合规自相关计算
        R = calc_autocorr_manual(frame, k_min, k_max)
        
        # 寻找峰值（手动实现）
        # 找到第一个过零后的最大值
        max_value = R[0]
        max_local_idx = 0
        
        # 搜索局部最大值（从第2个点开始）
        for j in range(1, len(R) - 1):
            if R[j] > R[j-1] and R[j] > R[j+1]:
                if R[j] > max_value:
                    max_value = R[j]
                    max_local_idx = j
        
        # 计算归一化自相关峰值
        if R0 > 1e-10:  # 防止除以零
            normalized_peak = max_value / R0
        else:
            normalized_peak = 0
        
        # 清浊音判别（符合课程PPT条件）
        # 条件：归一化峰值 > 0.3 且能量足够
        voicing_threshold = 0.3
        
        if normalized_peak > voicing_threshold and R0 > energy_threshold:
            pitch_period = max_local_idx + k_min  # 转换为绝对采样点数
            pitch_periods[i] = pitch_period
            voiced_decisions[i] = 1
        else:
            pitch_periods[i] = 0
            voiced_decisions[i] = 0
    
    total_time = time.time() - start_time
    voiced_count = np.sum(voiced_decisions)
    
    print(f"\nPitch detection completed in: {total_time:.2f} seconds")
    print(f"Voiced frames: {voiced_count}/{num_frames} ({voiced_count/num_frames*100:.1f}%)")
    
    return pitch_periods, voiced_decisions, frame_shift

# ==========================================
# 第三部分：主程序（严格符合任务要求）
# ==========================================

def main():
    """主函数 - 严格按任务要求实现"""
    print("=" * 70)
    print("Speech Signal Processing - Task 3: Pitch Period Detection")
    print("Method: Autocorrelation + Center Clipping")
    print("=" * 70)
    
    # 1. 文件选择（兼容IDLE环境）
    try:
        root = tk.Tk()
        root.withdraw()
        current_dir = os.getcwd()
        
        audio_file = filedialog.askopenfilename(
            initialdir=current_dir,
            title="Select speech file for pitch detection",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
        )
        
        if not audio_file:
            print("No file selected, exiting...")
            return
    except Exception as e:
        print(f"File selection failed: {e}")
        print("Please place audio file in current directory and enter filename:")
        audio_file = input("Filename: ").strip()
        if not os.path.exists(audio_file):
            print("File not found, exiting...")
            return
    
    print(f"\nProcessing file: {os.path.basename(audio_file)}")
    
    # 2. 加载音频
    try:
        fs, data = load_wav(audio_file)
        duration = len(data) / fs
        print(f"Sample rate: {fs} Hz")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Total samples: {len(data)}")
    except Exception as e:
        print(f"Failed to load audio: {e}")
        return
    
    # 3. 基音检测
    print("\n" + "-" * 70)
    print("Starting pitch period detection...")
    
    pitch_periods, voiced, frame_shift = pitch_detection_compliant(data, fs)
    
    # 4. 计算统计信息
    valid_pitches = pitch_periods[voiced == 1]
    if len(valid_pitches) > 0:
        mean_period = np.mean(valid_pitches)
        std_period = np.std(valid_pitches)
        mean_freq = fs / mean_period if mean_period > 0 else 0
        
        print(f"\nPitch period statistics (in samples):")
        print(f"  Mean: {mean_period:.1f} samples")
        print(f"  Std: {std_period:.1f} samples")
        print(f"  Range: {np.min(valid_pitches):.0f} - {np.max(valid_pitches):.0f} samples")
        print(f"  Corresponding mean frequency: {mean_freq:.1f} Hz")
    
    print("-" * 70)
    
    # 5. 绘图 - 严格按任务要求
    print("\nGenerating results plots...")
    
    # 计算关键时间参数，统一横坐标范围
    total_duration = len(data) / fs  # 音频总时长
    frame_times = np.arange(len(pitch_periods)) * (frame_shift / fs)  # 帧时间
    frame_duration = len(pitch_periods) * (frame_shift / fs)  # 分析总时长
    
    # 确定横坐标的统一范围
    x_min = 0
    x_max = max(total_duration, frame_times[-1] if len(frame_times) > 0 else total_duration)
    
    plt.figure(figsize=(12, 10))
    
    # 图表1：原始语音波形（参考图）
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(data)) / fs
    plt.plot(time_axis, data, 'gray', linewidth=0.5, alpha=0.7)
    plt.title(f"Original Speech Waveform - {os.path.basename(audio_file)}")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)  # 统一横坐标范围
    plt.xticks(np.arange(0, x_max + 0.1, max(0.1, x_max/10)))  # 统一刻度
    
    # 图表2：每一帧的基音周期（采样点数）- 任务要求
    plt.subplot(3, 1, 2)
    
    # 绘制浊音帧的基音周期
    voiced_indices = voiced == 1
    plt.plot(frame_times[voiced_indices], pitch_periods[voiced_indices], 
             'ro', markersize=3, alpha=0.7, label='Pitch Period (Voiced)')
    
    # 绘制清音/静音帧（周期为0）
    unvoiced_indices = voiced == 0
    plt.plot(frame_times[unvoiced_indices], 
             np.zeros(np.sum(unvoiced_indices)), 
             'kx', markersize=2, alpha=0.3, label='Unvoiced/Silence')
    
    plt.title("Task 3: Pitch Period per Frame (in Samples)")
    plt.ylabel("Pitch Period (Samples)")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)
    
    # 添加基音周期范围参考线
    plt.axhline(y=fs/500, color='g', linestyle='--', alpha=0.5, 
                linewidth=1, label=f'500Hz ({fs/500:.0f} samples)')
    plt.axhline(y=fs/60, color='b', linestyle='--', alpha=0.5, 
                linewidth=1, label=f'60Hz ({fs/60:.0f} samples)')
    
    # 统一横坐标范围
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(0, x_max + 0.1, max(0.1, x_max/10)))  # 统一刻度
    
    # 添加图例（放在右下角）
    plt.legend(loc='upper right', fontsize=8)
    
    # 图表3：清浊音判别结果 - 任务要求
    plt.subplot(3, 1, 3)
    
    # 使用阶梯图显示清浊音判别
    plt.step(frame_times, voiced, where='post', 
             color='blue', linewidth=1.5, label='V/U Decision')
    
    # 填充浊音区域
    for i in range(len(voiced) - 1):
        if voiced[i] == 1:
            plt.fill_between([frame_times[i], frame_times[i+1]], 
                             0, 1, color='green', alpha=0.3)
    
    plt.title("Task 3: Voiced/Unvoiced Detection Result")
    plt.xlabel("Time (s)")
    plt.ylabel("Decision")
    plt.yticks([0, 1], ['Unvoiced/Silence', 'Voiced'])
    plt.ylim(-0.1, 1.1)
    plt.grid(True, alpha=0.3)
    
    # 统一横坐标范围
    plt.xlim(x_min, x_max)
    plt.xticks(np.arange(0, x_max + 0.1, max(0.1, x_max/10)))  # 统一刻度
    
    # 添加浊音比例标注
    voiced_percent = np.sum(voiced) / len(voiced) * 100
    plt.text(0.02, 0.95, f'Voiced Frames: {voiced_percent:.1f}%',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    # 确保子图之间紧密排列，消除空白
    plt.tight_layout()
    
    print("\n" + "=" * 70)
    print("Task 3 Completed!")
    print("Note: Chart 1 is reference waveform, Charts 2-3 are required outputs")
    print("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()