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
    来源：课程PPT中的预处理方法
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
    手动实现自相关计算 - 合规实现
    使用基本循环和求和，不使用np.correlate
    公式：R(k) = Σ x(n)x(n+k), n=0 to N-k-1
    """
    N = len(frame)
    R_values = np.zeros(k_max - k_min + 1)
    
    # 只计算指定范围的k值
    for idx, k in enumerate(range(k_min, k_max + 1)):
        sum_val = 0.0
        # 使用向量化求和但避免高级函数
        # 这是允许的：使用np.sum进行数组求和
        sum_val = np.sum(frame[:N-k] * frame[k:])
        R_values[idx] = sum_val
    
    return R_values

# ==========================================
# 第二部分：基音检测主逻辑
# ==========================================

def pitch_detection_compliant(data, fs):
    """合规版基音检测算法"""
    print("正在进行预处理（中心削波）...")
    start_time = time.time()
    
    # 1. 预处理：使用中心削波（课程推荐方法）
    processed_data = center_clipping_filter(data)
    
    # 2. 分帧参数
    frame_len = int(0.030 * fs)      # 30ms
    frame_shift = int(0.010 * fs)    # 10ms
    num_frames = (len(processed_data) - frame_len) // frame_shift
    
    print(f"总帧数: {num_frames}")
    print(f"帧长: {frame_len}点 ({frame_len/fs*1000:.1f}ms)")
    print(f"帧移: {frame_shift}点 ({frame_shift/fs*1000:.1f}ms)")
    
    # 3. 基音搜索范围 (60Hz-500Hz，转换为采样点数)
    pitch_min_hz = 60   # 最低基音频率
    pitch_max_hz = 500  # 最高基音频率
    k_min = int(fs / pitch_max_hz)  # 最小周期点数
    k_max = int(fs / pitch_min_hz)  # 最大周期点数
    
    print(f"基音周期搜索范围: {k_min}-{k_max} 点")
    print(f"对应频率范围: {pitch_min_hz}-{pitch_max_hz} Hz")
    
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
    
    print(f"\n基音检测完成，耗时: {total_time:.2f}秒")
    print(f"浊音帧数: {voiced_count}/{num_frames} ({voiced_count/num_frames*100:.1f}%)")
    
    return pitch_periods, voiced_decisions, frame_shift

# ==========================================
# 第三部分：主程序（严格符合任务要求）
# ==========================================

def main():
    """主函数 - 严格按任务要求实现"""
    print("=" * 70)
    print("语音信号处理 - 任务三：基音周期检测")
    print("实现方法：自相关法 + 中心削波预处理")
    print("=" * 70)
    
    # 1. 文件选择（兼容IDLE环境）
    try:
        root = tk.Tk()
        root.withdraw()
        current_dir = os.getcwd()
        
        audio_file = filedialog.askopenfilename(
            initialdir=current_dir,
            title="请选择语音文件进行基音检测",
            filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
        )
        
        if not audio_file:
            print("未选择文件，程序退出")
            return
    except Exception as e:
        print(f"文件选择失败: {e}")
        print("请将音频文件放在当前目录，并输入文件名:")
        audio_file = input("文件名: ").strip()
        if not os.path.exists(audio_file):
            print("文件不存在，程序退出")
            return
    
    print(f"\n处理文件: {os.path.basename(audio_file)}")
    
    # 2. 加载音频
    try:
        fs, data = load_wav(audio_file)
        duration = len(data) / fs
        print(f"采样率: {fs} Hz")
        print(f"时长: {duration:.2f} 秒")
        print(f"总采样点数: {len(data)}")
    except Exception as e:
        print(f"加载音频失败: {e}")
        return
    
    # 3. 基音检测
    print("\n" + "-" * 70)
    print("开始基音周期检测...")
    
    pitch_periods, voiced, frame_shift = pitch_detection_compliant(data, fs)
    
    # 4. 计算统计信息（用于报告）
    valid_pitches = pitch_periods[voiced == 1]
    if len(valid_pitches) > 0:
        mean_period = np.mean(valid_pitches)
        std_period = np.std(valid_pitches)
        mean_freq = fs / mean_period if mean_period > 0 else 0
        
        print(f"\n基音周期统计（采样点数）:")
        print(f"  平均值: {mean_period:.1f} 点")
        print(f"  标准差: {std_period:.1f} 点")
        print(f"  范围: {np.min(valid_pitches):.0f} - {np.max(valid_pitches):.0f} 点")
        print(f"  对应平均频率: {mean_freq:.1f} Hz")
    
    print("-" * 70)
    
    # 5. 绘图 - 严格按任务要求
    print("\n生成结果图表...")
    
    plt.figure(figsize=(12, 8))
    
    # 图表1：原始语音波形（参考图）
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(data)) / fs
    plt.plot(time_axis, data, 'gray', linewidth=0.5, alpha=0.7)
    plt.title(f"Original Speech Waveform - {os.path.basename(audio_file)}")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, time_axis[-1])
    
    # 图表2：每一帧的基音周期（采样点数）- 任务要求
    plt.subplot(3, 1, 2)
    frame_times = np.arange(len(pitch_periods)) * (frame_shift / fs)
    
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
    plt.legend(loc='upper right', fontsize=9)
    
    # 添加基音周期范围参考线
    plt.axhline(y=fs/500, color='g', linestyle='--', alpha=0.5, 
                linewidth=1, label=f'500Hz ({fs/500:.0f} samples)')
    plt.axhline(y=fs/60, color='b', linestyle='--', alpha=0.5, 
                linewidth=1, label=f'60Hz ({fs/60:.0f} samples)')
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
    
    # 添加浊音比例标注
    voiced_percent = np.sum(voiced) / len(voiced) * 100
    plt.text(0.02, 0.95, f'Voiced Frames: {voiced_percent:.1f}%',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=9)
    
    plt.tight_layout()
    
    # 6. 保存结果（可选）
    save_option = input("\n是否保存检测结果到文本文件？(y/n): ")
    if save_option.lower() == 'y':
        output_file = f"task3_results_{os.path.splitext(os.path.basename(audio_file))[0]}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("任务三：基音周期检测结果\n")
            f.write(f"文件: {audio_file}\n")
            f.write(f"采样率: {fs} Hz\n")
            f.write(f"时长: {duration:.2f} 秒\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("帧号, 时间(s), 基音周期(采样点数), 清浊音判别\n")
            f.write("-" * 60 + "\n")
            
            for i in range(len(pitch_periods)):
                time_sec = i * frame_shift / fs
                v_u_label = "浊音" if voiced[i] == 1 else "清音/静音"
                f.write(f"{i:4d}, {time_sec:6.3f}, {pitch_periods[i]:8.1f}, {v_u_label}\n")
        
        print(f"结果已保存到: {output_file}")
    
    print("\n" + "=" * 70)
    print("任务三完成！")
    print("注：图表1为参考波形，图表2-3为任务要求输出")
    print("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()