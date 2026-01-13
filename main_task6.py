import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import time

# ==========================================
# 第一部分：核心算法工具库 (完全自研实现)
# ==========================================

def load_wav(filename):
    """读取标准 PCM WAV"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        data = np.frombuffer(wf.readframes(-1), dtype=np.int16)
        return fs, data.astype(np.float32) / 32768.0

def apply_hamming_window(frame):
    """应用汉明窗"""
    N = len(frame)
    n = np.arange(N)
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))
    return frame * hamming

def manual_correlation(x):
    """手动实现自相关计算"""
    N = len(x)
    R = np.zeros(N)
    for k in range(N):
        if k == 0:
            R[k] = np.sum(x * x)
        else:
            R[k] = np.sum(x[:N-k] * x[k:])
    return R

def center_clipping(frame, clipping_level=0.65):
    """
    中心削波预处理
    幅度削波比例: clipping_level (0-1)
    """
    max_amp = np.max(np.abs(frame))
    threshold = clipping_level * max_amp
    
    clipped = frame.copy()
    # 手动实现削波
    for i in range(len(clipped)):
        if clipped[i] > threshold:
            clipped[i] = clipped[i] - threshold
        elif clipped[i] < -threshold:
            clipped[i] = clipped[i] + threshold
        else:
            clipped[i] = 0.0
    return clipped

def pitch_detection(frame, fs):
    """
    任务三算法：基于自相关的基音检测
    包含中心削波预处理
    """
    # 1. 应用汉明窗
    windowed = apply_hamming_window(frame)
    
    # 2. 中心削波预处理 (去除共振峰干扰)
    clipped = center_clipping(windowed, clipping_level=0.65)
    
    # 3. 手动计算自相关函数
    autocorr = manual_correlation(clipped)
    
    # 4. 归一化自相关（最大值为1）
    if autocorr[0] > 0:
        autocorr_norm = autocorr / autocorr[0]
    else:
        autocorr_norm = autocorr
    
    # 5. 在有效基频范围内寻找峰值
    # 基频范围: 60Hz-500Hz
    min_period = int(fs / 500)  # 500Hz对应的最小周期
    max_period = int(fs / 60)   # 60Hz对应的最大周期
    
    # 确保不超过帧长的一半
    max_period = min(max_period, len(frame) // 2)
    
    if max_period <= min_period:
        return 0, 0.0, False  # 无效范围
    
    # 在有效范围内寻找最大峰值
    search_range = autocorr_norm[min_period:max_period+1]
    if len(search_range) == 0:
        return 0, 0.0, False
    
    max_peak_idx = np.argmax(search_range)
    max_peak_value = search_range[max_peak_idx]
    pitch_period = max_peak_idx + min_period
    
    # 6. V/U判别
    # 基于能量和自相关峰值
    frame_energy = np.sum(frame * frame)
    
    # 自适应阈值
    energy_threshold = 0.001  # 能量阈值
    correlation_threshold = 0.3  # 自相关峰值阈值
    
    is_voiced = (frame_energy > energy_threshold) and (max_peak_value > correlation_threshold)
    
    return pitch_period, max_peak_value, is_voiced

def durbin_algorithm(R, p):
    """
    手动实现Levinson-Durbin递推算法
    """
    E = np.zeros(p + 1)
    a = np.zeros((p + 1, p + 1))
    
    # 初始化
    E[0] = R[0]
    
    for i in range(1, p + 1):
        # 计算反射系数 k_i
        sum_val = 0.0
        for j in range(1, i):
            sum_val += a[i-1][j] * R[i-j]
        
        ki = (R[i] - sum_val) / (E[i-1] + 1e-12)
        
        # 手动截断反射系数
        if ki > 0.999:
            ki = 0.999
        elif ki < -0.999:
            ki = -0.999
        
        a[i][i] = ki
        
        # 更新系数
        for j in range(1, i):
            a[i][j] = a[i-1][j] - ki * a[i-1][i-j]
        
        # 更新误差能量
        E[i] = (1 - ki * ki) * E[i-1]
    
    return a[p][1:]

def ar_to_reflection_manual(ar_coeffs):
    """
    AR系数递推求反射系数
    """
    p = len(ar_coeffs)
    a = np.zeros((p + 1, p + 1))
    a[p, 1:] = ar_coeffs
    k = np.zeros(p)
    
    for i in range(p, 0, -1):
        ki = a[i, i]
        
        # 手动截断
        if ki > 0.999:
            ki = 0.999
        elif ki < -0.999:
            ki = -0.999
        
        k[i-1] = ki
        
        if i > 1:
            denom = 1 - ki * ki + 1e-12
            for j in range(1, i):
                a[i-1, j] = (a[i, j] + ki * a[i, i-j]) / denom
    
    return k

class LatticeSynthesizer:
    """
    带记忆的格型合成滤波器
    """
    def __init__(self, p=12):
        self.p = p
        self.reset_state()
    
    def reset_state(self):
        """重置滤波器状态"""
        self.f = np.zeros(self.p + 1)  # 前向预测误差
        self.b = np.zeros(self.p + 1)  # 后向预测误差
    
    def synthesize(self, excitation, k_coeffs):
        """
        合成一帧语音
        k_coeffs: 反射系数数组
        """
        frame_len = len(excitation)
        output = np.zeros(frame_len)
        
        for n in range(frame_len):
            # 输入激励信号
            self.f[0] = excitation[n]
            
            # 格型滤波器递归
            for i in range(1, self.p + 1):
                # 更新后向预测误差
                self.b[i] = self.b[i-1] - k_coeffs[i-1] * self.f[i-1]
                # 更新前向预测误差
                self.f[i] = self.f[i-1] + k_coeffs[i-1] * self.b[i-1]
            
            # 输出是最后一个前向预测误差
            output[n] = self.f[self.p]
        
        return output

def generate_excitation(frame_len, is_voiced, pitch_period, fs):
    """
    根据V/U判别生成激励源
    is_voiced: True/False
    pitch_period: 基音周期（采样点数）
    """
    excitation = np.zeros(frame_len)
    
    if is_voiced and pitch_period > 0:
        # 浊音：周期脉冲序列
        # 在脉冲位置放置汉明窗脉冲，避免冲击响应
        pulse_width = min(5, pitch_period // 4)
        pulse = np.hanning(pulse_width * 2 + 1)[:pulse_width]
        
        for pos in range(0, frame_len, pitch_period):
            if pos + pulse_width <= frame_len:
                excitation[pos:pos+pulse_width] = pulse
            elif pos < frame_len:
                remaining = frame_len - pos
                excitation[pos:pos+remaining] = pulse[:remaining]
    else:
        # 清音：白噪声
        excitation = np.random.randn(frame_len)
    
    return excitation

# ==========================================
# 第二部分：主处理逻辑
# ==========================================

def main():
    # ==========================================
    # 1. 初始化窗口并选择语音文件
    # ==========================================
    root = tk.Tk()
    root.withdraw()
    
    print("=" * 70)
    print("语音信号处理 - 任务六：基于码本与发声模型的语音重合成")
    print("=" * 70)
    
    print("正在选择待合成的原始音频文件...")
    audio_file = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="任务六：选择要重合成的音频",
        filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
    )
    if not audio_file:
        print("未选择音频文件，程序退出。")
        return

    # ==========================================
    # 2. 选择码本文件
    # ==========================================
    print("\n正在选择任务五生成的码本文件 (.npy)...")
    cb_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="任务六：选择矢量量化码本",
        filetypes=(("Numpy files", "*.npy"), ("All files", "*.*"))
    )
    
    if not cb_path:
        print("未选择码本文件，程序退出。")
        return
        
    try:
        codebook = np.load(cb_path)
        print(f"码本加载成功: {os.path.basename(cb_path)} | 形状: {codebook.shape}")
    except Exception as e:
        print(f"码本加载失败: {e}")
        return

    # ==========================================
    # 3. 编解码分析参数准备
    # ==========================================
    fs, data = load_wav(audio_file)
    p = 12  # 线性预测阶数
    frame_len = int(0.025 * fs)   # 25ms 帧长
    frame_shift = int(0.01 * fs)  # 10ms 帧移
    num_frames = (len(data) - frame_len) // frame_shift
    
    # 初始化合成信号和格型合成器
    synthesized_signal = np.zeros(len(data))
    synthesizer = LatticeSynthesizer(p)
    
    # 创建重叠相加的窗口
    window = np.hanning(frame_len)
    
    print(f"音频采样率: {fs} Hz")
    print(f"帧长: {frame_len} 采样点 ({frame_len/fs*1000:.1f} ms)")
    print(f"预计处理帧数: {num_frames}")
    print("开始执行编解码过程...")
    
    # 统计信息收集
    voiced_count = 0
    pitch_periods = []
    correlation_peaks = []

    # ==========================================
    # 4. 核心处理循环 (分析与重合成)
    # ==========================================
    for i in range(num_frames):
        start = i * frame_shift
        end = start + frame_len
        frame = data[start:end]
        
        # --- 编码阶段 (Analysis) ---
        # 应用汉明窗
        windowed_frame = apply_hamming_window(frame)
        
        # 计算帧能量
        energy = np.sum(windowed_frame * windowed_frame)
        
        if energy < 1e-6:  # 静音段处理
            # 生成低能量噪声作为激励
            excitation = np.random.randn(frame_len) * 0.01
            # 使用中性反射系数（全零）
            k_quantized = np.zeros(p)
        else:
            # --- 基音检测和V/U判别（任务三算法）---
            pitch_period, corr_peak, is_voiced = pitch_detection(windowed_frame, fs)
            
            # 统计信息
            pitch_periods.append(pitch_period)
            correlation_peaks.append(corr_peak)
            if is_voiced:
                voiced_count += 1
            
            # --- LPC分析 ---
            # 计算自相关函数（用于LPC）
            R = np.zeros(p + 1)
            for k in range(p + 1):
                R[k] = np.sum(windowed_frame[:frame_len-k] * windowed_frame[k:])
            
            # 使用Durbin算法计算AR系数
            ar_coeffs = durbin_algorithm(R, p)
            
            # 转换为反射系数
            k_current = ar_to_reflection_manual(ar_coeffs)
            
            # --- 矢量量化 ---
            # 在码本中寻找最接近的特征
            distances = np.zeros(len(codebook))
            for j in range(len(codebook)):
                distances[j] = np.sum((codebook[j] - k_current) * (codebook[j] - k_current))
            
            idx_min = np.argmin(distances)
            k_quantized = codebook[idx_min]
            
            # --- 激励生成 ---
            excitation = generate_excitation(frame_len, is_voiced, pitch_period, fs)
            
            # 增益平衡
            excitation_energy = np.sum(excitation * excitation) + 1e-12
            gain = np.sqrt(energy / excitation_energy)
            excitation = excitation * gain
        
        # --- 解码阶段 (Synthesis) ---
        # 通过格型合成滤波器恢复语音
        resyn_frame = synthesizer.synthesize(excitation, k_quantized)
        
        # 应用窗口并重叠相加
        synthesized_signal[start:end] += resyn_frame * window
        
        # 进度显示
        if i % 100 == 0:
            print(f"处理进度: {i}/{num_frames} 帧")
            if i > 0 and len(pitch_periods) > 0:
                valid_pitches = [p for p in pitch_periods if p > 0]
                if valid_pitches:
                    avg_pitch = np.mean(valid_pitches)
                    print(f"  平均基音周期: {avg_pitch:.1f} 采样点")

    # ==========================================
    # 5. 后处理和统计信息
    # ==========================================
    print("\n=== 处理统计信息 ===")
    print(f"总处理帧数: {num_frames}")
    print(f"浊音帧数: {voiced_count} ({voiced_count/max(1,num_frames)*100:.1f}%)")
    
    if len(pitch_periods) > 0:
        valid_pitches = [p for p in pitch_periods if p > 0]
        if valid_pitches:
            avg_pitch = np.mean(valid_pitches)
            print(f"有效基音周期范围: {min(valid_pitches)}-{max(valid_pitches)} 采样点")
            print(f"平均基音周期: {avg_pitch:.1f} 采样点 ({fs/avg_pitch:.1f} Hz)")
    
    # 归一化合成信号
    max_amp = np.max(np.abs(synthesized_signal))
    if max_amp > 0:
        synthesized_signal = synthesized_signal / max_amp * 0.8
    
    # ==========================================
    # 6. 保存合成结果（在显示图表之前）
    # ==========================================
    # 获取原始文件名（不含扩展名）
    original_name = os.path.splitext(os.path.basename(audio_file))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = f"{original_name}_resynthesized_{timestamp}.wav"
    
    try:
        # 转换为16位整数
        synthesized_int16 = (synthesized_signal * 32767).astype(np.int16)
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16位 = 2字节
            wf.setframerate(fs)
            wf.writeframes(synthesized_int16.tobytes())
        
        file_size = os.path.getsize(output_file)
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"\n合成语音已保存为: {output_file}")
        print(f"文件大小: {file_size:,} bytes ({file_size_mb:.2f} MB)")
        print(f"保存位置: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"保存文件失败: {e}")
        output_file = None
    
    # ==========================================
    # 7. 结果可视化
    # ==========================================
    plt.figure(figsize=(15, 12))
    
    # 1. 原始vs合成语音对比
    time_axis = np.arange(len(data)) / fs
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, data, 'b', alpha=0.7, label="Original", linewidth=1)
    plt.plot(time_axis, synthesized_signal, 'r', alpha=0.7, label="Synthesized", linewidth=1)
    plt.title("Original vs Synthesized Speech Waveform", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 添加时间标记
    if len(time_axis) > 0:
        plt.xlim(0, time_axis[-1])
    
    # 2. 短时能量对比
    plt.subplot(3, 1, 2)
    
    # 计算短时能量
    energy_original = []
    energy_synthesized = []
    window_size = int(0.02 * fs)  # 20ms窗口
    hop_size = window_size // 2
    
    for i in range(0, len(data) - window_size, hop_size):
        energy_original.append(np.sum(data[i:i+window_size]**2))
        energy_synthesized.append(np.sum(synthesized_signal[i:i+window_size]**2))
    
    energy_time = np.arange(len(energy_original)) * hop_size / fs
    plt.plot(energy_time, energy_original, 'b', alpha=0.6, label="Original Energy", linewidth=1)
    plt.plot(energy_time, energy_synthesized, 'r', alpha=0.6, label="Synthesized Energy", linewidth=1)
    plt.title("Short-term Energy Comparison", fontweight='bold')
    plt.xlabel("Time (s)")
    plt.ylabel("Energy")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 3. 频谱对比（使用短时傅里叶变换）
    plt.subplot(3, 1, 3)
    
    # 选择中间2秒的片段进行频谱分析
    mid_point = len(data) // 2
    analysis_duration = int(2 * fs)  # 2秒
    start_idx = max(0, mid_point - analysis_duration // 2)
    end_idx = min(len(data), start_idx + analysis_duration)
    
    # 原始音频频谱
    original_segment = data[start_idx:end_idx]
    original_fft = np.abs(np.fft.rfft(original_segment))
    original_freqs = np.fft.rfftfreq(len(original_segment), 1/fs)
    
    # 合成音频频谱
    synthesized_segment = synthesized_signal[start_idx:end_idx]
    synthesized_fft = np.abs(np.fft.rfft(synthesized_segment))
    
    # 绘制频谱对比
    plt.plot(original_freqs, 20*np.log10(original_fft+1e-12), 'b', alpha=0.6, label="Original", linewidth=1)
    plt.plot(original_freqs, 20*np.log10(synthesized_fft+1e-12), 'r', alpha=0.6, label="Synthesized", linewidth=1)
    plt.title("Spectrum Comparison (2-second segment)", fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, 4000)  # 只显示0-4kHz
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 添加总标题
    plt.suptitle(f'Speech Resynthesis Analysis: {original_name}\nTotal Frames: {num_frames}, Voiced: {voiced_count} ({voiced_count/max(1,num_frames)*100:.1f}%)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 显示弹窗提示（在显示图表之前）
    if output_file:
        abs_path = os.path.abspath(output_file)
        messagebox.showinfo(
            "处理完成",
            f"语音重合成已完成！\n\n"
            f"生成文件: {os.path.basename(output_file)}\n"
            f"保存位置: {abs_path}\n\n也就是在代码的目录文件\n\n"
            f"总帧数: {num_frames}\n"
            f"浊音帧: {voiced_count} ({voiced_count/max(1,num_frames)*100:.1f}%)\n\n"
            f"请点击确认来查看对比图表"
        )
    
    print("\n" + "=" * 70)
    print("正在显示对比图表...")
    print("=" * 70)
    
    # 显示图表
    plt.show()
    
    # 程序结束提示
    print("\n" + "=" * 70)
    print("程序执行完毕！")
    print(f"原始音频: {audio_file}")
    if output_file:
        print(f"合成音频: {output_file}")
        print(f"工作目录: {os.getcwd()}")
    print("=" * 70)

if __name__ == "__main__":
    main()