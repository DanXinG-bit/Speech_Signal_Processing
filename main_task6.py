import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 第一部分：自研算法工具库
# ==========================================

def load_wav(filename):
    """读取标准 PCM WAV"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        data = np.frombuffer(wf.readframes(-1), dtype=np.int16)
        return fs, data.astype(np.float32) / 32768.0

def durbin_step(data, p=12):
    """手动实现 Durbin 算法求 AR 系数"""
    N = len(data)
    R = np.zeros(p + 1)
    
    # 1. 计算自相关
    for k in range(p + 1):
        R[k] = np.sum(data[:N-k] * data[k:])
    
    if R[0] < 1e-12:  # 修正：检查R[0]而不是R
        return None
    
    # 2. Durbin递推
    E = R[0]
    a = np.zeros((p + 1, p + 1))
    
    for i in range(1, p + 1):
        # 计算反射系数 ki
        sum_term = 0.0
        for j in range(1, i):
            sum_term += a[i-1, j] * R[i-j]
        
        ki = (R[i] - sum_term) / (E + 1e-12)
        ki = np.clip(ki, -0.999, 0.999) 
        a[i, i] = ki
        
        # 更新系数
        for j in range(1, i):
            a[i, j] = a[i-1, j] - ki * a[i-1, i-j]
        
        # 更新误差能量
        E = (1 - ki**2) * E
    
    return a[p, 1:]

def ar_to_reflection(ar_coeffs):
    """AR 系数递推求反射系数 ki"""
    p = len(ar_coeffs)
    a = np.zeros((p + 1, p + 1))
    a[p, 1:] = ar_coeffs
    k = np.zeros(p)
    
    for i in range(p, 0, -1):
        ki = np.clip(a[i, i], -0.999, 0.999)
        k[i-1] = ki
        
        if i > 1:
            denom = 1 - ki**2 + 1e-12
            for j in range(1, i):
                a[i-1, j] = (a[i, j] + ki * a[i, i-j]) / denom
    
    return k

def apply_hamming_window(frame):
    """应用汉明窗"""
    N = len(frame)
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    return frame * hamming

def calculate_frame_energy(frame):
    """计算帧能量"""
    return np.sum(frame**2)

def calculate_zero_crossing_rate(frame):
    """计算过零率"""
    return np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1]))) / 2

def lattice_synthesis_filter(excitation, k):
    """
    格型合成滤波器 - 修正版本
    根据语音信号处理原理实现
    """
    p = len(k)
    N = len(excitation)
    output = np.zeros(N)
    
    # 状态变量
    f = np.zeros(p + 1)  # 前向预测误差
    b = np.zeros(p + 1)  # 后向预测误差
    
    for n in range(N):
        # 输入信号
        f[0] = excitation[n]
        
        # 格型滤波器递归
        for i in range(1, p + 1):
            # 更新后向预测误差
            b[i] = b[i-1] - k[i-1] * f[i-1]
            # 更新前向预测误差
            f[i] = f[i-1] + k[i-1] * b[i-1]
        
        # 输出是最后一个前向预测误差
        output[n] = f[p]
    
    return output

# ==========================================
# 第二部分：编解码主逻辑
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
    
    # 初始化合成信号
    synthesized_signal = np.zeros(len(data))
    
    # 创建重叠相加的窗口
    window = np.hanning(frame_len)
    
    print(f"音频采样率: {fs} Hz | 预计处理帧数: {num_frames}")
    print("开始执行编解码过程...")
    
    # 预先计算全局统计信息用于清浊音判断
    frame_energies = []
    for i in range(num_frames):
        start = i * frame_shift
        frame = data[start:start+frame_len]
        frame_energies.append(calculate_frame_energy(frame))
    
    energy_threshold = np.percentile(frame_energies, 70)  # 使用70分位数作为阈值

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
        
        # 计算帧特征
        energy = calculate_frame_energy(windowed_frame)
        zcr = calculate_zero_crossing_rate(windowed_frame)
        
        if energy < 1e-6:  # 静音段处理
            continue 
        
        # 提取当前帧AR系数
        ar_coeffs = durbin_step(windowed_frame, p)
        if ar_coeffs is None:
            continue
        
        # 转换为反射系数
        k_current = ar_to_reflection(ar_coeffs)
        
        # 矢量量化：在码本中寻找最接近的特征索引
        distances = np.sum((codebook - k_current)**2, axis=1)
        idx_min = np.argmin(distances)
        k_quantized = codebook[idx_min]

        # --- 解码阶段 (Synthesis) ---
        # 清浊音判断
        # 浊音：高能量、低过零率；清音：低能量、高过零率
        is_voiced = (energy > energy_threshold) and (zcr < frame_len * 0.1)
        
        # 根据发声模型生成激励
        if is_voiced:
            # 浊音：周期脉冲序列
            # 简单基音周期估计（实际应从任务三获取）
            pitch_period = int(fs / 100)  # 假设基频100Hz
            excitation = np.zeros(frame_len)
            
            # 生成脉冲序列
            for pos in range(0, frame_len, pitch_period):
                if pos < frame_len:
                    excitation[pos] = 1.0
        else:
            # 清音：伪随机噪声
            excitation = np.random.randn(frame_len)
        
        # 增益平衡：保持合成帧与原始帧能量一致
        gain = np.sqrt(energy / (np.sum(excitation**2) + 1e-12))
        excitation *= gain
        
        # 通过格型合成滤波器恢复语音
        resyn_frame = lattice_synthesis_filter(excitation, k_quantized)
        
        # 应用窗口并重叠相加
        synthesized_signal[start:end] += resyn_frame * window

        # 进度显示
        if i % 50 == 0:
            print(f"处理进度: {i}/{num_frames} 帧")

    # ==========================================
    # 5. 结果展示
    # ==========================================
    print("语音重合成完成。")
    
    # 归一化合成信号
    synthesized_signal = synthesized_signal / np.max(np.abs(synthesized_signal)) * 0.8
    
    plt.figure(figsize=(12, 8))
    
    # 原始语音
    plt.subplot(3, 1, 1)
    time_axis = np.arange(len(data)) / fs
    plt.plot(time_axis, data, 'b', alpha=0.7, label="Original Speech")
    plt.title(f"Original Speech: {os.path.basename(audio_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 合成语音
    plt.subplot(3, 1, 2)
    plt.plot(time_axis, synthesized_signal, 'r', alpha=0.7, label="Synthesized Speech")
    plt.title("Task 6: Synthesized Speech (Vocoder Model)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 对比图（重叠显示）
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, data, 'b', alpha=0.5, label="Original")
    plt.plot(time_axis, synthesized_signal, 'r', alpha=0.5, label="Synthesized")
    plt.title("Original vs Synthesized Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存合成结果
    # 获取原始文件名（不含扩展名）
    original_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = f"{original_name}_after.wav"
    
    try:
        # 转换为16位整数
        synthesized_int16 = (synthesized_signal * 32767).astype(np.int16)
        
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16位 = 2字节
            wf.setframerate(fs)
            wf.writeframes(synthesized_int16.tobytes())
        
        print(f"\n合成语音已保存为: {output_file}")
        print(f"文件大小: {os.path.getsize(output_file)} bytes")
        
    except Exception as e:
        print(f"保存文件失败: {e}")

if __name__ == "__main__":
    main()