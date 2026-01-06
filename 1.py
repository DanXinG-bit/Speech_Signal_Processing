import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 第一部分：自研核心算法工具库 (严格符合规范 [1-3])
# ==========================================

def load_wav(filename):
    """读取标准 PCM WAV [4]"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        data = np.frombuffer(str_data, dtype=np.int16)
        return fs, data.astype(np.float32) / 32768.0

def apply_hamming_window(frame):
    """汉明窗分析 [5, 6]"""
    N = len(frame)
    n = np.arange(N)
    return frame * (0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1)))

def center_clipping(frame, ratio=0.68):
    """中心削波预处理 [7, 8]"""
    max_amp = np.max(np.abs(frame))
    th = ratio * max_amp
    clipped = np.zeros_like(frame)
    for i in range(len(frame)):
        if frame[i] > th: 
            clipped[i] = frame[i] - th
        elif frame[i] < -th: 
            clipped[i] = frame[i] + th
        # 否则保持为0
    return clipped

def pitch_detection_task3(frame, fs):
    """复现任务三：自相关基音检测与V/U判别 [9-11]"""
    if len(frame) == 0:
        return 0, False
    
    # 1. 应用汉明窗
    windowed = apply_hamming_window(frame)
    
    # 2. 中心削波
    clipped = center_clipping(windowed)
    
    # 3. 计算自相关
    N = len(clipped)
    # 搜索范围 60Hz-500Hz
    k_min = int(fs / 500)  # 最小周期（500Hz）
    k_max = int(fs / 60)   # 最大周期（60Hz）
    
    # 确保搜索范围有效
    k_max = min(k_max, N // 2)
    if k_max <= k_min:
        return 0, False
    
    # 手动计算自相关（前k_max+1个点）
    R = np.zeros(k_max + 1)
    for k in range(k_max + 1):
        if k == 0:
            R[k] = np.sum(clipped * clipped)
        else:
            R[k] = np.sum(clipped[:N-k] * clipped[k:])
    
    if R[0] < 1e-10:  # 修复：检查R[0]而不是R
        return 0, False
    
    # 4. 在搜索范围内寻找最大峰值
    max_val = -1
    max_k = k_min
    
    for k in range(k_min, k_max + 1):
        if k < len(R) and R[k] > max_val:
            max_val = R[k]
            max_k = k
    
    # 5. 归一化并判断清浊音
    R0 = R[0]
    if R0 < 1e-10:
        return 0, False
    
    normalized_peak = max_val / R0
    frame_energy = np.sum(frame * frame)
    
    # 判断准则
    is_voiced = (normalized_peak > 0.35) and (frame_energy > 0.0005)
    
    return max_k, is_voiced

def durbin_algorithm(frame, p=12):
    """手动实现 Durbin 算法 [12, 13]"""
    if len(frame) == 0:
        return None
    
    win_frame = apply_hamming_window(frame)
    N = len(win_frame)
    
    # 计算自相关
    R = np.zeros(p + 1)
    for k in range(p + 1):
        if k == 0:
            R[k] = np.sum(win_frame * win_frame)
        else:
            R[k] = np.sum(win_frame[:N-k] * win_frame[k:])
    
    if R[0] < 1e-12:  # 修复：检查R[0]
        return None
    
    # Durbin递推
    E = R[0]  # 初始误差能量
    a = np.zeros((p + 1, p + 1))
    
    for i in range(1, p + 1):
        # 计算反射系数
        sum_prev = 0.0
        for j in range(1, i):
            sum_prev += a[i-1, j] * R[i-j]
        
        ki = (R[i] - sum_prev) / (E + 1e-12)
        
        # 稳定性截断（手动实现，不用np.clip）
        if ki > 0.999: 
            ki = 0.999
        elif ki < -0.999: 
            ki = -0.999
        
        a[i, i] = ki
        
        # 更新系数
        for j in range(1, i):
            a[i, j] = a[i-1, j] - ki * a[i-1, i-j]
        
        # 更新误差能量
        E = (1.0 - ki * ki) * E
    
    # 返回AR系数（a1到ap）
    return a[p, 1:]

def ar_to_reflection(ar_coeffs):
    """AR系数转反射系数 [15, 16]"""
    if ar_coeffs is None or len(ar_coeffs) == 0:
        return np.zeros(12)
    
    p = len(ar_coeffs)
    k = np.zeros(p)
    
    # 复制AR系数到矩阵
    a = np.zeros((p + 1, p + 1))
    a[p, 1:] = ar_coeffs
    
    # 反向递推计算反射系数
    for i in range(p, 0, -1):
        ki = a[i, i]
        
        # 手动截断
        if ki > 0.999: 
            ki = 0.999
        elif ki < -0.999: 
            ki = -0.999
        
        k[i-1] = ki
        
        if i > 1:
            denom = 1.0 - ki * ki
            if denom < 1e-12:
                denom = 1e-12
            
            for j in range(1, i):
                a[i-1, j] = (a[i, j] + ki * a[i, i-j]) / denom
    
    return k

# ==========================================
# 第二部分：合成优化增强 (解决波形瘦长与含混问题)
# ==========================================

class LatticeSynthesizer:
    """带状态记忆的格型合成滤波器 [17-19]"""
    def __init__(self, p=12):
        self.p = p
        self.b = np.zeros(p + 1)  # 后向预测误差缓存
        self.f = np.zeros(p + 1)  # 前向预测误差缓存
        self.de_emph_state = 0.0  # 去加重状态
    
    def reset(self):
        """重置滤波器状态"""
        self.b = np.zeros(self.p + 1)
        self.f = np.zeros(self.p + 1)
        self.de_emph_state = 0.0
    
    def synthesize_frame(self, excitation, k_coeffs):
        """合成一整帧语音"""
        frame_len = len(excitation)
        output = np.zeros(frame_len)
        
        for n in range(frame_len):
            # 输入激励
            self.f[0] = excitation[n]
            
            # 格型滤波器递归
            for i in range(1, self.p + 1):
                # 更新前向误差
                self.f[i] = self.f[i-1] + k_coeffs[i-1] * self.b[i-1]
                # 更新后向误差
                b_temp = self.b[i-1] - k_coeffs[i-1] * self.f[i-1]
                # 保存状态
                self.b[i] = b_temp
            
            # 去加重滤波 H(z) = 1/(1 - 0.97z^-1)
            # 实现：y[n] = x[n] + 0.97 * y[n-1]
            out_temp = self.f[self.p] + 0.97 * self.de_emph_state
            self.de_emph_state = out_temp
            
            output[n] = out_temp
        
        return output

def generate_excitation(frame_len, is_voiced, pitch_period):
    """生成平滑激励源 [21, 22]"""
    if not is_voiced or pitch_period < 10:
        # 清音：高斯白噪声
        return np.random.randn(frame_len)
    
    # 浊音：汉宁窗脉冲序列
    excitation = np.zeros(frame_len)
    
    # 脉冲宽度设置为基音周期的1/8，最小3点，最大15点
    pulse_width = max(3, min(15, pitch_period // 8))
    
    # 创建汉宁窗脉冲
    pulse = np.hanning(pulse_width * 2 - 1)
    pulse = pulse[pulse_width//2 : -pulse_width//2]
    
    # 放置脉冲序列
    for pos in range(0, frame_len, pitch_period):
        start_idx = pos - len(pulse) // 2
        end_idx = start_idx + len(pulse)
        
        # 确保索引在有效范围内
        if start_idx < 0:
            pulse_start = -start_idx
            start_idx = 0
        else:
            pulse_start = 0
        
        if end_idx > frame_len:
            pulse_end = len(pulse) - (end_idx - frame_len)
            end_idx = frame_len
        else:
            pulse_end = len(pulse)
        
        if pulse_end > pulse_start:
            excitation[start_idx:end_idx] += pulse[pulse_start:pulse_end]
    
    return excitation

def calculate_rms_energy(signal):
    """计算RMS能量"""
    if len(signal) == 0:
        return 0.0
    return np.sqrt(np.sum(signal * signal) / len(signal))

# ==========================================
# 第三部分：主程序逻辑
# ==========================================

def main():
    root = tk.Tk()
    root.withdraw()
    
    # 选择音频文件
    audio_file = filedialog.askopenfilename(
        title="选择原始语音", 
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    if not audio_file:
        print("未选择音频文件")
        return
    
    # 选择码本文件
    cb_file = filedialog.askopenfilename(
        title="选择128级码本", 
        filetypes=[("NPY files", "*.npy"), ("All files", "*.*")]
    )
    if not cb_file:
        print("未选择码本文件")
        return
    
    # 加载数据
    fs, data = load_wav(audio_file)
    try:
        codebook = np.load(cb_file)
        print(f"码本加载成功，形状: {codebook.shape}")
    except Exception as e:
        print(f"码本加载失败: {e}")
        return
    
    # 参数设置
    p = 12
    frame_len = int(0.025 * fs)   # 25ms
    frame_shift = int(0.01 * fs)  # 10ms
    num_frames = (len(data) - frame_len) // frame_shift
    
    # 初始化
    synthesizer = LatticeSynthesizer(p)
    output_signal = np.zeros(len(data) + frame_len)  # 留有余量
    
    # 创建重叠相加窗口
    window = np.hanning(frame_len)
    
    print(f"开始处理: 采样率={fs}Hz, 帧长={frame_len}, 总帧数={num_frames}")
    
    # 主处理循环
    for i in range(num_frames):
        pos = i * frame_shift
        frame = data[pos : pos + frame_len]
        
        # 1. 基音检测和V/U判别
        pitch_period, is_voiced = pitch_detection_task3(frame, fs)
        
        # 2. 计算AR系数
        ar_coeffs = durbin_algorithm(frame, p)
        if ar_coeffs is None:
            # 静音帧处理
            continue
        
        # 3. AR系数转反射系数
        k_current = ar_to_reflection(ar_coeffs)
        
        # 4. 矢量量化
        distances = np.sum((codebook - k_current)**2, axis=1)
        best_idx = np.argmin(distances)
        k_quantized = codebook[best_idx].copy()
        
        # 5. 生成激励信号
        excitation = generate_excitation(frame_len, is_voiced, pitch_period)
        
        # 6. 增益匹配（RMS能量）
        if len(excitation) > 0:
            # 计算原始帧能量
            frame_energy = np.sum(frame * frame)
            # 计算激励能量
            exc_energy = np.sum(excitation * excitation)
            
            if exc_energy > 1e-10 and frame_energy > 1e-10:
                gain = np.sqrt(frame_energy / exc_energy)
                excitation = excitation * gain
        
        # 7. 格型合成
        resyn_frame = synthesizer.synthesize_frame(excitation, k_quantized)
        
        # 8. 重叠相加
        output_signal[pos : pos + frame_len] += resyn_frame * window
        
        # 进度显示
        if i % 100 == 0:
            print(f"进度: {i+1}/{num_frames} 帧")
    
    # 裁剪到原始长度
    output_signal = output_signal[:len(data)]
    
    # 全局能量归一化
    max_amp = np.max(np.abs(output_signal))
    if max_amp > 0:
        output_signal = output_signal / max_amp * 0.8
    
    # 保存结果
    out_name = os.path.splitext(os.path.basename(audio_file))[0] + "_synthesized.wav"
    
    try:
        with wave.open(out_name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(fs)
            wf.setnframes(len(output_signal))
            wf.writeframes((output_signal * 32767).astype(np.int16).tobytes())
        
        print(f"合成完成，保存为: {out_name}")
        
    except Exception as e:
        print(f"保存失败: {e}")
        return
    
    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 原始语音
    plt.subplot(2, 1, 1)
    time_axis = np.arange(len(data)) / fs
    plt.plot(time_axis, data, 'b', alpha=0.7, label='Original')
    plt.title(f"Original Speech: {os.path.basename(audio_file)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 合成语音
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, output_signal, 'r', alpha=0.7, label='Synthesized')
    plt.title("Synthesized Speech")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()