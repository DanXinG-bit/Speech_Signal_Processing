import wave
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# ==========================================
# 第一部分：自研算法工具库
# ==========================================

def load_wav(filename):
    """读取音频并归一化"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        return fs, wave_data.astype(np.float32) / 32768.0

def apply_hamming_window(frame):
    """应用汉明窗以减少频谱泄露"""
    N = len(frame)
    # 汉明窗公式：w(n) = 0.54 - 0.46 * cos(2πn/(N-1))
    hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
    return frame * hamming

def calc_autocorr_p(frame, p):
    """
    计算0到p阶的自相关系数Rn(k)
    """
    # 先应用汉明窗
    windowed_frame = apply_hamming_window(frame)
    
    R = np.zeros(p + 1)
    N = len(windowed_frame)
    for k in range(p + 1):
        # R(k) = sum(x(n) * x(n+k))
        R[k] = np.sum(windowed_frame[:N-k] * windowed_frame[k:])
    return R

def durbin_algorithm(R, p, stability_threshold=0.999):
    """
    手动实现Levinson-Durbin递推算法
    输入: R (自相关序列, 0~p阶)
    输出: a (AR系数数组, 1~p阶), E (误差能量数组, 0~p阶)
    """
    E = np.zeros(p + 1)
    a = np.zeros((p + 1, p + 1))
    
    # 初始化
    E[0] = R[0]  # E(0) = R(0)
    
    # 递推开始
    for i in range(1, p + 1):
        # 计算反射系数k_i
        sum_val = 0.0
        for j in range(1, i):
            sum_val += a[i-1][j] * R[i-j]
            
        k_i = (R[i] - sum_val) / (E[i-1] + 1e-10)  # 防止除零
        
        # 稳定性截断：确保|k_i| < 1
        if abs(k_i) >= stability_threshold:
            if k_i > 0:
                k_i = stability_threshold - 1e-6
            else:
                k_i = -stability_threshold + 1e-6
        
        a[i][i] = k_i
        
        # 更新预测系数a_{i,j}
        for j in range(1, i):
            a[i][j] = a[i-1][j] - k_i * a[i-1][i-j]
            
        # 更新残差能量E_i
        E[i] = (1 - k_i**2) * E[i-1]
        
    # 返回最终阶数p的系数(a1, a2, ..., ap)和误差能量数组
    return a[p][1:], E

# ==========================================
# 第二部分：任务主逻辑
# ==========================================

def main():
    # 1. 文件选择
    root = tk.Tk()
    root.withdraw()
    audio_file = filedialog.askopenfilename(
        initialdir=os.getcwd(), 
        title="Select audio file for AR analysis"
    )
    if not audio_file: 
        print("No file selected, exiting.")
        return

    fs, data = load_wav(audio_file)
    p = 12  # AR阶数
    
    print(f"Audio file: {os.path.basename(audio_file)}")
    print(f"Sample rate: {fs} Hz")
    print(f"Data length: {len(data)} samples")
    print(f"AR order: p={p}")
    
    # 2. 分帧与选择
    frame_len = int(0.025 * fs)  # 25ms
    frame_shift = int(0.010 * fs)  # 10ms
    
    # 计算所有帧的能量和过零率
    num_frames = (len(data) - frame_len) // frame_shift
    energies = []
    zcrs = []
    
    for i in range(num_frames):
        f = data[i*frame_shift : i*frame_shift + frame_len]
        energies.append(np.sum(f**2))
        zcrs.append(np.sum(np.abs(np.sign(f[1:]) - np.sign(f[:-1]))) / 2)
    
    # 选择浊音帧（最大能量）和清音帧（高过零率）
    idx_v = np.argmax(energies)
    
    # 选择清音帧（高过零率但能量较低）
    sorted_zcr_indices = np.argsort(zcrs)[::-1]
    idx_u = sorted_zcr_indices[0]
    for idx in sorted_zcr_indices[1:]:
        if energies[idx] < 0.3 * energies[idx_v]:
            idx_u = idx
            break
    
    voiced_frame = data[idx_v*frame_shift : idx_v*frame_shift + frame_len]
    unvoiced_frame = data[idx_u*frame_shift : idx_u*frame_shift + frame_len]
    
    print(f"\nFrame selection:")
    print(f"  Voiced frame: position {idx_v*frame_shift} samples (high energy)")
    print(f"  Unvoiced frame: position {idx_u*frame_shift} samples (high ZCR)")
    print(f"  Frame length: {frame_len} samples ({frame_len/fs*1000:.1f} ms)")
    print(f"  Voiced energy: {energies[idx_v]:.6f}")
    print(f"  Unvoiced energy: {energies[idx_u]:.6f}")

    # 3. 计算AR系数和误差曲线
    print("\nCalculating AR coefficients and error curves...")
    
    R_v = calc_autocorr_p(voiced_frame, p)
    ar_v, E_v = durbin_algorithm(R_v, p)  # 现在返回两个值：系数和误差
    
    R_u = calc_autocorr_p(unvoiced_frame, p)
    ar_u, E_u = durbin_algorithm(R_u, p)  # 现在返回两个值：系数和误差
    
    print("AR coefficients and error curves calculated successfully.")
    
    # 打印误差曲线分析信息
    print("\nError Curve Analysis:")
    print(f"Voiced - Initial error E(0): {E_v[0]:.6f}")
    print(f"Voiced - Final error E({p}): {E_v[p]:.6f}")
    print(f"Voiced - Error reduction: {100*(1-E_v[p]/E_v[0]):.1f}%")
    print(f"Unvoiced - Initial error E(0): {E_u[0]:.6f}")
    print(f"Unvoiced - Final error E({p}): {E_u[p]:.6f}")
    print(f"Unvoiced - Error reduction: {100*(1-E_u[p]/E_u[0]):.1f}%")
    
    # 4. 绘图展示（显示时域图、AR系数图和误差曲线）
    plt.figure(figsize=(15, 12))
    
    # 时域波形
    plt.subplot(3, 2, 1)
    plt.plot(voiced_frame, 'b-', linewidth=1.5)
    plt.title("Voiced Frame (Time Domain)", fontsize=12, fontweight='bold')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(voiced_frame))
    
    plt.subplot(3, 2, 2)
    plt.plot(unvoiced_frame, 'r-', linewidth=1.5)
    plt.title("Unvoiced Frame (Time Domain)", fontsize=12, fontweight='bold')
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, len(unvoiced_frame))
    
    # AR系数
    plt.subplot(3, 2, 3)
    markerline, stemlines, baseline = plt.stem(
        range(1, p+1), ar_v, 
        linefmt='b-', markerfmt='bo', basefmt='k-'
    )
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=6)
    plt.title(f"Voiced AR Coefficients (p={p})", fontsize=12, fontweight='bold')
    plt.xlabel("Coefficient Index (i)")
    plt.ylabel("Value (ai)")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    markerline, stemlines, baseline = plt.stem(
        range(1, p+1), ar_u, 
        linefmt='r-', markerfmt='ro', basefmt='k-'
    )
    plt.setp(stemlines, linewidth=1.5)
    plt.setp(markerline, markersize=6)
    plt.title(f"Unvoiced AR Coefficients (p={p})", fontsize=12, fontweight='bold')
    plt.xlabel("Coefficient Index (i)")
    plt.ylabel("Value (ai)")
    plt.grid(True, alpha=0.3)
    
    # 新增：误差曲线
    plt.subplot(3, 2, 5)
    # 绘制归一化误差：E(i)/E(0)
    plt.plot(range(p + 1), E_v / E_v[0], 'b-o', linewidth=2, markersize=6, label='Voiced Error')
    plt.plot(range(p + 1), E_u / E_u[0], 'r-s', linewidth=2, markersize=6, label='Unvoiced Error')
    plt.title("Normalized Prediction Error vs. Order", fontsize=12, fontweight='bold')
    plt.xlabel("Order (i)")
    plt.ylabel("Normalized Error E(i)/R(0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(0, 1.1)  # 设置y轴范围，便于观察
    
    # 新增：误差下降曲线（对数坐标）
    plt.subplot(3, 2, 6)
    plt.plot(range(p + 1), 10 * np.log10(E_v / E_v[0]), 'b-o', linewidth=2, markersize=6, label='Voiced Error (dB)')
    plt.plot(range(p + 1), 10 * np.log10(E_u / E_u[0]), 'r-s', linewidth=2, markersize=6, label='Unvoiced Error (dB)')
    plt.title("Error Reduction in dB Scale", fontsize=12, fontweight='bold')
    plt.xlabel("Order (i)")
    plt.ylabel("Error Reduction (dB)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 添加关键点标注（转折点）
    # 找出误差下降趋于平缓的点（通常在第8阶左右）
    for i in range(4, p):
        if i >= 8:
            # 在转折点添加标记
            plt.annotate(f'i={i}', xy=(i, 10*np.log10(E_v[i]/E_v[0])), 
                        xytext=(i+1, 10*np.log10(E_v[i]/E_v[0])-2),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6),
                        fontsize=9, color='blue')
            plt.annotate(f'i={i}', xy=(i, 10*np.log10(E_u[i]/E_u[0])), 
                        xytext=(i+1, 10*np.log10(E_u[i]/E_u[0])+2),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.6),
                        fontsize=9, color='red')
            break
    
    plt.suptitle(f"AR Analysis with Error Curves: {os.path.basename(audio_file)}", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    
    # 5. 额外分析：误差下降率
    print("\nDetailed Error Analysis:")
    print("Order  Voiced_Norm  Unvoiced_Norm  Voiced_dB  Unvoiced_dB")
    print("----------------------------------------------------------")
    for i in range(0, p+1):
        if i % 2 == 0 or i <= 5:  # 显示关键阶数的数据
            voiced_norm = E_v[i] / E_v[0]
            unvoiced_norm = E_u[i] / E_u[0]
            voiced_db = 10 * np.log10(voiced_norm + 1e-10)
            unvoiced_db = 10 * np.log10(unvoiced_norm + 1e-10)
            print(f"{i:3d}    {voiced_norm:.6f}     {unvoiced_norm:.6f}      {voiced_db:7.2f} dB  {unvoiced_db:7.2f} dB")
    
    # 分析最优阶数建议
    print("\nOptimal Order Analysis (based on error reduction):")
    # 找出误差下降趋于平缓的点（当连续3阶误差下降小于0.5dB时）
    for i in range(2, p-2):
        if (10*np.log10(E_v[i]/E_v[i-1]) > -0.5 and 
            10*np.log10(E_v[i+1]/E_v[i]) > -0.5 and
            10*np.log10(E_v[i+2]/E_v[i+1]) > -0.5):
            print(f"  Voiced: Optimal order around i={i} (error reduction slows down)")
            break
    else:
        print(f"  Voiced: Could use higher order than p={p}")
    
    for i in range(2, p-2):
        if (10*np.log10(E_u[i]/E_u[i-1]) > -0.5 and 
            10*np.log10(E_u[i+1]/E_u[i]) > -0.5 and
            10*np.log10(E_u[i+2]/E_u[i+1]) > -0.5):
            print(f"  Unvoiced: Optimal order around i={i} (error reduction slows down)")
            break
    else:
        print(f"  Unvoiced: Could use higher order than p={p}")

if __name__ == "__main__":
    main()