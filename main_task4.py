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
    """读取音频并归一化 [6]"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        return fs, wave_data.astype(np.float32) / 32768.0

def calc_autocorr_p(frame, p):
    """计算 0 到 p 阶的自相关系数 Rn(k) [7, 8]"""
    R = np.zeros(p + 1)
    N = len(frame)
    for k in range(p + 1):
        # R(k) = sum(x(n) * x(n+k))
        R[k] = np.sum(frame[:N-k] * frame[k:])
    return R

def durbin_algorithm(R, p):
    """
    手动实现 Levinson-Durbin 递推算法 [1]
    输入: R (自相关序列, 0~p阶)
    输出: a (AR系数数组, 1~p阶)
    """
    E = np.zeros(p + 1)
    a = np.zeros((p + 1, p + 1))
    
    # 初始化: i = 0
    E = R
    
    # 递推开始
    for i in range(1, p + 1):
        # 计算反射系数 k_i (即代码中的 a[i][i])
        # 公式: k_i = [R(i) - sum_{j=1}^{i-1} a_{i-1,j} * R(i-j)] / E_{i-1}
        sum_val = 0.0
        for j in range(1, i):
            sum_val += a[i-1][j] * R[i-j]
            
        k_i = (R[i] - sum_val) / E[i-1]
        a[i][i] = k_i
        
        # 更新预测系数 a_{i,j}
        # 公式: a_{i,j} = a_{i-1,j} - k_i * a_{i-1, i-j}
        for j in range(1, i):
            a[i][j] = a[i-1][j] - k_i * a[i-1][i-j]
            
        # 更新残差能量 E_i
        # 公式: E_i = (1 - k_i^2) * E_{i-1}
        E[i] = (1 - k_i**2) * E[i-1]
        
    # 返回最终阶数 p 的系数 (a1, a2, ..., ap)
    return a[p][1:]

# ==========================================
# 第二部分：任务主逻辑 (选择过程与计算)
# ==========================================

def main():
    # 1. 窗口化文件选择 [5, 9]
    root = tk.Tk(); root.withdraw()
    audio_file = filedialog.askopenfilename(initialdir=os.getcwd(), title="任务四：求解AR系数")
    if not audio_file: return

    fs, data = load_wav(audio_file)
    p = 12  # 预测阶数，来源 [10, 11] 建议 8-12 阶
    
    # 2. 分帧与选择过程 (Selection Process) [12]
    # 逻辑：通过短时能量和过零率自动寻找一帧典型的浊音和清音
    frame_len = int(0.025 * fs)
    frame_shift = int(0.010 * fs)
    # 简单计算所有帧的能量和过零率用于自动“选择”
    num_frames = (len(data) - frame_len) // frame_shift
    energies = []
    zcrs = []
    
    for i in range(num_frames):
        f = data[i*frame_shift : i*frame_shift + frame_len]
        energies.append(np.sum(f**2))
        zcrs.append(np.sum(np.abs(np.sign(f[1:]) - np.sign(f[:-1]))) / 2)
    
    # 选择逻辑：浊音=最大能量帧；清音=高过零率且能量适中的帧
    idx_v = np.argmax(energies)
    idx_u = np.argmax(zcrs)
    
    voiced_frame = data[idx_v*frame_shift : idx_v*frame_shift + frame_len]
    unvoiced_frame = data[idx_u*frame_shift : idx_u*frame_shift + frame_len]
    
    print(f"选择过程：")
    print(f"  浊音帧位置: {idx_v*frame_shift} 采样点 (高能量)")
    print(f"  清音帧位置: {idx_u*frame_shift} 采样点 (高过零率)")

    # 3. 计算 AR 系数
    # 先求自相关，再用 Durbin 递推
    R_v = calc_autocorr_p(voiced_frame, p)
    ar_v = durbin_algorithm(R_v, p)
    
    R_u = calc_autocorr_p(unvoiced_frame, p)
    ar_u = durbin_algorithm(R_u, p)

    # 4. 绘图展示
    plt.figure(figsize=(12, 8))
    
    # 浊音部分
    plt.subplot(2, 2, 1)
    plt.plot(voiced_frame)
    plt.title("Voiced Frame (Time Domain)")
    plt.subplot(2, 2, 2)
    plt.stem(range(1, p+1), ar_v)
    plt.title(f"Voiced AR Coefficients (p={p})")
    plt.xlabel("Coefficient Index (i)")
    plt.ylabel("Value (ai)")

    # 清音部分
    plt.subplot(2, 2, 3)
    plt.plot(unvoiced_frame)
    plt.title("Unvoiced Frame (Time Domain)")
    plt.subplot(2, 2, 4)
    plt.stem(range(1, p+1), ar_u)
    plt.title(f"Unvoiced AR Coefficients (p={p})")
    plt.xlabel("Coefficient Index (i)")
    plt.ylabel("Value (ai)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()