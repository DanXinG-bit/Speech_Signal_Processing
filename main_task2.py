import wave
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# ==========================================
# 第一部分：自研算法工具库 (底层实现)
# ==========================================

def get_hamming_window(window_len):
    """手动实现汉明窗公式 [1]"""
    n = np.arange(window_len)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_len - 1))

def enframe(signal, frame_len, frame_shift):
    """分帧处理 [1]"""
    signal_len = len(signal)
    num_frames = int(np.ceil((signal_len - frame_len) / frame_shift)) + 1
    pad_len = (num_frames - 1) * frame_shift + frame_len
    padded_signal = np.concatenate((signal, np.zeros(pad_len - signal_len)))
    
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_len, 1)).T
    return padded_signal[indices.astype(np.int32)]

def calc_autocorr(frame):
    """手动实现短时自相关函数 Rn(k) [2]
    公式: R(k) = sum_{m=0}^{N-1-k} x(m) * x(m+k)
    """
    N = len(frame)
    # 结果对称，只需计算正半部分
    res = np.zeros(N)
    for k in range(N):
        # 计算延迟为 k 时的乘加和
        res[k] = np.sum(frame[:N-k] * frame[k:])
    return res

def load_wav(filename):
    """使用标准库 wave 读取音频 [3]"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        wave_data = wave_data.astype(np.float32) / 32768.0 # 归一化
        return fs, wave_data

# ==========================================
# 第二部分：任务主逻辑 (选择过程与绘图)
# ==========================================

def main():
    # 1. 窗口化文件选择
    root = tk.Tk()
    root.withdraw()
    audio_file = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="任务二：选择语音文件提取清浊音帧",
        filetypes=(("WAV files", "*.wav"), ("All files", "*.*"))
    )
    if not audio_file: return

    fs, data = load_wav(audio_file)
    
    # 2. 预处理：分帧并加汉明窗 (25ms帧长, 10ms帧移) [1]
    frame_len = int(0.025 * fs)
    frame_shift = int(0.01 * fs)
    # 注意：求自相关前通常需要加窗以减少边缘效应
    window = get_hamming_window(frame_len)
    raw_frames = enframe(data, frame_len, frame_shift)
    windowed_frames = raw_frames * window

    # 3. 帧选择过程 (Selection Process) [4, 5]
    # 逻辑：通过短时能量和过零率寻找典型帧
    energies = np.sum(windowed_frames**2, axis=1)
    # 计算过零率
    zcrs = np.zeros(len(windowed_frames))
    for i in range(len(windowed_frames)):
        zcrs[i] = np.sum(np.abs(np.sign(raw_frames[i, 1:]) - np.sign(raw_frames[i, :-1]))) / 2

    # --- 筛选浊音帧 (Voiced): 能量极高且过零率低的帧 ---
    v_idx = np.intersect1d(np.where(energies > np.mean(energies)*1.5), 
                           np.where(zcrs < np.mean(zcrs)*0.8))[6] # 取中间典型帧
    voiced_frame = windowed_frames[v_idx]

    # --- 筛选清音帧 (Unvoiced): 能量较低但过零率显著高的帧 ---
    u_idx = np.intersect1d(np.where(energies < np.mean(energies)), 
                           np.where(zcrs > np.mean(zcrs)*1.2))[6]
    unvoiced_frame = windowed_frames[u_idx]

    print(f"选择过程完成：")
    print(f"  浊音帧索引: {v_idx} (高能量, 低过零率)")
    print(f"  清音帧索引: {u_idx} (低能量, 高过零率)")

    # 4. 计算自相关 [2]
    r_voiced = calc_autocorr(voiced_frame)
    r_unvoiced = calc_autocorr(unvoiced_frame)

    # 5. 绘图展示
    plt.figure(figsize=(12, 10))
    
    # 浊音展示
    plt.subplot(2, 2, 1)
    plt.plot(voiced_frame)
    plt.title("Voiced Frame (Time Domain)")
    plt.subplot(2, 2, 2)
    plt.plot(r_voiced)
    plt.title("Autocorrelation of Voiced Frame")
    
    # 清音展示
    plt.subplot(2, 2, 3)
    plt.plot(unvoiced_frame)
    plt.title("Unvoiced Frame (Time Domain)")
    plt.subplot(2, 2, 4)
    plt.plot(r_unvoiced)
    plt.title("Autocorrelation of Unvoiced Frame")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()