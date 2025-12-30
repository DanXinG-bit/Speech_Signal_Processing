import numpy as np

def get_hamming_window(window_len):
    """手动实现汉明窗公式: w(n) = 0.54 - 0.46 * cos(2*pi*n / (N-1))"""
    n = np.arange(window_len)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_len - 1))
    return window

def enframe(signal, frame_len, frame_shift):
    """分帧处理，并手动应用汉明窗 [3]"""
    signal_len = len(signal)
    num_frames = int(np.ceil((signal_len - frame_len) / frame_shift)) + 1
    pad_len = (num_frames - 1) * frame_shift + frame_len
    padded_signal = np.concatenate((signal, np.zeros(pad_len - signal_len)))
    
    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_shift, frame_shift), (frame_len, 1)).T
    frames = padded_signal[indices.astype(np.int32)]
    
    window = get_hamming_window(frame_len)
    return frames * window

def calc_short_time_energy(frames):
    """计算短时能量 [4]"""
    return np.sum(frames**2, axis=1)

def calc_short_time_zcr(frames):
    """计算短时平均过零率 [4]"""
    num_frames, frame_len = frames.shape
    zcr = np.zeros(num_frames)
    for i in range(num_frames):
        frame = frames[i]
        count = 0
        for j in range(1, frame_len):
            if frame[j] * frame[j-1] < 0: # 符号变化
                count += 1
        zcr[i] = count
    return zcr