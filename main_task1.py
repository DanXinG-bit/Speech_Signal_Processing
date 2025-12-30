import wave
import numpy as np
import matplotlib.pyplot as plt
import functions_task1 as func

def load_wav_with_wave(filename):
    """使用标准库 wave 读取音频数据，不依赖 scipy [1, 5]"""
    with wave.open(filename, 'rb') as wf:
        # 获取采样率
        fs = wf.getparams().framerate
        # 读取全部帧数据 (字节流)
        n_frames = wf.getparams().nframes
        str_data = wf.readframes(n_frames)
        # 转换字节流为 16 位有符号整数
        wave_data = np.frombuffer(str_data, dtype=np.int16)
        # 归一化到 [-1, 1]
        wave_data = wave_data.astype(np.float32) / 32768.0
    return fs, wave_data

def endpoint_detection(energy, zcr):
    """双门限法逻辑实现 [6]"""
    # 阈值设置：基于前 10 帧背景噪声
    noise_energy = np.mean(energy[:10])
    noise_zcr = np.mean(zcr[:10])
    
    ITU = noise_energy * 5.0   # 高能量门限
    ITL = noise_energy * 2.0   # 低能量门限
    IZCT = noise_zcr * 3.0     # 过零率门限
    
    segments = []
    in_voiced = False
    temp_start = 0
    
    for i in range(len(energy)):
        if not in_voiced and energy[i] > ITU:
            in_voiced = True
            temp_start = i
        elif in_voiced and energy[i] < ITL:
            in_voiced = False
            # 简单记录起始和结束帧
            segments.append([temp_start, i])
    
    return segments, [ITU, ITL, IZCT]

def main():
    # 注意：请确保将音频文件放在代码同一目录下，满足路径简单化要求 [1, 7]
    # 需要选择两端静音且中间至少有 3 段静音的文件 [8]
    audio_file = "test_audio.wav" 
    
    try:
        fs, data = load_wav_with_wave(audio_file)
        print(f"成功加载文件: {audio_file}, 采样率: {fs}")
    except Exception as e:
        print(f"读取错误: {e}。请确认文件是否存在于当前目录。")
        return

    # 1. 特征提取 (25ms帧长, 10ms帧移) [3]
    frame_len = int(0.025 * fs)
    frame_shift = int(0.01 * fs)
    frames = func.enframe(data, frame_len, frame_shift)
    
    energy = func.calc_short_time_energy(frames)
    zcr = func.calc_short_time_zcr(frames)
    
    # 2. 执行端点检测算法 [6, 8]
    segments, thresholds = endpoint_detection(energy, zcr)
    
    # 3. 结果可视化 (Matplotlib) [7, 8]
    plt.figure(figsize=(10, 6))
    
    # 子图1：原始波形与端点标注
    plt.subplot(2, 1, 1)
    plt.plot(data, color='gray', alpha=0.5)
    for start_frame, end_frame in segments:
        # 转换帧序号为采样点位置
        start_sample = start_frame * frame_shift
        end_sample = end_frame * frame_shift
        plt.axvline(x=start_sample, color='red', linestyle='--')
        plt.axvline(x=end_sample, color='green', linestyle='--')
    plt.title("Speech Waveform and Endpoints")
    
    # 子图2：能量曲线
    plt.subplot(2, 1, 2)
    plt.plot(energy)
    plt.axhline(y=thresholds, color='red', label='ITU')
    plt.axhline(y=thresholds[8], color='orange', label='ITL')
    plt.title("Short-time Energy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()