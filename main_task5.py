import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import time

# ==========================================
# 第一部分：自研算法工具库 (底层实现，符合规范)
# ==========================================

def load_wav(filename):
    """读取标准 PCM WAV 文件并归一化 [2]"""
    try:
        with wave.open(filename, 'rb') as wf:
            fs = wf.getparams().framerate
            n_channels = wf.getparams().nchannels
            n_frames = wf.getparams().nframes
            
            # 读取数据
            str_data = wf.readframes(n_frames)
            wave_data = np.frombuffer(str_data, dtype=np.int16)
            
            # 如果是多通道，取第一个通道
            if n_channels > 1:
                wave_data = wave_data[::n_channels]
                print(f"  警告: {filename} 是多通道音频，将使用第一个通道")
            
            # 转换为浮点数并归一化到 [-1, 1]
            wave_data = wave_data.astype(np.float32) / 32768.0
            return fs, wave_data
    except Exception as e:
        print(f"  加载 {filename} 失败: {e}")
        return None, None

def durbin_step(data, p=12):
    """手动实现 Durbin 算法求解 AR 系数 [1]"""
    N = len(data)
    if N < p + 1:
        return None
    
    # 1. 计算自相关 R(0)...R(p) - 手动实现
    R = np.zeros(p + 1)
    for k in range(p + 1):
        sum_val = 0.0
        for i in range(N - k):
            sum_val += data[i] * data[i + k]
        R[k] = sum_val
    
    # 2. 检查 R[0] 是否太小（防止除以零）
    if R[0] < 1e-10:
        return None
    
    # 3. Durbin 递归算法
    E = np.zeros(p + 1)
    E[0] = R[0]
    a = np.zeros((p + 1, p + 1))
    
    for i in range(1, p + 1):
        # 计算反射系数 ki
        sum_prev = 0.0
        for j in range(1, i):
            sum_prev += a[i-1, j] * R[i-j]
        
        ki = (R[i] - sum_prev) / (E[i-1] + 1e-12)  # 防止除以零
        
        # 稳定性截断 [1]
        if ki >= 0.999:
            ki = 0.999
        elif ki <= -0.999:
            ki = -0.999
        
        a[i, i] = ki
        
        # 更新其他系数
        for j in range(1, i):
            a[i, j] = a[i-1, j] - ki * a[i-1, i-j]
        
        # 更新误差能量
        E[i] = (1.0 - ki * ki) * E[i-1]
    
    # 返回第 p 阶的 AR 系数
    return a[p, 1:p+1]

def ar_to_reflection(ar_coeffs):
    """由 AR 系数递推求反射系数 ki [1]"""
    if ar_coeffs is None or len(ar_coeffs) == 0:
        return None
    
    p = len(ar_coeffs)
    a = np.zeros((p + 1, p + 1))
    a[p, 1:p+1] = ar_coeffs
    
    k = np.zeros(p)
    
    # 反向递推求反射系数
    for i in range(p, 0, -1):
        ki = a[i, i]
        
        # 稳定性截断
        if ki >= 0.999:
            ki = 0.999
        elif ki <= -0.999:
            ki = -0.999
        
        k[i-1] = ki
        
        if i > 1:
            denom = 1.0 - ki * ki
            if abs(denom) < 1e-12:
                denom = 1e-12
            
            # 更新低阶系数
            for j in range(1, i):
                a[i-1, j] = (a[i, j] + ki * a[i, i-j]) / denom
    
    return k

# ==========================================
# 第二部分：优化版矢量量化 (LBG 算法)
# ==========================================

def lbg_train_optimized(features, cb_size=16, max_iterations=10):
    """优化版 LBG 分裂聚类算法 [1]"""
    if len(features) == 0:
        print("错误: 特征数据为空，无法训练")
        return np.array([])
    
    n_features = len(features)
    feature_dim = features.shape[1]
    
    print(f"  LBG训练: {n_features} 个特征, 维度 {feature_dim}, 目标码本大小 {cb_size}")
    
    # 1. 初始码本：全局形心
    codebook = np.zeros((1, feature_dim))
    
    # 手动计算全局均值
    for d in range(feature_dim):
        sum_val = 0.0
        for i in range(n_features):
            sum_val += features[i, d]
        codebook[0, d] = sum_val / n_features
    
    # 2. 分裂过程
    split_iteration = 0
    while codebook.shape[0] < cb_size:
        split_iteration += 1
        old_size = codebook.shape[0]
        
        # 分裂当前码本
        new_codebook = np.zeros((old_size * 2, feature_dim))
        epsilon = 0.01  # 分裂扰动因子
        
        for i in range(old_size):
            for d in range(feature_dim):
                new_codebook[2*i, d] = codebook[i, d] * (1.0 + epsilon)
                new_codebook[2*i + 1, d] = codebook[i, d] * (1.0 - epsilon)
        
        codebook = new_codebook
        current_size = codebook.shape[0]
        
        print(f"  分裂迭代 {split_iteration}: 码本大小 {old_size} → {current_size}")
        
        # 3. 迭代优化
        for iteration in range(max_iterations):
            # 计算每个特征到每个码字的距离
            distances = np.zeros((n_features, current_size))
            
            for i in range(n_features):
                for j in range(current_size):
                    # 计算欧氏距离平方
                    dist_sq = 0.0
                    for d in range(feature_dim):
                        diff = features[i, d] - codebook[j, d]
                        dist_sq += diff * diff
                    distances[i, j] = dist_sq
            
            # 为每个特征分配最近的码字
            nearest_idx = np.zeros(n_features, dtype=int)
            for i in range(n_features):
                min_dist = distances[i, 0]
                min_idx = 0
                for j in range(1, current_size):
                    if distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        min_idx = j
                nearest_idx[i] = min_idx
            
            # 计算新的形心
            new_cb = np.zeros((current_size, feature_dim))
            counts = np.zeros(current_size)
            
            # 累加每个胞腔的特征
            for i in range(n_features):
                idx = nearest_idx[i]
                for d in range(feature_dim):
                    new_cb[idx, d] += features[i, d]
                counts[idx] += 1
            
            # 计算均值并更新码本
            codebook_changed = False
            for i in range(current_size):
                if counts[i] > 0:
                    for d in range(feature_dim):
                        old_val = codebook[i, d]
                        new_val = new_cb[i, d] / counts[i]
                        codebook[i, d] = new_val
                        if abs(old_val - new_val) > 1e-6:
                            codebook_changed = True
                else:
                    # 空胞腔处理：使用随机值
                    for d in range(feature_dim):
                        codebook[i, d] = np.random.uniform(-0.5, 0.5)
                    codebook_changed = True
            
            # 如果码本变化很小，提前结束迭代
            if not codebook_changed:
                break
    
    print(f"  LBG训练完成，最终码本大小: {codebook.shape[0]}")
    return codebook

# ==========================================
# 第三部分：增强版主逻辑
# ==========================================

def main():
    print("=" * 60)
    print("语音信号处理任务五：LPC特征提取与矢量量化")
    print("=" * 60)
    
    start_time = time.time()
    
    # 1. 路径处理
    data_dir = "d:/speechdata/"
    if not os.path.exists(data_dir):
        print(f"提示: 默认路径 '{data_dir}' 不存在")
        print("请选择包含语音文件的文件夹...")
        root = tk.Tk()
        root.withdraw()
        data_dir = filedialog.askdirectory(title="请选择语音数据集文件夹")
        if not data_dir:
            print("未选择目录，程序退出")
            return
    
    print(f"数据目录: {data_dir}")
    
    # 2. 收集所有WAV文件
    wav_files = []
    for root_dir, _, files in os.walk(data_dir):
        for filename in files:
            if filename.lower().endswith('.wav'):
                full_path = os.path.join(root_dir, filename)
                wav_files.append(full_path)
    
    if not wav_files:
        print("错误: 未找到任何 .wav 文件")
        return
    
    print(f"找到 {len(wav_files)} 个WAV文件")
    
    # 3. 设置处理参数
    p = 12  # LPC阶数
    max_files = 100  # 最大处理文件数
    all_reflection_coeffs = []
    
    print(f"\n开始处理前 {min(max_files, len(wav_files))} 个文件...")
    
    # 4. 特征提取
    processed_count = 0
    for file_idx, file_path in enumerate(wav_files[:max_files]):
        file_start_time = time.time()
        print(f"\n[{file_idx+1}/{min(max_files, len(wav_files))}] 处理: {os.path.basename(file_path)}")
        
        # 加载音频
        fs, data = load_wav(file_path)
        if fs is None or data is None:
            continue
        
        # 检查数据有效性
        if len(data) < int(0.025 * fs) * 10:  # 至少10帧
            print(f"  跳过: 音频太短 ({len(data)/fs:.2f}秒)")
            continue
        
        # 计算帧参数
        frame_len = int(0.025 * fs)  # 25ms
        frame_shift = int(0.01 * fs)  # 10ms
        num_frames = (len(data) - frame_len) // frame_shift
        
        print(f"  采样率: {fs} Hz, 总帧数: {num_frames}")
        
        # 计算能量门限 (优化版)
        # 只检查前200帧以减少计算量
        sample_frames = min(200, num_frames)
        max_energy = 0.0
        
        for i in range(sample_frames):
            start = i * frame_shift
            frame = data[start:start+frame_len]
            frame_energy = 0.0
            for j in range(frame_len):
                frame_energy += frame[j] * frame[j]
            if frame_energy > max_energy:
                max_energy = frame_energy
        
        if max_energy < 1e-10:  # 可能是静音文件
            print(f"  跳过: 音频能量过低 (可能是静音)")
            continue
        
        # 设置自适应门限
        threshold = max_energy * 0.02  # 2% 的峰值能量
        print(f"  能量门限: {threshold:.6e}")
        
        # 提取特征
        features_from_file = 0
        for i in range(num_frames):
            start = i * frame_shift
            frame = data[start:start+frame_len]
            
            # 计算帧能量
            frame_energy = 0.0
            for j in range(frame_len):
                frame_energy += frame[j] * frame[j]
            
            # 能量判断
            if frame_energy > threshold:
                # 计算AR系数
                ar_coeffs = durbin_step(frame, p)
                
                if ar_coeffs is not None:
                    # 转换为反射系数
                    reflection_coeffs = ar_to_reflection(ar_coeffs)
                    
                    if reflection_coeffs is not None and len(reflection_coeffs) == p:
                        all_reflection_coeffs.append(reflection_coeffs)
                        features_from_file += 1
        
        processed_count += 1
        file_time = time.time() - file_start_time
        print(f"  提取特征: {features_from_file} 个, 耗时: {file_time:.2f}秒")
    
    # 5. 检查特征数据
    if len(all_reflection_coeffs) == 0:
        print("\n" + "=" * 60)
        print("错误: 未提取到任何有效特征")
        print("可能原因:")
        print("1. 音频文件可能全是静音或能量过低")
        print("2. 音频格式不支持或损坏")
        print("3. 能量门限设置可能不合理")
        print("=" * 60)
        return
    
    print(f"\n特征提取完成!")
    print(f"处理文件数: {processed_count}")
    print(f"总特征数: {len(all_reflection_coeffs)}")
    
    # 6. 转换为numpy数组
    features = np.array(all_reflection_coeffs)
    print(f"特征矩阵形状: {features.shape}")
    
    # 7. LBG聚类训练
    print("\n开始LBG聚类训练...")
    lbg_start_time = time.time()
    
    codebook = lbg_train_optimized(features, cb_size=16, max_iterations=10)
    
    if len(codebook) == 0:
        print("聚类失败: 无法生成码本")
        return
    
    lbg_time = time.time() - lbg_start_time
    print(f"LBG训练完成，耗时: {lbg_time:.2f}秒")
    print(f"最终码本形状: {codebook.shape}")
    
    # 8. 保存码本
    try:
        np.save("codebook.npy", codebook)
        print(f"码本已保存到: codebook.npy")
    except Exception as e:
        print(f"保存码本失败: {e}")
    
    # 9. 可视化结果 (使用英文确保可移植性)
    print("\n生成可视化结果...")
    plt.figure(figsize=(14, 6))
    
    # 图1: 特征空间散点图 (前两维)
    plt.subplot(1, 2, 1)
    plt.scatter(features[:, 0], features[:, 1], 
                s=1, c='gray', alpha=0.1, label='Feature Points')
    plt.scatter(codebook[:, 0], codebook[:, 1], 
                c='red', marker='x', s=100, linewidth=2, label='Codebook Vectors')
    plt.xlabel("Reflection Coefficient K1")
    plt.ylabel("Reflection Coefficient K2")
    plt.title("Feature Space Distribution (First Two Dimensions)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 图2: 码本矩阵热力图
    plt.subplot(1, 2, 2)
    plt.imshow(codebook.T, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.xlabel("Codebook Index (0-15)")
    plt.ylabel("Reflection Coefficient Order (1-12)")
    plt.title("Codebook Matrix Visualization")
    plt.colorbar(label="Coefficient Value")
    
    # 添加网格线
    plt.gca().set_xticks(np.arange(-0.5, 16, 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, 12, 1), minor=True)
    plt.grid(which='minor', color='w', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    # 10. 计算并显示总体统计信息
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("任务完成统计:")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"处理文件: {processed_count}个")
    print(f"提取特征: {len(features)}个")
    print(f"码本大小: {codebook.shape[0]}x{codebook.shape[1]}")
    print(f"特征维度: {features.shape[1]}阶")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()