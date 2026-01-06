import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog
import time

# ==========================================
# 第一部分：使用numpy向量化
# ==========================================

def load_wav(filename):
    """读取WAV文件"""
    try:
        with wave.open(filename, 'rb') as wf:
            fs = wf.getparams().framerate
            n_channels = wf.getparams().nchannels
            str_data = wf.readframes(wf.getparams().nframes)
            wave_data = np.frombuffer(str_data, dtype=np.int16)
            
            # 多通道处理
            if n_channels > 1:
                wave_data = wave_data.reshape(-1, n_channels)[:, 0]  # 取第一通道
            
            return fs, wave_data.astype(np.float32) / 32768.0
    except Exception as e:
        print(f"  加载失败: {e}")
        return None, None

def durbin_step_optimized(frame, p=12):
    """Durbin算法 (使用numpy向量化)"""
    N = len(frame)
    if N < p + 1:
        return None
    
    # 1. 向量化计算自相关 R(0)...R(p)
    R = np.zeros(p + 1)
    for k in range(p + 1):
        R[k] = np.sum(frame[:N-k] * frame[k:])
    
    # 能量检查
    if R[0] < 1e-12:
        return None
    
    # 2. 向量化Durbin算法
    E = np.zeros(p + 1)
    E[0] = R[0]
    a = np.zeros((p + 1, p + 1))
    
    for i in range(1, p + 1):
        # 向量化计算反射系数
        if i == 1:
            ki = R[1] / R[0]
        else:
            # 使用向量点积计算
            ki = (R[i] - np.dot(a[i-1, 1:i], R[i-1:0:-1])) / E[i-1]
        
        # 稳定性截断
        ki = np.clip(ki, -0.999, 0.999)
        a[i, i] = ki
        
        # 向量化更新系数
        if i > 1:
            a[i, 1:i] = a[i-1, 1:i] - ki * a[i-1, i-1:0:-1]
        
        E[i] = (1.0 - ki * ki) * E[i-1]
    
    return a[p, 1:p+1]

def ar_to_reflection_optimized(ar_coeffs):
    """AR到反射系数转换"""
    if ar_coeffs is None or len(ar_coeffs) == 0:
        return None
    
    p = len(ar_coeffs)
    a = np.zeros((p + 1, p + 1))
    a[p, 1:] = ar_coeffs
    k = np.zeros(p)
    
    # 反向递推
    for i in range(p, 0, -1):
        ki = np.clip(a[i, i], -0.999, 0.999)
        k[i-1] = ki
        
        if i > 1:
            denom = 1.0 - ki * ki
            if denom < 1e-12:
                denom = 1e-12
            
            # 向量化更新
            a[i-1, 1:i] = (a[i, 1:i] + ki * a[i, i-1:0:-1]) / denom
    
    return k

# ==========================================
# 第二部分：矢量量化 (LBG)
# ==========================================

def lbg_train_vectorized(features, cb_size=16, max_iterations=10):
    """向量化LBG算法 (性能优化版)"""
    if len(features) == 0:
        return np.array([])
    
    n_samples, n_dims = features.shape
    print(f"  LBG训练: {n_samples}个样本, {n_dims}维, 目标码本: {cb_size}")
    
    # 1. 初始码本：全局均值
    codebook = np.mean(features, axis=0, keepdims=True)
    
    # 2. 分裂训练
    while codebook.shape[0] < cb_size:
        # 分裂：添加扰动
        epsilon = 0.01
        codebook = np.vstack([
            codebook * (1 + epsilon),
            codebook * (1 - epsilon)
        ])
        
        current_size = codebook.shape[0]
        
        # 3. 迭代优化
        for iteration in range(max_iterations):
            # 向量化计算所有距离 (广播机制)
            # (x - c)^2 = x^2 + c^2 - 2*x·c
            x_sq = np.sum(features ** 2, axis=1, keepdims=True)  # (n_samples, 1)
            c_sq = np.sum(codebook ** 2, axis=1)  # (current_size,)
            x_dot_c = np.dot(features, codebook.T)  # (n_samples, current_size)
            
            # 距离矩阵: (n_samples, current_size)
            distances = x_sq + c_sq - 2 * x_dot_c
            
            # 分配最近码字
            nearest_idx = np.argmin(distances, axis=1)
            
            # 更新码本
            new_codebook = np.zeros_like(codebook)
            counts = np.zeros(current_size)
            
            # 向量化统计和更新
            for i in range(current_size):
                mask = nearest_idx == i
                if np.any(mask):
                    new_codebook[i] = np.mean(features[mask], axis=0)
                    counts[i] = np.sum(mask)
                else:
                    # 空胞腔：随机扰动
                    new_codebook[i] = codebook[i] * (1 + np.random.uniform(-0.1, 0.1))
                    counts[i] = 0
            
            # 检查收敛
            if np.allclose(codebook, new_codebook, atol=1e-6):
                break
            
            codebook = new_codebook
        
        print(f"    码本大小: {current_size}, 非空胞腔: {np.sum(counts > 0)}")
    
    return codebook

# ==========================================
# 第三部分：特征提取
# ==========================================

def extract_features_from_file(file_path, p=12, energy_threshold_ratio=0.02):
    """从单个文件高效提取特征 (向量化)"""
    fs, data = load_wav(file_path)
    if fs is None:
        return None, 0
    
    # 参数设置
    frame_len = int(0.025 * fs)  # 25ms
    frame_shift = int(0.01 * fs)  # 10ms
    
    # 检查数据长度
    if len(data) < frame_len * 10:
        return None, 0
    
    # 1. 向量化分帧
    num_frames = (len(data) - frame_len) // frame_shift
    indices = np.arange(frame_len).reshape(1, -1) + \
              np.arange(0, num_frames * frame_shift, frame_shift).reshape(-1, 1)
    frames = data[indices.astype(int)]
    
    # 2. 向量化能量计算
    energies = np.sum(frames ** 2, axis=1)
    
    # 3. 自适应能量门限
    # 只考虑前200帧计算最大能量（避免静音段影响）
    sample_size = min(200, num_frames)
    max_energy = np.max(energies[:sample_size])
    
    if max_energy < 1e-12:
        return None, 0
    
    threshold = max_energy * energy_threshold_ratio
    
    # 4. 选择有效帧
    valid_mask = energies > threshold
    valid_frames = frames[valid_mask]
    
    if len(valid_frames) == 0:
        return None, 0
    
    # 5. 批量计算特征
    features_list = []
    for frame in valid_frames:
        ar_coeffs = durbin_step_optimized(frame, p)
        if ar_coeffs is not None:
            k_coeffs = ar_to_reflection_optimized(ar_coeffs)
            if k_coeffs is not None:
                features_list.append(k_coeffs)
    
    if len(features_list) == 0:
        return None, 0
    
    features = np.array(features_list)
    return features, len(valid_frames)

# ==========================================
# 第四部分：主程序
# ==========================================

def main():
    print("=" * 70)
    print("语音信号处理任务五：LPC特征提取与矢量量化")
    print("=" * 70)
    
    total_start = time.time()

    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(title="选择语音数据集文件夹")
    if not data_dir:
        print("程序退出")
        return
    
    print(f"数据目录: {data_dir}")
    
    # 2. 收集WAV文件
    wav_files = []
    for root_dir, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root_dir, f))
    
    if not wav_files:
        print("错误: 未找到WAV文件")
        return
    
    print(f"找到 {len(wav_files)} 个WAV文件")
    
    # 3. 设置处理参数
    p = 12
    max_files = min(100, len(wav_files))  # 最多处理30个文件
    all_features = []
    file_stats = []
    
    print(f"\n开始处理前 {max_files} 个文件...")
    print("-" * 70)
    
    # 4. 并行化特征提取 (逐个文件处理，但文件内向量化)
    successful_files = 0
    for i, file_path in enumerate(wav_files[:max_files]):
        file_start = time.time()
        
        print(f"[{i+1:2d}/{max_files}] {os.path.basename(file_path):30s}", end="", flush=True)
        
        features, num_valid_frames = extract_features_from_file(file_path, p)
        
        if features is not None:
            all_features.append(features)
            successful_files += 1
            file_time = time.time() - file_start
            
            stats = {
                'name': os.path.basename(file_path),
                'features': len(features),
                'valid_frames': num_valid_frames,
                'time': file_time
            }
            file_stats.append(stats)
            
            print(f" ✓ {len(features):4d}特征 ({file_time:.2f}s)")
        else:
            file_time = time.time() - file_start
            print(f" ✗ 无有效特征 ({file_time:.2f}s)")
    
    # 5. 合并所有特征
    if len(all_features) == 0:
        print("\n错误: 未提取到任何特征")
        print("检查: 1.音频格式 2.能量门限 3.算法参数")
        return
    
    features_matrix = np.vstack(all_features)
    print(f"\n特征提取完成!")
    print(f"成功处理文件: {successful_files}/{max_files}")
    print(f"总特征数量: {len(features_matrix):,}")
    print(f"特征维度: {features_matrix.shape[1]}阶")
    
    # 6. LBG训练
    print("\n" + "-" * 70)
    print("开始LBG矢量量化训练...")
    lbg_start = time.time()
    
    codebook = lbg_train_vectorized(features_matrix, cb_size=16)
    
    if len(codebook) == 0:
        print("LBG训练失败")
        return
    
    lbg_time = time.time() - lbg_start
    print(f"LBG训练完成! 耗时: {lbg_time:.2f}秒")
    print(f"码本大小: {codebook.shape[0]}x{codebook.shape[1]}")
    
    # 7. 保存结果
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"codebook_{timestamp}.npy"
        np.save(output_file, codebook)
        print(f"码本已保存: {output_file}")
    except Exception as e:
        print(f"保存失败: {e}")
    
    # 8. 统计信息
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("性能统计:")
    print(f"总处理时间: {total_time:.2f}秒")
    print(f"平均每文件: {total_time/max_files:.2f}秒")
    print(f"特征提取速率: {len(features_matrix)/total_time:.1f} 特征/秒")
    
    if file_stats:
        best_file = max(file_stats, key=lambda x: x['features'])
        worst_file = min(file_stats, key=lambda x: x['features'])
        print(f"\n最佳文件: {best_file['name']} ({best_file['features']}特征)")
        print(f"最差文件: {worst_file['name']} ({worst_file['features']}特征)")
    
    # 9. 可视化
    print("\n生成可视化结果...")
    plt.figure(figsize=(15, 5))
    
    # 图1: 特征分布
    plt.subplot(1, 3, 1)
    plt.scatter(features_matrix[:, 0], features_matrix[:, 1], 
                s=1, alpha=0.1, c='blue', label='Features')
    plt.scatter(codebook[:, 0], codebook[:, 1],
                s=100, marker='X', c='red', edgecolors='black', 
                linewidth=1.5, label='Codebook')
    plt.xlabel('Reflection Coefficient K1')
    plt.ylabel('Reflection Coefficient K2')
    plt.title('Feature Distribution (K1 vs K2)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 图2: 码本热力图
    plt.subplot(1, 3, 2)
    im = plt.imshow(codebook.T, aspect='auto', cmap='viridis', 
                    interpolation='nearest')
    plt.colorbar(im, label='Coefficient Value')
    plt.xlabel('Codebook Index (0-15)')
    plt.ylabel('Coefficient Order (1-12)')
    plt.title('Codebook Matrix')
    plt.xticks(range(16))
    plt.yticks(range(12))
    
    # 图3: 反射系数统计
    plt.subplot(1, 3, 3)
    mean_coeffs = np.mean(features_matrix, axis=0)
    std_coeffs = np.std(features_matrix, axis=0)
    
    x_pos = np.arange(1, p + 1)
    plt.errorbar(x_pos, mean_coeffs, yerr=std_coeffs, 
                 fmt='o-', capsize=5, linewidth=2, markersize=6,
                 label='Mean ± Std')
    plt.xlabel('Coefficient Order')
    plt.ylabel('Coefficient Value')
    plt.title('Reflection Coefficients Statistics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(x_pos)
    
    plt.suptitle(f'LPC-VQ Analysis (Total Features: {len(features_matrix):,})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print("\n" + "=" * 70)
    print("任务完成! 请查看可视化结果...")
    print("=" * 70)
    
    plt.show()

if __name__ == "__main__":
    main()