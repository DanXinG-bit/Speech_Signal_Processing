import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog, messagebox
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
    """Durbin算法"""
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
# 第二部分：矢量量化 (LBG) - 修改以记录误差曲线和码字分配
# ==========================================

def lbg_train_with_stats(features, cb_size=128, max_iterations=10):
    """向量化LBG算法，记录归一化误差曲线和每个码字分配的矢量数量"""
    if len(features) == 0:
        return np.array([]), [], np.array([])
    
    n_samples, n_dims = features.shape
    print(f"  LBG训练: {n_samples}个样本, {n_dims}维, 目标码本: {cb_size}")
    
    # 1. 初始码本：全局均值
    codebook = np.mean(features, axis=0, keepdims=True)
    
    # 记录归一化误差曲线
    distortion_history = []
    
    # 计算初始畸变
    initial_dist = np.mean(np.sum((features - codebook)**2, axis=1))
    distortion_history.append(initial_dist)
    
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
            x_sq = np.sum(features ** 2, axis=1, keepdims=True)  # (n_samples, 1)
            c_sq = np.sum(codebook ** 2, axis=1)  # (current_size,)
            x_dot_c = np.dot(features, codebook.T)  # (n_samples, current_size)
            
            # 距离矩阵: (n_samples, current_size)
            distances = x_sq + c_sq - 2 * x_dot_c
            
            # 分配最近码字
            nearest_idx = np.argmin(distances, axis=1)
            
            # 计算当前平均畸变
            current_dist = np.mean(np.min(distances, axis=1))
            distortion_history.append(current_dist)
            
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
    
    # 计算最终分配
    x_sq = np.sum(features ** 2, axis=1, keepdims=True)
    c_sq = np.sum(codebook ** 2, axis=1)
    x_dot_c = np.dot(features, codebook.T)
    distances = x_sq + c_sq - 2 * x_dot_c
    nearest_idx = np.argmin(distances, axis=1)
    
    # 统计每个码字分配的矢量数量
    final_counts = np.bincount(nearest_idx, minlength=len(codebook))
    
    # 计算归一化误差曲线
    norm_error_curve = np.array(distortion_history) / initial_dist
    
    return codebook, norm_error_curve, final_counts

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
    max_files = min(100, len(wav_files))
    # 最多处理N个文件
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
    
    # 修改：调用新的函数获取误差曲线和分配统计
    codebook, error_curve, code_counts = lbg_train_with_stats(features_matrix, cb_size=128)
    
    if len(codebook) == 0:
        print("LBG训练失败")
        return
    
    lbg_time = time.time() - lbg_start
    print(f"LBG训练完成! 耗时: {lbg_time:.2f}秒")
    print(f"码本大小: {codebook.shape[0]}x{codebook.shape[1]}")
    print(f"总迭代步数: {len(error_curve)}")
    
    # 7. 保存结果（在显示图表之前）
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"codebook_{timestamp}.npy"
        output_path = os.path.join(os.getcwd(), output_file)
        np.save(output_file, codebook)
        file_size = os.path.getsize(output_path)
        file_size_mb = file_size / (1024 * 1024)
        print(f"码本已保存: {output_file}")
        print(f"文件大小: {file_size:,} bytes ({file_size_mb:.2f} MB)")
    except Exception as e:
        print(f"保存失败: {e}")
        output_file = None
        output_path = None
    
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
    
    # 打印码字分配的详细统计
    print("\n码字分配详细统计:")
    print("-" * 40)
    print(f"空码字数: {np.sum(code_counts == 0)}")
    print(f"分配1-10个矢量的码字数: {np.sum((code_counts >= 1) & (code_counts <= 10))}")
    print(f"分配11-100个矢量的码字数: {np.sum((code_counts >= 11) & (code_counts <= 100))}")
    print(f"分配100+个矢量的码字数: {np.sum(code_counts > 100)}")
    
    # 显示前10个最常用的码字
    print("\n前10个最常用的码字:")
    print("Index | Vector Count")
    print("-" * 25)
    for i, idx in enumerate(np.argsort(code_counts)[::-1][:10]):
        print(f"{idx:5d} | {code_counts[idx]:12d}")
    
    # 显示弹窗提示（在显示图表之前）
    if output_file:
        abs_path = os.path.abspath(output_path)
        messagebox.showinfo(
            "处理完成",
            f"任务五：LPC特征提取与矢量量化已完成！\n\n"
            f"✓ 特征提取: {len(features_matrix):,}个特征\n"
            f"✓ 码本训练: {codebook.shape[0]}x{codebook.shape[1]}\n"
            f"✓ 训练时间: {lbg_time:.1f}秒\n\n"
            f"生成文件: {output_file}\n"
            f"保存位置: {abs_path}\n\n也就是代码所在的目录\n\n"
            f"请点击确定来查看后续分析图表"
        )
    
    # 9. 可视化 - 根据要求修改为三个图
    print("\n生成可视化结果...")
    
    # 创建三个子图
    plt.figure(figsize=(15, 5))
    
    # 图1: 归一化误差曲线
    plt.subplot(1, 3, 1)
    plt.plot(error_curve, 'b-', linewidth=2)
    plt.xlabel('Iteration Step')
    plt.ylabel('Normalized Distortion D(n)/D(0)')
    plt.title('Normalized Quantization Error Curve')
    plt.grid(True, alpha=0.3)
    
    # 标记关键点
    if len(error_curve) > 20:
        # 找出误差下降显著变缓的点（通常在前10-20步）
        for i in range(10, len(error_curve)-5):
            if (error_curve[i+1] - error_curve[i]) / error_curve[i] > -0.001:
                plt.axvline(x=i, color='r', linestyle='--', alpha=0.5, 
                           label=f'Slowing at step {i}')
                plt.text(i, error_curve[i]*0.9, f'Step {i}', 
                        rotation=90, fontsize=9)
                break
    
    # 添加统计信息
    final_error = error_curve[-1]
    initial_error = error_curve[0]
    reduction_rate = (1 - final_error) * 100
    plt.text(0.05, 0.95, f'Final D/D(0): {final_error:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    plt.text(0.05, 0.88, f'Reduction: {reduction_rate:.1f}%', 
             transform=plt.gca().transAxes, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 图2: 每个码字包含的矢量数量统计（条形图）
    plt.subplot(1, 3, 2)
    
    # 按数量排序，便于观察分布
    sorted_indices = np.argsort(code_counts)[::-1]
    sorted_counts = code_counts[sorted_indices]
    
    bars = plt.bar(range(len(sorted_counts)), sorted_counts, 
                   color='steelblue', edgecolor='black')
    
    # 添加数值标签在前10个柱子上
    for i, count in enumerate(sorted_counts[:10]):
        plt.text(i, count + 5, str(int(count)), 
                ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Codebook Index (Sorted)')
    plt.ylabel('Number of Vectors Assigned')
    plt.title(f'Vector Count per Codebook Entry\n(Total Vectors: {np.sum(code_counts):,})')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加统计信息
    max_count = np.max(code_counts)
    min_count = np.min(code_counts)
    avg_count = np.mean(code_counts)
    std_count = np.std(code_counts)
    
    stats_text = (f'Max: {max_count}\n'
                  f'Min: {min_count}\n'
                  f'Mean: {avg_count:.1f}\n'
                  f'Std: {std_count:.1f}')
    
    plt.text(0.65, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="white", alpha=0.8))
    
    # 图3: 码本矩阵热力图（保留核心表征）
    plt.subplot(1, 3, 3)
    im = plt.imshow(codebook.T, aspect='auto', cmap='viridis', 
                    interpolation='nearest')
    plt.colorbar(im, orientation='vertical', pad=0.02, label='Coefficient Value')
    plt.xlabel('Codebook Index (0-127)')
    plt.ylabel('Reflection Coefficient Order (1-12)')
    plt.title('Codebook Matrix (128x12)')
    
    # 设置刻度
    plt.xticks(range(0, 128, 16), [str(i) for i in range(0, 128, 16)])
    plt.yticks(range(12), range(1, 13))
    
    plt.suptitle(f'LPC-VQ Analysis Results (Total Features: {len(features_matrix):,})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    print("\n" + "=" * 70)
    print("正在显示分析图表...")
    print("=" * 70)
    
    # 显示图表
    plt.show()
    
    # 程序结束提示
    print("\n" + "=" * 70)
    print("任务五执行完毕！")
    if output_file:
        print(f"码本文件: {output_file}")
        print(f"保存位置: {os.path.abspath(output_path)}")
        print(f"工作目录: {os.getcwd()}")
    print("=" * 70)

if __name__ == "__main__":
    main()