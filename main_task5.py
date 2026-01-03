import wave
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import filedialog

# ==========================================
# 第一部分：自研算法工具库 (底层数学实现)
# ==========================================

def load_wav(filename):
    """读取音频并归一化 [2]"""
    with wave.open(filename, 'rb') as wf:
        fs = wf.getparams().framerate
        str_data = wf.readframes(wf.getparams().nframes)
        # 转换为浮点数并归一化到 [-1, 1]
        return fs, np.frombuffer(str_data, dtype=np.int16).astype(np.float32) / 32768.0

def durbin_step(data, p=12):
    """手写 Durbin 算法求解 AR 系数 (来源文档 p107 逻辑) [1]"""
    N = len(data)
    R = np.zeros(p + 1)
    
    # 计算自相关 R(0)...R(p) - 手动实现
    for k in range(p + 1):
        sum_val = 0.0
        for i in range(N - k):
            sum_val += data[i] * data[i + k]
        R[k] = sum_val
    
    # 检查 R[0] 是否太小，而不是检查整个数组
    if R[0] < 1e-10: 
        return None
    
    # 初始化变量
    E = np.zeros(p + 1)
    E[0] = R[0]
    
    # 初始化系数矩阵
    a = np.zeros((p + 1, p + 1))
    
    # Durbin 算法主循环
    for i in range(1, p + 1):
        # 计算反射系数 ki
        sum_prev = 0.0
        for j in range(1, i):
            sum_prev += a[i-1, j] * R[i-j]
        ki = (R[i] - sum_prev) / (E[i-1] + 1e-10)  # 防止除以零
        
        # 稳定性检查：反射系数需 < 1
        if abs(ki) >= 1.0: 
            ki = np.sign(ki) * 0.999
        
        # 更新当前阶系数
        a[i, i] = ki
        
        # 更新其他系数
        for j in range(1, i):
            a[i, j] = a[i-1, j] - ki * a[i-1, i-j]
        
        # 更新误差
        E[i] = (1.0 - ki * ki) * E[i-1]
    
    # 返回第 p 阶的 AR 系数
    return a[p, 1:p+1]

def ar_to_reflection(ar_coeffs):
    """由 AR 系数递推求反射系数 ki (来源文档 p139 公式) [1]"""
    p = len(ar_coeffs)
    if p == 0:
        return np.array([])
    
    # 创建系数矩阵
    a = np.zeros((p + 1, p + 1))
    a[p, 1:p+1] = ar_coeffs
    
    # 反射系数数组
    k = np.zeros(p)
    
    # 反向递推求反射系数
    for i in range(p, 0, -1):
        ki = a[i, i]
        if abs(ki) >= 1.0: 
            ki = np.sign(ki) * 0.999
        k[i-1] = ki
        
        if i > 1:
            denom = 1.0 - ki * ki
            if abs(denom) < 1e-10:  # 防止除以零
                denom = 1e-10
            
            # 更新低阶系数
            for j in range(1, i):
                a[i-1, j] = (a[i, j] + ki * a[i, i-j]) / denom
    
    return k

# ==========================================
# 第二部分：矢量量化 (LBG 算法实现)
# ==========================================

def lbg_train(features, cb_size=16):
    """LBG 分裂聚类算法 (来源文档 p172 逻辑) [1]"""
    if len(features) == 0:
        return np.array([])
    
    # 初始码本：全局形心
    feature_dim = features.shape[1]
    codebook = np.zeros((1, feature_dim))
    
    # 手动计算全局均值
    for i in range(feature_dim):
        sum_val = 0.0
        for j in range(len(features)):
            sum_val += features[j, i]
        codebook[0, i] = sum_val / len(features)
    
    # 不断分裂直到达到目标大小
    while codebook.shape[0] < cb_size:
        # 1. 分裂 (Split) - 添加小扰动
        new_codebook = np.zeros((codebook.shape[0] * 2, feature_dim))
        epsilon = 0.01  # 分裂扰动因子
        
        for i in range(codebook.shape[0]):
            for j in range(feature_dim):
                new_codebook[2*i, j] = codebook[i, j] * (1.0 + epsilon)
                new_codebook[2*i + 1, j] = codebook[i, j] * (1.0 - epsilon)
        
        codebook = new_codebook
        
        # 2. 迭代优化
        for iteration in range(10):
            # 计算每个特征向量到每个码字的距离
            distances = np.zeros((len(features), codebook.shape[0]))
            
            # 手动计算欧氏距离平方
            for i in range(len(features)):
                for j in range(codebook.shape[0]):
                    dist_sq = 0.0
                    for k in range(feature_dim):
                        diff = features[i, k] - codebook[j, k]
                        dist_sq += diff * diff
                    distances[i, j] = dist_sq
            
            # 为每个特征找到最近的码字
            nearest_idx = np.zeros(len(features), dtype=int)
            for i in range(len(features)):
                min_dist = distances[i, 0]
                min_idx = 0
                for j in range(1, codebook.shape[0]):
                    if distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        min_idx = j
                nearest_idx[i] = min_idx
            
            # 更新形心
            new_cb = np.zeros((codebook.shape[0], feature_dim))
            counts = np.zeros(codebook.shape[0])
            
            # 统计每个胞腔的特征点
            for i in range(len(features)):
                idx = nearest_idx[i]
                for j in range(feature_dim):
                    new_cb[idx, j] += features[i, j]
                counts[idx] += 1
            
            # 计算新的形心
            for i in range(codebook.shape[0]):
                if counts[i] > 0:
                    for j in range(feature_dim):
                        codebook[i, j] = new_cb[i, j] / counts[i]
                else:
                    # 防止空胞腔，使用随机值
                    for j in range(feature_dim):
                        codebook[i, j] = np.random.randn() * 0.1
    
    return codebook

# ==========================================
# 第三部分：任务主程序
# ==========================================

def main():
    print("=== 语音信号处理任务五：LPC特征提取与矢量量化 ===")
    
    # 严格遵循老师要求的路径 [2]
    data_dir = "d:/speechdata/"
    if not os.path.exists(data_dir):
        print(f"提示：默认路径 {data_dir} 不存在。")
        print("请选择包含多个子文件夹的语音文件夹...")
        root = tk.Tk()
        root.withdraw()
        data_dir = filedialog.askdirectory(title="请选择包含多个子文件夹的语音文件夹")
        if not data_dir: 
            print("未选择目录，程序退出。")
            return

    all_reflection_coeffs = []
    p = 12
    
    # 遍历目录下的所有 wav 文件
    wav_files = []
    print(f"正在扫描目录: {data_dir}")
    for root_p, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith('.wav'):
                wav_files.append(os.path.join(root_p, f))

    if not wav_files:
        print("错误：选定目录下未找到任何 .wav 文件。")
        return

    print(f"找到 {len(wav_files)} 个WAV文件，开始处理前10个...")
    
    processed_count = 0
    for f_idx, f_path in enumerate(wav_files[:10]):
        try:
            print(f"\n处理文件 {f_idx+1}/{min(10, len(wav_files))}: {os.path.basename(f_path)}")
            
            fs, data = load_wav(f_path)
            print(f"  采样率: {fs} Hz, 时长: {len(data)/fs:.2f}秒")
            
            # 帧参数
            frame_len = int(0.025 * fs)  # 25ms
            frame_shift = int(0.01 * fs)  # 10ms
            num_frames = (len(data) - frame_len) // frame_shift
            
            # 计算能量门限
            max_energy = 0.0
            for i in range(0, min(100, num_frames)):  # 只检查前100帧找最大值
                start = i * frame_shift
                end = start + frame_len
                frame_energy = 0.0
                for j in range(frame_len):
                    frame_energy += data[start + j] * data[start + j]
                if frame_energy > max_energy:
                    max_energy = frame_energy
            
            threshold = max_energy * 0.05 
            print(f"  能量门限: {threshold:.6f}")
            
            # 提取特征
            count = 0
            for i in range(num_frames):
                start = i * frame_shift
                end = start + frame_len
                frame = data[start:end]
                
                # 计算帧能量
                frame_energy = 0.0
                for j in range(frame_len):
                    frame_energy += frame[j] * frame[j]
                
                # 只处理有声音的帧
                if frame_energy > threshold:
                    ar_coeffs = durbin_step(frame, p)
                    if ar_coeffs is not None:
                        k_coeffs = ar_to_reflection(ar_coeffs)
                        if len(k_coeffs) == p:  # 确保长度正确
                            all_reflection_coeffs.append(k_coeffs)
                            count += 1
            
            print(f"  提取到 {count} 个有效特征")
            processed_count += 1
            
        except Exception as e:
            print(f"  处理文件 {os.path.basename(f_path)} 时出错: {e}")
            continue

    if len(all_reflection_coeffs) == 0:
        print("\n最终结果：未提取到任何有效特征。")
        print("可能原因：")
        print("1. 音频文件可能全是静音")
        print("2. 能量门限设置太高")
        print("3. 音频格式不支持")
        return

    print(f"\n特征提取完成，共获得 {len(all_reflection_coeffs)} 个特征向量")
    
    # 转换为numpy数组
    features = np.array(all_reflection_coeffs)
    print(f"特征矩阵形状: {features.shape}")
    
    # 聚类求取码本
    print("\n开始 LBG 聚类训练...")
    codebook = lbg_train(features, cb_size=16)
    
    if len(codebook) == 0:
        print("聚类失败，无有效码本生成")
        return
    
    print(f"码本生成完成，形状: {codebook.shape}")

    # 绘图展示结果
    plt.figure(figsize=(12, 5))
    
    # 图1: 特征空间散点图 (只显示前两维)
    plt.subplot(1, 2, 1)
    plt.scatter(features[:, 0], features[:, 1], s=1, c='gray', alpha=0.1, label='Feature Points')
    plt.scatter(codebook[:, 0], codebook[:, 1], c='red', marker='x', s=100, linewidth=2, label='Codebook Vectors')
    plt.xlabel("Reflection Coefficient K1")
    plt.ylabel("Reflection Coefficient K2")
    plt.title("Feature Space Distribution (First Two Dimensions)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 图2: 码本矩阵可视化
    plt.subplot(1, 2, 2)
    plt.imshow(codebook.T, aspect='auto', cmap='viridis')
    plt.xlabel("Codebook Index (0-15)")
    plt.ylabel("Reflection Coefficient Order (1-12)")
    plt.title("Codebook Matrix (12-Order Reflection Coefficients)")
    plt.colorbar(label="Coefficient Value")
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== 任务五完成 ===")

if __name__ == "__main__":
    main()