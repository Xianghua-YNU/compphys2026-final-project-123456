import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

# ====== 读取 Excel ======
df = pd.read_excel('data.xlsx')

# ====== 数据列 ======
datasets = {
    '2K': ('B_2K', 'R_2K'),
    '3K': ('B_3K', 'R_3K'),
    '4K': ('B_4K', 'R_4K'),
    '5K': ('B_5K', 'R_5K'),
    '6K': ('B_6K', 'R_6K'),
    '7K': ('B_7K', 'R_7K')
}

# ====== 指定颜色 ======
colors = {
    '2K': 'black',
    '3K': 'red',
    '4K': 'green',
    '5K': 'blue',
    '6K': 'orange',
    '7K': 'purple'
}

# ====== 循环处理 ======
for temp, (B_col, R_col) in datasets.items():

    B = df[B_col].dropna().values
    R = df[R_col].dropna().values

    # 转 1/B
    invB = 1 / B

    # 排序
    idx = np.argsort(invB)
    invB = invB[idx]
    R = R[idx]

    # 插值
    invB_uniform = np.linspace(invB.min(), invB.max(), 3000)
    interp_func = interp1d(invB, R, kind='linear')
    R_uniform = interp_func(invB_uniform)

    # 去背景
    p = np.polyfit(invB_uniform, R_uniform, 5)
    background = np.polyval(p, invB_uniform)
    dR = R_uniform - background

    # 平滑 + 窗函数
    
    dR = savgol_filter(dR, 151, 3)
    window = np.hanning(len(dR))
    dR = dR * window


    # ====== FFT 补零 ======
    N_original = len(dR)          
    N_fft = 65536                 
    d_invB = invB_uniform[1] - invB_uniform[0]

    # 对 dR 补零后进行 FFT
    yf = fft(dR, n=N_fft)         
    xf = fftfreq(N_fft, d_invB)   # 注意这里传入 N_fft

    # 取正频率部分
    mask = xf > 0
    xf = xf[mask]
    yf = np.abs(yf[mask])


# ====== 限制绘图范围（比如 0~200 T）======
    xlim_max = 200
    plot_mask = xf <= xlim_max
    xf_plot = xf[plot_mask]
    yf_plot = yf[plot_mask]

    # ====== 寻峰（核心新增）======
    # 1. 计算频率步长，用于设置最小峰间距（避免在同一个宽峰上标多个点）
    freq_step = xf_plot[1] - xf_plot[0]
    
    # 2. 设定峰之间的最小间隔（单位：T）。如果数据噪声大，可以设大一点，比如 3.0
    min_freq_separation = 2.0  # 单位：T⁻¹ (即 1/T)
    min_distance_indices = int(min_freq_separation / freq_step)

    # 3. 设定峰的最小显著性（prominence）。
    prominence_threshold = np.max(yf_plot) * 0.07

    # 4. 执行寻峰
    peaks, properties = find_peaks(yf_plot, 
                                   prominence=prominence_threshold,
                                   distance=min_distance_indices)

    # ====== 画图 ======
    plt.plot(xf_plot, yf_plot, label=temp, color=colors[temp])

    # ====== 标记峰值 ======
    # 用散点标记峰值位置
    plt.scatter(xf_plot[peaks], yf_plot[peaks], 
                color=colors[temp], s=30, zorder=5, marker='v')

    # 在峰值上方标注频率数值
    for p in peaks:
        freq_val = xf_plot[p]
        amp_val = yf_plot[p]
        plt.annotate(f'{freq_val:.1f}', 
                     (freq_val, amp_val),
                     textcoords="offset points",
                     xytext=(0, 8),   
                     ha='center',
                     fontsize=8,
                     color=colors[temp],
                     weight='bold')

    # 打印该温度下检测到的所有峰值频率
    print(f"{temp} 峰值频率 (T): {', '.join([f'{xf_plot[p]:.2f}' for p in peaks])}")




# ====== 图设置 ======
plt.xlabel('Frequency (T)')
plt.ylabel('Amplitude')
plt.xlim(0, 200)
plt.legend()
plt.show()
