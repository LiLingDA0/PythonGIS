import matplotlib.pyplot as plt
import os
from ClusterWays import exponential_compression as root



def plot_sorted_array(data_array, chart_title="Sorted Data Plot", save_path=None, rootV=1, ignoreV=None):
    """
    绘制排序后数据的折线图（始终显示图表）

    参数：
    data_array (list/np.array): 输入的一维数据数组
    chart_title (str, optional): 图表标题
    save_path (str, optional): 图片存储路径

    示例：
    plot_sorted_array([3,1,4,2], "My Data", "output.png")
    """

    data = data_array.copy()

    # 第一步：过滤特定值
    if ignoreV is not None:
        data = [x for x in data if x != ignoreV]
        if not data:  # 空数据检查
            raise ValueError("过滤后数据为空，无法绘制图表")

    if rootV != 1:
        data= root(data, rootV)
    # 对数组进行排序
    data = sorted(data)
    

    # 创建画布和坐标轴
    # 配置 matplotlib 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体或其他支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 解负号显示问题
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制折线图
    ax.plot(data,
           marker='o', linestyle='-', color='steelblue',
           markersize=6, linewidth=1.5)

    # 设置图表元素
    ax.set_title(chart_title, fontsize=14)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 自动调整布局
    plt.tight_layout()

    # 保存图表（如果指定路径）
    if save_path:
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')



    # 始终显示图表
    plt.show()

    # 关闭图形释放内存
    plt.close(fig)
