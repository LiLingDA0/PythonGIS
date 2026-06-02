import pandas as pd

# 读取CSV文件
file_path = 'C:\\Users\\32001\\Desktop\\work\\Python\\GBD\\GBDEX5\\Data\\hebei.csv'
data = pd.read_csv(file_path)

# 选择相关列
columns_of_interest = [
    '二氧化碳排放总量（万吨）',
    '人均二氧化碳排放量（吨/人）',
    '二氧化碳排放强度（吨/万元）',
    '地区生产总值(亿元)',
    '地区生产总值－第一产业(亿元)',
    '地区生产总值－第二产业(亿元)',
    '地区生产总值－第三产业(亿元)',
    '人均地区生产总值(元)',
    '社会消费品零售总额(亿元)',
    '全社会固定资产投资(亿元)',
    '年底人口数(万人)'
]

# 计算统计量
statistics = data[columns_of_interest].describe()

# 打印统计结果
print(statistics)
# 输出结果到CSV文件
output_file_path = 'C:\\Users\\32001\\Desktop\\work\\Python\\GBD\\GBDEX5\\Result\\hebei_statistics.csv'
statistics.to_csv(output_file_path, encoding='utf-8-sig')

print(f"统计结果已保存到 {output_file_path}")