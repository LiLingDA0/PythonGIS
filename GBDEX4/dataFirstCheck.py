import pandas as pd

# 读取CSV文件
df1 = pd.read_csv('ChineAirQuality1.csv')
df2 = pd.read_csv('ChinaAirQuality2.csv')

# 检测并转换日期格式
df1['Date'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d')
df2['Date'] = pd.to_datetime(df2['Date'], format='%Y-%m-%d')

# 合并数据
merged_df = pd.concat([df1, df2], ignore_index=True)

# 检查数据类型
merged_df['Ctnb'] = merged_df['Ctnb'].astype(str)
merged_df['Ctn'] = merged_df['Ctn'].astype(str)
merged_df['Prvn'] = merged_df['Prvn'].astype(str)
merged_df['AQIind'] = merged_df['AQIind'].astype(int)
merged_df['Qltlv'] = merged_df['Qltlv'].astype(str)
merged_df['AQIrnk'] = merged_df['AQIrnk'].astype(int)
merged_df['24hPM25avg'] = merged_df['24hPM25avg'].astype(int)
merged_df['24hPM10avg'] = merged_df['24hPM10avg'].astype(int)
merged_df['24hSO2avg'] = merged_df['24hSO2avg'].astype(int)
merged_df['24hNO2avg'] = merged_df['24hNO2avg'].astype(int)
merged_df['24hCOavg'] = merged_df['24hCOavg'].astype(float)
merged_df['24hO3avg'] = merged_df['24hO3avg'].astype(int)

# 检查异常数据
# 检查日期范围是否合理
min_date = pd.Timestamp('2000-01-01')  # 假设最小日期为2000-01-01
max_date = pd.Timestamp('2050-12-31')  # 假设最大日期为2050-12-31
invalid_dates = merged_df[(merged_df['Date'] < min_date) | (merged_df['Date'] > max_date)]
if not invalid_dates.empty:
    print("发现无效日期：")
    print(invalid_dates)

# 检查AQIind是否为正整数
invalid_aqiind = merged_df[merged_df['AQIind'] < 0]
if not invalid_aqiind.empty:
    print("发现无效AQIind值：")
    print(invalid_aqiind)

# 检查AQIrnk是否为正整数
invalid_aqirnk = merged_df[merged_df['AQIrnk'] <= 0]
if not invalid_aqirnk.empty:
    print("发现无效AQIrnk值：")
    print(invalid_aqirnk)

# 检查24小时平均值是否为非负数
pollutants = ['24hPM25avg', '24hPM10avg', '24hSO2avg', '24hNO2avg', '24hCOavg', '24hO3avg']
for pollutant in pollutants:
    invalid_values = merged_df[merged_df[pollutant] < 0]
    if not invalid_values.empty:
        print(f"发现{pollutant}的无效值：")
        print(invalid_values)
    # 将负值替换为0
    merged_df[pollutant] = merged_df[pollutant].clip(lower=0)


# 检查Qltlv是否为预期的字符串值
expected_quality_levels = ['优', '良', '轻度污染', '中度污染', '重度污染', '严重污染']
invalid_quality_levels = merged_df[~merged_df['Qltlv'].isin(expected_quality_levels)]
if not invalid_quality_levels.empty:
    print("发现无效Qltlv值：")
    print(invalid_quality_levels)

# 新建PM25Class字段
merged_df['PM25Class'] = merged_df['24hPM25avg'].apply(lambda x: '空气污染' if x > 100 else '空气无污染')

merged_df.to_parquet("Data/ChinaAirQuality.parquet",engine="pyarrow")

shijiazhuang_df = merged_df[merged_df['Ctn'] == '石家庄市']
# 保存为Parquet文件
shijiazhuang_df.to_parquet("Data/SJZAirQuality.parquet", engine="pyarrow")