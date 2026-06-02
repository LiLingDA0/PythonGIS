import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ClusterWays

# 读取数据并去重
df = pd.read_parquet('data/selectedDataPreP.parquet')
df = df.drop_duplicates(subset='app_id', keep='first')  # 确保app_id唯一


price_judge_KM =ClusterWays.KMCluster(data=df,
                                     StrID = 'app_id',
                                     StrCluster='original_price_lubi',
                                     Knum=4,
                                     label_str=['免费', '便宜', '一般', '昂贵'],
                                     ignoreV=0)
review_judege_KM = ClusterWays.KMCluster(data=df,
                                         StrID = 'app_id',
                                         StrCluster='overall_review_count',
                                         Knum=3,
                                         label_str=['无人问津', '关注较少', '关注较多'])


price_judge_Slipt = ClusterWays.SliptClusters(data=df,
                                              StrID='app_id',
                                              StrCluster='original_price_lubi',
                                              SplitValues=[1,1000,3000],
                                              label_str=['免费', '便宜', '一般', '昂贵'],
                                              )

review_judge_Slipt = ClusterWays.SliptClusters(data=df,
                                              StrID='app_id',
                                              StrCluster='overall_review_count',
                                              SplitValues=[10000,200000],
                                              label_str=['无人问津', '关注较少', '关注较多'],
                                              )

price_judge_Jenk_V3 = ClusterWays.JenkCluster(data=df,
                                              StrID='app_id',
                                              StrCluster='original_price_lubi',
                                              Knum=4,
                                              label_str=['免费', '便宜', '一般', '昂贵'],
                                              ignoreV=0,
                                              rootV=3
                                              )

review_judege_Jenk_V3 = ClusterWays.JenkCluster(data=df,
                                            StrID='app_id',
                                            StrCluster='overall_review_count',
                                             Knum=3,
                                             label_str=['无人问津', '关注较少', '关注较多'],
                                                rootV =3)

price_judge_KM_V3 =ClusterWays.KMCluster(data=df,
                                     StrID = 'app_id',
                                     StrCluster='original_price_lubi',
                                     Knum=4,
                                     label_str=['免费', '便宜', '一般', '昂贵'],
                                     ignoreV=0,
                                        rootV=3)
review_judege_KM_V3 = ClusterWays.KMCluster(data=df,
                                         StrID = 'app_id',
                                         StrCluster='overall_review_count',
                                         Knum=3,
                                         label_str=['无人问津', '关注较少', '关注较多'],
                                            rootV=3)

merged_df = df.merge(
    price_judge_KM.rename(columns={f'original_price_lubi_judge':'price_judge_KM'}),
    on='app_id',
    how='left'
).merge(
    review_judege_KM.rename(columns={f'overall_review_count_judge': 'review_judege_KM'}),
    on='app_id',
    how='left'
)
merged_df = merged_df.merge(
    price_judge_Jenk_V3.rename(columns={f'original_price_lubi_judge': 'price_judge_Jenk_V3'}),
    on='app_id',
    how='left'
).merge(
review_judege_Jenk_V3.rename(columns={f'overall_review_count_judge': 'review_judege_Jenk_V3'}),
    on='app_id',
    how='left'
)
merged_df = merged_df.merge(
    price_judge_Slipt.rename(columns={f'original_price_lubi_judge': 'price_judge_Slipt'}),
    on='app_id',
    how='left'
).merge(
review_judge_Slipt.rename(columns={f'overall_review_count_judge': 'review_judge_Slipt'}),
    on='app_id',
    how='left'
)

merged_df = merged_df.merge(
    price_judge_KM_V3.rename(columns={f'original_price_lubi_judge': 'price_judge_KM_V3'}),
    on='app_id',
    how='left'
).merge(
review_judege_KM_V3.rename(columns={f'overall_review_count_judge': 'review_judege_KM_V3'}),
    on='app_id',
)

merged_df.to_parquet('data/selectedDataJudge.parquet', engine='pyarrow')