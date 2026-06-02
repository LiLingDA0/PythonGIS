import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
import pandas as pd
from pyproj import CRS# 添加热力图
from folium.plugins import HeatMap


# 读取shp数据（WGS84）
sjz_map = gpd.read_file("shp/石家庄市/石家庄市.shp")
sjz_map = sjz_map.to_crs(epsg=4326)  # 保持WGS84坐标系

# 读取表格数据（CGCS2000转WGS84）
df = pd.read_parquet("data/0311_filter.parquet")
points_gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.lon, df.lat),
    crs=CRS.from_epsg(4490)  # CGCS2000
).to_crs(epsg=4326)  # 转换为WGS84

# 创建底图
m = folium.Map(location=[38.0428, 114.5149], zoom_start=10)

# 添加行政区划
folium.GeoJson(
    sjz_map,
    name='石家庄行政区划',
    style_function=lambda x: {'color':'blue','fillOpacity':0.1}
).add_to(m)

# 创建点聚类
marker_cluster = MarkerCluster().add_to(m)

# 添加带交互的标记点
for idx, row in points_gdf.iterrows():
    popup_html = f"""
    <b>标题:</b> {row['title']}<br>
    <b>类别:</b> {row['category_fst']}<br>
    <b>签到数:</b> {row['checkin_num']}<br>
    <b>照片数:</b> {row['photo_num']}
    """
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(popup_html, max_width=250),
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(marker_cluster)

heat_data = [[point.y, point.x, row['checkin_num']]
            for point, row in zip(points_gdf.geometry, points_gdf.to_dict('records'))]

HeatMap(heat_data, name='签到热力图').add_to(m)

# 添加图层控制
folium.LayerControl().add_to(m)
m.save('map.html')
