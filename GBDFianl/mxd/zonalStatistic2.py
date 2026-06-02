
# -*- coding: utf-8 -*-
>>> 
>>> # -*- coding: utf-8 -*-
... import arcpy
... import os
... 
... # 0. 参数区 —— 按需改
... arcpy.env.overwriteOutput = True
... arcpy.env.workspace = r"C:\Users\32001\Desktop\work\Class50118LLH\GISAppDA\EX3\Temp50118LLH.gdb"     # 工作空间
... zone_fc = r"服务区"                             # 要素类/图层
... zone_id = "FID"                                 # 唯一字段
... pop_ras = r"修正老年人口分布"                   # 栅格
... out_csv = r"C:\Users\32001\Desktop\work\Class50118LLH\GISAppDA\EX3\Temp50118LLH.gdb\ZoneSum.csv"     # 输出 CSV
... 
... # 1. 确保能调用 Spatial Analyst 许可
... arcpy.CheckOutExtension("Spatial")
... 
... # 2. 准备空列表存结果
... result = [["FID", "SUM"]]
... 
... # 3. 开始逐要素循环
... with arcpy.da.SearchCursor(zone_fc, ["OID@", "SHAPE@"]) as cur:
...     for oid, geom in cur:
...         # 3.1 选出当前要素
...         arcpy.SelectLayerByAttribute_management(zone_fc, "NEW_SELECTION",
...                                                 "{0} = {1}".format(zone_id, oid))
...         # 3.2 临时保存到 in_memory（省磁盘 IO）
...         tmp_fc = r"in_memory\zone_" + str(oid)
...         arcpy.CopyFeatures_management(zone_fc, tmp_fc)
... 
...         # 3.3 区域统计——只取 SUM
...         #     返回的是栅格属性对象，.maximum 就是总和（单值）
...         sum_val = arcpy.sa.ZonalStatistics(tmp_fc, zone_id, pop_ras, "SUM", "NODATA")
...         sum_value = float(sum_val.maximum)   # 单值栅格，maximum 即总和
... 
...         # 3.4 记录
...         result.append([oid, sum_value])
... 
...         # 3.5 清内存
...         arcpy.Delete_management(tmp_fc)
...         arcpy.SelectLayerByAttribute_management(zone_fc, "CLEAR_SELECTION")
... 
... # 4. 写出 CSV（ArcMap 自带纯文本即可读）
... import csv
... with open(out_csv, 'wb') as f:
...     writer = csv.writer(f)
...     writer.writerows(result)
... 
... print("逐要素统计完成，结果见：{}".format(out_csv))
