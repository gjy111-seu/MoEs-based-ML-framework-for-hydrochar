from pyecharts.charts import Map
from pyecharts import options as opts
import pandas as pd
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot


df = pd.read_excel("Data file path") #Energy or emission reduction data of each province in China
province_data = list(zip(df["province"], df["data"]))

map_chart = (
    Map()
    .add("distribution map", province_data, "china", is_map_symbol_show=False,
         label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Heat map of Chinese provinces"), visualmap_opts=opts.VisualMapOpts(
            is_piecewise=False,
            min_=0,
            max_=800,# setting according to data
            range_color=["#DBE7EA", "#588B99"]
        ),
    )
)
# map_chart.render("china_heatmap.html")
make_snapshot(snapshot, map_chart.render(), "china_map_2.png")
