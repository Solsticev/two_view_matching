### two-view matching

#### 实验内容

1. 利用OpenCV进行SIFT特征提取
2. 实现对特征点的暴力匹配并展示匹配结果
3. 实现规范化八点法，利用特征匹配结果计算fundamental matrix
4. 利用fundamental matrix计算epipolar lines并可视化（使用ChatGPT3.5辅助实现）

#### 代码说明

* `read_and_show`用于读取两个视角的图片并进行灰度处理，可以设置`show`参数进行两张灰度图展示
* `feature_extraction`使用OpenCV基于SIFT对图片特征进行提取，返回关键点和对应的描述符
* `brute_force_match`用于对得到的图片特征进行匹配，基于关键点描述符的相似度进行匹配，可以设置`threshold`来对特征进行进一步筛选
* `show_matching`用于对匹配的结果进行可视化展示，默认展示200个匹配点
* `cal_fundamental_mtx`用于计算fundamental matrix，可以设置`method`来用不同计算方法进行计算，其中设置`method=8POINT`代表使用自己实现的规范化八点法进行计算，其它则是使用OpenCV中的库函数来实现（用于对比）
* `compute_epilines`和`show_epipolar_lines`分别用于基于基础矩阵计算epipolar lines和展示epipolarlines(基于ChatGPT3.5实现)

* 所有的结果均保存在res路径中

#### 结果展示

##### 特征匹配结果(只展示200对匹配)

<div align=center>
<img src=./res/building_matching.png width=70%>
</div>

<div align=center>
<img src=./res/Tempo_matching.png width=70%>
</div>

<div align=center>
<img src=./res/desk_matching.png width=70%>
</div>

##### 基于使用规范化八点法计算fundamental matrix的epipolar lines的结果

<div align=center>
<img src=./res/building_epipolar_lines_8POINT.png width=90%>
</div>

<div align=center>
<img src=./res/Tempo_epipolar_lines_8POINT.png width=90%>
</div>

<div align=center>
<img src=./res/desk_epipolar_lines_8POINT.png width=90%>
</div>

##### 基于使用Least-Median of Squares计算fundamental matrix的epipolar lines的结果

<div align=center>
<img src=./res/building_epipolar_lines_FM_LMEDS.png width=90%>
</div>

<div align=center>
<img src=./res/Tempo_epipolar_lines_FM_LMEDS.png width=90%>
</div>

<div align=center>
<img src=./res/desk_epipolar_lines_FM_LMEDS.png width=90%>
</div>

可见，使用不同的方法计算fundamental matrix确实会产生一定差异
