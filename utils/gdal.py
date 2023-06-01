# -*- encoding: utf-8 -*-
'''
@Time       : 12/28/21 10:35 AM
@Author     : Jiang.xx
@Email      : cxyth@live.com
@Description: 遥感影像(tif)及矢量数据(shp)处理工具集
'''

import os
import sys
import numpy as np
import cv2 as cv
from skimage import transform
from osgeo import osr, ogr, gdal


def read_gdal(path):
    '''
        读取一个tiff图像
    :param path: 要读取的图像路径(包括后缀名)
    :type path: string
    :return im_data: 返回图像矩阵(h, w, c)
    :rtype im_data: numpy
    :return im_proj: 返回投影信息
    :rtype im_proj: ?
    :return im_geotrans: 返回坐标信息
    :rtype im_geotrans: ?
    '''
    image = gdal.Open(path, gdal.GA_ReadOnly)  # 打开该图像
    if image == None:
        print(path + "文件无法打开")
        return
    img_w = image.RasterXSize  # 栅格矩阵的列数
    img_h = image.RasterYSize  # 栅格矩阵的行数
    # im_bands = image.RasterCount  # 波段数
    im_proj = image.GetProjection()  # 获取投影信息
    im_geotrans = image.GetGeoTransform()  # 仿射矩阵
    im_data = image.ReadAsArray(0, 0, img_w, img_h)

    # 二值图一般是二维，需要添加一个维度
    if len(im_data.shape) == 2:
        im_data = im_data[np.newaxis, :, :]

    im_data = im_data.transpose((1, 2, 0))
    return im_data, im_proj, im_geotrans


def write_gdal(im_data, path, im_proj=None, im_geotrans=None, nodata=None):
    '''
        重新写一个tiff图像
    :param im_data: 图像矩阵(h, w, c)
    :type im_data: numpy
    :param im_proj: 要设置的投影信息(默认None)
    :type im_proj: ?
    :param im_geotrans: 要设置的坐标信息(默认None)
    :type im_geotrans: ?
    :param path: 生成的图像路径(包括后缀名)
    :type path: string
    :return: None
    :rtype: None
    '''
    im_data = im_data.transpose((2, 0, 1))
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    elif 'float32' in im_data.dtype.name:
        datatype = gdal.GDT_Float32
    else:
        datatype = gdal.GDT_Float64
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, im_width, im_height, im_bands, datatype)
    if (dataset != None):
        if im_geotrans == None or im_proj == None:
            pass
        else:
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
            dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        if nodata is not None:
            dataset.GetRasterBand(i + 1).SetNoDataValue(nodata)
    del dataset


# 写入shp文件,polygon
def writeShp():
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 属性表字段支持中文
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    # 注册驱动
    ogr.RegisterAll()
    # 创建shp数据
    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        return "驱动不可用："+strDriverName
    # 创建数据源
    oDS = oDriver.CreateDataSource("polygon.shp")
    if oDS == None:
        return "创建文件失败：polygon.shp"
    # 创建一个多边形图层，指定坐标系为WGS84
    papszLCO = []
    geosrs = osr.SpatialReference()
    geosrs.SetWellKnownGeogCS("WGS84")
    # 线：ogr_type = ogr.wkbLineString
    # 点：ogr_type = ogr.wkbPoint
    ogr_type = ogr.wkbPolygon
    # 面的类型为Polygon，线的类型为Polyline，点的类型为Point
    oLayer = oDS.CreateLayer("Polygon", geosrs, ogr_type, papszLCO)
    if oLayer == None:
        return "图层创建失败！"
    # 创建属性表
    # 创建id字段
    oId = ogr.FieldDefn("id", ogr.OFTInteger)
    oLayer.CreateField(oId, 1)
    # 创建name字段
    oName = ogr.FieldDefn("name", ogr.OFTString)
    oLayer.CreateField(oName, 1)
    oDefn = oLayer.GetLayerDefn()
    # 创建要素
    # 数据集
    # wkt_geom id name
    features = ['test0;POLYGON((-1.58 0.53, -0.79 0.55, -0.79 -0.23, -1.57 -0.25, -1.58 0.53))',
                'test1;POLYGON((-1.58 0.53, -0.79 0.55, -0.79 -0.23, -1.57 -0.25, -1.58 0.53))']
    for index, f in enumerate(features):
        oFeaturePolygon = ogr.Feature(oDefn)
        oFeaturePolygon.SetField("id",index)
        oFeaturePolygon.SetField("name",f.split(";")[0])
        geomPolygon = ogr.CreateGeometryFromWkt(f.split(";")[1])
        oFeaturePolygon.SetGeometry(geomPolygon)
        oLayer.CreateFeature(oFeaturePolygon)
    # 创建完成后，关闭进程
    oDS.Destroy()
    return "数据集创建完成！"

# 读shp文件
def readShp():
    # 支持中文路径
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    # 支持中文编码
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")
    # 注册所有的驱动
    ogr.RegisterAll()
    # 打开数据
    ds = ogr.Open("polygon.shp", 0)
    if ds == None:
        return "打开文件失败！"
    # 获取数据源中的图层个数，shp数据图层只有一个，gdb、dxf会有多个
    iLayerCount = ds.GetLayerCount()
    print("图层个数 = ", iLayerCount)
    # 获取第一个图层
    oLayer = ds.GetLayerByIndex(0)
    if oLayer == None:
        return "获取图层失败！"
    # 对图层进行初始化
    oLayer.ResetReading()
    # 输出图层中的要素个数
    num = oLayer.GetFeatureCount(0)
    print("要素个数 = ", num)
    result_list = []
    # 获取要素
    for i in range(0, num):
        ofeature = oLayer.GetFeature(i)
        id = ofeature.GetFieldAsString("id")
        name = ofeature.GetFieldAsString('name')
        geom = str(ofeature.GetGeometryRef())
        result_list.append([id,name,geom])
    ds.Destroy()
    del ds
    return result_list


# 通过tif文件直接获取转换参数
def get_TransformPara(path):
    '''
        通过tif文件直接获取转换参数
    :param path: tif路径
    :return: 变换参数
    '''
    # load the image
    image = gdal.Open(path)  # 打开该图像
    if image == None:
        print(path + "文件无法打开")
        return
    print(path + "文件打开了")

    '''左上角地理坐标'''
    adfGeoTransform = image.GetGeoTransform()
    # 打印左上角地理坐标
    print(adfGeoTransform[0])
    print(adfGeoTransform[3])
    return adfGeoTransform


# 打开shp文件
def open_shp(path):
    '''
        打开shp文件
    :param path: shp文件路径
    :return: shp文件
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')  # 载入驱动
    filename = path  # 不止需要.shp文件，还需要附带的其它信息文件
    dataSource = driver.Open(filename, 0)  # 第二个参数为0是只读，为1是可写
    if dataSource is None:  # 判断是否成功打开
        print('could not open')
        sys.exit(1)
    else:
        print('done!')
        return dataSource


def trans_shp_geo_to_xy(Geoshp, Geotif):
    '''
        地理坐标转像素坐标
    :param Geoshp: shp文件路径
    :param Geotif: tif文件路径
    :return: 转换好的像素坐标系坐标
    '''
    dataSource = open_shp(Geoshp)  # 打开shp文件
    layer = dataSource.GetLayer(0)  # 读取第一个图层

    '''读出上下左右边界，坐标系为地理坐标系'''
    extent = layer.GetExtent()
    print('extent:', extent)
    print('ul:', extent[0], extent[1])  # 左右边界
    print('lr:', extent[2], extent[3])  # 下上边界

    n = layer.GetFeatureCount()  # 该图层中有多少个要素
    print('feature count:', n)

    TransformPara = get_TransformPara(Geotif)  # 获取变换参数

    '''循环遍历所有的该图层中所有的要素'''
    arrRowcol = []  # 存储转换出的行列坐标的数组
    for i in range(n):
        feat = layer.GetNextFeature()  # 读取下一个要素
        geom = feat.GetGeometryRef()  # 提取该要素的轮廓坐标
        # print(i, ":")
        # print(geom)     # 输出的多边形轮廓坐标

        # 对多边形Geometry格式进行字符串裁剪，截取并放入geom_str的list中
        geom_replace = str(geom).replace('(', '')  # 首先需将Geometry格式转换为字符串
        geom_replace = geom_replace.replace(')', '')
        geom_replace = geom_replace.replace(' ', ',')
        geom_str = geom_replace.split(',')[1:]  # Geometry格式中首个字符串为POLYGON，需跳过，故从1开始

        # print(geom_str)  # 打印geom_str List
        geom_x = geom_str[0::2]  # 在list中输出经度坐标
        # print(geom_str[0::2])
        geom_y = geom_str[1::2]  # 在list中输出纬度坐标
        # print(geom_str[1::2])

        row = []  # 存储单个行列坐标数据，每次循环重置
        '''对每个坐标进行转换'''
        for j in range(len(geom_x)):
            dTemp = TransformPara[1] * TransformPara[5] - TransformPara[2] * TransformPara[4];
            Xpixel = (TransformPara[5] * (float(geom_x[j]) - TransformPara[0]) - TransformPara[2] * (
                    float(geom_y[j]) - TransformPara[3])) / dTemp + 0.5;
            Yline = (TransformPara[1] * (float(geom_y[j]) - TransformPara[3]) - TransformPara[4] * (
                    float(geom_x[j]) - TransformPara[0])) / dTemp + 0.5;

            col = [Xpixel, Yline]  # 接收行列坐标数据
            # print("col:",col)
            row.append(col)  # 放入row中
            # print("row:",row)
        arrRowcol.append(row)  # 将每一行的坐标放入存储像素坐标系的数组中

    # print("arrRowcol:",arrRowcol) # 打印转换好的像素坐标系坐标，三维数组
    return arrRowcol


def trans_shp_xy_to_geo(arrayXY, Geotif):
    '''
        像素坐标转地理坐标
    :param arrayXY: 像素坐标系的坐标
    :param Geotif: tif文件路径
    :return: 地理坐标系的坐标
    '''
    togeo = arrayXY  # 接收行列系坐标

    TransformPara = get_TransformPara(Geotif)  # 获取变换参数

    arrSlope = []  # 用于存储地理坐标
    for i in range(len(togeo)):
        row = []
        for j in range(len(togeo[i])):
            togeo_x = togeo[i][j][0]  # 在list中输出X坐标
            # print(togeo_x)
            togeo_y = togeo[i][j][1]  # 在list中输出Y坐标
            # print(togeo_y)

            '''转换公式'''
            px = TransformPara[0] + float(togeo_x) * TransformPara[1] + float(togeo_y) * TransformPara[2]
            py = TransformPara[3] + float(togeo_x) * TransformPara[4] + float(togeo_y) * TransformPara[5]
            col = [px, py]
            row.append(col)
        arrSlope.append(row)

    return arrSlope


# 打开shp文件
def read_shp(path, is_writable=0):
    '''
        打开矢量文件
    :param path: 矢量文件路径
    :type path: string
    :return: 矢量文件对象
    :rtype: shp object
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')  # 载入驱动
    filename = path  # 不止需要.shp文件，还需要附带的其它信息文件
    dataSource = driver.Open(filename, is_writable)  # 第二个参数为0是只读，为1是可写
    if dataSource is None:  # 判断是否成功打开
        raise Exception('could not open the shp file!')
    else:
        return dataSource


def shp2tif(inShp, asRaster, outRaster, outType:["men", "tif"], idField:str, noData=255):
    '''
        将shp文件栅格化，根据类别id字段赋予栅格值
    :param inShp: 输入的shape文件
    :param asRaster: 对应的底图
    :param outRaster: 输出文件名
    :param outType: ["mem", "tif"]之一，mem表示仅保存在内存，tif表示以文件形式保存到磁盘
    :param idField: 类别ID字段, 字段的属性值将为栅格值
    :param noData: 设定无效值
    :return: None
    '''
    raster = gdal.Open(asRaster, gdal.GA_ReadOnly)
    geoTransform = raster.GetGeoTransform()
    spatialRef = raster.GetSpatialRef()
    ncol = raster.RasterXSize
    nrow = raster.RasterYSize
    
    # Open the data source
    source_ds = ogr.Open(inShp)
    source_layer = source_ds.GetLayer()

    # Create the destination data source
    target_ds = gdal.GetDriverByName('MEM').Create('', ncol, nrow, 1, gdal.GDT_Byte)

    target_ds.SetGeoTransform(geoTransform)
    target_ds.SetProjection(spatialRef.ExportToWkt())
    band = target_ds.GetRasterBand(1)

    # Define NoData value of new raster
    band.SetNoDataValue(noData)
    
    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, options=[f"ATTRIBUTE={idField}"])

    if outType == "mem":
        # Read as array
        binary = band.ReadAsArray()
        del source_ds, target_ds
        return binary
    else:
        gdal.GetDriverByName('GTiff').CreateCopy(outRaster, target_ds)
        del source_ds, target_ds


def tif2shp(tif_path, shp_path, field_name='GRIDCODE', band_select=1):
    '''
        栅格图像转换成矢量图(tif转shp)
    :param tif_path: 栅格图像路径
    :type tif_path: string
    :param shp_path: 保存的矢量图路径
    :type shp_path: string
    :param field_name: 字段名
    :type field_name: string
    :param band_select: 选择波段数(一般二值图都是第1个波段)
    :type band_select: int
    :return: None
    :rtype: None
    '''
    # this allows GDAL to throw Python Exceptions
    gdal.UseExceptions()
    src_ds = gdal.Open(tif_path)
    if src_ds is None:
        print('Unable to open %s' % tif_path)
        exit()
    try:
        srcband = src_ds.GetRasterBand(band_select)
        maskband = srcband.GetMaskBand()
    except Exception as e:
        # for example, try GetRasterBand(10)
        print('Band  %i  not found' % band_select)
        print(e)
        exit()

    drv = ogr.GetDriverByName("ESRI Shapefile")
    # Remove output shapefile if it already exists
    if os.path.exists(shp_path):
        print('Remove output shapefile because of it already has existed')
        drv.DeleteDataSource(shp_path)

    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjectionRef())
    dst_ds = drv.CreateDataSource(shp_path)
    dst_layer = dst_ds.CreateLayer(shp_path, geom_type=ogr.wkbPolygon, srs=srs)

    gridcodeField = ogr.FieldDefn(field_name, ogr.OFTInteger)
    dst_layer.CreateField(gridcodeField)
    dst_field = 0

    gdal.Polygonize(srcband, maskband, dst_layer, dst_field, ['8CONNECTED=8'], callback=None)


def remapping_classid(inShp, outShp, field, map_dict:dict):
    '''
        将shp文件的类别id重新分配
    :param inShp: 输入的shp文件
    :param outShp: 保存的shp文件
    :param field: 类别id字段
    :param map_dict: 映射规则
    :return: None
    '''
    source_ds = ogr.Open(inShp)
    source_layer = source_ds.GetLayer()
    spatialRef = source_layer.GetSpatialRef()
    infeature = source_layer.GetFeature(0)
    fielddefn = infeature.GetFieldDefnRef(field)  # 需要提取的字段

    driver = ogr.GetDriverByName('ESRI Shapefile')
    target_ds = driver.CreateDataSource(outShp)
    name = os.path.split(outShp)[-1].rstrip('.shp')
    target_layer = target_ds.CreateLayer(name, spatialRef, geom_type=ogr.wkbMultiPolygon)
    target_layer.CreateField(fielddefn)
    featuredefn = target_layer.GetLayerDefn()

    for i in range(source_layer.GetFeatureCount()):
        feature = source_layer.GetFeature(i)
        attr = feature.GetField(field)
        new_attr = map_dict[attr] if attr in map_dict.keys() else attr
        newfeature = ogr.Feature(featuredefn)
        geom = feature.GetGeometryRef()
        newfeature.SetGeometry(geom)
        newfeature.SetField(field, new_attr)
        target_layer.CreateFeature(newfeature)
    del feature, newfeature
    del source_ds, target_ds


def resampling_by_scale(inRaster, outRaster, scale=1.0):
    '''
        栅格重采样,采样方法自选。(gdal.gdalconst.GRA_NearestNeighbour/GRA_Bilinear...
    :param inRaster:
    :param outRaster:
    :param scale:
    :return:
    '''
    assert scale > 0.
    src_ds = gdal.Open(inRaster, gdal.GA_ReadOnly)
    cols = src_ds.RasterXSize  # 列数
    rows = src_ds.RasterYSize  # 行数
    bands = src_ds.RasterCount  # 波段数
    dst_cols = int(cols * scale)    # 创建重采样后的栅格
    dst_rows = int(rows * scale)
    data_type = src_ds.GetRasterBand(1).DataType

    if os.path.exists(outRaster) and os.path.isfile(outRaster):
        os.remove(outRaster)      # 如果已存在同名文件则删除

    dst_ds = src_ds.GetDriver().Create(outRaster, xsize=dst_cols, ysize=dst_rows, bands=bands, eType=data_type)

    geoTransform = src_ds.GetGeoTransform()
    geoTransform = list(src_ds.GetGeoTransform())
    geoTransform[1] = geoTransform[1] / scale  # 像元宽度变为原来的scale倍
    geoTransform[5] = geoTransform[5] / scale  # 像元高度变为原来的scale倍
    dst_ds.SetGeoTransform(geoTransform)

    projection = src_ds.GetProjection()
    dst_ds.SetProjection(projection)

    # for i in range(bands):
    #     out_band = dst_ds.GetRasterBand(i + 1)
    #     out_band.SetNoDataValue(0)
    #     out_band.SetNoDataValue(src_ds.GetRasterBand(i + 1).GetNoDataValue())
    gdal.ReprojectImage(src_ds, dst_ds, projection, projection, gdal.gdalconst.GRA_Bilinear, 0.0, 0.0)


# def resampling_by_scale(inRaster, outRaster, scale=1.0):
#     '''
#         根据缩放比例对影像重采样
#     :param inRaster: 输入tif文件路径
#     :param outRaster: 保存的tif文件路径
#     :param scale: 缩放系数
#     :return:
#     '''
#     assert scale > 0.
#     src_ds = gdal.Open(inRaster, gdal.GA_ReadOnly)
#     cols = src_ds.RasterXSize  # 列数
#     rows = src_ds.RasterYSize  # 行数
#     bands = src_ds.RasterCount  # 波段数
#
#     dst_cols = int(cols * scale)  # 计算新的行列数
#     dst_rows = int(rows * scale)
#
#     geoTransform = list(src_ds.GetGeoTransform())
#     geoTransform[1] = geoTransform[1] / scale  # 像元宽度变为原来的scale倍
#     geoTransform[5] = geoTransform[5] / scale  # 像元高度变为原来的scale倍
#
#     if os.path.exists(outRaster) and os.path.isfile(outRaster):
#         os.remove(outRaster)      # 如果已存在同名文件则删除
#
#     data_type = src_ds.GetRasterBand(1).DataType
#     dst_ds = src_ds.GetDriver().Create(outRaster, xsize=dst_cols, ysize=dst_rows, bands=bands, eType=data_type)
#     dst_ds.SetProjection(src_ds.GetProjection())  # 设置投影坐标
#     dst_ds.SetGeoTransform(geoTransform)  # 设置地理变换参数
#
#     for i in range(bands):
#         data = src_ds.GetRasterBand(i+1).ReadAsArray(buf_xsize=dst_cols, buf_ysize=dst_rows)
#         out_band = dst_ds.GetRasterBand(i + 1)
#         # out_band.SetNoDataValue(0)
#         # out_band.SetNoDataValue(src_ds.GetRasterBand(i + 1).GetNoDataValue())
#         out_band.WriteArray(data)  # 写入数据到新影像中
#         out_band.FlushCache()
#         out_band.ComputeBandStats(False)  # 计算统计信息
#     del src_ds, dst_ds



def shp2geojson(shpPath, geojsonPath):
    ds = ogr.Open(shpPath, update=1)
    if ds == None:
        print("shp null")
        return
    dv = ogr.GetDriverByName("GeoJSON");
    if dv == None:
        print("GeoJSON create failed")
        return
    dv.CopyDataSource(ds, geojsonPath)


def world2Pixel(geoMatrix, x, y):
    """
    使用GDAL库的geomatrix对象((gdal.GetGeoTransform()))计算地理坐标的像素位置
    """
    ulx = geoMatrix[0]
    uly = geoMatrix[3]
    xDist = geoMatrix[1]
    yDist = geoMatrix[5]
    rtnX = geoMatrix[2]
    rtnY = geoMatrix[4]
    pixel = int((x - ulx) / xDist)
    line = int((uly - y) / abs(yDist))
    return (pixel, line)


def cutRegion(inShp, inRaster, outRaster):
    '''
        裁剪shp要素的最小外接矩形
    :param inShp: 输入的shp文件
    :param inRaster: 输入的底图
    :param outRaster: 裁剪后保存路径
    :return:
    '''
    # 读取要切的原图
    in_ds = gdal.Open(inRaster)
    geoTrans = in_ds.GetGeoTransform()
    # 读取原图中的每个波段
    in_band1 = in_ds.GetRasterBand(1)
    in_band2 = in_ds.GetRasterBand(2)
    in_band3 = in_ds.GetRasterBand(3)

    # 使用PyShp库打开shp文件
    r = shapefile.Reader(inShp)
    # 将图层扩展转换为图片像素坐标
    minX, minY, maxX, maxY = r.bbox
    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)
    # 计算新图片的尺寸
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)

    # 从每个波段中切需要的矩形框内的数据(注意读取的矩形框不能超过原图大小)
    out_band1 = in_band1.ReadAsArray(ulX, ulY, pxWidth, pxHeight)
    out_band2 = in_band2.ReadAsArray(ulX, ulY, pxWidth, pxHeight)
    out_band3 = in_band3.ReadAsArray(ulX, ulY, pxWidth, pxHeight)

    # 获取Tif的驱动，为创建切出来的图文件做准备
    gtif_driver = gdal.GetDriverByName("GTiff")

    # 创建切出来的要存的文件（3代表3个不都按，最后一个参数为数据类型，跟原文件一致）
    out_ds = gtif_driver.Create(outRaster, pxWidth, pxHeight, 3, in_band1.DataType)

    # 获取原图的原点坐标信息
    ori_transform = in_ds.GetGeoTransform()
    if ori_transform:
        print(ori_transform)
        print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
        print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))

    # 读取原图仿射变换参数值
    top_left_x = ori_transform[0]  # 左上角x坐标
    w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
    top_left_y = ori_transform[3]  # 左上角y坐标
    n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + ulX * w_e_pixel_resolution
    top_left_y = top_left_y + ulY * n_s_pixel_resolution

    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, ori_transform[1], ori_transform[2], top_left_y, ori_transform[4], ori_transform[5])

    # 设置裁剪出来图的原点坐标
    out_ds.SetGeoTransform(dst_transform)

    # 设置SRS属性（投影信息）
    out_ds.SetProjection(in_ds.GetProjection())

    # 写入目标文件
    out_ds.GetRasterBand(1).WriteArray(out_band1)
    out_ds.GetRasterBand(2).WriteArray(out_band2)
    out_ds.GetRasterBand(3).WriteArray(out_band3)

    # 将缓存写入磁盘
    out_ds.FlushCache()
    print("FlushCache succeed")
    # 计算统计值
    # for i in range(1, 3):
    #     out_ds.GetRasterBand(i).ComputeStatistics(False)
    # print("ComputeStatistics succeed")
    del out_ds


# def cutRegion(inShp, inRaster, outRaster):
#     '''
#         根据shape文件的矢量范围（外接矩形）裁剪底图
#     :param inShp: 输入shp文件路径
#     :param inRaster: 输入tif文件路径
#     :param outRaster: 保存的tif文件路径
#     :return:
#     '''
#     input_raster=gdal.Open(inRaster)
#     ds = gdal.Warp(outRaster,
#                     input_raster,
#                     format = 'GTiff',
#                     cutlineDSName = inShp,
#                     # cutlineWhere="FIELD = 'whatever'",
#                     dstNodata = 0)
#     # 关闭文件
#     ds = None


def shp_4_delete_gridcode(tmp_shp_path, pFieldName="GRIDCODE", pFieldValue=0):
    '''
        根据gridcode选择的FieldValue值删除矢量图中对应的要素(Tips：delete是基于原矢量图进行操作，故使用前记得备份原矢量图)
    :param tmp_shp_path: 要修改的矢量图路径
    :type tmp_shp_path: string
    :param pFieldName: 要删除的属性表字段，默认为"GRIDCODE"
    :type pFieldName: string
    :param pFieldValue: 默认为0，即删除背景(也可以使用字符串，但是要有对应的属性表字段和数值类型)
    :type pFieldValue: int
    :return: None
    :rtype: None
    '''
    pFeatureDataset = read_shp(tmp_shp_path, is_writable=1)
    pFeaturelayer = pFeatureDataset.GetLayer(0)

    if isinstance(pFieldValue, str):
        strFilter = pFieldName + " = '" + str(pFieldValue) + "'"
    else:
        strFilter = pFieldName + " = " + str(pFieldValue)
    pFeaturelayer.SetAttributeFilter(strFilter)

    for pFeature in pFeaturelayer:
        pFeatureFID = pFeature.GetFID()
        pFeaturelayer.DeleteFeature(int(pFeatureFID))
    strSQL = "REPACK " + str(pFeaturelayer.GetName())
    pFeatureDataset.ExecuteSQL(strSQL, None, "")
    pFeaturelayer = None
    pFeatureDataset = None

    # filter
    # ioShpFile = ogr.Open(outShp, update=1)
    # lyr = ioShpFile.GetLayerByIndex(0)
    # lyr.ResetReading()
    # for i in lyr:
    #     lyr.SetFeature(i)
    #     # if area is less than inMinSize or if it isn't forest, remove polygon
    #     if i.GetField('Class') != 1:
    #         lyr.DeleteFeature(i.GetFID())
    # ioShpFile.Destroy()


def uint16_to_8(im_data, lower_percent=0.001, higher_percent=99.999, per_channel=True):
    '''
        将uint 16bit转换成uint 8bit (压缩法)
    :param im_data: 图像矩阵(h, w, c)
    :type im_data: numpy
    :param lower_percent: np.percentile的最低百分位
    :type lower_percent: float
    :param higher_percent: np.percentile的最高百分位
    :type higher_percent: float
    :return: 返回图像矩阵(h, w, c)
    :rtype: numpy
    '''
    if per_channel:
        out = np.zeros_like(im_data, dtype=np.uint8)
        for i in range(im_data.shape[2]):
            a = 0  # np.min(band)
            b = 255  # np.max(band)
            c = np.percentile(im_data[:, :, i], lower_percent)
            d = np.percentile(im_data[:, :, i], higher_percent)
            if (d - c) == 0:
                out[:, :, i] = im_data[:, :, i]
            else:
                t = a + (im_data[:, :, i] - c) * (b - a) / (d - c)
                t = np.clip(t, a, b)
                out[:, :, i] = t
    else:
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(im_data, lower_percent)
        d = np.percentile(im_data, higher_percent)
        out = a + (im_data - c) * (b - a) / (d - c)
        out = np.clip(out, a, b).astype(np.uint8)
    return out


def single_set_proj_trans(ori_path, target_path):
    '''
        为 target_path 影像设置 ori_path 影像的投影、坐标信息
    :param ori_path: 获取 ori_path 影像路径
    :type ori_path: string
    :param target_path: 获取 target_path 影像路径
    :type target_path: string
    :return: None
    :rtype: None
    '''
    # 原图像导入
    _, im_proj, im_geotrans = read_gdal(ori_path)
    # 目标二值图导入
    im_data, _, _ = read_gdal(target_path)
    # 地理信息写入
    write_gdal(im_data, target_path, im_proj, im_geotrans)


def set_proj_trans(GET_DIR, OUT_DIR, in_endwiths):
    '''
        给 OUT_DIR 文件夹下的影像 设置 GET_DIR 文件夹下的影像的投影、坐标信息
    :param GET_DIR: 获取该文件夹下影像的投影、坐标信息
    :type GET_DIR: string
    :param OUT_DIR: 设置该文件夹下影像的投影、坐标信息
    :type OUT_DIR: string
    :param in_endwiths: 输入影像的格式（后缀名）
    :type in_endwiths: string
    :return: None
    :rtype: None
    '''
    endwiths = ['tif', 'tiff']
    if in_endwiths in endwiths:
        pic = get_filelist(GET_DIR, in_endwiths)
        for i in range(len(pic)):
            # os.path.join(GET_DIR, pic[i])
            # os.path.join(OUT_DIR, pic[i])
            get_path = GET_DIR + '/' + pic[i]
            target_path = OUT_DIR + '/' + pic[i]
            single_set_proj_trans(get_path, target_path)
    else:
        print('该功能仅支持.tif或.tiff格式')



def get_array_from_polygons(inShape, inRaster):
    '''
        根据shp文件内的面要素（如矩形区域）对栅格影像进行切割（返回array矩阵，不带地理信息）
    :param inShape: 输入的shp文件路径
    :param inRaster: 输入的栅格文件路径
    :return: array
    '''
    shpdata = GeoDataFrame.from_file(inShape)
    rasterdata = rio.open(inRaster)
    # 投影变换，使矢量数据与栅格数据投影参数一致
    shpdata = shpdata.to_crs(rasterdata.crs)

    arrays = []
    for i in range(0, len(shpdata)):
        # 获取矢量数据的features
        geo = shpdata.geometry[i]
        feature = [geo.__geo_interface__]
        # 通过feature裁剪栅格影像
        out_image, out_transform = rio.mask.mask(rasterdata, feature, all_touched=True, crop=True,
                                                 nodata=rasterdata.nodata)
        out_image = out_image.transpose((1, 2, 0))
        arrays.append(out_image)
    return arrays


def rotate_gdal(x, angle=0):
    '''
        旋转图像 (h*w*c) ps:需要注意输入输出的位数
    :param x: 图像矩阵
    :type x: numpy
    :param angle: 旋转角度
    :type angle: int
    :return: 旋转后图像矩阵
    :rtype: numpy
    '''
    re = 0  # 用于记录变换后像素值恢复乘数
    tmp_type = x.dtype  # 暂存矩阵输入时的位数
    if tmp_type == 'uint8':
        re = 255
    elif tmp_type == 'uint16':
        re = 511
    elif tmp_type == 'float32':
        re = 1023
    elif tmp_type == 'float64':
        re = 2047
    dst = transform.rotate(x, angle=angle)
    dst = dst * re
    dst = np.array(dst, dtype=tmp_type)
    return dst


def resize_gdal(x, shape=None, order=1, mode='reflect'):
    '''
        缩放图像 (h*w*c) ps:需要注意输入输出的位数
    :param x: 图像矩阵
    :type x: numpy
    :param shape: 缩放后形状 (h,w)
    :type shape: tuple
    :param order: 插值方法
         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic
    :type order: int
    :param mode: 填充模式{'constant', 'edge', 'symmetric', 'reflect', 'wrap'}
    :type mode: string
    :return: 缩放后图像矩阵
    :rtype: numpy
    '''
    re = 0  # 用于记录变换后像素值恢复乘数
    tmp_type = x.dtype  # 暂存矩阵输入时的位数
    if tmp_type == 'uint8':
        re = 255
    elif tmp_type == 'uint16':
        re = 511
    elif tmp_type == 'float32':
        re = 1023
    elif tmp_type == 'float64':
        re = 2047
    dst = transform.resize(x, output_shape=shape, order=order, mode=mode)
    dst = dst * re
    dst = np.array(dst, dtype=tmp_type)
    return dst


def RasterToPoly(rasterName, shpName):
    """
        栅格转矢量
        :param rasterName: 输入分类后的栅格名称
        :param shpName: 输出矢量名称
        :return:
   """

    def deleteBackground(shpName, backGroundValue):
        """
        删除背景,一般背景的像素值为0
        """
        driver = ogr.GetDriverByName('ESRI Shapefile')
        pFeatureDataset = driver.Open(shpName, 1)
        pFeaturelayer = pFeatureDataset.GetLayer(0)
        strValue = backGroundValue

        strFilter = "Value = '" + str(strValue) + "'"
        pFeaturelayer.SetAttributeFilter(strFilter)
        # pFeatureDef = pFeaturelayer.GetLayerDefn()
        # pLayerName = pFeaturelayer.GetName()
        # pFieldName = "Value"
        # pFieldIndex = pFeatureDef.GetFieldIndex(pFieldName)

        for pFeature in pFeaturelayer:
            pFeatureFID = pFeature.GetFID()
            pFeaturelayer.DeleteFeature(int(pFeatureFID))

        strSQL = "REPACK " + str(pFeaturelayer.GetName())
        pFeatureDataset.ExecuteSQL(strSQL, None, "")
        del pFeaturelayer
        del pFeatureDataset
        return

    inraster = gdal.Open(rasterName)  # 读取路径中的栅格数据
    inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段，如果是单波段数据的话那就都是1
    prj = osr.SpatialReference()
    prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息，用来为后面生成的矢量做准备

    outshp = shpName
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outshp):  # 若文件已经存在，则删除它继续重新做一遍
        drv.DeleteDataSource(outshp)
    Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
    Poly_layer = Polygon.CreateLayer(shpName[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)  # 对shp文件创建一个图层，定义为多个面类
    newField = ogr.FieldDefn('Value', ogr.OFTReal)  # 给目标shp文件添加一个字段，用来存储原始栅格的pixel value
    Poly_layer.CreateField(newField)

    gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数，执行的就是栅格转矢量操作
    Polygon.SyncToDisk()
    del Polygon

    deleteBackground(shpName, 0)  # 删除背景


if __name__ == '__main__':
    # to solve the problem of 'ERROR 1: PROJ: pj_obj_create: Open of /opt/conda/share/proj failed'
    # os.environ['PROJ_LIB'] = '/opt/conda/share/proj'
    os.environ['PROJ_LIB'] = r'C:\Users\AI\anaconda3\envs\torch17\Library\share\proj'

    pass
    # im_data, _, _ = read_gdal(r'C:\Users\obt_ai05\Desktop\dog_2.jpg')
    # print(im_data.shape)
    # # write_gdal(im_data, r'C:\Users\obt_ai05\Desktop\dog_3.jpg')
    #
    # im_data = np.transpose(im_data, (1, 2, 0))  # (h,w,c)
    # print(im_data.shape)
    #
    # # 记录下底层图像以rgb形式
    # # 0:3 mean 1,2,3 bands
    # # 1:4 mean 2,3,4 bands
    # # check!!!!!!!!!!!!!!!!!!!
    # img2 = im_data[:, :, 0:3]
    # print(img2.shape)
    # out = img2[:, :, ::1]  # rgb->bgr
    # print(out.shape)
    a = 'E:\\Jiang\\changedetection\\datasets\\foshan-bigmap\\daliang_tta\\bin.tif'
    b = 'E:\\Jiang\\changedetection\\datasets\\foshan-bigmap\\daliang_tta\\bin1.shp'
    RasterToPoly(a, b)



