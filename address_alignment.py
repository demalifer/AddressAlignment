# -*- coding: utf-8 -*-
"""
地址对齐/标准化入口。

这个文件做的事情可以拆成三步：
1. 调用地址标注模型，对输入地址的每个字预测一个标签，例如 prov、city、district。
2. 把连续相同类别的字拼成地址片段，并归并为固定层级：省、市、区县、街道、详细地址。
3. 查询 MySQL 中的 region 行政区划表，校验并补全省/市/区县/街道之间的父子关系。

注意：这里的“对齐”主要指把模型识别出来的非标准地名，对齐到数据库中的标准行政区名称。
"""

import config
import pymysql
from models_def import AddressTagging, load_params


def address_alignment(text, model):
    """
    将一条原始地址文本拆分并标准化为结构化地址。

    参数:
        text: 原始地址字符串，例如“浙江省杭州市余杭区葛墩路27号楼”。
        model: 已加载参数的 AddressTagging 模型实例，用来预测每个字的地址成分标签。

    返回:
        dict: 包含“省份、城市、区县、街道、详细地址”的结构化结果。

    整体流程:
        1. 模型输出逐字标签。
        2. label_map 把细粒度标签映射为系统内部使用的地址层级编号。
        3. 扫描标签序列，将连续属于同一层级的文本片段写入 address。
        4. 调用 check_address，用数据库中的行政区划关系校正省市区街道。
    """
    # tagging 是模型对 text 的逐字预测结果。理想情况下，它的长度应与 text 的字符数一致。
    # 例如 text 中“浙江省”的三个字可能都被预测为 prov。
    if not text:
        return {"省份": None, "城市": None, "区县": None, "街道": None, "详细地址": None}

    tagging = model.predict(text)

    # 将模型输出的标签名称归并到本文件使用的地址层级编号。
    # 0: 非地址成分/忽略
    # 2: 省级
    # 3: 市级
    # 4: 区县级
    # 5: 街道/乡镇/道路等较粗的基层位置
    # 6: 门牌号、小区、POI、楼层、辅助信息等详细地址
    label_map = {"": 0,
                "prov": 2,
                "city": 3,
                "district": 4,
                "road": 5,
                "intersection": 5,
                "town": 5,
                "roadno": 6,
                "cellno": 6,
                "community": 6,
                "houseno": 6,
                "poi": 6,
                "subpoi": 6,
                "assist": 6,
                "distance": 6,
                "village_group": 6,
                "floorno": 6,
                "devzone": 6,
                "detail": 6,
                "name": 0,
                "phone": 0
                 }

    # address 使用数字 key 保存中间结果，和 label_map 的层级编号对应。
    # 后面返回给接口前，再转换成中文字段名。
    address = {2 : None, 3: None, 4: None, 5: None, 6: None}

    # start_pos 记录当前连续标签片段的起始位置。
    # end_pos 会从左到右扫描 tagging，一旦发现“下一个字的地址层级变了”，
    # 就把 start_pos 到 end_pos 之间的文本切出来，作为一个地址片段。
    start_pos = 0
    tag_len = len(tagging)
    for end_pos in range(tag_len):
        # 到达最后一个字符，或下一个字符的层级与当前片段起点的层级不同时，
        # 当前片段结束。
        current_label = label_map.get(tagging[start_pos], 0)
        if(end_pos == tag_len-1) or (label_map.get(tagging[end_pos + 1], 0) != current_label):
            # 层级为 0 的片段不是有效地址成分，不写入 address。
            if current_label != 0:
                address[current_label] = text[start_pos:end_pos + 1]
            start_pos = end_pos + 1

    # 对省、市、区县、街道逐级做数据库校验。
    # 详细地址通常不是行政区划表中的标准区域，所以不参与 check_address。
    original_address = address.copy()
    try:
        for region_type_id in [2, 3, 4, 5]:
            if not address[region_type_id]:
                continue
            check_address(region_type_id, address)
    except pymysql.MySQLError as exc:
        # 数据库不可用时不要让 Web 接口直接 500。
        # 这里回退为模型识别出的原始地址片段，只是跳过行政区划标准化。
        print(f"行政区划数据库校验失败，保留模型识别结果: {exc}")
        address = original_address

    # 对外返回更容易理解的中文字段名。
    return {"省份":address[2],
            "城市":address[3],
            "区县":address[4],
            "街道":address[5],
            "详细地址":address[6]
            }


def check_address(region_type_id, address, parent_id=None):
    """
    校验并修正某一级行政区划。

    参数:
        region_type_id: 当前要校验的行政区层级。
            2=省，3=市，4=区县，5=街道/乡镇。
        address: address_alignment 中构造的地址字典，会在本函数内被原地修改。
        parent_id: 可选的父级区域 id。递归校验父级时用它限定查询范围。

    返回:
        bool:
            True 表示当前层级可以保留，或数据库中没有查到时已清空该层级。
            False 表示当前层级与已知父级冲突，需要调用方回滚相关层级。

    这个函数的核心作用:
        - 把模型识别出的地名替换为数据库中的标准名称。
        - 如果缺少父级，例如只有“杭州市”，可以从数据库反推并补上“浙江省”。
        - 如果父子关系冲突，例如地址里写的省和数据库中该市的父级不一致，则清空冲突层级。
    """
    # 查询数据库，找到当前层级名称对应的标准区域，以及它的父级区域。
    res =query_parent(region_type_id,address[region_type_id],
                      parent_id)

    # 数据库查不到该区域时，认为模型识别结果不可靠，直接清空这一层。
    if not res:
        address[region_type_id] = None
        return True

    # 省级没有更高的父级需要校验，查到后直接替换成标准名称。
    if region_type_id== 2:
        address[region_type_id] =res["name"]
        return True

    # 如果上一级已经存在，并且与数据库中的父级名称一致，说明父子关系合法。
    if address[region_type_id-1] and address[region_type_id-1] ==res["parent_name"]:
        address[region_type_id] =res["name"]
        return True

    # 如果上一级已经存在，但和数据库中的父级不一致，说明当前层级和父级冲突。
    # 例如识别出“广东省杭州市”，但数据库中“杭州市”的父级是“浙江省”。
    if address[region_type_id-1] and address[region_type_id-1] !=res["parent_name"]:
        address[region_type_id] = None
        return False

    # 如果上一级缺失，则先用数据库中的 parent_name 补齐父级，
    # 再递归校验补出来的父级是否也能在数据库中对齐。
    if not address[region_type_id-1]:
        address[region_type_id] =res["name"]
        address[region_type_id-1] =res["parent_name"]
        done =check_address(region_type_id- 1, address, res["parent_id"])

    # 递归校验成功，保留当前层级和补齐的父级。
    if done:
        return True

    # 递归校验失败，说明补齐出来的父级也无法成立；
    # 回滚当前层级和刚才补齐的父级，避免返回错误的行政区划。
    address[region_type_id] = None
    address[region_type_id-1] = None
    return False


def query_parent(region_type_id,region_name,region_id=None):
    """
    从 region 表中查询某个区域及其父级区域。

    参数:
        region_type_id: 要查询的区域类型，和 check_address 中的层级编号一致。
        region_name: 模型识别出的区域名称，可以是完整名称，也可以是名称片段。
        region_id: 可选区域 id。递归校验父级时用于把查询限定到指定父级记录。

    返回:
        dict | None:
            查到时返回包含 parent_id、parent_name、name 的字典；
            查不到时返回 None。

    数据库表假设:
        region 表至少包含 id、parent_id、name、region_type 字段。
        region.parent_id 指向父级区域的 id。
    """
    with pymysql.connect(**config.MYSQL_CONFIG) as mysql_conn:
        with mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 自连接 region 表：
            # - region 表示当前要查询的区域；
            # - region_parent 表示它的父级区域。
            sql = (
                "select "
                "region_parent.id as parent_id,"
                "region_parent.name as parent_name,"
                "region.name as name "
                "from region "
                "left join region as region_parent on region_parent.id=region.parent_id "
                "where region.region_type=%s and region.name like %s"
            )

            # 默认使用 like 模糊查询，允许模型只识别出地名的一部分。
            params = (region_type_id, f"%{region_name}%")

            # 如果传入 region_id，就进一步限制当前 region.id。
            # 这里主要服务于 check_address 的递归场景，用数据库 id 校验父级链条。
            if region_id:
                sql += " and region.id=%s"
                params = (region_type_id, f"%{region_name}%", region_id)

            cursor.execute(sql, params)
            results = cursor.fetchall()
            res = None

            # 多条候选结果中，优先选择名称完全一致的记录；
            # 如果没有完全一致的记录，则退而使用第一条模糊匹配结果。
            if results:
                for r in results:
                    if region_name==r["name"]:
                        res = r
                        break
                else:
                    res =results[0]
            return res


if __name__ == "__main__":
    # 直接运行本文件时执行一个简单的本地测试：
    # 1. 创建模型结构；
    # 2. 加载微调后的参数；
    # 3. 对若干示例地址做结构化解析并打印结果。
    model = AddressTagging(config.ADDRESS_TAGGING_PRETRAINED_DIR, config.LABELS)
    load_params(model, config.ADDRESS_TAGGING_PARAMS_PATH)
    text = [
        "中国浙江省杭州市余杭区葛墩路27号楼",
        "北京市通州区永乐店镇27号楼",
        "北京市市辖区高地街道27号楼",
        "新疆维吾尔自治区划阿拉尔市金杨镇27号楼",
        "甘肃省南市文县碧口镇27号楼",
        "陕西省渭南市华阴市罗镇27号楼",
        "西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        "广州市花都区花东镇27号楼",
    ]
    for i in text:
        print(address_alignment(i, model))
