import config
import pymysql
from models_def import AddressTagging, load_params

def address_alignment(text, model):
    tagging = model.predict(text)
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
                "devzone": 6
                 }
    address = {2 : None, 3: None, 4: None, 5: None, 6: None}
    start_pos = 0
    tag_len = len(tagging)
    for end_pos in range(tag_len):
        if(end_pos == tag_len-1) or (label_map[tagging[end_pos + 1]] != label_map[tagging[start_pos]]):
            if label_map [tagging[start_pos]] != 0:
                address[label_map[tagging[start_pos]]] = text[start_pos:end_pos + 1]
                start_pos = end_pos + 1

    for region_type_id in [2, 3, 4, 5]:
        if not address[region_type_id]:
            continue
        check_address(region_type_id, address)

    return {"省份":address[2],
            "城市":address[3],
            "区县":address[4],
            "街道":address[5],
            "详细地址":address[6]
            }

def check_address(region_type_id, address, parent_id=None):
    res =query_parent(region_type_id,address[region_type_id],
                      parent_id)
    if not res:
        address[region_type_id] = None
        return True
    if region_type_id== 2:
        address[region_type_id] =res["name"]
        return True
    if address[region_type_id-1] and address[region_type_id-1] ==res["parent_name"]:
        address[region_type_id] =res["name"]
        return True
    if address[region_type_id-1] and address[region_type_id-1] !=res["parent_name"]:
        address[region_type_id] = None
        return False
    if not address[region_type_id-1]:
        address[region_type_id] =res["name"]
        address[region_type_id-1] =res["parent_name"]
        done =check_address(region_type_id- 1, address, res["parent_id"])

    if done:
        return True
    address[region_type_id] = None
    address[region_type_id-1] = None
    return False

def query_parent(region_type_id,region_name,region_id=None):
    with pymysql.connect(**config.MYSQL_CONFIG) as mysql_conn:
        with mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql = (
            "select "
            "region_parent.id as parent_id,"
            "region_parent.name as parent_name,"
            "region.name as name "
            "from region "
            "join region as region_parent on region_parent.id=region.parent_id "
            "where region.region_type=%s and region.name like %s"
            )
        filter= (region_type_id,f"%{region_name}%")

        if region_id:
            sql += " and region.id=%s"
            filter= (region_type_id,f"%{region_name}%", region_id)
        cursor.execute(sql,filter)
        results = cursor.fetchall()
        res = None

        if results:
            for r in results:
                if region_name==r["name"]:
                    res = r
                    break
            else:
                res =results[0]
        return res

if __name__ == "__main__":
    model = AddressTagging(config.PRETRAINED_DIR/"roberta-small-wwm-chinese-cluecorpussmall", config.LABELS)
    load_params(model, config.FINETUNED_DIR/"address_tagging.pt")
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