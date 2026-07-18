# -*- coding: utf-8 -*-
"""
项目全局配置。

这里集中保存数据库连接信息、项目路径、运行设备和标签集合。
其他模块通过 import config 读取这些配置，避免在多个文件里重复写路径和标签。
"""

import torch
from pathlib import Path

# MySQL 连接配置。
# address_alignment.query_parent 会使用这个配置连接数据库，并查询 region 行政区划表。
# 实际部署时，密码这类敏感信息通常建议放到环境变量或单独的本地配置文件中。
MYSQL_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '020410',
    'database': 'test',
    'charset': 'utf8mb4',
    # 数据库没有启动或连接失败时，快速失败，避免 Web 请求长时间卡住。
    'connect_timeout': 3,
}

# 路径设置。
# BASE_DIR 是当前 config.py 所在目录，也就是项目根目录。
BASE_DIR = Path(__file__).parent

# 原始数据目录，通常用于保存还没有转换成训练格式的数据。
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'

# 当前项目已有的序列标注训练数据。
# 每行是一条 JSON 样本，格式为 {"text": [...], "labels": [...]}。
ADDRESS_TAGGING_RAW_DATA_PATH = RAW_DATA_DIR / 'data.jsonl'

# 预处理后的数据目录，当前项目里 Processor 会把 train/valid/test 保存到磁盘。
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'

# 本地预训练模型目录，例如 bert-base-chinese、roberta-small-wwm-chinese-cluecorpussmall。
PRETRAINED_DIR = BASE_DIR / 'pretrained'

# 当前项目实际使用的地址标注预训练模型目录。
ADDRESS_TAGGING_PRETRAINED_DIR = PRETRAINED_DIR / 'bert-base-chinese'

# 微调模型参数保存目录。
FINETUNE_DIR = BASE_DIR / 'finetune'

# 地址标注模型的默认微调参数路径。
ADDRESS_TAGGING_PARAMS_PATH = FINETUNE_DIR / 'address_tagging.pt'

# TensorBoard 日志目录。
LOGS_DIR = BASE_DIR / 'logs'

# 默认训练参数。main.py 支持通过命令行参数覆盖这些值。
MAX_SEQ_LEN = 96
BATCH_SIZE = 32
EPOCHS = 1

# 自动选择训练/推理设备：有 CUDA 就用 GPU，否则回退到 CPU。
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 地址标注任务的标签集合。
# 标签采用 BIOES 风格：
# - B: 一个多字实体的开始
# - I: 一个多字实体的中间
# - E: 一个多字实体的结束
# - S: 单字实体
# - O: 非地址实体
#
# 例如 “B-city/I-city/E-city” 表示一个城市名的开始、中间、结束。
# models_def.AddressTagging.predict 会把 “B-city” 这类标签裁掉前两个字符，得到 “city”。
#
# 这里的标签集合要和 data/raw/data.jsonl 中出现的标签保持一致。
# name/phone 是训练数据中的收件人和手机号，address_alignment.py 会在结构化地址时忽略它们。
LABELS = [
    "O",
    "B-name",
    "I-name",
    "E-name",
    "B-phone",
    "I-phone",
    "E-phone",
    "B-prov",
    "I-prov",
    "E-prov",
    "B-city",
    "I-city",
    "E-city",
    "B-district",
    "I-district",
    "S-district",
    "E-district",
    "B-town",
    "I-town",
    "E-town",
    "B-detail",
    "I-detail",
    "S-detail",
    "E-detail",
]
