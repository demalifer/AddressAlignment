"""
数据准备脚本占位文件。

当前项目的训练入口 main.py 默认直接读取：
    config.PRETRAINED_DIR / 'data.txt'

如果后续需要从原始地址数据生成模型训练文件，可以把这些逻辑放到这里，例如：
1. 读取原始地址样本。
2. 清洗异常字符、空行、重复样本。
3. 转换成“字 标签”的序列标注格式。
4. 保存为 AddressTaggingProcessor 可以读取的 data.txt。
"""
