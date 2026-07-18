# -*- coding: utf-8 -*-
"""
训练、验证、测试和简单推理的脚本入口。

这个文件把模型、数据处理器和训练器串起来：
1. 创建 AddressTagging 模型。
2. 用 AddressTaggingProcessor 构造 DataLoader。
3. 用 AddressTaggingTrainer 执行训练、验证和测试。
4. 可选地对 text 中的示例地址做推理。
"""

import config
import argparse
from datetime import datetime
from train import AddressTaggingTrainer
from preprocess import AddressTaggingProcessor
from models_def import AddressTagging, load_params
from torch.utils.tensorboard.writer import SummaryWriter

# 训练和推理使用的设备，以及优化器学习率。
device = config.DEVICE
learning_rate = 1e-5


def model_go(train=None, test=None, inference=None, model_params_path=None, epochs=None, batch_size=None, max_examples=None):
    """
    统一控制训练、测试和推理流程。

    参数:
        train: 为真时执行训练，并在验证集 f1 提升时保存模型参数。
        test: 为真时执行测试集评估。
        inference: 为真时对下方 text 示例列表做预测。
        model_params_path: 预训练/微调参数路径；会先尝试加载，再继续后续流程。
        epochs: 训练轮数；不传时使用 config.EPOCHS。
        batch_size: DataLoader 批大小；不传时使用 config.BATCH_SIZE。
        max_examples: 每个 split 最多抽取多少条样本；用于快速烟测。
    """

    epochs = epochs or config.EPOCHS
    batch_size = batch_size or config.BATCH_SIZE

    # 训练、日志、预处理数据目录都在这里确保存在，避免第一次运行时报目录不存在。
    config.FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 创建地址标注模型。模型结构由 BERT + dropout + 线性分类层组成。
    model = AddressTagging(
        config.ADDRESS_TAGGING_PRETRAINED_DIR,
        config.LABELS,
        max_seq_len=config.MAX_SEQ_LEN,
    )

    # 数据处理器负责读取 data/raw/data.jsonl、切分 train/valid/test，并产出 DataLoader。
    processor = AddressTaggingProcessor(
        config.ADDRESS_TAGGING_RAW_DATA_PATH,
        save_dir=config.PROCESSED_DATA_DIR,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        label_list=config.LABELS,
        max_seq_len=config.MAX_SEQ_LEN,
    )

    # 训练器封装了训练循环、验证指标计算和模型保存逻辑。
    trainer = AddressTaggingTrainer(model, device, epochs, learning_rate)

    # writer 只在训练时创建，用于记录 TensorBoard 日志。
    writer = None

    # 每次训练用当前时间生成一个保存名，避免覆盖旧模型。
    save_name = f'address_tagging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'

    # 如果传入 model_params_path，就先加载已有参数；加载失败时 load_params 会使用默认参数。
    load_params(model, model_params_path)

    if train:
        writer = SummaryWriter(config.LOGS_DIR / save_name)

        # 训练阶段需要 train 和 valid 两个 DataLoader。
        dataloader = {
            'train': processor.get_dataloader('train', max_examples=max_examples),
            'valid': processor.get_dataloader('valid', max_examples=max_examples),
        }

        # 最优模型保存到 app.py 默认加载的位置。
        # 这样训练完成后，重启 Web 服务就能直接使用最新微调参数。
        model_params_path = config.ADDRESS_TAGGING_PARAMS_PATH
        trainer(dataloader, model_params_path, writer)

    if test:
        # 测试阶段只需要 test DataLoader，Trainer 会输出 loss/precision/recall/f1。
        test_dataloader = processor.get_dataloader('test', max_examples=max_examples)
        trainer({'test': test_dataloader}, writer=writer, is_test=True)

    if writer:
        # 训练结束后关闭 TensorBoard writer，确保日志写入完成。
        writer.close()

    if inference:
        # 对下面定义的示例地址做推理。
        # predict 对多条地址会返回二维列表：每条地址对应一组逐字标签。
        res = model.predict(text, device)
        for a_text, a_res in zip(text, res):
            for t, r in zip(a_text, a_res):
                print(f'{t}-{r}', end='\t')
            print('\n')


# 用于本地训练/测试/推理的示例地址。
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

def parse_args():
    """解析命令行参数，让训练脚本既能完整训练，也能快速烟测。"""

    parser = argparse.ArgumentParser(description='地址标注模型训练/测试/推理入口')
    parser.add_argument('--train', action='store_true', help='执行训练')
    parser.add_argument('--test', action='store_true', help='执行测试集评估')
    parser.add_argument('--inference', action='store_true', help='执行示例地址推理')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='DataLoader 批大小')
    parser.add_argument('--max-examples', type=int, default=None, help='每个 split 最多使用多少条样本，便于快速烟测')
    return parser.parse_args()


# 直接运行 main.py 时才开始训练、测试和推理；
# 从其他模块 import main.py 时不会自动启动训练。
if __name__ == "__main__":
    args = parse_args()

    # 不加任何开关时，默认走训练、测试、推理流程，符合原脚本行为。
    run_all = not (args.train or args.test or args.inference)
    model_go(
        train=args.train or run_all,
        test=args.test or run_all,
        inference=args.inference or run_all,
        model_params_path=config.ADDRESS_TAGGING_PARAMS_PATH,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_examples=args.max_examples,
    )
