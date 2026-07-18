# -*- coding: utf-8 -*-
"""
模型训练入口和训练器定义。

这个文件既可以被 main.py 导入，也可以直接运行：

    python train.py
    python train.py --epochs 3 --batch-size 16
    python train.py --epochs 1 --batch-size 1 --max-examples 2 --no-tensorboard

直接运行时的默认行为是训练模型，并把验证集 f1 最好的参数保存到：

    finetune/address_tagging.pt

训练流程：
1. 读取 config 中的模型目录、标签集合和数据路径。
2. 使用 AddressTaggingProcessor 构造 train/valid/test DataLoader。
3. 使用 AddressTaggingTrainer 训练 BERT 序列标注模型。
4. 每个 epoch 后在验证集上计算 precision/recall/f1。
5. 验证集 f1 变好时保存模型参数。
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import tqdm
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.tensorboard.writer import SummaryWriter

import config
from models_def import AddressTagging, load_params
from preprocess import AddressTaggingProcessor


SAMPLE_TEXTS = [
    "中国浙江省杭州市余杭区葛墩路27号楼",
    "北京市通州区永乐店镇27号楼",
    "北京市市辖区高地街道27号楼",
    "新疆维吾尔自治区划阿拉尔市金杨镇27号楼",
    "甘肃省南市文县碧口镇27号楼",
    "陕西省渭南市华阴市罗镇27号楼",
    "西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
    "广州市花都区花东镇27号楼",
]


def set_seed(seed):
    """
    固定随机种子，让数据抽样、参数初始化和训练过程尽量可复现。

    注意：GPU 上的部分底层算子仍可能存在非完全确定性，但这个设置足够满足本项目调试。
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    通用训练器基类。

    这个类只处理训练循环的公共部分：
    - epoch 循环
    - train/valid/test 阶段切换
    - loss 统计
    - 反向传播和 optimizer 更新
    - checkpoint 保存
    - TensorBoard 日志记录

    子类需要实现三个任务相关方法：
    - forward: 一个 batch 如何前向计算并返回 loss
    - update_records: 验证/测试阶段如何收集预测和标签
    - compute_metrics: 如何把 records 转换成 precision/recall/f1 等指标
    """

    def __init__(self, model, device, epochs, learning_rate, checkpoint_steps=200, max_grad_norm=1.0):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.checkpoint_steps = checkpoint_steps
        self.max_grad_norm = max_grad_norm

        # Adam 是这里的基础优化器。学习率由命令行或 config 控制。
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def __call__(self, dataloader: dict, model_params_path=None, writer=None, is_test=False):
        """
        启动训练或测试流程。

        参数:
            dataloader: DataLoader 字典。训练时需要 train/valid，测试时需要 test。
            model_params_path: 模型参数保存路径。训练时必须提供。
            writer: 可选 TensorBoard SummaryWriter。
            is_test: 为 True 时只跑测试集，不训练、不保存模型。

        返回:
            dict: 本次训练或测试得到的指标，方便 main.py 或其它脚本继续处理。
        """

        self.dataloader = dataloader
        self.model_params_path = model_params_path
        self.writer = writer
        self.model.to(self.device)
        self.global_step = 0

        if is_test:
            metrics = self.run_epoch('test')
            for k, v in metrics.items():
                print(f'Test {k}:', v)
            return {'test': metrics}

        if self.model_params_path is None:
            raise ValueError('缺少模型参数保存路径')

        Path(self.model_params_path).parent.mkdir(parents=True, exist_ok=True)

        best_valid_metric = -1.0
        last_metrics = {}
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch + 1}/{self.epochs}')

            train_metrics = self.run_epoch('train', epoch)
            for k, v in train_metrics.items():
                print(f'Train {k}:', v)

            valid_metrics = self.run_epoch('valid', epoch)
            for k, v in valid_metrics.items():
                print(f'Valid {k}:', v)

            last_metrics = {'train': train_metrics, 'valid': valid_metrics}

            # 以验证集 f1 作为“最佳模型”的选择标准。
            if valid_metrics['f1'] >= best_valid_metric:
                best_valid_metric = valid_metrics['f1']
                torch.save(self.model.state_dict(), self.model_params_path)
                print(f'保存最佳模型到: {self.model_params_path}')

        last_metrics['best_valid_f1'] = best_valid_metric
        return last_metrics

    def run_epoch(self, phase, epoch=0):
        """
        运行一个阶段：train、valid 或 test。

        train 阶段会启用梯度并更新参数；
        valid/test 阶段只前向计算，并收集预测结果用于计算指标。
        """

        if phase not in self.dataloader:
            raise KeyError(f'缺少 {phase} 对应的 DataLoader')

        self.model.train() if phase == 'train' else self.model.eval()
        total_loss = 0.0
        total_examples = 0
        records = {}

        with torch.set_grad_enabled(phase == 'train'):
            for inputs in tqdm.tqdm(self.dataloader[phase], desc=phase):
                # Processor 已经把 input_ids/attention_mask/labels 转成 Tensor；
                # 这里统一移动到 CPU/GPU 设备上。
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs, loss = self.forward(inputs)

                if phase == 'train':
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()

                    # 梯度裁剪可以降低偶发梯度爆炸导致训练不稳定的概率。
                    if self.max_grad_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()

                    if self.writer:
                        self.writer.add_scalar(f'Loss/{phase}', loss.item(), self.global_step)

                    self.global_step += 1
                    if self.checkpoint_steps and self.global_step % self.checkpoint_steps == 0:
                        checkpoint_path = str(self.model_params_path) + '.checkpoint'
                        torch.save(self.model.state_dict(), checkpoint_path)
                        print(f'保存 checkpoint 到: {checkpoint_path}')

                current_batch_size = inputs['input_ids'].size(0)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size

                if phase != 'train':
                    self.update_records(inputs, outputs, records)

        if total_examples == 0:
            raise ValueError(f'{phase} DataLoader 中没有样本')

        metrics = {'loss': total_loss / total_examples}
        if phase != 'train':
            self.compute_metrics(metrics, records)
            if self.writer:
                for metric_name, value in metrics.items():
                    self.writer.add_scalar(f'{phase}/{metric_name}', value, epoch)

        return metrics

    def forward(self, inputs):
        """子类实现：定义一个 batch 如何前向计算并返回 loss。"""

        raise NotImplementedError

    def update_records(self, inputs, outputs, records):
        """子类实现：在验证/测试阶段收集计算指标需要的预测结果。"""

        raise NotImplementedError

    def compute_metrics(self, metrics, records):
        """子类实现：根据 records 计算任务指标，并写入 metrics。"""

        raise NotImplementedError


class AddressTaggingTrainer(Trainer):
    """地址序列标注任务训练器。"""

    def forward(self, inputs):
        """
        调用 AddressTagging.forward。

        AddressTagging.forward 返回 {'logits': ..., 'loss': ...}，
        这里把完整 outputs 和 loss 同时交回给通用 Trainer。
        """

        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
        )
        return outputs, outputs['loss']

    def update_records(self, inputs, outputs, records):
        """
        收集验证/测试阶段的预测结果和真实标签。

        只保留真实有效 token：
        - attention_mask == 1: 不是 padding
        - labels != -100: 不是 [CLS]、[SEP]、padding 或被忽略的子词位置
        """

        predictions = outputs['logits'].argmax(dim=-1)
        labels = inputs['labels']
        mask = (inputs['attention_mask'] == 1) & (labels != -100)

        records.setdefault('predictions', []).append(predictions[mask].view(-1).detach().cpu())
        records.setdefault('labels', []).append(labels[mask].view(-1).detach().cpu())

    def compute_metrics(self, metrics, records):
        """计算宏平均 precision、recall、f1，并合并到 metrics 字典中。"""

        if not records.get('predictions') or not records.get('labels'):
            metrics.update({'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
            return

        all_predictions = torch.cat(records['predictions']).numpy()
        all_labels = torch.cat(records['labels']).numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='macro',
            zero_division=0,
        )
        metrics.update({'precision': precision, 'recall': recall, 'f1': f1})


def build_dataloaders(processor, train=False, test=False, max_examples=None):
    """
    根据运行模式构造 DataLoader。

    训练需要 train/valid；
    测试只需要 test。
    """

    dataloaders = {}
    if train:
        dataloaders['train'] = processor.get_dataloader('train', max_examples=max_examples)
        dataloaders['valid'] = processor.get_dataloader('valid', max_examples=max_examples)
    if test:
        dataloaders['test'] = processor.get_dataloader('test', max_examples=max_examples)
    return dataloaders


def create_processor(data_path, processed_dir, tokenizer, batch_size, max_seq_len):
    """创建地址标注数据处理器。"""

    return AddressTaggingProcessor(
        data_path,
        save_dir=processed_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        label_list=config.LABELS,
        max_seq_len=max_seq_len,
    )


def train_model(args):
    """
    命令行训练主流程。

    这个函数把模型、数据处理器、训练器和 TensorBoard writer 串起来。
    """

    set_seed(args.seed)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    model = AddressTagging(args.pretrained_dir, config.LABELS, max_seq_len=args.max_seq_len)
    load_params(model, args.init_params)

    processor = create_processor(
        args.data_path,
        args.processed_dir,
        model.tokenizer,
        args.batch_size,
        args.max_seq_len,
    )

    trainer = AddressTaggingTrainer(
        model,
        args.device,
        args.epochs,
        args.learning_rate,
        checkpoint_steps=args.checkpoint_steps,
        max_grad_norm=args.max_grad_norm,
    )

    run_train = args.train or not (args.train or args.test or args.inference)
    run_test = args.test

    writer = None
    if run_train and not args.no_tensorboard:
        log_name = f'address_tagging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        writer = SummaryWriter(config.LOGS_DIR / log_name)

    try:
        metrics = {}
        if run_train:
            dataloaders = build_dataloaders(processor, train=True, max_examples=args.max_examples)
            metrics.update(trainer(dataloaders, model_params_path=args.output, writer=writer))

        if run_test:
            # 如果刚训练过，先加载验证集 f1 最好的参数再测试；
            # 如果只测试，load_params 已经在前面把 args.init_params 加载进来了。
            if run_train and args.output.exists():
                load_params(model, args.output)
            dataloaders = build_dataloaders(processor, test=True, max_examples=args.max_examples)
            metrics.update(trainer(dataloaders, writer=writer, is_test=True))

        if args.inference:
            predictions = model.predict(SAMPLE_TEXTS, device=args.device, batch_size=args.batch_size)
            for address_text, labels in zip(SAMPLE_TEXTS, predictions):
                for char, label in zip(address_text, labels):
                    print(f'{char}-{label}', end='\t')
                print('\n')

        return metrics
    finally:
        if writer:
            writer.close()


def parse_args():
    """解析 train.py 的命令行参数。"""

    parser = argparse.ArgumentParser(description='训练地址序列标注模型')
    parser.add_argument('--train', action='store_true', help='执行训练；不指定任何模式时默认训练')
    parser.add_argument('--test', action='store_true', help='执行测试集评估')
    parser.add_argument('--inference', action='store_true', help='对内置示例地址做推理')

    parser.add_argument('--data-path', type=Path, default=config.ADDRESS_TAGGING_RAW_DATA_PATH, help='训练数据路径')
    parser.add_argument('--processed-dir', type=Path, default=config.PROCESSED_DATA_DIR, help='预处理数据保存目录')
    parser.add_argument('--pretrained-dir', type=Path, default=config.ADDRESS_TAGGING_PRETRAINED_DIR, help='本地预训练模型目录')
    parser.add_argument('--init-params', type=Path, default=config.ADDRESS_TAGGING_PARAMS_PATH, help='启动时尝试加载的模型参数')
    parser.add_argument('--output', type=Path, default=config.ADDRESS_TAGGING_PARAMS_PATH, help='训练后保存最佳模型的路径')

    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE, help='批大小')
    parser.add_argument('--max-seq-len', type=int, default=config.MAX_SEQ_LEN, help='最大序列长度')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='学习率')
    parser.add_argument('--checkpoint-steps', type=int, default=200, help='每隔多少 step 保存 checkpoint；0 表示不保存')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='梯度裁剪阈值；0 表示不裁剪')
    parser.add_argument('--max-examples', type=int, default=None, help='每个 split 最多使用多少条样本，便于快速烟测')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no-tensorboard', action='store_true', help='不写 TensorBoard 日志')

    args = parser.parse_args()
    args.device = config.DEVICE

    if args.checkpoint_steps <= 0:
        args.checkpoint_steps = None
    if args.max_grad_norm <= 0:
        args.max_grad_norm = None

    return args


if __name__ == '__main__':
    train_model(parse_args())
