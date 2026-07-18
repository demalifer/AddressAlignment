# -*- coding: utf-8 -*-
"""
数据预处理模块。

这个文件负责把序列标注数据转换成模型可训练的 Dataset/DataLoader。
处理流程大致是：
1. 从 data.jsonl 或 data.txt 读取样本。
2. 把文本和标签转换成 HuggingFace Dataset。
3. 用 tokenizer 编码文本，并把字符级标签对齐到 token 级。
4. 切分 train/valid/test，并保存到磁盘，后续可直接 load_from_disk。
"""

import json
import os
import random
from datasets import Dataset
from torch.utils.data import Subset
from datasets import load_from_disk
from torch.utils.data import DataLoader


class Processor:
    """
    通用数据处理器基类。

    子类需要实现 _make_dataset，用来定义如何从原始数据生成 Dataset。
    本类提供公共能力：数据切分、保存到磁盘、读取 DataLoader、抽样调试。
    """

    def __init__(self, data_path, save_dir, tokenizer, batch_size, max_seq_len, train_ratio=0.8, test_ratio=0.1):
        # data_path: 原始训练文件路径。
        self.data_path = data_path

        # save_dir: 处理后的 train/valid/test 数据保存目录。
        self.save_dir = save_dir

        # tokenizer: 与模型匹配的 HuggingFace tokenizer。
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        # train_ratio/test_ratio 基于原始数据总量计算。
        # valid 集由剩余数据中扣除 train/test 后得到。
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def process(self):
        """构建 Dataset，切分数据集，并将 train/valid/test 保存到磁盘。"""

        self.save_dir.mkdir(parents=True, exist_ok=True)
        dataset = self._make_dataset()
        dataset = self._split_dataset(dataset)

        # 分别保存三个 split。save_to_disk 后，下次训练可以直接 load_from_disk 读取。
        for type in ['train', 'valid', 'test']:
            dataset[type].save_to_disk(self.save_dir / type)

    def get_dataloader(self, type, max_examples=None):
        """
        获取指定 split 的 DataLoader。

        参数:
            type: 'train'、'valid' 或 'test'。
            max_examples: 可选抽样数量，便于快速调试小数据集。
        """

        # 如果对应 split 还没有生成，就先执行完整预处理。
        if not os.path.exists(self.save_dir / type):
            self.process()

        dataset = load_from_disk(self.save_dir / type)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        # max_examples 用于随机抽取部分样本，加快本地调试速度。
        if max_examples:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            max_examples = min(max_examples, len(dataset))
            indices = indices[:max_examples]
            dataset = Subset(dataset, indices)

        # 只有训练集需要 shuffle；验证集和测试集保持固定顺序，方便复现实验结果。
        return DataLoader(dataset, self.batch_size, shuffle=(type == 'train'))

    def _make_dataset(self):
        """由子类实现：从原始数据构造 HuggingFace Dataset。"""

        raise NotImplementedError

    def _split_dataset(self, dataset: Dataset):
        """按照 train_ratio/test_ratio 将数据集切分为 train、valid、test。"""

        # train_size 是训练集样本数，不是比例。
        train_size = int(dataset.num_rows * self.train_ratio)

        # 第一次切分出 test；剩余部分会继续切成 train 和 valid。
        dataset = dataset.train_test_split(test_size=self.test_ratio)

        # 第二次切分：从非 test 的部分中取 train_size 个样本作为训练集，其余作为验证集。
        dataset['train'], dataset['valid'] = (dataset['train'].train_test_split(train_size=train_size).values())
        return dataset


class AddressTaggingProcessor(Processor):
    """地址标注任务专用的数据处理器。"""

    def __init__(self, data_path, save_dir, tokenizer, batch_size, label_list, max_seq_len=64, train_ratio=0.8, test_ratio=0.1):
        super().__init__(data_path, save_dir, tokenizer, batch_size, max_seq_len, train_ratio, test_ratio)

        # label2id 将 config.LABELS 中的标签字符串映射成训练用的整数 id。
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

    def _make_dataset(self):
        """生成地址标注 Dataset，并完成 tokenizer 编码和标签对齐。"""

        # Dataset.from_generator 会逐条调用 _generate_examples 产出样本。
        dataset = Dataset.from_generator(self._generate_examples)

        # map 会把原始 text/labels 转换成 input_ids/attention_mask/labels。
        # remove_columns 删除原始字段，避免 DataLoader 里混入字符串列表。
        dataset = dataset.map(self._map_fn, batched=True, remove_columns=['text', 'labels'])
        return dataset

    def _generate_examples(self):
        """
        从 data.jsonl 或 data.txt 读取样本。

        文件格式假设:
            浙 B-prov
            江 I-prov
            省 E-prov

            杭 B-city
            州 I-city
            市 E-city

        空行用于分隔不同地址样本。
        """

        if str(self.data_path).endswith('.jsonl'):
            yield from self._generate_examples_from_jsonl()
        else:
            yield from self._generate_examples_from_text_file()

    def _generate_examples_from_jsonl(self):
        """读取 data/raw/data.jsonl 这种一行一条 JSON 样本的数据。"""

        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                item = json.loads(line)
                text = item['text']
                raw_labels = item['labels']
                if len(text) != len(raw_labels):
                    raise ValueError(f'第 {line_no} 行 text 和 labels 长度不一致')

                labels = [self._label_to_id(label, line_no) for label in raw_labels]
                yield {'text': text, 'labels': labels}

    def _generate_examples_from_text_file(self):
        """读取空行分隔的“字 标签”文本数据。"""

        with open(self.data_path, 'r', encoding='utf-8') as f:
            blocks = f.read().split('\n\n')
            for block_no, block in enumerate(blocks, start=1):
                text, labels = [], []
                lines = block.split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    word, label = line.strip().split()
                    text.append(word)
                    labels.append(self._label_to_id(label, block_no))

                # text 是字符列表，labels 是同长度的标签 id 列表。
                if text:
                    yield {'text': text, 'labels': labels}

    def _label_to_id(self, label, line_no):
        """将标签字符串转换成 id，并在标签集不匹配时给出明确错误。"""

        try:
            return self.label2id[label]
        except KeyError as exc:
            raise KeyError(f'标签 {label!r} 不在 config.LABELS 中，数据位置: {line_no}') from exc

    def _map_fn(self, example):
        """
        tokenizer 编码，并将字符级 labels 对齐到 token 级。

        HuggingFace tokenizer 可能会加入 [CLS]、[SEP]、padding，也可能把一个字符拆成多个 token。
        所以不能直接把原始 labels 塞给模型，必须通过 word_ids 做对齐。
        """

        inputs = self.tokenizer(example['text'], is_split_into_words=True, max_length=self.max_seq_len, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        all_labels = []
        for i, labels in enumerate(example['labels']):
            # word_ids 表示每个 token 对应原始文本中的第几个字符；
            # 特殊 token 和 padding 的 word_id 是 None。
            word_ids = inputs.word_ids(batch_index=i)
            aligned_labels = self._align_labels_with_tokens(labels, word_ids)
            all_labels.append(aligned_labels)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': all_labels}

    def _align_labels_with_tokens(self, labels, word_ids):
        """
        将原始字符级标签对齐到 tokenizer 之后的 token 序列。

        返回值中 -100 表示该位置不参与损失计算；
        PyTorch CrossEntropyLoss 默认 ignore_index=-100。
        """

        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            # 当前逻辑只给“新的原始字符”的第一个 token 分配标签，
            # 特殊 token、padding、重复子词位置都标为 -100。
            # word_idx == 0 是第一个原始字符，也需要保留标签。
            if word_idx is not None and word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        return aligned_labels
