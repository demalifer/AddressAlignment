# -*- coding: utf-8 -*-
"""
地址序列标注模型定义。

这个文件包含两个核心部分：
1. load_params: 从磁盘加载模型参数。
2. AddressTagging: 基于 BERT 的逐字地址成分标注模型。

模型任务是给地址中的每个字预测一个标签，例如 B-prov、I-city、E-detail、O。
推理阶段会把 BIOES 前缀去掉，只保留 prov、city、detail、name、phone 等类别。
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel


def load_params(model, model_params_path):
    """
    加载已经训练好的模型参数。

    参数:
        model: 要加载参数的 PyTorch 模型。
        model_params_path: 参数文件路径，通常是 .pt 文件。

    兼容逻辑:
        - 文件不存在或路径为空时，保留模型默认初始化参数。
        - 如果参数是在 GPU 上保存、当前只能用 CPU 加载，则用 map_location 回退到 CPU。
        - 如果标签数量变化导致分类层形状不匹配，也保留默认参数，避免服务启动失败。
    """

    if not model_params_path:
        print('模型参数路径为空，使用默认参数')
        return

    try:
        # 统一加载到 CPU，之后由训练器或 predict 再把模型移动到目标设备。
        state_dict = torch.load(model_params_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except (FileNotFoundError, AttributeError):
        print('模型参数不存在，使用默认参数')
    except RuntimeError as exc:
        print(f'模型参数加载失败，使用默认参数: {exc}')


class AddressTagging(nn.Module):
    """
    BERT 地址成分标注模型。

    结构:
        tokenizer: 与预训练模型匹配的分词器。
        bert: 预训练 BERT 主干，用来提取每个 token 的上下文表示。
        dropout: 防止训练阶段过拟合。
        classifier: 对每个 token 输出一个标签类别。
        loss_fn: 训练时使用的交叉熵损失。
        max_seq_len: tokenizer 最大序列长度，超过该长度会被截断。
    """

    def __init__(self, model_name, label_list, max_seq_len=96):
        super(AddressTagging, self).__init__()

        # 标签列表由 config.LABELS 传入，标签下标就是训练时使用的类别 id。
        self.label_list = label_list
        self.num_labels = len(self.label_list)

        # 和训练预处理中的 max_seq_len 保持一致，保证训练和推理看到的最大长度相同。
        self.max_seq_len = max_seq_len

        # tokenizer 和 bert 必须来自同一个预训练模型目录，否则 token id 和模型词表可能不匹配。
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        # BERT 每个位置输出 hidden_size 维向量，线性层把它转换成 num_labels 个类别分数。
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        # CrossEntropyLoss 默认会忽略标签值为 -100 的位置；
        # preprocess.py 正是用 -100 标记 [CLS]、[SEP]、padding 等不需要计算损失的位置。
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向计算。

        参数:
            input_ids: tokenizer 输出的 token id，形状通常是 [batch_size, seq_len]。
            attention_mask: 有效 token 为 1、padding 为 0 的掩码。
            labels: 可选真实标签。训练/验证时传入，纯推理时可以不传。

        返回:
            dict:
                logits: 每个 token 对每个标签的预测分数。
                loss: 如果传入 labels，则为交叉熵损失；否则为 0.0。
        """

        # last_hidden_state 的形状是 [batch_size, seq_len, hidden_size]。
        outputs = self.bert(input_ids, attention_mask)

        # 对序列中每个 token 单独做分类，得到 [batch_size, seq_len, num_labels]。
        logits = self.classifier(self.dropout(outputs.last_hidden_state))
        loss = 0.0
        if labels is not None:
            # view(-1, ...) 把 batch 和 seq_len 拉平成一个维度，
            # 让 CrossEntropyLoss 可以按 token 级别计算分类损失。
            loss += self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return {'logits': logits, 'loss': loss}

    @torch.inference_mode()
    def predict(self, text: str | list[str], device=torch.device('cpu'), batch_size=32):
        """
        对单条或多条地址文本做标签预测。

        参数:
            text: 字符串或字符串列表。
            device: 推理设备，默认 CPU。
            batch_size: 批量推理大小。

        返回:
            单条输入返回一维标签列表，多条输入返回二维标签列表。
            每个字符对应一个去掉 BIOES 前缀后的标签，例如 “B-city” 会变成 “city”，
            “O” 会变成空字符串。
        """

        # 切换到评估模式并移动到指定设备。
        self.eval()
        self.to(device)

        res: list[list[str]] = []

        # 统一转换为列表，方便下面按 batch 处理；最后再按输入类型恢复返回结构。
        is_single_text = isinstance(text, str)
        input_texts = [text] if is_single_text else text
        for i in range(0, len(input_texts), batch_size):
            batch_texts = input_texts[i:i + batch_size]

            # 地址标注是逐字任务，因此这里把每条地址拆成字符列表。
            # is_split_into_words=True 会让 tokenizer 保留“原始字”和 token 之间的对应关系。
            batch_words = [list(t) for t in batch_texts]
            inputs = self.tokenizer(
                batch_words,
                is_split_into_words=True,
                max_length=self.max_seq_len,
                truncation=True,
                padding=True,
                return_tensors='pt'
            ).to(device)

            outputs = self(inputs['input_ids'], inputs['attention_mask'])

            # logits 取最大值所在下标，得到每个 token 的预测标签 id。
            preds = torch.argmax(outputs['logits'], dim=-1).detach().cpu()
            for batch_idx, pred in enumerate(preds):
                # word_ids 用来把 tokenizer 产生的 token 位置映射回原始字符位置。
                # [CLS]、[SEP]、padding 这类特殊 token 的 word_id 是 None。
                word_ids = inputs.word_ids(batch_idx)
                current_word = None
                tokens_pred = []
                for idx, word_id in enumerate(word_ids):
                    # 只在遇到新的原始字符时记录一次标签，避免一个字符被拆成多个子词后重复记录。
                    if word_id is not None and word_id != current_word:
                        current_word = word_id
                        label = self.label_list[pred[idx].item()]

                        # 去掉 BIOES 前缀：
                        # - "B-city"[2:] -> "city"
                        # - "O"[2:] -> ""，后续 address_alignment 会把空字符串当作非地址成分。
                        tokens_pred.append(label[2:])
                res.append(tokens_pred)

        return res[0] if is_single_text else res
