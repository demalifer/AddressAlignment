import os
import random
from datasets import Dataset
from torch.utils.data import Subset
from datasets import load_from_disk
from torch.utils.data import DataLoader

class Processor:
    def __init__(self, data_path, save_dir, tokenizer, batch_size, max_seq_len, train_ratio=0.8, test_ratio=0.1):
        self.data_path = data_path
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def process(self):
        dataset = self._make_dataset()
        dataset = self._split_dataset(dataset)
        for type in ['train', 'valid', 'test']:
            dataset[type].save_to_disk(self.save_dir / type)

    def get_dataloader(self, type, max_examples=None):
        if not os.path.exists(self.save_dir / type):
            self.process()
        dataset = load_from_disk(self.save_dir / type)
        if max_examples:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            max_examples = min(max_examples, len(dataset))
            indices = indices[:max_examples]
            dataset = Subset(dataset, indices)
        return DataLoader(dataset, self.batch_size, shuffle=(type == 'train'))

    def _make_dataset(self):
        raise NotImplementedError

    def _split_dataset(self, dataset: Dataset):
        train_size = int(dataset.num_rows * self.train_ratio)
        dataset = dataset.train_test_split(test_size=self.test_ratio)
        dataset['train'], dataset['valid'] = (dataset['train'].train_test_split(train_size=train_size).values())
        return dataset

class AddressTaggingProcessor(Processor):
    def __init__(self, data_path, save_dir, tokenizer, batch_size, label_list, max_seq_len=64, train_ratio=0.8, test_ratio=0.1):
        super().__init__(data_path, save_dir, tokenizer, batch_size, max_seq_len, train_ratio, test_ratio)
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

    def _make_dataset(self):
        dataset = Dataset.from_generator(self._generate_examples)
        dataset = dataset.map(self._map_fn, batched=True, remove_columns=['text', 'labels'])
        return dataset

    def _generate_examples(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            blocks = f.read().split('\n\n')
            for block in blocks:
                text, labels = [], []
                lines = block.split('\n')
                for line in lines:
                    if not line.strip():
                        continue
                    word, label = line.strip().split()
                    text.append(word)
                    labels.append(self.label2id[label])
                yield {'text': text, 'labels': labels}

    def _map_fn(self, example):
        inputs = self.tokenizer(example['text'], is_split_into_words=True, max_length=self.max_seq_len, padding='max_length', truncation=True)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        all_labels = []
        for i, labels in enumerate(example['labels']):
            word_ids = inputs.word_ids(batch_index=i)
            aligned_labels = self._align_labels_with_tokens(labels, word_ids)
            all_labels.append(aligned_labels)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': all_labels}

    def _align_labels_with_tokens(self, labels, word_ids):
        aligned_labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx and word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        return aligned_labels