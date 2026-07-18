import tqdm
import torch
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support

class Trainer:
    def __init__(self, model, device, epochs, learning_rate, checkpoint_steps=200):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.checkpoint_steps = checkpoint_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def __call__(self, dataloader:dict, model_params_path=None, writer=None, is_test=False):
        self.dataloader = dataloader
        self.model_params_path = model_params_path
        self.writer = writer
        self.is_test = is_test
        self.model.to(self.device)
        self.global_step = 0

        if is_test:
            for k, v in self.run_epoch('test').items():
                print(f'Test {k}:', v)
            return

        assert self.model_params_path is not None, '缺少模型参数保存路径'
        best_valid_metric = 0
        for epoch in range(self.epochs):
            print(f'Epoch: {epoch}')

            train_metrics = self.run_epoch('train', epoch)
            for k, v in train_metrics.items():
                print(f'Train {k}:', v)

            valid_metrics = self.run_epoch('valid', epoch)
            for k, v in valid_metrics.items():
                print(f'Valid {k}:', v)

            if valid_metrics['f1'] >= best_valid_metric:
                best_valid_metric = valid_metrics['f1']
                torch.save(self.model.state_dict(), self.model_params_path)

    def run_epoch(self, phase, epoch=0):
        self.model.train() if phase == 'train' else self.model.eval()
        total_loss = 0.0
        total_examples = 0
        records = {}

        with torch.set_grad_enabled(phase == 'train'):
            for inputs in tqdm.tqdm(self.dataloader[phase], desc=phase):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs, loss = self.forward(inputs)
                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    if self.writer:
                        self.writer.add_scalar(f'Loss/{phase}', loss.item(), self.global_step)
                    self.global_step += 1
                    if self.checkpoint_steps and self.global_step % self.checkpoint_steps == 0:
                        checkpoint_path = str(self.model_params_path) + '.checkpoint'
                        torch.save(self.model.state_dict(), checkpoint_path)
                current_batch_size = inputs['input_ids'].size(0)
                total_loss += loss.item() * current_batch_size
                total_examples += current_batch_size
                if phase != 'train':
                    self.update_records(inputs, outputs, records)
        metrics = {'loss': total_loss / total_examples}
        if phase != 'train':
            self.compute_metrics(metrics, records)
            if self.writer:
                for metric_name, value in metrics.items():
                    self.writer.add_scalar(f'{phase}/{metric_name}', value, epoch)
        return metrics

    def forward(self, inputs):
        raise NotImplementedError

    def update_records(self, inputs, outputs, records):
        raise NotImplementedError

    def compute_metrics(self, metrics, records):
        raise NotImplementedError

class AddressTaggingTogether(Trainer):
    def __init__(self, inputs):
        outputs = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels'])
        return outputs, outputs['loss']

    def update_records(self, inputs, outputs, records):
        predictions = outputs['logits'].argmax(dim=-1)
        labels = inputs['labels']
        mask = (inputs['attention_mask'] == 1) & (labels != -100)
        predictions = predictions[mask].view(-1).detach().cpu()
        labels = labels[mask].view(-1).detach().cpu()
        records.setdefault('predictions', []).append(predictions)
        records.setdefault('labels', []).append(labels)

    def compute_metrics(self, metrics, records):
        all_predictions = torch.cat(records['predictions'])
        all_labels = torch.cat(records['labels'])
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=0)
        metrics.update({'precision': precision, 'recall': recall, 'f1': f1})