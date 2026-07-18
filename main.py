import config
from datetime import datetime
from train import AddressTaggingTrainer
from preprocess import AddressTaggingProcessor
from models_def import AddressTagging, load_params
from torch.utils.tensorboard.writer import SummaryWriter

device = config.DEVICE
learning_rate = 1e-5

def model_go(train=None, test=None, inference=None, model_params_path=None):
    model = AddressTagging(config.PRETRAINED_DIR / 'roberta-small-wwm-chinese-cluecorpussmall', config.LABELS)
    processor = AddressTaggingProcessor(config.PRETRAINED_DIR / 'data.txt', save_dir=config.PRETRAINED_DIR, tokenizer=model.tokenizer, batch_size=128, label_list=config.LABELS)
    trainer = AddressTaggingTrainer(model, device, 10, learning_rate)
    writer = None
    save_name = f'address_tagging-{datetime.now().strftime("%Y-%m-%d,%H:%M:%S")}'
    load_params(model, model_params_path)
    if train:
        writer = SummaryWriter(config.LOGS_DIR / save_name)
        dataloader = {
            'train': processor.get_dataloader('train'),
            'valid': processor.get_dataloader('valid'),
        }
        model_params_path = config.FINETUNE_DIR / f'{save_name}.pt'
        trainer(dataloader, model_params_path, writer)

    if test:
        test_dataloader = processor.get_dataloader('test')
        trainer({'test': test_dataloader}, writer, is_test=True)

    if writer:
        writer.close()

    if inference:
        res = model.predict(text, device)
        if isinstance(res, str):
            print(res)
        elif isinstance(res, dict):
            for a_text, a_res in zip(text, res):
                for t, r in zip(a_text, a_res):
                    print(f'{t}-{r}', end='\t')
                print('\n')
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
model_go(1, 1, 1, config.BASE_DIR / 'finetune' / 'address_tagging.pt')