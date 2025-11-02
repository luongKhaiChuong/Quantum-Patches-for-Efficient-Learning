import warnings
import torch
import os, sys, multiprocessing, time

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"

from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
#from keras.utils import Progbar
import logging
import json


sys.path.append(os.path.join(os.path.dirname(__file__), "scripts"))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'training_log.txt')),
        logging.StreamHandler()
    ]
)
from QuantumApply import QuantumApply
from QuanSaliencyMix import QuanSaliencyMix
from SaliencyMix import SaliencyMix
from evaluate import evaluate
from resnet18 import build_resnet18
from WarmupCosineLR import WarmupCosineLR
#The last argument in the data_path and aug_path are the name 
data_path = os.path.join(project_root, 'data', """Subset_all_class\\sub_processed_CIFAR_10""")
aug_path = os.path.join(project_root, 'data', """Subset_all_class\\sub_transformed_dataset.pt""")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resize_size = (128, 128)
split_ratio = (0.8, 0.2)
normalize=True

mean = (0.4914, 0.4822, 0.4465)
std = (0.2471, 0.2435, 0.2616)
random_state = 0
def get_num_workers():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return multiprocessing.cpu_count()

num_workers = get_num_workers()

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label, idx 

    def __len__(self):
        return len(self.data)
    

def load_datasets(batch_size):
    dataset = datasets.ImageFolder(root=data_path, transform=T.ToTensor())    
    images, targets = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for img_batch, label_batch in loader:
        images.append(img_batch)
        targets.append(label_batch)

    images = torch.cat(images, dim=0)
    targets = torch.cat(targets, dim=0)
    del loader
    custom_dataset = CustomDataset(data=images, labels=targets)
    train_size = int(split_ratio[0] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        custom_dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    train_subset, test_subset = random_split(
    custom_dataset, [train_size, test_size],
    generator=torch.Generator().manual_seed(random_state)
)

    train_dataset = CustomDataset(
        data=[custom_dataset[i][0] for i in train_subset.indices],
        labels=[custom_dataset[i][1] for i in train_subset.indices],
        transform=T.Compose([
                    T.ToPILImage(),
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean, std)])
    )

    quantum_train_dataset = CustomDataset(
        data=[custom_dataset[i][0] for i in train_subset.indices],
        labels=[custom_dataset[i][1] for i in train_subset.indices],
        transform=T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize(mean, std)])
    )

    test_dataset = CustomDataset(
        data=[custom_dataset[i][0] for i in test_subset.indices],
        labels=[custom_dataset[i][1] for i in test_subset.indices],
        transform=T.Compose([
                    T.ToPILImage(),
                    T.ToTensor(),
                    T.Normalize(mean, std)])
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=torch.Generator().manual_seed(random_state),
        pin_memory=True
    )
    quantum_train_loader = DataLoader(
        quantum_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=torch.Generator().manual_seed(random_state),
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, quantum_train_loader, test_loader, dataset


def train(model, 
          save_name="baseline_model", epochs=1, beta=1, p=0.5, is_quantum=False, 
          criterion=getattr(nn, "CrossEntropyLoss")(), 
          optimizer=lambda params: optim.SGD(params, lr=1e-2, weight_decay=1e-2, momentum=0.9, nesterov=True)
    ): # optimizer and scheduler from resnet huyvnphan

    logging.info(f"Initialize {save_name} run... \n")
    history = defaultdict(list)
    np.random.seed(random_state)
    
    optimizer = optimizer(model.parameters())
    scheduler=lambda optimizer: WarmupCosineLR(optimizer=optimizer, warmup_epochs=int(epochs*0.3), max_epochs=100, warmup_start_lr=1e-8, eta_min=1e-8)
    print (f"Num warm_up: {int(epochs*0.3)}")
    scheduler = scheduler(optimizer)
    best_val_accuracy = 0.0
    for epoch in range(epochs):
        begin_time = time.time()
        
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss, correct_samples, total_samples = 0.0, 0, 0
        best_model_state_dict = model.state_dict()
        training_loader = train_loader if is_quantum==False else quantum_loader
        for batch_idx, (inputs, labels, idxs) in enumerate(train_loader):
            sub_augmented = []
            if (is_quantum==True):
                for _ in idxs:
                    sub_augmented.extend(dataset_augmented[_*4:_*4+4])
                sub_augmented = torch.stack(sub_augmented)  
            if np.random.rand(1) < p: 
                inputs, label_a, label_b, lam = QuanSaliencyMix(inputs, sub_augmented, labels, beta) if is_quantum==True else SaliencyMix(inputs, labels, beta) 
                inputs, label_a, label_b = inputs.to(device, non_blocking=True), label_a.to(device, non_blocking=True), label_b.to(device, non_blocking=True)
                labels = label_a

                # if is_quantum==True:
                #     _, axes = plt.subplots(1, 4, figsize=(16, 4))
                #     for i in range(4):
                #         img = inputs[i]
                #         img = img.permute(1, 2, 0).cpu().numpy()
                        
                #         axes[i].imshow(img)
                #         axes[i].axis('off')
                #         axes[i].set_title(f"Image {i+1}")

                #     plt.show()

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, label_a) * lam + criterion(outputs, label_b) * (1 - lam) 
            else:
                if (is_quantum==True and p==0.0):
                    inputs, labels = QuantumApply(inputs, sub_augmented, labels)
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)                    
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct_samples += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            avg_train_loss = running_loss / (batch_idx + 1)
            avg_train_accuracy = correct_samples / total_samples
            #progbar.update(batch_idx + 1, values=[("loss", avg_train_loss), ("accuracy", avg_train_accuracy)])
        
        if scheduler:
            scheduler.step()

        val_loss, val_accuracy = evaluate(criterion=criterion, model=model, loader=test_loader, device=device)
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        logging.info(f"Epoch {epoch + 1} - Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.2%}")
        logging.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2%}")
        
        end_time = time.time()
        logging.info(f"Time taken: {end_time - begin_time}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            logging.info("Best model weights saved.")
            
    model.load_state_dict(best_model_state_dict)
    
    val_loss, val_accuracy = evaluate(criterion=criterion, model=model, loader=test_loader, device=device)
    logging.info(f"Best model info: Loss: {val_loss}, Acc: {val_accuracy}")
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 6))
    plot_info = [['train_loss', 'val_loss', 'Loss'], ['train_acc', 'val_acc', 'Accuracy']]
    for _ in range(2):
        plt.subplot(1, 2, _ + 1)
        plt.plot(epochs, history[f'{plot_info[_][0]}'], label=f'{plot_info[_][0]}', color='orange', linewidth=2, marker='o', markersize=3)
        plt.plot(epochs, history[f'{plot_info[_][1]}'], label=f'{plot_info[_][1]}', color='blue', linewidth=2, marker='o', markersize=3)
        plt.xlabel('Epochs')
        plt.ylabel(plot_info[_][2])
        plt.title(f'Training vs Validation {plot_info[_][2]}')
        plt.legend()
    plt.tight_layout()

    # Save plot
    plot_filename = os.path.join(project_root, save_name + ' training plot.png')
    plt.savefig(plot_filename)
    logging.info(f"Training history plot saved as {plot_filename}")

    # Save whole model
    model_save_path = os.path.join(project_root, save_name + ' model.pt')
    torch.save(model, model_save_path)
    logging.info(f"Full model saved to {model_save_path}")

    # Save model's state_dict
    state_dict_path = os.path.join(project_root, save_name + """ model state_dict.pt""")
    torch.save(best_model_state_dict, state_dict_path)
    logging.info(f"Model state dict saved to {state_dict_path}")

    #Save history
    with open(os.path.join(project_root, save_name + ' history.json'), 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model")
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--p', type=float, default=0)
    parser.add_argument('--quantum', action='store_true')
    parser.add_argument('--allin', action='store_true')
    parser.add_argument('--save_name', type=str, default="baseline_model")

    args = parser.parse_args()
    train_loader, quantum_loader, test_loader, original_dataset = load_datasets(args.batch_size)
    dataset_augmented = []
    
    dataset_augmented = torch.load(aug_path)
    if args.allin == True:
        epochs_options = [100, 150]
        for _ in epochs_options:

            cur_epochs = _
            logging.info (f"Running the {cur_epochs} ver")

            model = build_resnet18()
            model = model.to(device)
            save_name_base_quan = f"basline_Quantum_ver{cur_epochs}"
            train(model=model,epochs=cur_epochs, p=0, is_quantum=True, save_name=save_name_base_quan)
            
            model = build_resnet18()
            model = model.to(device)
            save_name_Quan_mix = f"Quantum_Saliency_Mix_ver{cur_epochs}"
            train(model=model,epochs=cur_epochs,beta=args.beta,p=0.5,is_quantum=True,save_name=save_name_Quan_mix)

            model = build_resnet18()
            model = model.to(device)
            save_name_Sal_mix = f"Saliency_Mix_ver{cur_epochs}"
            train(model=model, epochs=cur_epochs, beta=args.beta,p=0.5,is_quantum=False,save_name=save_name_Sal_mix)

            
            save_name_base_line = f"basline_CNN_ver{cur_epochs}"
            model = build_resnet18()
            model = model.to(device)
            train(model=model, epochs=cur_epochs, p=0, is_quantum=False, save_name=save_name_base_line)
    else:
        model = build_resnet18()
        model = model.to(device)
        train(model=model, 
              epochs=args.epochs, 
              p=args.p, 
              is_quantum=args.quantum, 
              save_name=args.save_name)


"""
Baseline: python main_script.py --epochs 100  --save_name basline_run  
Saliency: python main_script.py --epochs 100  --p 0.5 --save_name saliencymix_run  
Quantum:  python main_script.py --epochs 100  --p 0.5 quantum --save_name quantum_run  
"""