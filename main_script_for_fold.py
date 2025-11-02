import warnings
import torch
import os, sys, multiprocessing, time
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchvision import datasets, transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
#from keras.utils import Progbar
import logging
import json

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false"



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
    

def load_full_set(batch_size):
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
    
    return custom_dataset


def train(model, 
          save_name="baseline_model", epochs=1, beta=1, p=0.5, is_quantum=False, 
          criterion=getattr(nn, "CrossEntropyLoss")(), 
          optimizer=lambda params: optim.SGD(params, lr=1e-2, weight_decay=1e-2, momentum=0.9, nesterov=True)
    ): # optimizer and scheduler from resnet huyvnphan

    logging.info(f"Initialize {save_name} run... \n")
    history = defaultdict(list)
    np.random.seed(random_state)
    
    optimizer = optimizer(model.parameters())
    scheduler=lambda optimizer: WarmupCosineLR(optimizer=optimizer, warmup_epochs=int(epochs*0.3), max_epochs=epochs, warmup_start_lr=1e-8, eta_min=1e-8)
    print (f"Num warm_up: {int(epochs*0.3)}")
    scheduler = scheduler(optimizer)
    for epoch in range(epochs):
        begin_time = time.time()
        
        logging.info(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        running_loss, correct_samples, total_samples = 0.0, 0, 0
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
        return history

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train model")
    
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--p', type=float, default=0)
    parser.add_argument('--quantum', action='store_true')
    parser.add_argument('--save_name', type=str, default="all")

    args = parser.parse_args()
    full_set = load_full_set(args.batch_size)

    kfold = KFold(n_splits=10, shuffle=True, random_state=random_state)
    dataset_augmented = []
    dataset_augmented = torch.load(aug_path)
    
    all_runs_history = {
        "baseline_cnn": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
        "quantum_baseline": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []},
        "quantum_saliencymix": {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    }
    folds = kfold.split(full_set)
    for fold, (train_indices, val_indices) in enumerate(folds):
        logging.info(f"Fold {fold + 1}/10 ---")
        train_dataset = CustomDataset(
                    data=[full_set[i][0] for i in train_indices],
                    labels=[full_set[i][1] for i in train_indices],
                    transform=T.Compose([
                        T.ToPILImage(),
                        T.RandomCrop(32, padding=4),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(mean, std)
                    ]))
        
        quantum_train_dataset = CustomDataset(
            data=[full_set[i][0] for i in train_indices],
            labels=[full_set[i][1] for i in train_indices],
            transform=T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean, std)
            ]))

        val_dataset = CustomDataset(
            data=[full_set[i][0] for i in val_indices],
            labels=[full_set[i][1] for i in val_indices],
            transform=T.Compose([
                T.ToPILImage(),
                T.ToTensor(),
                T.Normalize(mean, std)
            ]))

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, generator=torch.Generator().manual_seed(random_state),
            pin_memory=True
        )
        quantum_loader = DataLoader(
            quantum_train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=num_workers, generator=torch.Generator().manual_seed(random_state),
            pin_memory=True
        )
        test_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        model = build_resnet18()
        model = model.to(device)
        save_name_base_quan = f"basline_Quantum"
        baseline_quan_history = train(model=model,epochs=args.epochs, p=0, is_quantum=True, save_name=save_name_base_quan)
        for key in baseline_quan_history:
            all_runs_history['quantum_baseline'][key].append(baseline_quan_history[key])

        model = build_resnet18()
        model = model.to(device)
        save_name_Quan_mix = f"Quantum_Saliency_Mix"
        mix_quan_history = train(model=model,epochs=args.epochs,beta=args.beta,p=args.p,is_quantum=True,save_name=save_name_Quan_mix)
        for key in mix_quan_history:
            all_runs_history["quantum_saliencymix"][key].append(mix_quan_history[key])

        save_name_base_line = f"basline_CNN"
        model = build_resnet18()
        model = model.to(device)
        baseline_history = train(model=model, epochs=args.epochs, p=0, is_quantum=False, save_name=save_name_base_line)
        for key in baseline_history:
            all_runs_history["baseline_cnn"][key].append(baseline_history[key])
    logging.info("\n--- Aggregating Results and Generating Plots ---")
    epochs_range = range(1, args.epochs + 1)
    plot_metrics = [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_acc', 'val_acc', 'Accuracy')
    ]

    for run_name, run_data in all_runs_history.items():
        plt.figure(figsize=(12, 6))
        logging.info(f"Plotting for: {run_name}")

        for i, (train_key, val_key, title_suffix) in enumerate(plot_metrics):
            # Convert list of lists (each inner list is history for one fold) to numpy array
            train_values = np.array(run_data[train_key])
            val_values = np.array(run_data[val_key])

            # Calculate mean and standard deviation across folds for each epoch
            mean_train_values = np.mean(train_values, axis=0)
            std_train_values = np.std(train_values, axis=0)
            mean_val_values = np.mean(val_values, axis=0)
            std_val_values = np.std(val_values, axis=0)

            plt.subplot(1, 2, i + 1)
            plt.plot(epochs_range, mean_train_values, label=f'Mean Train {title_suffix}', color='orange', linewidth=2)
            plt.fill_between(epochs_range, mean_train_values - std_train_values, mean_train_values + std_train_values, color='orange', alpha=0.2, label=f'Std Dev Train {title_suffix}')

            plt.plot(epochs_range, mean_val_values, label=f'Mean Val {title_suffix}', color='blue', linewidth=2)
            plt.fill_between(epochs_range, mean_val_values - std_val_values, mean_val_values + std_val_values, color='blue', alpha=0.2, label=f'Std Dev Val {title_suffix}')

            plt.xlabel('Epochs')
            plt.ylabel(title_suffix)
            # Create a more readable title from the run_name
            readable_run_name = run_name.replace("_", " ").title().replace("Cnn", "CNN").replace("Mix", "Mix")
            plt.title(f'Mean Training vs Validation {title_suffix} for {readable_run_name}')
            plt.legend()
        
        plt.tight_layout() # Adjust plot to prevent labels overlapping
        plot_filename = os.path.join(project_root, f"{args.save_name}_{run_name}_mean_plot.png")
        plt.savefig(plot_filename)
        logging.info(f"Mean history plot for {run_name} saved as {plot_filename}")
        plt.close() # Close figure to free up memory

    # Save the aggregated history to a JSON file for later analysis
    history_json_path = os.path.join(project_root, f"{args.save_name}_all_runs_history.json")
    # Convert defaultdicts to regular dicts for JSON serialization compatibility
    serializable_history = {
        k: {inner_k: inner_v for inner_k, inner_v in v.items()}
        for k, v in all_runs_history.items()
    }
    with open(history_json_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)
    logging.info(f"Aggregated history saved to {history_json_path}")

    logging.info("\n--- K-Fold Cross-Validation Run Completed ---")

"""
Baseline: python main_script.py --epochs 100  --save_name basline_run  
Saliency: python main_script.py --epochs 100  --p 0.5 --save_name saliencymix_run  
Quantum:  python main_script.py --epochs 100  --p 0.5 quantum --save_name quantum_run  
"""