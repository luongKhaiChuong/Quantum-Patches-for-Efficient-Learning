import torch

def evaluate(model, criterion=None, loader=None, device="cpu"):
    model.eval()
    running_loss, correct_samples, total_samples = 0.0, 0, 0
    dataloader = loader
    with torch.no_grad():
        for inputs, labels, idxs in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_samples += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = running_loss / total_samples if criterion else None
    accuracy = correct_samples / total_samples
    return (avg_loss, accuracy) if avg_loss is not None else accuracy

if __name__=="__main__":
    print ("eval ran.")