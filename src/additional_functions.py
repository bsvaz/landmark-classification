import torch
from src.optimization import get_loss

def accuracy(data_loaders, model, valid_size, loss):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    
    with torch.no_grad():
        model.eval()
        model.to(device)
        
        loss = get_loss()
        
        correct = 0
        loss_value = 0
        for images, labels in data_loaders['train']:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss_value += loss(logits, labels)
            
            pred = torch.argmax(logits, axis=1)
            correct += torch.sum(pred == labels).cpu()
        
        train_loss = loss_value/ len(data_loaders['train'])    
        train_acc = 100 * correct / (len(data_loaders['train'].dataset) * (1 - valid_size))
        
        correct = 0
        loss_value = 0
        for images, labels in data_loaders['valid']:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss_value += loss(logits, labels)
            
            pred = torch.argmax(logits, axis=1)
            correct += torch.sum(pred == labels).cpu()
        
        val_loss = loss_value / len(data_loaders['valid'])
        val_acc = 100 * correct / (len(data_loaders['valid'].dataset) * valid_size)
        
        print(f'Train loss: {train_loss}. Validation loss: {val_loss}')
        print('\nTrain accuracy: {:.2f}%. Validation accuracy: {:.2f}%'.format(train_acc, val_acc))
              
    return train_acc, train_loss, val_acc, val_loss