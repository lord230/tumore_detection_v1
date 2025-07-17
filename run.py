from train import train_and_test, evaluate , test
from dataset import train_loader, val_loader, test_loader

train_and_test(train_loader, val_loader, test_loader, num_epochs=40)
acc, _ = test(test_loader)
print(f"Test Accuracy = {acc}")
acc , test_report = evaluate(val_loader)
print(f"Validation acc = {acc}, Validation report = {test_report}")