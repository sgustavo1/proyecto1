from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset

class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        return x, y

def load_iris_data(test_size=0.2, random_state=42):
    iris = load_iris()
    features = iris.data
    labels = iris.target

    # Escalar los datos
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    # Convertir los datos a tensores de PyTorch
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).long()
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test).long()

    train_dataset = IrisDataset(X_train, y_train)
    test_dataset = IrisDataset(X_test, y_test)

    return train_dataset, test_dataset
