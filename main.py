import torch
from torch.optim import SGD
from torch.utils.data import DataLoader
from datos.dataloader import load_iris_data
from modelos.neural_network import NeuralNetwork
from entrenamiento.train import train_model
from entrenamiento.evaluacion import evaluate

# Cargar los datos
train_dataset, test_dataset = load_iris_data()

# Crear los data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Crear una instancia de la red neuronal
model = NeuralNetwork()

# Definir la función de pérdida y el optimizador
criterion = torch.nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Entrenar el modelo
num_epochs = 1000
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluar el modelo en el conjunto de prueba
accuracy = evaluate(model, test_loader)
print(f'Precisión en el conjunto de prueba: {accuracy}')


