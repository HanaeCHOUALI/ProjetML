import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lib.LogisticRegressionModel import LogisticRegressionModel
from lib.NeuralNetwork import NeuralNetwork
from lib.PCA import PrincipalComponentAnalysis
from lib.RandomForestModel import RandomForestModel
from lib.SVMModel import SVMModel

if __name__ == '__main__':
    data = pd.read_csv('data/data.csv', sep=",", index_col=0)
    labels = pd.read_csv('data/labels.csv', sep=",", index_col=0)

    labels_unique_values = {
        'PRAD': 1,
        'LUAD': 2,
        'BRCA': 3,
        'KIRC': 4,
        'COAD': 5
    }

    labels['Class'] = labels['Class'].map(labels_unique_values)

    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42
    )

    # PCA - 500 components
    pca = PrincipalComponentAnalysis(500, X_train, X_test)
    pca.fit(scaler=StandardScaler())
    pca.test()
    x_train_pca = pca.x_train
    x_test_pca = pca.x_test

    # Build Random Forest model
    rf_parameters = {'rf__max_depth': np.arange(1, 101, 5)}
    print('\nModel: Random Forest')
    rf_model = RandomForestModel(
        X_train,
        X_test,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        rf_parameters
    )
    rf_model.train(scaler=StandardScaler(), cv=3)
    rf_model.test()

    # Build Random Forest model - PCA
    rf_parameters = {'rf__max_depth': np.arange(1, 101, 5)}
    print('\nModel: Random Forest - PCA with n_components = 100')
    rf_model_pca = RandomForestModel(
        x_train_pca,
        x_test_pca,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        rf_parameters
    )
    rf_model_pca.train(scaler=StandardScaler(), cv=3)
    rf_model_pca.test()

    # Build Logistic Regression model
    lr_parameters = {'logreg__C': np.logspace(-2, 2, 5, base=2)}
    print('\nModel: Logistic Regression')
    lr_model = LogisticRegressionModel(
        X_train,
        X_test,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        lr_parameters
    )
    lr_model.train(StandardScaler(), max_iter=3000, cv=3)
    lr_model.test()

    # Build SVM model
    svm_parameters = {'svc__C': np.logspace(-2, 2, 5, base=2)}
    print('\nModel: Support Vector Machine')
    svm_model = SVMModel(
        X_train,
        X_test,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        svm_parameters
    )
    svm_model.train(StandardScaler(), max_iter=3000, cv=3)
    svm_model.test()

    # Neural network
    train_tensor = TensorDataset(
        torch.tensor(X_train.values.astype(np.float32)),
        torch.tensor(y_train.values.astype(np.float32))
    )

    trainloader = DataLoader(
        train_tensor,
        batch_size=32,
        shuffle=True
    )

    test_tensor = TensorDataset(
        torch.tensor(X_test.values.astype(np.float32)),
        torch.tensor(y_test.values.astype(np.float32))
    )

    testloader = DataLoader(
        test_tensor,
        batch_size=32,
        shuffle=False
    )

    # Build Neural Network
    print('\nModel: Neural Network')

    neural_network = NeuralNetwork(
        input_size=20531,
        hidden_sizes=[1024, 256],
        output_size=5
    )

    learning_rate = 0.001
    num_epochs = 5
    optimizer = optim.SGD(neural_network.parameters(), lr=learning_rate, momentum=0.9)

    neural_network.train_model(trainloader, optimizer, num_epochs)  # Train the neural network
    neural_network.test_model(testloader)  # Test the neural network

