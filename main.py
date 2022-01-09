import time
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import KernelPCA, PCA
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
    # Read data
    data = pd.read_csv('data/data.csv', sep=",", index_col=0)
    labels = pd.read_csv('data/labels.csv', sep=",", index_col=0)

    # Replace labels names with numerical values
    labels_unique_values = {
        'PRAD': 1,
        'LUAD': 2,
        'BRCA': 3,
        'KIRC': 4,
        'COAD': 5
    }

    labels['Class'] = labels['Class'].map(labels_unique_values)

    # Create training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42
    )

    # Build PCA model considering only the 3 first components
    model = PCA(n_components=3)
    pca = PrincipalComponentAnalysis(model, X_train, X_test, kernel=False)
    pca.fit(scaler=StandardScaler())

    # Apply the dimension reduction to the training and testing sets
    x_train_pca = pca.x_train
    x_test_pca = pca.x_test

    # Build Kernel PCA model considering only the 3 first components and polynomial kernel
    model = KernelPCA(n_components=3, kernel='poly')
    pca_kernel = PrincipalComponentAnalysis(model, X_train, X_test, kernel=True)
    pca_kernel.fit(scaler=StandardScaler())

    # Apply the dimension reduction to the training and testing sets
    x_train_pca_kernel = pca_kernel.x_train
    x_test_pca_kernel = pca_kernel.x_test

    # Build Random Forest model without PCA
    rf_parameters = {'rf__max_depth': np.arange(1, 101, 5)}
    print('\nModel: Random Forest without PCA')
    rf_model = RandomForestModel(
        X_train,
        X_test,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        rf_parameters
    )
    init_time = time.time()
    rf_model.train(scaler=StandardScaler(), cv=3)
    training_time = time.time() - init_time
    rf_model.test()
    print('Training time:', training_time)

    # Build Random Forest model with PCA
    print('\nModel: Random Forest with PCA')
    rf_model_pca = RandomForestModel(
        x_train_pca,
        x_test_pca,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        rf_parameters
    )
    init_time = time.time()
    rf_model_pca.train(scaler=StandardScaler(), cv=3)
    training_time = time.time() - init_time
    rf_model_pca.test()
    print('Training time:', training_time)

    # Build Random Forest model with Kernel PCA
    print('\nModel: Random Forest with Kernel PCA')
    rf_model_pca_kernel = RandomForestModel(
        x_train_pca_kernel,
        x_test_pca_kernel,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        rf_parameters
    )
    init_time = time.time()
    rf_model_pca_kernel.train(scaler=StandardScaler(), cv=3)
    training_time = time.time() - init_time
    rf_model_pca_kernel.test()
    print('Training time:', training_time)

    # Build Logistic Regression model without PCA
    lr_parameters = {'logreg__C': np.logspace(-2, 2, 5, base=2)}
    print('\nModel: Logistic Regression without PCA')
    lr_model = LogisticRegressionModel(
        X_train,
        X_test,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        lr_parameters
    )
    init_time = time.time()
    lr_model.train(StandardScaler(), max_iter=3000, cv=3)
    training_time = time.time() - init_time
    lr_model.test()
    print('Training time:', training_time)

    # Build Logistic Regression model with PCA
    print('\nModel: Logistic Regression with PCA')
    lr_model_pca = LogisticRegressionModel(
        x_train_pca,
        x_test_pca,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        lr_parameters
    )
    init_time = time.time()
    lr_model_pca.train(StandardScaler(), max_iter=5000, cv=3)
    training_time = time.time() - init_time
    lr_model_pca.test()
    print('Training time:', training_time)

    # Build Logistic Regression model with Kernel PCA
    print('\nModel: Logistic Regression with Kernel PCA')
    lr_model_pca_kernel = LogisticRegressionModel(
        x_train_pca_kernel,
        x_test_pca_kernel,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        lr_parameters
    )
    init_time = time.time()
    lr_model_pca_kernel.train(StandardScaler(), max_iter=5000, cv=3)
    training_time = time.time() - init_time
    lr_model_pca_kernel.test()
    print('Training time:', training_time)

    # Build SVM model without PCA
    svm_parameters = {'svc__C': np.logspace(-2, 2, 5, base=2)}
    print('\nModel: Support Vector Machine without PCA')
    svm_model = SVMModel(
        X_train,
        X_test,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        svm_parameters
    )
    init_time = time.time()
    svm_model.train(StandardScaler(), max_iter=3000, cv=3)
    training_time = time.time() - init_time
    svm_model.test()
    print('Training time:', training_time)

    # Build SVM model with PCA
    print('\nModel: Support Vector Machine with PCA')
    svm_model_pca = SVMModel(
        x_train_pca,
        x_test_pca,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        svm_parameters
    )
    init_time = time.time()
    svm_model_pca.train(StandardScaler(), max_iter=5000, cv=3)
    training_time = time.time() - init_time
    svm_model_pca.test()
    print('Training time:', training_time)

    # Build SVM model with Kernel PCA
    print('\nModel: Support Vector Machine with Kernel PCA')
    svm_model_pca_kernel = SVMModel(
        x_train_pca_kernel,
        x_test_pca_kernel,
        np.array(np.transpose(y_train)).reshape(-1),
        np.array(np.transpose(y_test)).reshape(-1),
        svm_parameters
    )
    init_time = time.time()
    svm_model_pca_kernel.train(StandardScaler(), max_iter=5000, cv=3)
    training_time = time.time() - init_time
    svm_model_pca_kernel.test()
    print('Training time:', training_time)

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

