import pickle
import os
from itertools import product


LAYERS = [
    (8,), (16,), (32,), (64,), (128,), 
    (8, 4), (16, 8), (32, 16), (64, 32), (128, 64), (256, 128),
    (8, 4, 2), (16, 8, 4), (32, 16, 8), (64, 32, 16), (128, 64, 32), (256, 128, 64),
    (64, 32, 16, 8), (128, 64, 32, 16), (256, 128, 64, 32),
    (128, 64, 32, 16, 8), (256, 128, 64, 32, 16)
]

param_grid = {
    'CLASSIFICADOR': ['LTSM', 'GRU'],
    'LSTM_LAYERS': LAYERS,
    'BATCH_SIZE': [8, 16, 32, 64, 128],
    'LEARNING_RATE': [0.001, 0.01, 0.1],
    'EPOCHS': [25, 50, 100, 200, 400],
    'OPTIMIZER': ['adam', 'sgd']
}

def grid_search(param_grid):
    
    path_grid_search = os.path.join('grid_search/')
    # create directory for grid search results
    if not os.path.exists(path_grid_search):
        os.mkdir(path_grid_search)

    # get the keys and values for the grid search
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # generate all combinations of parameters
    combinacoes = list(product(*values))

    # transform combinations into a list of dictionaries
    grid_completo = [dict(zip(keys, combo)) for combo in combinacoes]
    


def train_model(config):
    # define file paths
    path = "datasets/"
    test_data_file = "bp_sv_co_test_dataset.pkl"
    train_data_file = "bp_sv_co_train_dataset.pkl"

    # load data
    x_train, y_train, x_test, y_test = load_and_split_data(path, train_data_file, test_data_file)

    # build model
    model = build_model(config)

    # compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=config['EPOCHS'],
        batch_size=config['BATCH_SIZE']
    )

    return model, history

def load_and_split_data(path, train_data_path, test_data_path):
    # load train and test data
    train_data = pickle.load(open(f"{path}{train_data_path}", "rb"))
    test_data = pickle.load(open(f"{path}{test_data_path}", "rb"))

    # x and Y split
    x_train = train_data[0]
    y_train = train_data[1]

    x_test = test_data[0]
    y_test = test_data[1]

    return x_train, y_train, x_test, y_test

def build_model(config):
