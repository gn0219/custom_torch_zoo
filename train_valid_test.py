from sklearn.model_selection import train_test_split
import itertools
import pandas as pd

# Feature Engineering (Add more features)
def add_features(X):
    new_X = []
    for X_ in X:
        X_['e4.acc.mag'] = np.sqrt(X_['e4.acc.x']**2 + X_['e4.acc.y']**2 + X_['e4.acc.z']**2)

        # Recommended by GPT
        X_['e4.acc.mean'] = np.mean([X_['e4.acc.x'], X_['e4.acc.y'], X_['e4.acc.z']], axis=0)
        X_['e4.acc.std'] = np.std([X_['e4.acc.x'], X_['e4.acc.y'], X_['e4.acc.z']], axis=0)

        X_['e4.acc.jerk.x'] = np.diff(X_['e4.acc.x'], prepend=X_['e4.acc.x'][0])
        X_['e4.acc.jerk.y'] = np.diff(X_['e4.acc.y'], prepend=X_['e4.acc.y'][0])
        X_['e4.acc.jerk.z'] = np.diff(X_['e4.acc.z'], prepend=X_['e4.acc.z'][0])
        X_['e4.acc.jerk.mag'] = np.sqrt(X_['e4.acc.jerk.x']**2 + X_['e4.acc.jerk.y']**2 + X_['e4.acc.jerk.z']**2)

        X_['e4.hr.diff'] = np.diff(X_['e4.hr'], prepend=X_['e4.hr'][0])
        X_['e4.bvp.diff'] = np.diff(X_['e4.bvp'], prepend=X_['e4.bvp'][0])
        X_['e4.eda.diff'] = np.diff(X_['e4.eda'], prepend=X_['e4.eda'][0])
        X_['e4.temp.rate'] = np.diff(X_['e4.temp'], prepend=X_['e4.temp'][0])

        X_['brain.alpha.power'] = np.mean([X_['brain.alpha.low'], X_['brain.alpha.high']], axis=0)
        X_['brain.beta.power'] = np.mean([X_['brain.beta.low'], X_['brain.beta.high']], axis=0)
        X_['brain.gamma.power'] = np.mean([X_['brain.gamma.low'], X_['brain.gamma.mid']], axis=0)
        X_['brain.alpha.theta.ratio'] = X_['brain.alpha.low'] / X_['brain.theta']
        X_['brain.beta.alpha.ratio'] = X_['brain.beta.low'] / X_['brain.alpha.low']

        X_['brain.asymmetry'] = X_['brain.alpha.high'] - X_['brain.alpha.low']

        new_X.append(X_)
    return new_X

def get_resampled_features(X, pids, target_N=20, use_features=['e4.hr', 'e4.eda', 'e4.temp']):
    X_res = []
    for x, pid in zip(X, pids):
        row = []
        for dtype in use_features:
            sig = resampling(x = x[dtype], target_N=target_N)
            row.append(sig)
        X_res.append(np.column_stack(row))

    X_res = np.asarray(X_res)
    return X_res

def model_call(params, model_name='baseline'):
    if model_name == 'simplednn':
        model = SimpleDNN(target_N=params['target_N'], n_of_features=len(params['use_features'])).to(device)
    elif model_name == 'baseline':
        model = Baseline(target_N=params['target_N'], n_of_features=len(params['use_features'])).to(device)
    elif model_name == 'convlstm':
        model = ConvLSTM(input_channels=params['target_N']).to(device)
    elif model_name == 'cnnbilstm':
        model = CNNBiLSTM3b(target_N=params['target_N'], n_of_features=len(params['use_features'])).to(device)
    return model

def get_search_all_params(search_params):
    param_combinations = itertools.product(*search_params.values())
    param_combinations_dicts = []
    for combination in param_combinations:
        param_dict = dict(zip(search_params.keys(), combination))
        param_combinations_dicts.append(param_dict)
    return param_combinations_dicts

def train_valid(new_X, pids, params, fold=0, return_best_model=False):
    
    X_res = get_resampled_features(new_X, pids, target_N=params['target_N'], use_features=params['use_features'])

    # DataLoader
    X_train, y_train = X_res[I_TRAINS[fold]], y_simple[I_TRAINS[fold]]
    # X_test, y_test = X_res[I_TESTS[fold]], y_simple[I_TESTS[fold]]

    # X_train, X_valid, y_train, y_valid = X_train[120:], X_train[:120], y_train[120:], y_train[:120]
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_dataset = TensorDataset(torch.tensor(X_valid, dtype=torch.float32), torch.tensor(y_valid, dtype=torch.long))
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # print(next(iter(train_loader))[0].shape)
    # Call model
    model = model_call(params, model_name=params['model_name'])

    # Hyperparameter for model tuning
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_norm'])
    early_stopping = EarlyStopping(patience=params['early_stopping_patience'], verbose=params['verbose'])
    gradient_clipping = nn.utils.clip_grad_norm_ # add gradient clip

    val_accuracies=[]
    for epoch in range(params['epochs']):

        num_correct, num_samples, val_num_correct, val_num_samples = 0, 0, 0, 0
        
        model.train()
        train_loss=[]
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)

            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == target).sum()
            num_samples += predictions.size(0)

            loss = criterion(scores, target)
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), 1)
            optimizer.step()

            train_loss.append(loss.item())
            
        train_acc = (num_correct / num_samples).item()
        
        model.eval()
        val_loss=[]
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                
                scores = model(data)
                _, predictions = scores.max(1)
                val_num_correct += (predictions == target).sum()
                val_num_samples += predictions.size(0)
                loss = criterion(scores, target)
                val_loss.append(loss.item())

        val_acc = (val_num_correct / val_num_samples).item()
        val_accuracies.append(val_acc)
        if params['verbose']:
            print(f'Epoch: [{epoch:0>2}] Train Loss : [{np.mean(train_loss):.4f}] Valid Loss : [{np.mean(val_loss):.4f}] Train Acc: [{train_acc*100:.2f}%] Valid Acc: [{val_acc*100:.2f}%]')

        early_stopping(np.mean(val_loss), model)

        if early_stopping.early_stop:
            if params['verbose']:
                print(f"Epoch: [{epoch:0>2}] Early Stopping with best validation accuracy {val_accuracies[-params['early_stopping_patience']]*100:.2f}")
            best_models = early_stopping.load_best_model()
            break

    if return_best_model:
        return best_models
    else:
        _params = params
        _params['use_features'] = str(_params['use_features'])
        _params['val_accuracy'] = val_accuracies[-params['early_stopping_patience']]
        return _params


new_X = add_features(X)
search_params = {
    'model_name': ['simplednn', 'convlstm'],
    'use_features': [[i] for i in new_X[0].keys()],
    'target_N': [20],
    'batch_size': [64],
    'learning_rate': [1e-3],
    'l2_norm': [1e-2],
    'early_stopping_patience': [15],
    'epochs': [100],
    'verbose': [False]
}

results=[]
all_params = get_search_all_params(search_params)
for params in tqdm(all_params):
    results.append(train_valid(new_X, pids, params))

val_results = pd.DataFrame(results)
val_results.sort_values(by='val_accuracy', ascending=False).head(10)

search_params = {
    'model_name': ['simplednn', 'baseline', 'convlstm', 'cnnbilstm3b'],
    'use_features': [
        ['e4.eda', 'e4.hr', 'e4.temp'],
        ['e4.eda', 'e4.hr', 'e4.temp', 'e4.acc.mag', 'brain.attention', 'brain.meditation'],
        ['e4.eda', 'e4.hr', 'e4.temp', 'e4.acc.mag', 'e4.bvp.diff', 'brain.attention', 'brain.meditation', 'brain.gamma.power', 'brain.theta', 'brain.alpha.low'],
    ],
    'target_N': [10, 20, 40, 80],
    'batch_size': [64],
    'learning_rate': [1e-3],
    'l2_norm': [1e-2],
    'early_stopping_patience': [15],
    'epochs': [100],
    'verbose': [False]
}

results=[]
all_params = get_search_all_params(search_params)
for params in tqdm(all_params):
    results.append(train_valid(new_X, pids, params))

val_results2 = pd.DataFrame(results)

## after tuning
params = {
    'model_name': 'baseline',
    'use_features': ['e4.eda', 'e4.hr', 'e4.temp', 'e4.acc.mag', 'brain.attention', 'brain.meditation'],
    # 'use_features': ['e4.eda', 'e4.hr', 'e4.temp'],
    'target_N': 20,
    'batch_size': 64,
    'learning_rate': 1e-3,
    'l2_norm': 1e-2,
    'early_stopping_patience': 15,
    'epochs': 200,
    'verbose': True
}

# Train and save model with tuned parameters
MODELS=[]
for fold in range(4):
    MODELS.append(train_valid(new_X, pids, params, fold=fold, return_best_model=True))

# Test
X_res = get_resampled_features(new_X, pids, target_N=params['target_N'], use_features=params['use_features'])

accuracies = []
losses = []
for model, I_test in zip(MODELS, I_TESTS):
    X_test, y_test = X_res[I_test], y_simple[I_test]

    acc_, loss_ = evaluate_model(model, X_test, y_test, batch_size=params['batch_size'])
    accuracies.append(acc_)
    losses.append(loss_)

# Calculate and print the mean and standard deviation of accuracies
mean_accuracy = np.mean(accuracies)
std_dev_accuracy = np.std(accuracies)
mean_loss = np.mean(losses)
std_dev_loss = np.std(losses)

print(f"Mean Accuracy: {mean_accuracy * 100:.2f}%, (SD={std_dev_accuracy})")
print(f"Mean Loss: {mean_loss:.4f}, (SD={std_dev_loss})")
###############################################
