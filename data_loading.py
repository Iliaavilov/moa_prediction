import pandas as pd
from sklearn.decomposition import PCA


def load_data(pca_features = None, add_non_scored = True):
    # Загрузка данных
    train_features = pd.read_csv('train_features.csv')
    test_features = pd.read_csv('test_features.csv')
    train_targets = pd.read_csv('train_targets_scored.csv')

    if add_non_scored == True:
        train_targets_nonscored = pd.read_csv('train_targets_nonscored.csv')

        # Добавляем неоцениваемые таргеты
        train_targets = pd.merge(train_targets, train_targets_nonscored, how='inner')

    if pca_features != None:

        # Добавляем pca фичи
        pca_train_features = PCA(n_components=pca_features)
        pca_train_matrix = pca_train_features.fit_transform(train_features.iloc[:, 4:])
        train_features = pd.concat([train_features, pd.DataFrame(pca_train_matrix)], axis='columns')

        pca_test_features = PCA(n_components=pca_features)
        pca_test_matrix = pca_test_features.fit_transform(test_features.iloc[:, 4:])
        test_features = pd.concat([test_features, pd.DataFrame(pca_test_matrix)], axis='columns')

    # Удаляем контрольную группу
    train_features = train_features[train_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    test_features = test_features[test_features['cp_type'] != 'ctl_vehicle'].reset_index(drop=True)
    train_targets = train_targets[train_targets['sig_id'].isin(train_features['sig_id'].values)].reset_index(drop=True)

    # OHE
    train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_time'], prefix='cp_time')], axis=1)
    train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_dose'], prefix='cp_dose')], axis=1)
    train_features = pd.concat([train_features, pd.get_dummies(train_features['cp_type'], prefix='cp_type')], axis=1)
    train_features = train_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)

    test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_time'], prefix='cp_time')], axis=1)
    test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_dose'], prefix='cp_dose')], axis=1)
    test_features = pd.concat([test_features, pd.get_dummies(test_features['cp_type'], prefix='cp_type')], axis=1)
    test_features = test_features.drop(['cp_type', 'cp_time', 'cp_dose'], axis=1)

    train_features = train_features.drop(['sig_id'], axis='columns')
#    test_features = test_features.drop(['sig_id'], axis='columns')
    train_targets = train_targets.drop(['sig_id'], axis='columns')

    return (train_targets, test_features, train_features)