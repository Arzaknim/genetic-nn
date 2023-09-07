import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def prepare_data(file_path):
    df = pd.read_csv(file_path, header=0)

    X = np.array(df.iloc[:, 1:-1])
    y = np.array(df.iloc[:, -1:])
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    new_X = []

    for x in X.transpose():
        if isinstance(x[0], str):
            x = encoder.fit_transform(x)
        new_X.append(np.array(x, dtype=np.float32))

    new_X = np.array(new_X).transpose()
    new_X = torch.from_numpy(new_X)
    y = torch.from_numpy(y)

    return train_test_split(new_X, y, test_size=0.25, random_state=2201, shuffle=True)
