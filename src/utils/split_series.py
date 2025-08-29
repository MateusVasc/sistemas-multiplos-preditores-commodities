import pandas as pd

def split_series(df):
    train_list = []
    test_list = []
    val_list = []

    for uid, group in df.groupby('unique_id'):
        n = len(group)
        
        train_end = int(n * 0.7)
        test_start = train_end
        
        val_start = int(train_end * 0.8)
        
        train = group.iloc[:val_start].copy()
        val = group.iloc[val_start:train_end].copy()
        test = group.iloc[test_start:].copy()
        
        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    df_train = pd.concat(train_list).reset_index(drop=True)
    df_val = pd.concat(val_list).reset_index(drop=True)
    df_test = pd.concat(test_list).reset_index(drop=True)
    print(f'Splitted data into TRAIN with size {len(df_train)}, VALIDATION with size {len(df_val)} and TEST with size {len(df_test)}')

    return df_train, df_val, df_test