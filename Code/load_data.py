import copy 
import pandas as pd 
import numpy as np
import copy 
from scipy import stats
from sklearn.preprocessing import  MinMaxScaler, StandardScaler
from category_encoders.cat_boost import CatBoostEncoder
import kagglehub
import geopandas as gpd 
import shapely as shp

def load_data_path(path, target_tb, format = 'csv'):
    path = kagglehub.dataset_download(path)
    if format =='csv':
        df = pd.read_csv(path + '/' +target_tb,encoding_errors='ignore')
    else:
        arrays = dict(np.load(path + '/' +target_tb))
        data = {k: [s.decode("utf-8") for s in v.tobytes().split(b"\x00")] if v.dtype == np.uint8 else v for k, v in arrays.items()}
        df = pd.DataFrame.from_dict(data)
    return df 


def scale_feats(df, scaler = StandardScaler, skip_coords = True, mask_col = 'split_type'):
    cols = df.columns
    if scaler is not None:
        s = scaler()
    else: # default
        s = StandardScaler()
    if skip_coords:
        cols = [col for col in df.columns if col not in ['lon', 'lat', mask_col]]
    if mask_col is not None:
        X = df[df[mask_col] == 0][cols]
    else:
        X = df[cols]
    s.fit(X)
    df.loc[:, cols] = s.transform(df[cols])
    return df, s, list(cols)  # also return scaler and scaled column names


# ==============================================================================
# INVERSE TRANSFORM UTILITIES
# ==============================================================================

def inverse_transform_label(pred_scaled: np.ndarray, scaler, cols: list) -> np.ndarray:
    """
    Revert predictions from normalized log space to original price/sqm scale.

    Reverse pipeline:  pred_norm -> * σ + μ -> log1p space -> expm1 -> original
    """
    pred_scaled = np.asarray(pred_scaled)
    label_idx = cols.index('label')
    label_mean = scaler.mean_[label_idx]
    label_std  = scaler.scale_[label_idx]
    pred_log = pred_scaled * label_std + label_mean
    return np.expm1(pred_log)


def inverse_transform_std(pred_scaled: np.ndarray, std_scaled: np.ndarray,
                          scaler, cols: list) -> np.ndarray:
    """
    Revert predicted std from normalized log space to original scale (delta method).

    The StandardScaler step is linear, so std only scales (no shift):
        std_log = std_norm * σ_scaler

    The expm1 step is non-linear; the first-order Jacobian gives:
        std_original ≈ exp(pred_log) * std_log = (pred_original + 1) * std_log

    Valid when uncertainty is small relative to the mean. For coverage/intervals
    use inverse_transform_interval_bounds() instead (exact, asymmetric).
    """
    pred_scaled = np.asarray(pred_scaled)
    std_scaled  = np.asarray(std_scaled)
    label_idx       = cols.index('label')
    label_std_scaler = scaler.scale_[label_idx]
    std_log      = std_scaled * label_std_scaler          # denorm std → log space
    pred_original = inverse_transform_label(pred_scaled, scaler, cols)
    return (pred_original + 1) * std_log                  # Jacobian of expm1


def inverse_transform_interval_bounds(pred_scaled: np.ndarray, std_scaled: np.ndarray,
                                      scaler, cols: list,
                                      n_sigma: float = 1.96) -> tuple:
    """
    Invert symmetric prediction intervals from normalized log space to original scale.

    Because expm1 is non-linear the resulting intervals are asymmetric in original
    space, which correctly reflects the lognormal shape of the distribution.
    """
    pred_scaled = np.asarray(pred_scaled)
    std_scaled  = np.asarray(std_scaled)
    lower_orig = inverse_transform_label(pred_scaled - n_sigma * std_scaled, scaler, cols)
    upper_orig = inverse_transform_label(pred_scaled + n_sigma * std_scaled, scaler, cols)
    return lower_orig, upper_orig


def train_val_test_split(split_rate, length, shuffle = False, return_type = 'feats'):
    tr_r, val_r, te_r = split_rate
    assert tr_r + val_r + te_r == 1
    if shuffle:
        indices = np.random.permutation(length)
    else:
        indices = np.arange(length)
    ix_ls = [indices[:int(tr_r*length)], indices[int(tr_r*length):int((val_r + tr_r)*length)], indices[int((val_r + tr_r)*length):]]
    if return_type == 'index':
        mask_ls = []
        for i in range(3):
            mask = np.zeros(length, dtype=bool)
            mask[ix_ls[i]] = True
            mask_ls.append(mask)
        return mask_ls
    elif return_type == 'feats':
        split_type =  np.zeros(length, dtype=int)
        for i in range(3):
            split_type[ix_ls[i]] = i # 0-train, 1-val, 2-test
        return split_type
    
def load_london( split_rate=None, scale =True, coords_only = False):
    df_raw = load_data_path("jakewright/house-price-data", 'kaggle_london_house_price_data.csv')
    df = copy.deepcopy(df_raw[['bathrooms', 'bedrooms', 'floorAreaSqM', 'livingRooms',
    'tenure', 'propertyType', 'currentEnergyRating']])
    category_feats = ['tenure', 'propertyType']
    df["label"] = df_raw["history_price"]/ df_raw["floorAreaSqM"]
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude'] 
    d = {'A' : 7, 'B':6, 'C':5, 'D':4, 'E':3, 'F':2, 'G':1, np.nan:0}
    df['currentEnergyRating'] = df['currentEnergyRating'].map(d)
    df["history_date"] = pd.to_numeric(df_raw["history_date"].str.replace('-',''), errors='coerce')
    df = df[df['history_date'] >= 20240301] # not too old data 
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    # remove outliers 
    df['label'] = np.log(df['label'] + 1)
    df = df[(stats.zscore(df["label"])<3) & (stats.zscore(df["label"])>-3)]
    df = df.sort_values(by=['history_date']) # temporal split 
    df['x'] = df['lon'].copy()
    df['y'] = df['lat'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'label'])
            df = encoder.transform(df)
    label_scaler = None
    label_cols = None
    if scale:
        df, label_scaler, label_cols = scale_feats(df, scaler = StandardScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'label', 'split_type']], label_scaler, label_cols
    else:
        return df, label_scaler, label_cols

def load_newyork( split_rate=None, scale =True, coords_only = False):
    df_raw = load_data_path("nelgiriyewithana/new-york-housing-market", 'NY-House-Dataset.csv')
    df = copy.deepcopy(df_raw[['BEDS', 'BATH', 'PROPERTYSQFT', 'TYPE']])
    category_feats = ['TYPE']
    df["label"] = df_raw['PRICE']/df_raw['PROPERTYSQFT']
    df["lon"] = df_raw["LONGITUDE"] 
    df["lat"] = df_raw['LATITUDE']
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df['label'] = np.log(df['label'] + 1)
    df = df[(stats.zscore(df["label"])<3) & (stats.zscore(df["label"])>-3)]
    df = df.sample(frac=1,  random_state=42).reset_index(drop=True) # random to shuffle the table because no temporal info
    df['x'] = df['lon'].copy()
    df['y'] = df['lat'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'label'])
            df = encoder.transform(df)
    label_scaler = None
    label_cols = None
    if scale:
        df, label_scaler, label_cols = scale_feats(df, scaler = StandardScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'label', 'split_type']], label_scaler, label_cols
    else:
        return df, label_scaler, label_cols

def load_paris( split_rate=None, scale =True, coords_only = False):
    df_raw =  load_data_path("benoitfavier/immobilier-france", 'transactions.npz', format='npz')
    df_raw = df_raw[df_raw['ville'].str.startswith("PARIS ")]
    df_raw['date_transaction'] = df_raw['date_transaction'].dt.strftime('%Y-%m-%d')
    
    df_raw = df_raw[df_raw['date_transaction'] >= '2024-01-01']
    df = copy.deepcopy(df_raw[['date_transaction', 'type_batiment','n_pieces',
       'surface_habitable']])
    category_feats = ['type_batiment']
    df["label"] = df_raw['prix']/df_raw['surface_habitable']
    df["lon"] = df_raw["longitude"] 
    df["lat"] = df_raw['latitude']
    df["date_transaction"] = pd.to_numeric(df_raw["date_transaction"].str.replace('-',''), errors='coerce')
    for i in df.columns:
        if i not in category_feats:
            df[i] = pd.to_numeric(df[i], errors='coerce')
    df = df.dropna() # delete nan first!
    df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
    df['label'] = np.log(df['label'] + 1)
    df = df[(stats.zscore(df["label"])<3) & (stats.zscore(df["label"])>-3)]
    df = df.sort_values(by=['date_transaction']) # temporal split 
    df['x'] = df['lon'].copy()
    df['y'] = df['lat'].copy()
    if split_rate is not None:
        split_type = train_val_test_split(split_rate, length=len(df), shuffle = False, return_type = 'feats')
        df['split_type'] = split_type
        if len(category_feats)>0:
            encoder = CatBoostEncoder(cols = category_feats)
            encoder.fit(df[df['split_type'] == 0], df.loc[df['split_type'] == 0, 'label'])
            df = encoder.transform(df)
    label_scaler = None
    label_cols = None
    if scale:
        df, label_scaler, label_cols = scale_feats(df, scaler = StandardScaler, skip_coords = True, mask_col = 'split_type')
    df = df[['lon', 'lat'] + [col for col in df.columns if col not in ['lat', 'lon']]]
    if coords_only:
        return df[['lon', 'lat', 'x', 'y', 'label', 'split_type']], label_scaler, label_cols
    else:
        return df, label_scaler, label_cols

def load_gdf(df):
    feats = list(df.columns)
    feats.remove('lon')
    feats.remove('lat')
    d_dict = df[feats].to_dict('list')
    p = [shp.Point(i) for i in np.array(df[['lon', 'lat']]) ] 
    # convert to geopandas 
    gdf = gpd.GeoDataFrame(geometry=p, data=d_dict)
    gdf_train = gdf[gdf['split_type'] == 0].drop(columns= ['split_type'])
    gdf_val = gdf[gdf['split_type'] == 1].drop(columns= ['split_type'])
    gdf_test = gdf[gdf['split_type'] == 2].drop(columns= ['split_type'])
    df_train = gdf[gdf['split_type'] == 0].drop(columns= ['split_type'])
    df_val = gdf[gdf['split_type'] == 1].drop(columns= ['split_type'])
    df_test = gdf[gdf['split_type'] == 2].drop(columns= ['split_type'])
    return (gdf_train, gdf_val, gdf_test), (df_train, df_val, df_test)





