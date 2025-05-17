import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
import json

# Charger les données
interactions = pd.read_csv("data_final_project/KuaiRec 2.0/data/big_matrix.csv")
kuairec_caption_category = pd.read_csv(
    "data_final_project/KuaiRec 2.0/data/kuairec_caption_category.csv",
    engine="python", sep=",", quotechar='"', on_bad_lines='skip'
)
item_categories = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_categories.csv")
item_daily_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_daily_features.csv")
social_network = pd.read_csv("data_final_project/KuaiRec 2.0/data/social_network.csv")
user_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/user_features.csv")

print(interactions.head())

# Créer un indicateur d'interaction positive basé sur le temps de visionnement
# Si l'utilisateur regarde au moins 50% de la vidéo, on considère que c'est un feedback positif
interactions['positive_interaction'] = (interactions['watch_ratio'] >= 0.5).astype(int)

social_network['num_friends'] = social_network['friend_list_parsed'].apply(len)

# Diviser les données en ensembles d'entraînement et de test
# Utiliser une division chronologique si possible, sinon utiliser une division aléatoire
if 'timestamp' in interactions.columns:
    # Trier par timestamp
    interactions = interactions.sort_values('timestamp')
    # Prendre les 80% premières interactions pour l'entraînement
    split_idx = int(len(interactions) * 0.8)
    train_data = interactions.iloc[:split_idx]
    test_data = interactions.iloc[split_idx:]
else:
    # Division aléatoire stratifiée par utilisateur
    unique_users = interactions['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    
    train_data = interactions[interactions['user_id'].isin(train_users)]
    test_data = interactions[interactions['user_id'].isin(test_users)]

print(f"Taille de l'ensemble d'entraînement: {train_data.shape}")
print(f"Taille de l'ensemble de test: {test_data.shape}")

# Sauvegarder les données prétraitées
train_data.to_csv('interactions_train.csv', index=False)
test_data.to_csv('interactions_test.csv', index=False)
# Training set size: (10024644, 9)
# Test set size: (2506162, 9)

train_videos = set(train_data['video_id'])
test_videos = set(test_data['video_id'])
overlap = train_videos.intersection(test_videos)
print(f"Overlap between train and test videos: {len(overlap)} videos")
# Overlap between train and test videos: 4049 videos

print(f"Average number of interactions per user in test: {test_data.groupby('user_id').size().mean()}")
# Average number of interactions per user in test: 361.06641694280364

test_users = test_data['user_id'].unique()
test_videos = interactions['video_id'].unique()

num_users = train_data['user_id'].nunique()
num_videos = train_data['video_id'].nunique()

print(f"Number of unique users in train data: {num_users}")
print(f"Number of unique videos in train data: {num_videos}")
# Number of unique users in train data: 7176
# Number of unique videos in train data: 9072


# Créer une matrice d'interactions utilisateur-item
def create_interaction_matrix(data, user_col='user_id', item_col='video_id', rating_col='positive_interaction'):
    """
    Crée une matrice d'interactions entre utilisateurs et items.
    """
    interactions = data.groupby([user_col, item_col])[rating_col].sum().unstack().fillna(0)
    return interactions

# Créer la matrice pour l'ensemble d'entraînement
train_matrix = create_interaction_matrix(train_data)
print(f"Dimensions de la matrice d'interactions d'entraînement: {train_matrix.shape}")
print(f"Densité de la matrice: {train_matrix.count().sum() / (train_matrix.shape[0] * train_matrix.shape[1]):.4f}")

# Sauvegarder la matrice d'interactions
train_matrix.to_pickle('train_interaction_matrix.pkl')
# Dimensions de la matrice d'interactions d'entraînement: (7176, 9072)
