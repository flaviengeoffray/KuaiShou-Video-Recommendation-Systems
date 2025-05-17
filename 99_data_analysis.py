# 1. EXPLORATION DES DONNÉES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Charger les données

# Index(['user_id', 'video_id', 'play_duration', 'video_duration', 'time',
#        'date', 'timestamp', 'watch_ratio'],
#       dtype='object')
interactions_train = pd.read_csv("data_final_project/KuaiRec 2.0/data/big_matrix.csv")
interactions_test = pd.read_csv("data_final_project/KuaiRec 2.0/data/small_matrix.csv")


# Index(['video_id', 'manual_cover_text', 'caption', 'topic_tag',
#        'first_level_category_id', 'first_level_category_name',
#        'second_level_category_id', 'second_level_category_name',
#        'third_level_category_id', 'third_level_category_name'],
#       dtype='object')
kuairec_caption_category = pd.read_csv(
    "data_final_project/KuaiRec 2.0/data/kuairec_caption_category.csv",
    engine="python",
    sep=",",
    quotechar='"',
    on_bad_lines='skip'      # alternative pour pandas ≥ 1.3
)


#Index(['video_id', 'feat'], dtype='object')
item_categories = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_categories.csv")


# Index(['video_id', 'date', 'author_id', 'video_type', 'upload_dt',
#        'upload_type', 'visible_status', 'video_duration', 'video_width',
#        'video_height', 'music_id', 'video_tag_id', 'video_tag_name',
#        'show_cnt', 'show_user_num', 'play_cnt', 'play_user_num',
#        'play_duration', 'complete_play_cnt', 'complete_play_user_num',
#        'valid_play_cnt', 'valid_play_user_num', 'long_time_play_cnt',
#        'long_time_play_user_num', 'short_time_play_cnt',
#        'short_time_play_user_num', 'play_progress', 'comment_stay_duration',
#        'like_cnt', 'like_user_num', 'click_like_cnt', 'double_click_cnt',
#        'cancel_like_cnt', 'cancel_like_user_num', 'comment_cnt',
#        'comment_user_num', 'direct_comment_cnt', 'reply_comment_cnt',
#        'delete_comment_cnt', 'delete_comment_user_num', 'comment_like_cnt',
#        'comment_like_user_num', 'follow_cnt', 'follow_user_num',
#        'cancel_follow_cnt', 'cancel_follow_user_num', 'share_cnt',
#        'share_user_num', 'download_cnt', 'download_user_num', 'report_cnt',
#        'report_user_num', 'reduce_similar_cnt', 'reduce_similar_user_num',
#        'collect_cnt', 'collect_user_num', 'cancel_collect_cnt',
#        'cancel_collect_user_num'],
#       dtype='object')
item_daily_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_daily_features.csv")


# Index(['user_id', 'friend_list'], dtype='object')
social_network = pd.read_csv("data_final_project/KuaiRec 2.0/data/social_network.csv")


# Index(['user_id', 'user_active_degree', 'is_lowactive_period',
#        'is_live_streamer', 'is_video_author', 'follow_user_num',
#        'follow_user_num_range', 'fans_user_num', 'fans_user_num_range',
#        'friend_user_num', 'friend_user_num_range', 'register_days',
#        'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
#        'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
#        'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10',
#        'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
#        'onehot_feat15', 'onehot_feat16', 'onehot_feat17'],
#       dtype='object')
user_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/user_features.csv")

# Explorer les données
print("\n--- Aperçu des interactions_train ---")
print(f"Dimensions: {interactions_train.shape}")
print(interactions_train.head())
print(interactions_train.info())
print("Statistiques descriptives des interactions_train:")
print(interactions_train.describe())

print("\n--- Aperçu des catégories d'items ---")
print(f"Dimensions: {item_categories.shape}")
print(item_categories.head())
# Analyser la structure des catégories d'items (si c'est un vecteur)
if 'feat' in item_categories.columns:
    # Prendre quelques exemples pour comprendre la structure
    sample_feats = item_categories['feat'].head(3)
    print("Exemples de vecteurs de caractéristiques des items:")
    for feat in sample_feats:
        print(feat[:100] + "..." if len(str(feat)) > 100 else feat)

print("\n--- Aperçu des catégories et légendes ---")
print(f"Dimensions: {kuairec_caption_category.shape}")
print(kuairec_caption_category.head())
print("Distribution des catégories principales:")
if 'first_level_category_name' in kuairec_caption_category.columns:
    print(kuairec_caption_category['first_level_category_name'].value_counts().head(10))

print("\n--- Aperçu des caractéristiques quotidiennes des items ---")
print(f"Dimensions: {item_daily_features.shape}")
print(item_daily_features.head())
print("Métriques d'engagement moyennes:")
engagement_cols = ['play_cnt', 'like_cnt', 'comment_cnt', 'share_cnt']
print(item_daily_features[engagement_cols].describe()) 

print("\nNumber of unique video_id values:", item_daily_features['video_id'].nunique()) # = 10728

print("\n--- Aperçu du réseau social ---")
print(f"Dimensions: {social_network.shape}")
print(social_network.head())
# Analyser la structure des listes d'amis
if 'friend_list' in social_network.columns:
    # Prendre quelques exemples pour comprendre la structure
    sample_friends = social_network['friend_list'].head(3)
    print("Exemples de listes d'amis:")
    for friends in sample_friends:
        print(friends[:100] + "..." if len(str(friends)) > 100 else friends)

print("\n--- Aperçu des caractéristiques utilisateurs ---")
print(f"Dimensions: {user_features.shape}")
print(user_features.head())
print("Distribution de l'activité des utilisateurs:")
if 'user_active_degree' in user_features.columns:
    print(user_features['user_active_degree'].value_counts())

print("\nNumber of user_id values:", user_features['user_id'].nunique()) # = 7176

# Analyser les interactions_train
print("\n--- Analyse des interactions_train ---")
print("Nombre d'utilisateurs uniques:", interactions_train['user_id'].nunique())
print("Nombre de vidéos uniques:", interactions_train['video_id'].nunique())
print("Nombre total d'interactions_train:", len(interactions_train))
print("Densité de la matrice utilisateur-item:", 
      len(interactions_train) / (interactions_train['user_id'].nunique() * interactions_train['video_id'].nunique()))

# Analyser les ratios de visionnement
if 'watch_ratio' in interactions_train.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(interactions_train['watch_ratio'], bins=20)
    plt.title('Distribution des ratios de visionnement')
    plt.xlabel('Ratio de visionnement')
    plt.ylabel('Fréquence')
    plt.savefig('watch_ratio_distribution.png')
    plt.close()
    
    # Relation entre durée de visionnement et durée de la vidéo
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=interactions_train.sample(1000), x='video_duration', y='play_duration', alpha=0.5)
    plt.title('Relation entre durée de la vidéo et durée de visionnement')
    plt.xlabel('Durée de la vidéo (s)')
    plt.ylabel('Durée de visionnement (s)')
    plt.savefig('play_vs_video_duration.png')
    plt.close()

# Analyser les vidéos les plus populaires
print("\n--- Top 10 des vidéos les plus regardées ---")
video_popularity = interactions_train.groupby('video_id').size().sort_values(ascending=False)
print(video_popularity.head(10))

# Analyser les utilisateurs les plus actifs
print("\n--- Top 10 des utilisateurs les plus actifs ---")
user_activity = interactions_train.groupby('user_id').size().sort_values(ascending=False)
print(user_activity.head(10))

# Vérifier s'il y a des valeurs manquantes
print("\n--- Valeurs manquantes dans les données ---")
print("interactions_train:", interactions_train.isnull().sum().sum())
print("Catégories d'items:", item_categories.isnull().sum().sum())
print("Catégories et légendes:", kuairec_caption_category.isnull().sum().sum())
print("Caractéristiques quotidiennes des items:", item_daily_features.isnull().sum().sum())
print("Réseau social:", social_network.isnull().sum().sum())
print("Caractéristiques utilisateurs:", user_features.isnull().sum().sum())
