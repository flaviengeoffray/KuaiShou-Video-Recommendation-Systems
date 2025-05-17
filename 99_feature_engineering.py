import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
from tqdm import tqdm
import pickle
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

# Load the preprocessed data
train_data = pd.read_csv('data_final_project/KuaiRec 2.0/data/small_matrix.csv')
test_data = pd.read_csv('data_final_project/KuaiRec 2.0/data/small_matrix.csv')

# Load the additional datasets to extract features
kuairec_caption = pd.read_csv(
    "data_final_project/KuaiRec 2.0/data/kuairec_caption_category.csv",
    engine="python", sep=",", quotechar='"', on_bad_lines='skip'
)
item_categories = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_categories.csv")
item_daily_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/item_daily_features.csv")
social_network = pd.read_csv("data_final_project/KuaiRec 2.0/data/social_network.csv")
user_features = pd.read_csv("data_final_project/KuaiRec 2.0/data/user_features.csv")


# Remove bad ids in kuairec_caption_category
kuairec_caption = kuairec_caption[pd.to_numeric(kuairec_caption['video_id'], errors='coerce').notna()]
kuairec_caption['video_id'] = kuairec_caption['video_id'].astype(int)
print(f"Number of unique video IDs in kuairec_caption: {kuairec_caption['video_id'].nunique()}")


# Process watch_ratio - clamp extreme values to keep them in a reasonable range
# Normally, a watch_ratio > 1 can indicate replays, but clamp to 2.0 as a reasonable max value
train_data['watch_ratio_clamped'] = np.clip(train_data['watch_ratio'], 0, 2.0)
test_data['watch_ratio_clamped'] = np.clip(test_data['watch_ratio'], 0, 2.0)

# Adjust the positive interaction threshold based on data analysis
POSITIVE_THRESHOLD = 0.8
train_data['positive_interaction'] = (train_data['watch_ratio_clamped'] >= POSITIVE_THRESHOLD).astype(int)
test_data['positive_interaction'] = (test_data['watch_ratio_clamped'] >= POSITIVE_THRESHOLD).astype(int)

print(f"Positive interactions in train set: {train_data['positive_interaction'].sum()} / {len(train_data)} ({train_data['positive_interaction'].mean()*100:.2f}%)")
print(f"Positive interactions in test set: {test_data['positive_interaction'].sum()} / {len(test_data)} ({test_data['positive_interaction'].mean()*100:.2f}%)")

# 1.1 Average viewing behavior per user
user_behavior = train_data.groupby('user_id').agg({
    'watch_ratio': ['mean', 'std', 'count'],
    'play_duration': ['mean', 'sum'],
    'positive_interaction': 'sum'
}).reset_index()

user_behavior.columns = ['user_id', 'avg_watch_ratio', 'std_watch_ratio', 'interaction_count', 
                        'avg_play_duration', 'total_play_duration', 'positive_interactions']

# display(user_behavior)

# 1.2 Video duration preference per user
user_duration_pref = train_data.groupby('user_id')['video_duration'].agg(['mean', 'std']).reset_index()
user_duration_pref.columns = ['user_id', 'preferred_video_duration', 'video_duration_std']

# display(user_duration_pref)

# 1.3 Merge with user features from the original dataset
user_features_subset = user_features[['user_id', 'user_active_degree', 'follow_user_num', 
                                     'fans_user_num', 'friend_user_num', 'register_days']]

# Merge all user features into a single DataFrame
user_features_enriched = user_behavior.merge(user_duration_pref, on='user_id', how='left')
user_features_enriched = user_features_enriched.merge(user_features_subset, on='user_id', how='left')

# display(user_features_enriched)


print("Extracting video features...")

# 1. Extract video metrics from item_daily_features
video_metrics = item_daily_features.groupby('video_id').agg({
    'video_duration': 'mean',
    'play_cnt': 'sum',
    'like_cnt': 'sum',
    'comment_cnt': 'sum',
    'share_cnt': 'sum',
    'follow_cnt': 'sum',
    'complete_play_cnt': 'sum',
    'valid_play_cnt': 'sum',
    'play_user_num': 'max',
    'like_user_num': 'max'
}).reset_index()

video_metrics['like_ratio'] = video_metrics['like_cnt'] / video_metrics['play_cnt'].clip(lower=1)
video_metrics['comment_ratio'] = video_metrics['comment_cnt'] / video_metrics['play_cnt'].clip(lower=1)
video_metrics['share_ratio'] = video_metrics['share_cnt'] / video_metrics['play_cnt'].clip(lower=1)
video_metrics['follow_ratio'] = video_metrics['follow_cnt'] / video_metrics['play_cnt'].clip(lower=1)
video_metrics['completion_ratio'] = video_metrics['complete_play_cnt'] / video_metrics['play_cnt'].clip(lower=1)
video_metrics['validity_ratio'] = video_metrics['valid_play_cnt'] / video_metrics['play_cnt'].clip(lower=1)

# MinMaxScaler is used here to scale the computed popularity and engagement scores to a [0, 1] range.
# To ensure the features are on the same scale as other normalized features,
video_metrics['popularity_score'] = MinMaxScaler().fit_transform(
    np.log1p(video_metrics[['play_cnt', 'like_cnt', 'comment_cnt', 'share_cnt']]).sum(axis=1).values.reshape(-1, 1)
)
video_metrics['engagement_score'] = MinMaxScaler().fit_transform(
    (video_metrics[['like_ratio', 'comment_ratio', 'share_ratio', 'follow_ratio']].mean(axis=1)).values.reshape(-1, 1)
)

# 2. Extract category information from item_categories
# Convert string representation of list to actual list
item_categories['feat'] = item_categories['feat'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)

# Create one-hot encoding for categories
category_features = pd.DataFrame(item_categories['video_id'])

# Get all unique categories
all_categories = set()
for categories in item_categories['feat']:
    if isinstance(categories, list):
        all_categories.update(categories)

# Initialize category columns with zeros
for category in all_categories:
    category_features[f'category_{category}'] = 0
    
# Fill in category values
for idx, row in item_categories.iterrows():
    if isinstance(row['feat'], list):
        for category in row['feat']:
            category_features.loc[idx, f'category_{category}'] = 1


# 3. Extract content features from kuairec_caption
caption_df = kuairec_caption.copy()
caption_df = caption_df[['video_id', 'caption']].dropna()
caption_df['caption'] = caption_df['caption'].astype(str)

tfidf = TfidfVectorizer(
    max_features=300,
)

tfidf_matrix = tfidf.fit_transform(caption_df['caption'])

tfidf_features_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=caption_df['video_id'].values,
    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
).reset_index().rename(columns={'index': 'video_id'})

# 4. Merge all features
video_features = video_metrics.merge(category_features, on='video_id', how='left')
video_features = video_features.merge(tfidf_features_df, on='video_id', how='left')

video_features = video_features.fillna(0)

# Create a mapping from video_id to index for quick lookup
video_id_map = {video_id: idx for idx, video_id in enumerate(video_features['video_id'])}

feature_matrix = video_features.drop('video_id', axis=1)

# Normalize features
scaler = StandardScaler()
normalized_features = scaler.fit_transform(feature_matrix)

video_features_matrix = normalized_features

print(f"Extracted features for {len(video_features)} videos with {video_features_matrix.shape[1]} features each")



# 3.1 Create a user-video interaction matrix
def create_interaction_matrix(data, user_col='user_id', item_col='video_id', rating_col='positive_interaction'):
    """
    Create a user-video interaction matrix.
    Returns a sparse matrix in CSR format.
    """
    # Create a mapping for user and video indices
    user_ids = data[user_col].unique()
    video_ids = data[item_col].unique()
    
    user_idx_map = {uid: i for i, uid in enumerate(user_ids)}
    video_idx_map = {vid: i for i, vid in enumerate(video_ids)}
    
    # Create indices for the sparse matrix
    user_indices = [user_idx_map[uid] for uid in data[user_col]]
    video_indices = [video_idx_map[vid] for vid in data[item_col]]
    
    # Create the sparse matrix
    ratings = data[rating_col].values
    interaction_matrix = csr_matrix((ratings, (user_indices, video_indices)), 
                                   shape=(len(user_ids), len(video_ids)))
    
    return interaction_matrix, user_ids, video_ids, user_idx_map, video_idx_map

# Create the interaction matrix for the training set
train_matrix, train_users, train_videos, user_map, video_map = create_interaction_matrix(train_data)
print(f"Sparse interaction matrix created: {train_matrix.shape}")
print(f"Matrix density: {train_matrix.count_nonzero() / (train_matrix.shape[0] * train_matrix.shape[1]):.6f}")

# 5. Processing features for models
print("\n--- Preparing features for models ---")

# Normalize numerical features
user_num_cols = ['avg_watch_ratio', 'interaction_count', 'avg_play_duration', 
                'total_play_duration', 'preferred_video_duration']
video_num_cols = ['avg_watch_ratio', 'interactions_count', 'positive_interactions', 
                 'popularity_score', 'play_cnt', 'like_cnt', 'comment_cnt', 'share_cnt',]

user_scaler = StandardScaler()
video_scaler = StandardScaler()

# Apply normalization
user_features_scaled = user_features.copy()
user_features_scaled[user_num_cols] = user_scaler.fit_transform(
    user_features[user_num_cols].fillna(0))

video_features_scaled = video_features.copy()
video_features_scaled[video_num_cols] = video_scaler.fit_transform(
    video_features[video_num_cols].fillna(0))

# display(user_features_scaled)
# display(video_features_scaled)

print('train_data.columns', train_data.columns)
print('user_features.columns', user_features.columns)
print('video_features.columns', video_features.columns)

print("Feature normalization completed")

# Save all generated features and matrices
user_features.to_csv('data/user_features.csv', index=False)
video_features.to_csv('data/video_features.csv', index=False)

# Save matrices and mappings for model training
with open('data/interaction_matrix.pkl', 'wb') as f:
    pickle.dump({
        'train_matrix': train_matrix,
        'train_users': train_users,
        'train_videos': train_videos,
        'user_map': user_map,
        'video_map': video_map
    }, f)

# Save scalers for prediction
with open('data/scalers.pkl', 'wb') as f:
    pickle.dump({
        'user_scaler': user_scaler,
        'video_scaler': video_scaler
    }, f)

print("Feature engineering completed. Files saved.")
