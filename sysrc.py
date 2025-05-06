# Import libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Index(['user_id', 'video_id', 'play_duration', 'video_duration', 'time',
#        'date', 'timestamp', 'watch_ratio'],
#       dtype='object')
interactions = pd.read_csv("data_final_project/KuaiRec 2.0/data/small_matrix.csv")


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
