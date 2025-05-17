## Project Objective

The project objective was to develop a recommender system that suggests short videos to users based on user preferences, interaction histories, and video content using the **KuaiRec dataset**. The challenge is to create a personalised and scalable recommendation engine similar to those used in platforms like TikTok or Kuaishou.

## Solution: Content-Based Video Recommendation System

This project implements a comprehensive content-based filtering recommender system tailored for short video platforms. It leverages user interaction history along with rich, heterogeneous metadata about videos to construct personalized user profiles and generate relevant video recommendations. The system's primary goal is to learn each user's content preferences and use them to rank new videos based on similarity.

## Dataset Overview

We will use the **KuaiRec dataset**, a large-scale, fully-observed dataset collected from the Kuaishou short-video platform.

It contains:

- **User interactions** (views, likes, etc.)
- **Video metadata** (video ID, tags, etc.)
- **Timestamps**

More info: [KuaiRec Paper](https://arxiv.org/abs/2202.10842)

### Download the dataset

You can download the dataset via a wget command:

```bash
wget https://nas.chongminggao.top:4430/datasets/KuaiRec.zip --no-check-certificate
unzip KuaiRec.zip
```

### Dataset Description 

KuaiRec contains millions of user-item interactions as well as side information including the item categories and a social network. Six files are included in the download data:

- `big_matrix.csv`: Historical interaction data capturing watch behavior between users and videos. Used as the training set for building user profiles.
- `small_matrix.csv`: Separate dataset containing interactions from a different time window. Used exclusively for evaluation. Note that it includes new user-video pairs not present in the training set.
- `item_daily_features.csv`: Aggregated video statistics such as play count, likes, shares, and completions across days. Critical for generating engagement features.
- `item_categories.csv`: Contains semantic tags assigned to videos. Enables category-based encoding of content.
- `kuairec_caption_category.csv`: Offers text captions and hierarchical topic tags, allowing the extraction of textual semantics via NLP techniques

## Methodology

