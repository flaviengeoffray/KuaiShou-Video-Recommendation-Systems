## Project Objective

The project objective was to develop a recommender system that suggests short videos to users based on user preferences, interaction histories, and video content using the **KuaiRec dataset**. The challenge is to create a personalised and scalable recommendation engine similar to those used in platforms like TikTok or Kuaishou.

---

# Content-Based Video Recommendation System

This project implements a comprehensive **content-based filtering recommender system** tailored for short video platforms. It leverages user interaction history along with rich, heterogeneous metadata about videos to construct personalized user profiles and generate relevant video recommendations. The system's primary goal is to learn each user's content preferences and use them to rank new videos based on similarity.

---

## Dataset Overview

**KuaiRec dataset** is a large-scale, fully-observed dataset collected from the Kuaishou short-video platform.

It contains:

- **User interactions** (views, likes, etc.)
- **Video metadata** (video ID, tags, etc.)
- **Timestamps**

More info: [KuaiRec Paper](https://arxiv.org/abs/2202.10842)

### Download the dataset

You can download the dataset via a wget command:

```bash
wget https://nas.chongminggao.top:4430/datasets/KuaiRec.zip --no-check-certificate
unzip KuaiRec.zip -d data_final_project
```

### Dataset Description 

KuaiRec contains millions of user-item interactions as well as side information including the item categories and a social network. Six files are included in the download data:

- `big_matrix.csv`: Historical interaction data capturing watch behavior between users and videos. Used as the training set for building user profiles.
- `small_matrix.csv`: Separate dataset containing interactions from a different time window. Used exclusively for evaluation. Note that it includes new user-video pairs not present in the training set.
- `item_daily_features.csv`: Aggregated video statistics such as play count, likes, shares, and completions across days. Critical for generating engagement features.
- `item_categories.csv`: Contains semantic tags assigned to videos. Enables category-based encoding of content.
- `kuairec_caption_category.csv`: Offers text captions and hierarchical topic tags, allowing the extraction of textual semantics via NLP techniques
- `social_network.csv`: Contains user relationships and social connections.
- `user_features.csv`: Aggregated user statistics such as watch time, likes, and shares across days. Important for generating user engagement features.


## Methodology

### **Data Preprocessing**

* **Data Cleaning**: Remove duplicates, handle missing values, and filter out irrelevant interactions.

### **Feature Engineering**

* **Engagement Metrics**: Derived from raw counters like play count, like count, share count, comment count, valid play count, etc. Ratios like `like_ratio`, `completion_ratio`, and `validity_ratio` are computed to normalize behavior.
* **Popularity & Engagement Scores**: Combined metrics transformed with `log1p` and normalized using MinMaxScaler to indicate video appeal.
* **Category Encoding**: Each video's categories are one-hot encoded to represent semantic tags.
* **Caption Embedding**: Captions are processed using a TF-IDF vectorizer, resulting in 300-dimensional sparse features capturing textual semantics.
* All features are **merged** and **standardized** using `StandardScaler`, forming a dense feature matrix for all videos.

### **User Profile Construction**

* Each user is represented as a vector in the same feature space as the videos.
* A user's profile is calculated as the **weighted average** of feature vectors of positively interacted videos. The weight is proportional to `watch_ratio_clamped`, emphasizing stronger preferences.
* Profiles are normalized to unit vectors to enable cosine similarity computation.

### **Recommendation Engine**

* For each user, candidate videos are scored using **cosine similarity** between their profile and all video feature vectors.
* Top-K videos with the highest similarity scores are selected as recommendations.
* Optionally, videos the user has already seen can be excluded from recommendation.

### **Evaluation Protocol**

* Users from the test set (`small_matrix.csv`) are evaluated based on videos they actually interacted with.
* A video is labeled as **liked** if its `watch_ratio_clamped` exceeds a configurable threshold (e.g., 0.7).
* For each user, the system ranks all test videos they've seen and evaluates:

  * **Precision\@K**: Fraction of top-K videos that were liked
  * **Recall\@K**: Fraction of liked videos retrieved in top-K
  * **NDCG\@K**: Gain that rewards relevant items appearing earlier in the list

---

## Experiments & Results

### Configuration:

* `positive_threshold = 0.7` (default value to define positive interaction)
* Evaluation conducted for `K = 10` (Top-10 ranked recommendations)

### Performance Metrics:

* **Precision\@10**: 0.8147 — High accuracy of top-K predictions
* **Recall\@10**: 0.0053 — Low due to large number of relevant items
* **NDCG\@10**: 0.9222 — Strong ranking quality, with relevant items near the top

> The high **Precision** and **NDCG** reflect that the system ranks the most relevant items correctly. Low **Recall** is expected due to the overwhelming number of positives in the test set (average: \~1850 per user).

### Additional Observations:

* Varying the positive threshold (e.g., 0.5, 0.8) affects precision-recall tradeoffs.
* Increasing `K` beyond 10 improves recall but slightly lowers precision.

---

## Strengths & Limitations

### Advantages:

* **Personalized without peer data**: Doesn’t require other users’ history (no cold-start for new users).
* **Explainable**: Recommendations are driven by interpretable content features.
* **Scalable and modular**: Easily extendable with additional features like audio, image, or deep embeddings.
* **Good top-K performance**: High precision and NDCG even with limited training data.

### Limitations:

* **Low recall** on large item pools: Hard to capture all relevant items with few top predictions.
* **Lack of collaborative signals**: Does not learn user-user correlations.
* **Static preferences**: User profile is fixed once computed; does not evolve over time.
* **Sensitive to feature quality**: Poor metadata or noisy captions can degrade performance.

---

## Conclusion

This content-based recommender system demonstrates strong performance in ranking and prioritizing videos that align with a user's known preferences.

The model is especially effective when item metadata is rich and diverse. However, limitations in recall and the absence of collaborative filtering effects suggest areas for future enhancement.

### Future Improvements:

* Integrating **collaborative filtering** (matrix factorization, embeddings)

---

## Author

**[Flavien Geoffray]**

