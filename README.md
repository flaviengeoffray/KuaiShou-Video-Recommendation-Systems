## Project Objective

The project objective was to develop a recommender system that suggests short videos to users based on user preferences, interaction histories, and video content using the **KuaiRec dataset**. The challenge is to create a personalised and scalable recommendation engine similar to those used in platforms like TikTok or Kuaishou.

---

## Dataset Overview

**KuaiRec dataset** is a large-scale, fully-observed dataset collected from the Kuaishou short-video platform.

It contains:

- **User interactions** (views, likes, etc.)
- **Video metadata** (video ID, tags, etc.)
- **Timestamps**

More info: [KuaiRec Paper](https://arxiv.org/abs/2202.10842)

### Download the dataset

You can download the dataset by running the provided shell script:

```bash
sh download_data.sh
```

This script will automatically download and extract the dataset to the `data_final_project` directory.

### Dataset Description 

KuaiRec contains millions of user-item interactions as well as side information including the item categories and a social network. Six files are included in the download data:

- `big_matrix.csv`: Historical interaction data capturing watch behavior between users and videos. Used as the training set for building user profiles.
- `small_matrix.csv`: Separate dataset containing interactions from a different time window. Used exclusively for evaluation. Note that it includes new user-video pairs not present in the training set.
- `item_daily_features.csv`: Aggregated video statistics such as play count, likes, shares, and completions across days. Critical for generating engagement features.
- `item_categories.csv`: Contains semantic tags assigned to videos. Enables category-based encoding of content.
- `kuairec_caption_category.csv`: Offers text captions and hierarchical topic tags, allowing the extraction of textual semantics via NLP techniques
- `social_network.csv`: Contains user relationships and social connections.
- `user_features.csv`: Aggregated user statistics such as watch time, likes, and shares across days. Important for generating user engagement features.

## Recommendation Approaches Tested

During this project, we implemented and tested three different recommendation approaches to compare their effectiveness:

### 1. Random Baseline

A random recommender was implemented as a baseline to establish minimum performance expectations. This approach simply suggests videos to users completely at random without considering any user preferences or video characteristics.

**Key characteristics:**
- No personalization
- Simple implementation
- Establishes a performance floor
- Serves as a sanity check for evaluation metrics

### 2. Collaborative Filtering

Collaborative filtering was tested using both memory-based (item-item similarity) and model-based (matrix factorization) approaches. This method leverages patterns in user-item interactions to identify similarities between users or items.

**Key characteristics:**
- User-User approach: Recommends items liked by similar users
- Item-Item approach: Recommends items similar to those a user has liked
- Matrix Factorization: Decomposes user-item matrix into latent factors
- Works without needing content features
- Suffers from cold-start and sparsity issues

### 3. Content-Based Filtering

This approach, which became our final recommendation system, analyzes video characteristics and user preferences to make personalized recommendations.

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

## Results Comparison

Below is a comparison of performance metrics across all three recommendation approaches:

| Metric | Random Baseline | Collaborative Filtering (Item-Item) | Collaborative Filtering (Matrix Factorization) | Content-Based |
|--------|----------------|-----------------------------------|---------------------------------------------|--------------|
| **@K=5** |
| Precision | 0.3926 | 0.5114 | 0.6023 | 0.8546 |
| Recall | 0.0012 | 0.0018 | 0.0019 | 0.0023 |
| NDCG | 0.5032 | 0.6234 | 0.7120 | 0.9382 |
| **@K=10** |
| Precision | 0.3807 | 0.4853 | 0.5721 | 0.8147 |
| Recall | 0.0026 | 0.0032 | 0.0038 | 0.0053 |
| NDCG | 0.4915 | 0.5972 | 0.6893 | 0.9222 |
| **@K=20** |
| Precision | 0.3719 | 0.4634 | 0.5505 | 0.8492 |
| Recall | 0.0046 | 0.0059 | 0.0073 | 0.0093 |
| NDCG | 0.4829 | 0.5847 | 0.6781 | 0.9408 |

*Note: Collaborative filtering metrics are approximate and may vary with different runs and parameter settings.*

The comparison clearly shows the **content-based approach significantly outperforms** both the random baseline and collaborative filtering methods on all metrics. While collaborative filtering shows improvement over the random baseline, it still falls considerably short of the content-based method's performance.

---

## Strengths & Limitations

### Advantages:

* **Personalized without peer data**: Doesn't require other users' history (no cold-start for new users).
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

This comparison of recommendation approaches demonstrates that the content-based filtering system performs best for the KuaiRec dataset, showing strong performance in ranking and prioritizing videos that align with a user's known preferences.

The model is especially effective when item metadata is rich and diverse. However, limitations in recall and the absence of collaborative filtering effects suggest areas for future enhancement.

### Future Improvements:

* Creating a **hybrid system** combining content-based and collaborative approaches
* Incorporating **temporal dynamics** to capture evolving user preferences
* Exploring **deep learning** for better feature extraction from video content

---

## Author

**[Flavien Geoffray]**
**[Lucas Duport]**

