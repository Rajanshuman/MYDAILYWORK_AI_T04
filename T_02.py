import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import numpy as np

# Sample movie ratings by users
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
    'movie_id': [1, 2, 3, 1, 3, 2, 3, 4, 1, 4],
    'rating': [5, 4, 3, 4, 2, 5, 4, 2, 3, 4]
}

ratings = pd.DataFrame(data)

# Convert to a user-item matrix
user_movie_matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill missing values with 0
user_movie_matrix.fillna(0, inplace=True)

print(user_movie_matrix)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert to a DataFrame for easier understanding
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

print(user_similarity_df)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert to a DataFrame for easier understanding
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

print(user_similarity_df)

def recommend_movies(user_id, num_recommendations=3):
    # Get the user's ratings
    user_ratings = user_movie_matrix.loc[user_id]

    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # Get ratings of similar users
    similar_user_ratings = user_movie_matrix.loc[similar_users.index]

    # Compute the weighted sum of ratings
    weighted_ratings = np.dot(similar_users, similar_user_ratings) / similar_users.sum()

    # Create a recommendation score for each movie
    recommendations = pd.Series(weighted_ratings, index=user_movie_matrix.columns)

    # Exclude movies that the user has already rated
    recommendations = recommendations[user_ratings == 0]

    # Return the top recommendations
    return recommendations.sort_values(ascending=False).head(num_recommendations)

# Example: Recommend movies for user 4
print(recommend_movies(2))
