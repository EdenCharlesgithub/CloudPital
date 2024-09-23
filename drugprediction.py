import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process

# Load the data from CSV file (replace 'final.csv' with your file)
data = pd.read_csv('final.csv')

# Assume the CSV file has columns 'disease' and 'drug'
diseases = data['disease']
medicines = data['drug']

# Function to find similar diseases and recommend medicine
def recommend_medicine(disease, diseases=diseases, medicines=medicines):
    # Use fuzzy matching to find the closest matching disease
    match = process.extractOne(disease, diseases, scorer=fuzz.partial_ratio)
    
    # Extract the best match and match score
    if match:
        best_match = match[0]  # Disease name
        match_score = match[1]  # Similarity score

        # If the match score is above a certain threshold (e.g., 70), we proceed
        if match_score > 70:
            print(f"Did you mean: {best_match}? (Match score: {match_score})\n")

            # Combine diseases and medicines into a single string for vectorization
            combined_features = diseases + " " + medicines

            # Use TF-IDF Vectorizer to convert text data into vectors
            vectorizer = TfidfVectorizer().fit_transform(combined_features)

            # Get the vector of the matched disease
            disease_idx = diseases[diseases == best_match].index[0]
            input_vector = vectorizer[disease_idx]
            similarity_scores = cosine_similarity(input_vector, vectorizer).flatten()

            # Get the indices of the most similar diseases (excluding the best match itself)
            similar_diseases_idx = similarity_scores.argsort()[::-1][1:4]  # top 3 similar diseases

            # Create a list to store similar medicines
            recommended_medicines = []
            for idx in similar_diseases_idx:
                recommended_medicines.append(medicines[idx])

            return recommended_medicines
        else:
            return
    
