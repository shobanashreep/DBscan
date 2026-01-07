import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

# ---------------------------
# 1️⃣ Load DBSCAN + preprocessing
# ---------------------------
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
dbscan = joblib.load('dbscan.pkl')
X_reduced = joblib.load('X_reduced.pkl')      # preprocessed training data
labels_train = joblib.load('labels_train.pkl')  # cluster labels for training data

# Feature names
features = ['StudyHours','Attendance','OnlineCourses',
            'AssignmentCompletion','StressLevel','Motivation']

# Cluster descriptions
cluster_description = {
    -1: "Unusual / rare behavior (outlier)",
     0: "Low-motivation routine learners",
     1: "Moderately motivated consistent learners",
     2: "Highly motivated structured learners"
}

# ---------------------------
# 2️⃣ App UI
# ---------------------------
st.title("DBSCAN Student Behavior Clustering with Outliers Highlighted")
st.write("Adjust the student's features and see the predicted cluster and position in the graph.")

user_input = []
for feature in features:
    if feature in ['StressLevel', 'Motivation']:
        val = st.slider(feature, 0, 10, 1)
    elif feature == 'OnlineCourses':
        val = st.number_input(feature, 0, 100, 10)
    elif feature == 'AssignmentCompletion':
        val = st.number_input(feature, 0, 100, 75)
    elif feature == 'Attendance':
        val = st.number_input(feature, 0, 100, 80)  # realistic max 100
    else:  # StudyHours or any other numeric features
        val = st.number_input(feature, 0, 50, 20)   # adjust range as needed
    user_input.append(val)

user_array = np.array(user_input).reshape(1, -1)

# ---------------------------
# 3️⃣ Scale & PCA transform
# ---------------------------
user_scaled = scaler.transform(user_array)
user_reduced = pca.transform(user_scaled)

# ---------------------------
# 4️⃣ Assign cluster using nearest core point
# ---------------------------
core_indices = np.where(labels_train != -1)[0]   # core points
core_points = X_reduced[core_indices]
core_labels = labels_train[core_indices]

distances = pairwise_distances(user_reduced, core_points)
min_dist = np.min(distances)
nearest_label = core_labels[np.argmin(distances)]

if min_dist <= dbscan.eps:
    assigned_cluster = nearest_label
else:
    assigned_cluster = -1

# Display cluster info
st.subheader("Predicted Behavior Pattern:")
st.write(f"Cluster: {assigned_cluster}")
st.write(f"Interpretation: {cluster_description.get(assigned_cluster, 'Unknown')}")

# ---------------------------
# 5️⃣ 2D Visualization (highlight outliers)
# ---------------------------
# Reduce training points to 2D for visualization
pca2d = PCA(n_components=2)
X_2d = pca2d.fit_transform(X_reduced)
user_2d = pca2d.transform(user_reduced)

plt.figure(figsize=(8,6))

# Plot each cluster
for cluster in np.unique(labels_train):
    mask = labels_train == cluster
    if cluster == -1:
        plt.scatter(X_2d[mask,0], X_2d[mask,1],
                    color='red', marker='x', s=80, alpha=0.7, label='Outliers (-1)')
    else:
        plt.scatter(X_2d[mask,0], X_2d[mask,1],
                    alpha=0.5, label=f'Cluster {cluster}')

# Highlight the new user
if assigned_cluster == -1:
    plt.scatter(user_2d[0,0], user_2d[0,1], color='darkred', s=150, marker='*', label='New Student (Outlier)')
else:
    plt.scatter(user_2d[0,0], user_2d[0,1], color='green', s=150, marker='*', label='New Student')

plt.title("DBSCAN Clusters (2D PCA Projection)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.grid(True)
st.pyplot(plt)