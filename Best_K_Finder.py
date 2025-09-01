from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt


# Read collected data
data = pd.read_csv('fake_field_random.csv')

# Prepare empty list to store results
rows = []

# Try different k values for elbow method
for k in range(1, 50):
    # Preprocess
    features = data[['soil_moisture', 'N', 'P', 'K', 'pH', 'temp']].values
    X_scaled = StandardScaler().fit_transform(features)  # normalize features

    # Apply k-means
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(X_scaled)  # fit model

    wcss = kmeans.inertia_
    print(f"WCSS for k={k}: {wcss}")

    rows.append({"K": k, "WCSS": wcss})

# Convert list of dicts to DataFrame
table = pd.DataFrame(rows)
print(table)

plt.figure(figsize=(8,6))
plt.plot(table["K"], table["WCSS"], marker='o')
plt.xticks(table["K"])
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal K')
plt.grid(True)
plt.show()