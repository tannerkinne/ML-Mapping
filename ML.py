from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd


# Read collected data
data1 = pd.read_csv('fake_field_pattern.csv')
data2 = pd.read_csv('fake_field_random.csv')
data_list = [data1, data2]  # put them in a list
first = True
for data in data_list:

    # Preprocess
    features = data[['soil_moisture','N','P','K','pH','temp']].values
    X_scaled = StandardScaler().fit_transform(features)  # normalize features:contentReference[oaicite:13]{index=13}

    # Apply k-means
    k = 4  # example number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)               # clustering:contentReference[oaicite:14]{index=14}



    # Assign zones and save
    data['zone'] = labels
    if first:
        data.to_csv('soil_zones_pattern.csv', index=False)
    else:
        data.to_csv('soil_zones_random.csv', index=False)



    import folium

    zone_color = {
        0: "blue",
        1: "green",
        2: "orange",
        3: "red"
    }

    # Center map on field mean location
    center = [data.lat.mean(), data.lon.mean()]
    m = folium.Map(location=center, zoom_start=17, tiles=None)  # offline mode:contentReference[oaicite:16]{index=16}
    # Add markers for each point colored by zone
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=4,
            color=zone_color[row.zone],
            fill=True,
            fill_opacity=0.7,
            popup=f"Zone: {row.zone}\nMoisture: {row.soil_moisture:.1f}%\nN: {row.N:.1f}"
        ).add_to(m)
    if first:
        m.save('field_zones_pattern.html')
        first = False
    else:
        m.save('field_zones_random.html')
