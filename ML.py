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
    X_scaled = StandardScaler().fit_transform(features)

    # Apply k-means
    k = 4  # example number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)



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


    import pandas as pd

    # Summarize average conditions per cluster
    summary = data.groupby("zone")[['soil_moisture', 'N', 'P', 'K', 'pH', 'temp']].mean()

    # Define thresholds and tips
    tips = {}
    for zone, row in summary.iterrows():
        recs = []
        # Soil moisture
        if row['soil_moisture'] < 25:
            recs.append("Low moisture → increase irrigation")
        elif row['soil_moisture'] > 70:
            recs.append("High moisture → reduce irrigation")
        # Nitrogen
        if row['N'] < 20:
            recs.append("Low nitrogen → add N fertilizer")
        elif row['N'] > 50:
            recs.append("High nitrogen → monitor over-fertilization")
        # Phosphorus
        if row['P'] < 15:
            recs.append("Low phosphorus → apply P fertilizer")
        # Potassium
        if row['K'] < 15:
            recs.append("Low potassium → apply K fertilizer")
        # pH
        if row['pH'] < 5.5:
            recs.append("Soil acidic → apply lime")
        elif row['pH'] > 7.5:
            recs.append("Soil alkaline → apply sulfur or acidifier")
        # Temperature (example thresholds)
        if row['temp'] < 10:
            recs.append("Cold zone → planting risk")
        elif row['temp'] > 35:
            recs.append("Hot zone → heat stress risk")
        # Default if no issues
        if not recs:
            recs.append("Conditions are optimal")

        tips[zone] = "; ".join(recs)

    print("Tips per zone:\n")
    for z, t in tips.items():
        print(f"Zone {z}: {t}")

        # Center map on field mean location
        center = [data.lat.mean(), data.lon.mean()]
        m = folium.Map(location=center, zoom_start=17, tiles=None)
        # Add markers for each point colored by zone
        for _, row in data.iterrows():
            folium.CircleMarker(
                location=[row.lat, row.lon],
                radius=4,
                color=zone_color[row.zone],
                fill=True,
                fill_opacity=0.7,
                popup=f"Zone: {row.zone}\nTips: {tips[row.zone]}"
            ).add_to(m)
        if first:
            m.save('field_zones_pattern.html')
            first = False
        else:
            m.save('field_zones_random.html')




