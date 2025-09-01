import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Settings
n_samples = 500
start_time = datetime.now()

# Rough GPS bounds for a rectangular "field" in PA
lat_start, lat_end = 40.0, 40.002
lon_start, lon_end = -75.0, -74.998

# Generate fake data
data = []
for i in range(n_samples):
    timestamp = start_time + timedelta(seconds=i * 5)

    lat = random.uniform(lat_start, lat_end)
    lon = random.uniform(lon_start, lon_end)

    # Soil sensor values (add noise for realism)
    soil_moisture = np.clip(np.random.normal(25, 5), 5, 45)  # %VWC
    nitrogen = np.clip(np.random.normal(30, 10), 5, 60)  # mg/kg
    phosphorus = np.clip(np.random.normal(20, 5), 5, 40)  # mg/kg
    potassium = np.clip(np.random.normal(40, 15), 10, 80)  # mg/kg
    pH = np.clip(np.random.normal(6.5, 0.5), 5.5, 8.0)  # pH units
    temp = np.clip(np.random.normal(20, 3), 10, 35)  # °C

    data.append([timestamp, lat, lon, soil_moisture, nitrogen, phosphorus, potassium, pH, temp])

# Create DataFrame
df = pd.DataFrame(data, columns=[
    "timestamp", "lat", "lon", "soil_moisture", "N", "P", "K", "pH", "temp"
])

# Save CSV
file_path = "fake_field_random.csv"
df.to_csv(file_path, index=False)




# Set random seed for reproducibility
np.random.seed(42)

# Create grid for field (lat/lon)
num_rows, num_cols = 20, 20  # 400 points
lat_base, lon_base = 40.0, -76.0  # arbitrary origin
lat = np.linspace(lat_base, lat_base + 0.01, num_rows)
lon = np.linspace(lon_base, lon_base + 0.01, num_cols)

# Create meshgrid
lat_grid, lon_grid = np.meshgrid(lat, lon)
lat_flat = lat_grid.flatten()
lon_flat = lon_grid.flatten()

# Generate synthetic soil data with patterns
# We'll make four zones with different NPK/pH/moisture
zone_size = len(lat_flat) // 4
soil_moisture = np.concatenate([
    np.random.normal(30, 2, zone_size),  # zone 0
    np.random.normal(40, 2, zone_size),  # zone 1
    np.random.normal(50, 2, zone_size),  # zone 2
    np.random.normal(60, 2, len(lat_flat) - 3*zone_size)  # zone 3
])

N = np.concatenate([
    np.random.normal(50, 5, zone_size),
    np.random.normal(80, 5, zone_size),
    np.random.normal(120, 5, zone_size),
    np.random.normal(150, 5, len(lat_flat) - 3*zone_size)
])

P = np.concatenate([
    np.random.normal(10, 2, zone_size),
    np.random.normal(15, 2, zone_size),
    np.random.normal(20, 2, zone_size),
    np.random.normal(25, 2, len(lat_flat) - 3*zone_size)
])

K = np.concatenate([
    np.random.normal(50, 5, zone_size),
    np.random.normal(100, 5, zone_size),
    np.random.normal(150, 5, zone_size),
    np.random.normal(200, 5, len(lat_flat) - 3*zone_size)
])

pH = np.concatenate([
    np.random.normal(6.0, 0.2, zone_size),
    np.random.normal(6.5, 0.2, zone_size),
    np.random.normal(7.0, 0.2, zone_size),
    np.random.normal(7.5, 0.2, len(lat_flat) - 3*zone_size)
])

temp = np.random.normal(25, 1, len(lat_flat))  # temperature can be uniform

# Build DataFrame
data = pd.DataFrame({
    'lat': lat_flat,
    'lon': lon_flat,
    'soil_moisture': soil_moisture,
    'N': N,
    'P': P,
    'K': K,
    'pH': pH,
    'temp': temp
})

# Save to CSV for testing
data.to_csv('fake_field_pattern.csv', index=False)
print("Fake patterned field data saved to 'fake_field_pattern.csv'")
