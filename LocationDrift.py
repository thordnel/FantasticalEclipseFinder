import math

# Parameters: adjust based on your location
plate_velocity_cm_per_year = 2.3  # Plate speed in cm/year (e.g., 2.3 cm/year for North American Plate)
plate_direction_deg = 270  # Movement direction (azimuth from north, e.g., west is 270°)

years = 100

# Convert to meters over 100 years
total_shift_m = (plate_velocity_cm_per_year / 100) * years

# Convert azimuth to radians
direction_rad = math.radians(plate_direction_deg)

# Calculate shifts in meters for north (latitude) and east (longitude)
delta_north_m = total_shift_m * math.cos(direction_rad)
delta_east_m = total_shift_m * math.sin(direction_rad)

# Convert shifts to degrees
deg_per_meter_lat = 1 / 111000  # 1 degree latitude ~111 km
deg_per_meter_lon = 1 / (111000 * math.cos(math.radians(45)))  # adjust for latitude; assume ~45° N

delta_lat_deg = delta_north_m * deg_per_meter_lat
delta_lon_deg = delta_east_m * deg_per_meter_lon

# Results
print(f"Total shift over {years} years: {total_shift_m:.2f} meters")
print(f"Northward shift: {delta_north_m:.2f} m (~{delta_lat_deg:.8f}° latitude)")
print(f"Eastward shift: {delta_east_m:.2f} m (~{delta_lon_deg:.8f}° longitude)")

