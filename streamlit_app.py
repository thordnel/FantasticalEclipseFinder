import streamlit as st
import datetime
import numpy as np
from skyfield.api import load, wgs84
from skyfield.constants import AU_KM
from scipy.optimize import minimize, minimize_scalar

# --- PAGE CONFIG ---
st.set_page_config(page_title="Eclipse Finder", page_icon="🌑")

# --- CACHED LOADING (Speeds up the app) ---
@st.cache_resource
def load_data():
    # Streamlit Cloud handles caching automatically
    eph = load('de421.bsp')
    ts = load.timescale()
    return eph, ts

eph, ts = load_data()
sun = eph['sun']
moon = eph['moon']
earth = eph['earth']

# --- HELPERS (Same as your logic) ---
DEFAULT_SUN_RADIUS_KM = 695700.0
DEFAULT_MOON_RADIUS_KM = 1737.4

def get_angular_diameter_arcsec(radius_km, distance_au):
    if not radius_km or distance_au <= 0: return 0
    return np.degrees(2 * radius_km / (distance_au * AU_KM)) * 3600

def get_altaz(time_obj, loc, body):
    app = loc.at(time_obj).observe(body).apparent()
    alt, az, _ = app.altaz()
    return alt, az, app

def shortest_angle_diff(a1, a2):
    return (a2 - a1 + 180) % 360 - 180

def check_alignment_at_offset(offset, lat1, lon1, t1_base, body1_name, lat2, lon2, t2_base, body2_name):
    b1 = eph[body1_name]
    b2 = eph[body2_name]
    
    t1_curr = t1_base + datetime.timedelta(seconds=offset)
    t2_curr = t2_base + datetime.timedelta(seconds=offset)
    
    st1 = ts.utc(t1_curr)
    st2 = ts.utc(t2_curr)
    
    loc1 = earth + wgs84.latlon(lat1, lon1)
    loc2 = earth + wgs84.latlon(lat2, lon2)

    alt1, az1, dist1 = get_altaz(st1, loc1, b1)
    alt2, az2, dist2 = get_altaz(st2, loc2, b2)
    
    d_alt = alt1.degrees - alt2.degrees
    d_az = shortest_angle_diff(az1.degrees, az2.degrees)
    sep = np.sqrt(d_alt**2 + d_az**2)
    
    return sep, t1_curr, t2_curr, dist1, dist2

def calculate_overlap(sep, dist1, body1_name, dist2, body2_name):
    r1 = DEFAULT_SUN_RADIUS_KM if body1_name == 'sun' else DEFAULT_MOON_RADIUS_KM
    r2 = DEFAULT_MOON_RADIUS_KM if body2_name == 'moon' else DEFAULT_SUN_RADIUS_KM
    ang1 = get_angular_diameter_arcsec(r1, dist1.distance().au)
    ang2 = get_angular_diameter_arcsec(r2, dist2.distance().au)
    rad1 = (ang1/2)/3600
    rad2 = (ang2/2)/3600
    sum_radii = rad1 + rad2
    if sep < sum_radii:
        return max(0.0, (sum_radii - sep) / sum_radii) * 100.0
    return 0.0

def optimize_time_offset(lat1, lon1, t1, body1, lat2, lon2, t2, body2, window_hours):
    def objective(x):
        sep, *rest = check_alignment_at_offset(x, lat1, lon1, t1, body1, lat2, lon2, t2, body2)
        return sep
    
    bounds = (-window_hours*3600, window_hours*3600)
    res = minimize_scalar(objective, bounds=bounds, method='bounded')
    best_offset = res.x if res.success else 0.0
    
    sep, ft1, ft2, d1, d2 = check_alignment_at_offset(best_offset, lat1, lon1, t1, body1, lat2, lon2, t2, body2)
    overlap = calculate_overlap(sep, d1, body1, d2, body2)
    
    return {'sep': sep, 'overlap': overlap, 'offset': best_offset, 't1': ft1, 't2': ft2}

# --- THE UI ---

st.title("🌒 Fantastical Eclipse Finder")
st.write("Scan historical dates for alignments between two locations.")

# Sidebar Inputs
with st.sidebar:
    st.header("Locations")
    lat1 = st.number_input("Loc1 (Sun) Lat", value=11.4517, format="%.4f")
    lon1 = st.number_input("Loc1 (Sun) Lon", value=120.9311, format="%.4f")
    st.markdown("---")
    lat2 = st.number_input("Loc2 (Moon) Lat", value=50.3785, format="%.4f")
    lon2 = st.number_input("Loc2 (Moon) Lon", value=4.2306, format="%.4f")
    
    st.header("Search Settings")
    ref_date = st.date_input("Fixed Date (Sun)", datetime.date(1922, 8, 24))
    start_year = st.number_input("Start Year", 1914, 1930, 1914)
    end_year = st.number_input("End Year", 1914, 1930, 1918)
    search_window = st.slider("Window (Hours)", 1, 12, 4)
    tolerance = st.slider("Min Overlap %", 0, 100, 1)

if st.button("Start Scan", type="primary"):
    
    # Progress Bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    # Setup
    offset_loc1 = datetime.timedelta(hours=8)
    offset_loc2 = datetime.timedelta(hours=1)
    
    # Date Setup
    dt1_base = datetime.datetime(ref_date.year, ref_date.month, ref_date.day, 23, 0, tzinfo=datetime.timezone.utc)
    
    total_months = (end_year - start_year + 1) * 12
    current_step = 0
    hits = 0

    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            # Update Progress
            current_step += 1
            progress_bar.progress(current_step / total_months)
            status_text.text(f"Scanning {y}-{m:02d}...")
            
            try:
                dt2_base = datetime.datetime(y, m, ref_date.day, 23, 0, tzinfo=datetime.timezone.utc)
            except: continue

            # Calc
            res = optimize_time_offset(lat1, lon1, dt1_base, 'sun', lat2, lon2, dt2_base, 'moon', search_window)
            
            if res['overlap'] >= tolerance:
                hits += 1
                
                # Formatting
                t1_utc = res['t1']
                t2_utc = res['t2']
                
                utc_str = t1_utc.strftime("%d %b %Y %H:%M:%S UTC")
                
                # Local Time Calcs
                loc1_local = t1_utc + offset_loc1
                loc2_local = t2_utc + offset_loc2
                
                loc1_str = loc1_local.strftime("%d %b %Y %I:%M:%S %p, %A")
                loc2_str = loc2_local.strftime("%d %b %Y %I:%M:%S %p, %A")
                
                # Google Maps Links
                link1 = f"http://maps.google.com/?q={lat1},{lon1}"
                link2 = f"http://maps.google.com/?q={lat2},{lon2}"

                with results_container:
                    st.success(f"**Match Found!** (Overlap: {res['overlap']:.1f}%)")
                    st.markdown(f"""
                    **Loc1 (Sun):** [{lat1:.4f}, {lon1:.4f}]({link1})  
                    🕓 {utc_str} (**{loc1_str}**)
                    
                    **Loc2 (Moon):** [{lat2:.4f}, {lon2:.4f}]({link2})  
                    🕓 {utc_str} (**{loc2_str}**)
                    """)
                    st.divider()

    status_text.text(f"Search Complete. Found {hits} matches.")
    progress_bar.progress(100)
