import streamlit as st
import datetime
import numpy as np
from skyfield.api import load, wgs84
from skyfield.constants import AU_KM
from scipy.optimize import minimize, minimize_scalar

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Eclipse Finder", page_icon="🌗", layout="wide")

# --- 2. CUSTOM CSS (Dark Mode Cards) ---
st.markdown("""
<style>
    .hit-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .hit-card h3 { margin-top: 0; color: #fff; }
    .hit-card p { margin: 5px 0; color: #e0e0e0; }
    
    /* Drift Box Styling */
    .drift-box {
        background-color: #0e1117;
        padding: 10px;
        margin-top: 10px;
        border-radius: 5px;
        font-size: 0.9em;
        border: 1px solid #444;
    }
    
    /* Link Styling */
    a { color: #4fc3f7 !important; text-decoration: underline; }
    a:hover { color: #81d4fa !important; }
</style>
""", unsafe_allow_html=True)

# --- 3. LOAD DATA (Cached for Speed) ---
@st.cache_resource
def load_skyfield_data():
    # Only loads the heavy file once
    eph = load('de421.bsp')
    ts = load.timescale()
    return eph, ts

eph, ts = load_skyfield_data()
sun = eph['sun']
moon = eph['moon']
earth = eph['earth']

# --- 4. HELPER FUNCTIONS ---

def format_latlon_link(lat, lon):
    """Returns a clickable Google Maps HTML link."""
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    text_label = f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"
    # Standard Google Maps Query URL
    url = f"https://www.google.com/maps?q={lat},{lon}"
    return f'<a href="{url}" target="_blank">{text_label}</a>'

def get_altaz(time_obj, loc, body):
    app = loc.at(time_obj).observe(body).apparent()
    alt, az, _ = app.altaz()
    return alt, az, app

def shortest_angle_diff(a1, a2):
    return (a2 - a1 + 180) % 360 - 180

def get_angular_diameter_arcsec(radius_km, distance_au):
    if not radius_km or distance_au <= 0: return 0
    return np.degrees(2 * radius_km / (distance_au * AU_KM)) * 3600

# --- 5. CORE OPTIMIZATION LOGIC ---

def check_alignment_at_offset(offset, lat1, lon1, t1_base, body1_obj, lat2, lon2, t2_base, body2_obj):
    # Calculate exact times
    t1_curr = t1_base + datetime.timedelta(seconds=offset)
    t2_curr = t2_base + datetime.timedelta(seconds=offset)
    
    st1 = ts.utc(t1_curr)
    st2 = ts.utc(t2_curr)
    
    loc1 = earth + wgs84.latlon(lat1, lon1)
    loc2 = earth + wgs84.latlon(lat2, lon2)

    alt1, az1, dist1 = get_altaz(st1, loc1, body1_obj)
    alt2, az2, dist2 = get_altaz(st2, loc2, body2_obj)
    
    d_alt = alt1.degrees - alt2.degrees
    d_az = shortest_angle_diff(az1.degrees, az2.degrees)
    sep = np.sqrt(d_alt**2 + d_az**2)
    
    return sep, t1_curr, t2_curr, dist1, dist2, alt1.degrees, alt2.degrees

def optimize_time_offset(lat1, lon1, t1, b1_obj, lat2, lon2, t2, b2_obj, window_hours):
    def objective(x):
        sep, *rest = check_alignment_at_offset(x, lat1, lon1, t1, b1_obj, lat2, lon2, t2, b2_obj)
        return sep
    
    bounds = (-window_hours*3600, window_hours*3600)
    res = minimize_scalar(objective, bounds=bounds, method='bounded')
    best_offset = res.x if res.success else 0.0
    
    sep, ft1, ft2, d1, d2, alt1, alt2 = check_alignment_at_offset(best_offset, lat1, lon1, t1, b1_obj, lat2, lon2, t2, b2_obj)
    
    # Calculate overlap
    r1 = 695700.0 # Sun Radius
    r2 = 1737.4   # Moon Radius
    ang1 = get_angular_diameter_arcsec(r1, d1.distance().au)
    ang2 = get_angular_diameter_arcsec(r2, d2.distance().au)
    sum_radii_deg = ((ang1/2 + ang2/2) / 3600)
    
    overlap = 0.0
    if sep < sum_radii_deg:
        overlap = max(0.0, (sum_radii_deg - sep) / sum_radii_deg) * 100.0

    return {
        'sep': sep, 'overlap': overlap, 'offset': best_offset,
        't1': ft1, 't2': ft2, 'alt1': alt1, 'alt2': alt2,
        'final_lat1': lat1, 'final_lon1': lon1,
        'final_lat2': lat2, 'final_lon2': lon2
    }

def optimize_spatial(base_res, lat1, lon1, lat2, lon2, t1_base, b1_obj, t2_base, b2_obj, limit_km):
    safe_limit = limit_km / 1.4142 
    lat_b = safe_limit / 111.32
    
    def get_lon_bound(lat_val):
        if abs(lat_val) >= 89.0: return 180.0
        return safe_limit / (111.32 * np.cos(np.radians(lat_val)))

    bounds = [(-lat_b, lat_b), (-get_lon_bound(lat1), get_lon_bound(lat1)), 
              (-lat_b, lat_b), (-get_lon_bound(lat2), get_lon_bound(lat2))]
    
    def objective(x):
        sep, *rest = check_alignment_at_offset(
            base_res['offset'], lat1+x[0], lon1+x[1], t1_base, b1_obj,
            lat2+x[2], lon2+x[3], t2_base, b2_obj
        )
        return sep

    opt = minimize(objective, [0.0]*4, bounds=bounds, method='L-BFGS-B')
    x = opt.x
    
    sep, ft1, ft2, d1, d2, alt1, alt2 = check_alignment_at_offset(
        base_res['offset'], lat1+x[0], lon1+x[1], t1_base, b1_obj,
        lat2+x[2], lon2+x[3], t2_base, b2_obj
    )
    
    # Recalculate overlap
    r1 = 695700.0 
    r2 = 1737.4 
    ang1 = get_angular_diameter_arcsec(r1, d1.distance().au)
    ang2 = get_angular_diameter_arcsec(r2, d2.distance().au)
    sum_radii_deg = ((ang1/2 + ang2/2) / 3600)
    overlap = max(0.0, (sum_radii_deg - sep) / sum_radii_deg) * 100.0 if sep < sum_radii_deg else 0.0

    drift1 = np.sqrt((x[0]*111.32)**2 + (x[1]*(111.32*np.cos(np.radians(lat1))))**2)
    drift2 = np.sqrt((x[2]*111.32)**2 + (x[3]*(111.32*np.cos(np.radians(lat2))))**2)
    
    base_res.update({
        'sep': sep, 'overlap': overlap, 'drift1': drift1, 'drift2': drift2,
        'final_lat1': lat1+x[0], 'final_lon1': lon1+x[1],
        'final_lat2': lat2+x[2], 'final_lon2': lon2+x[3]
    })
    return base_res

# --- 6. USER INTERFACE ---

st.title("🌗 Eclipse & Alignment Finder")

# Input Form
with st.container():
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🌞 Loc1 (Sun)**")
        # Defaults: Manila
        lat1 = st.number_input("Lat1", value=14.5995, format="%.4f")
        lon1 = st.number_input("Lon1", value=120.9842, format="%.4f")
        st.caption("Timezone: UTC + 8")
        
    with col2:
        st.markdown("**ME/ASC Loc2 (Moon)**")
        # Defaults: London
        lat2 = st.number_input("Lat2", value=51.5074, format="%.4f")
        lon2 = st.number_input("Lon2", value=-0.1278, format="%.4f")
        st.caption("Timezone: UTC + 1")

    st.markdown("---")
    
    c3, c4, c5 = st.columns(3)
    with c3:
        ref_date = st.date_input("Reference Date", value=datetime.date(1922, 8, 24))
        ref_time = st.time_input("Reference Time (UTC)", value=datetime.time(23, 0))
    with c4:
        start_year = st.number_input("Start Year", value=1920)
        end_year = st.number_input("End Year", value=1925)
    with c5:
        radius_km = st.number_input("Search Radius (km)", value=150.0)

# --- 7. EXECUTION LOGIC ---

if st.button("Start Calculation", type="primary"):
    
    # UI Elements for progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()

    # Define Offsets (UTC+8 and UTC+1)
    offset_loc1 = datetime.timedelta(hours=8)
    offset_loc2 = datetime.timedelta(hours=1)
    
    # Prepare Search Params
    dt1_base = datetime.datetime.combine(ref_date, ref_time).replace(tzinfo=datetime.timezone.utc)
    
    total_steps = (end_year - start_year + 1) * 12
    current_step = 0
    matches = 0

    status_text.write("Initializing search...")

    # Main Search Loop
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            current_step += 1
            progress_bar.progress(int((current_step / total_steps) * 100))
            status_text.write(f"Scanning {y}-{m:02d}...")

            try:
                dt2_base = datetime.datetime(y, m, ref_date.day, ref_time.hour, ref_time.minute, tzinfo=datetime.timezone.utc)
            except ValueError: continue

            # 1. Optimize Time
            res = optimize_time_offset(lat1, lon1, dt1_base, sun, lat2, lon2, dt2_base, moon, 12)
            
            # Check if bodies are visible (Altitude > -1 degree)
            if res['alt1'] > -1 and res['alt2'] > -1:
                final_res = res
                
                # 2. Optimize Space (if close enough)
                if res['sep'] < 5.0:
                    final_res = optimize_spatial(res, lat1, lon1, lat2, lon2, dt1_base, sun, dt2_base, moon, radius_km)

                # 3. Check Tolerance (Overlap > 0.1%)
                if final_res['overlap'] > 0.1:
                    matches += 1
                    
                    # --- Formatting ---
                    t1_utc = final_res['t1']
                    t2_utc = final_res['t2']
                    
                    # Apply Offsets
                    loc1_local = t1_utc + offset_loc1
                    loc2_local = t2_utc + offset_loc2
                    
                    # Format Strings
                    utc_str_1 = t1_utc.strftime("%d %b %Y %H:%M:%S UTC")
                    utc_str_2 = t2_utc.strftime("%d %b %Y %H:%M:%S UTC")
                    
                    loc1_str = loc1_local.strftime("%d %b %Y %I:%M:%S %p, %A")
                    loc2_str = loc2_local.strftime("%d %b %Y %I:%M:%S %p, %A")
                    
                    # Prepare Drift HTML
                    drift_html = ""
                    if final_res.get('drift1', 0) > 1.0 or final_res.get('drift2', 0) > 1.0:
                        drift_html = f"""
                        <div class="drift-box">
                            <strong>📍 Adjusted Locations (Drift):</strong><br>
                            <span style="color: #d32f2f;">Loc1 (Sun):</span> {format_latlon_link(final_res['final_lat1'], final_res['final_lon1'])} (Moved {final_res['drift1']:.1f} km)<br>
                            <span style="color: #1976d2;">Loc2 (Moon):</span> {format_latlon_link(final_res['final_lat2'], final_res['final_lon2'])} (Moved {final_res['drift2']:.1f} km)
                        </div>
                        """
                    
                    # Display Result Card
                    # NOTE: unsafe_allow_html=True is REQUIRED for the links and CSS to work
                    results_container.markdown(f"""
                    <div class="hit-card">
                        <h3>Match Found!</h3>
                        <p><strong>Angle: {final_res['sep']:.4f}°</strong> (Overlap: {final_res['overlap']:.1f}%)</p>
                        <p><strong>Loc1 (Sun):</strong> {utc_str_1} ( {loc1_str} )</p>
                        <p><strong>Loc2 (Moon):</strong> {utc_str_2} ( {loc2_str} )</p>
                        {drift_html}
                    </div>
                    """, unsafe_allow_html=True)

    status_text.write(f"✅ Search Complete. Found {matches} matches.")
    progress_bar.progress(100)