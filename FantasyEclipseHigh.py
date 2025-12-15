import eel
import time
import datetime
import numpy as np
from skyfield.api import Topos, load, wgs84
from skyfield.constants import AU_KM
from scipy.optimize import minimize, minimize_scalar

# --- Configuration ---
EPHEMERIS_FILE_GLOBAL = 'de421.bsp'
DEFAULT_SUN_RADIUS_KM = 695700.0
DEFAULT_MOON_RADIUS_KM = 1737.4
# Note: SEARCH_RADIUS_KM is now just a fallback default
DEFAULT_SEARCH_RADIUS_KM = 150.0 

eph_main_global = None
planets_main_dict_global = None
ts_main_global = None

# --- Helpers ---
def format_dms(degrees_float):
    """Formats decimal degrees to a standard DMS string."""
    if degrees_float is None or np.isnan(degrees_float): return "N/A"
    degrees_float_abs = abs(degrees_float)
    d = int(degrees_float_abs)
    m = int((degrees_float_abs - d) * 60)
    s = ((degrees_float_abs - d) * 60 - m) * 60
    return f"{d:02d}°{m:02d}'{s:05.2f}\""

def format_latlon(lat, lon):
    """Formats a coordinate pair into a nice N/S E/W string."""
    ns = 'N' if lat >= 0 else 'S'
    ew = 'E' if lon >= 0 else 'W'
    return f"{format_dms(lat)}{ns}, {format_dms(lon)}{ew}"

def get_angular_diameter_arcsec(radius_km, distance_au):
    if not radius_km or distance_au <= 0: return 0
    return np.degrees(2 * radius_km / (distance_au * AU_KM)) * 3600

def get_altaz(time_obj, loc, body):
    app = loc.at(time_obj).observe(body).apparent()
    alt, az, _ = app.altaz()
    return alt, az, app

def shortest_angle_diff(a1, a2):
    return (a2 - a1 + 180) % 360 - 180

def load_skyfield_objects():
    global eph_main_global, planets_main_dict_global, ts_main_global
    if eph_main_global is None:
        try: eel.updateProgress("Loading Ephemeris...")
        except: pass
        eph_main_global = load(EPHEMERIS_FILE_GLOBAL)
        planets_main_dict_global = {
            'sun': eph_main_global['sun'], 
            'moon': eph_main_global['moon'], 
            'earth': eph_main_global['earth']
        }
        ts_main_global = load.timescale()

# --- CORE LOGIC ---

def check_alignment_at_offset(offset, lat1, lon1, t1_base, body1_name, lat2, lon2, t2_base, body2_name):
    """Checks alignment when 'offset' seconds are added to BOTH t1_base and t2_base."""
    b1 = planets_main_dict_global[body1_name]
    b2 = planets_main_dict_global[body2_name]
    earth = planets_main_dict_global['earth']
    
    # Calculate exact times
    t1_curr = t1_base + datetime.timedelta(seconds=offset)
    t2_curr = t2_base + datetime.timedelta(seconds=offset)
    
    st1 = ts_main_global.utc(t1_curr)
    st2 = ts_main_global.utc(t2_curr)
    
    loc1 = earth + wgs84.latlon(lat1, lon1)
    loc2 = earth + wgs84.latlon(lat2, lon2)

    alt1, az1, dist1 = get_altaz(st1, loc1, b1)
    alt2, az2, dist2 = get_altaz(st2, loc2, b2)
    
    d_alt = alt1.degrees - alt2.degrees
    d_az = shortest_angle_diff(az1.degrees, az2.degrees)
    sep = np.sqrt(d_alt**2 + d_az**2)
    
    # Return everything needed for detailed display
    return sep, t1_curr, t2_curr, alt1, az1, alt2, az2, dist1, dist2

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

# --- OPTIMIZERS ---

def optimize_time_offset(lat1, lon1, t1, body1, lat2, lon2, t2, body2, window_hours):
    """Finds best time offset."""
    def objective(x):
        sep, *rest = check_alignment_at_offset(x, lat1, lon1, t1, body1, lat2, lon2, t2, body2)
        return sep
    
    bounds = (-window_hours*3600, window_hours*3600)
    res = minimize_scalar(objective, bounds=bounds, method='bounded')
    best_offset = res.x if res.success else 0.0
    
    sep, ft1, ft2, alt1, az1, alt2, az2, d1, d2 = check_alignment_at_offset(best_offset, lat1, lon1, t1, body1, lat2, lon2, t2, body2)
    overlap = calculate_overlap(sep, d1, body1, d2, body2)
    
    return {
        'sep': sep, 'overlap': overlap, 'offset': best_offset,
        't1': ft1, 't2': ft2, 'alt1': alt1.degrees, 'alt2': alt2.degrees,
        # Default coords (no drift yet)
        'final_lat1': lat1, 'final_lon1': lon1,
        'final_lat2': lat2, 'final_lon2': lon2
    }

def optimize_spatial(base_res, lat1, lon1, lat2, lon2, t1_base, body1, t2_base, body2, limit_km):
    """Wiggles locations to maximize overlap, STRICTLY constrained by limit_km."""
    
    # 1. We effectively create a "Square" search box.
    # To ensure the corner of the square doesn't exceed the radius,
    # we limit the side of the square to (Radius / sqrt(2)).
    safe_limit = limit_km / 1.4142 

    # 2. Calculate Latitude Bounds (1 deg Lat ~= 111.32 km everywhere)
    lat_b = safe_limit / 111.32

    # 3. Calculate Longitude Bounds separately for Loc1 and Loc2
    # (Longitude degrees shrink as you get closer to the poles)
    def get_lon_bound(lat_val):
        if abs(lat_val) >= 89.0: return 180.0
        # 1 deg Lon = 111.32 * cos(lat)
        km_per_deg = 111.32 * np.cos(np.radians(lat_val))
        return safe_limit / km_per_deg

    lon_b1 = get_lon_bound(lat1)
    lon_b2 = get_lon_bound(lat2)

    # Objective function for the optimizer
    def objective(x):
        # x is [dlat1, dlon1, dlat2, dlon2]
        sep, *rest = check_alignment_at_offset(
            base_res['offset'], 
            lat1+x[0], lon1+x[1], t1_base, body1,
            lat2+x[2], lon2+x[3], t2_base, body2
        )
        return sep

    # 4. Apply distinct bounds for each variable
    bounds = [
        (-lat_b, lat_b),    # Lat1
        (-lon_b1, lon_b1),  # Lon1
        (-lat_b, lat_b),    # Lat2
        (-lon_b2, lon_b2)   # Lon2 (Calculated specifically for Lat2)
    ]
    
    # Run Optimizer
    opt = minimize(objective, [0.0]*4, bounds=bounds, method='L-BFGS-B')
    x = opt.x
    
    # 5. Get final results
    sep, ft1, ft2, alt1, az1, alt2, az2, d1, d2 = check_alignment_at_offset(
        base_res['offset'], 
        lat1+x[0], lon1+x[1], t1_base, body1,
        lat2+x[2], lon2+x[3], t2_base, body2
    )
    overlap = calculate_overlap(sep, d1, body1, d2, body2)
    
    # Calculate actual drift in KM for display
    # (Approximation using 111km/deg)
    drift1 = np.sqrt((x[0]*111.32)**2 + (x[1]*(111.32*np.cos(np.radians(lat1))))**2)
    drift2 = np.sqrt((x[2]*111.32)**2 + (x[3]*(111.32*np.cos(np.radians(lat2))))**2)
    
    return {
        'sep': sep, 'overlap': overlap,
        'drift1': drift1,
        'drift2': drift2,
        'final_lat1': lat1+x[0], 'final_lon1': lon1+x[1],
        'final_lat2': lat2+x[2], 'final_lon2': lon2+x[3]
    }

# (Imports and helper functions remain the same as before...)
# ... paste helper functions like optimize_spatial here ...

@eel.expose
def start_calculation(params):
    try:
        if eph_main_global is None: load_skyfield_objects()
        
        # --- PARSE PARAMS ---
        lat1, lon1 = params['lat1'], params['lon1']
        lat2, lon2 = params['lat2'], params['lon2']
        search_window = params['search_hours_offset']
        radius_km = float(params.get('search_radius', 150.0))

        # Dynamic Reference Date
        ref_input = params.get('ref_date_str', '1922-08-24')
        try:
            y_str, m_str, d_str = ref_input.split('-')
            fixed_year, fixed_month, fixed_day = int(y_str), int(m_str), int(d_str)
        except:
            fixed_year, fixed_month, fixed_day = 1922, 8, 24
            eel.updateProgress(f"Warning: Invalid date, defaulting to 1922-08-24")

        # --- DYNAMIC SCAN RANGE ---
        start_year = int(params.get('start_year', 1914))
        end_year = int(params.get('end_year', 1918))
        
        # Parse Time
        try:
            # Skyfield times are UTC aware by default usually
            bd = datetime.datetime.fromisoformat(params['time1'].replace('Z', '+00:00'))
            sh, sm = bd.hour, bd.minute
        except: sh, sm = 23, 0 
        
        dt1_base = datetime.datetime(fixed_year, fixed_month, fixed_day, sh, sm, tzinfo=datetime.timezone.utc)
        
        eel.updateProgress(f"=== MODE: TIME TRAVEL SEARCH ===")
        eel.updateProgress(f"Loc1 (Sun): Fixed {fixed_year}-{fixed_month:02d}-{fixed_day:02d}")
        eel.updateProgress(f"Loc2 (Moon): Scanning {start_year}-{end_year}")
        
        # --- PROGRESS CALCULATION ---
        total_years = end_year - start_year + 1
        total_months = total_years * 12
        current_step = 0
        hits = 0
        
        # Define Offsets
        offset_loc1 = datetime.timedelta(hours=8) # Sun Location
        offset_loc2 = datetime.timedelta(hours=1) # Moon Location

        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                
                current_step += 1
                percent = (current_step / total_months) * 100
                eel.updateProgressBar(percent)
                eel.updateProgress(f"Scanning {y}-{m:02d}...")

                try: 
                    dt2_base = datetime.datetime(y, m, fixed_day, sh, sm, tzinfo=datetime.timezone.utc)
                except ValueError: continue 

                # 1. OPTIMIZE TIME
                res = optimize_time_offset(lat1, lon1, dt1_base, 'sun', lat2, lon2, dt2_base, 'moon', search_window)
                if res['alt1'] < -1 or res['alt2'] < -1: continue 

                # 2. OPTIMIZE SPACE
                final_res = res
                if res['sep'] < 5.0:
                    spatial_res = optimize_spatial(res, lat1, lon1, lat2, lon2, dt1_base, 'sun', dt2_base, 'moon', radius_km)
                    if spatial_res['sep'] < res['sep']:
                        final_res.update(spatial_res)

                # 3. REPORT HIT
                # UPDATE TOLERANCE HERE: Change 0.1 to a higher number if you want strictly better matches
                if final_res['overlap'] > 0.1: 
                    hits += 1
                    
                    # --- NEW FORMATTING LOGIC START ---
                    
                    # Get the Skyfield time objects (as Python datetime)
                    # Note: final_res['t1'] is already a UTC datetime object from the optimizer
                    t1_utc = final_res['t1'] 
                    t2_utc = final_res['t2']

                    # Calculate Local Times
                    loc1_local = t1_utc + offset_loc1
                    loc2_local = t2_utc + offset_loc2

                    # Format Strings
                    # Base UTC string
                    utc_str_1 = t1_utc.strftime("%d %b %Y %H:%M:%S UTC")
                    utc_str_2 = t2_utc.strftime("%d %b %Y %H:%M:%S UTC")

                    # Local Strings with Day Name and AM/PM
                    loc1_str = loc1_local.strftime("%d %b %Y %I:%M:%S %p, %A")
                    loc2_str = loc2_local.strftime("%d %b %Y %I:%M:%S %p, %A")
                    
                    # --- NEW FORMATTING LOGIC END ---

                    color = "#4caf50" if final_res['overlap'] > 20 else "#ff9800"
                    
                    drift_html = ""
                    if final_res.get('drift1', 0) > 1.0 or final_res.get('drift2', 0) > 1.0:
                        new_coords1 = format_latlon(final_res['final_lat1'], final_res['final_lon1'])
                        new_coords2 = format_latlon(final_res['final_lat2'], final_res['final_lon2'])
                        
                        drift_html = f"""
                        <div style="background-color: #2a2a2a; padding: 8px; margin-top: 5px; border-radius: 4px; font-size: 0.9em;">
                            <strong>📍 Adjusted Locations (Drift):</strong><br>
                            <span style="color: #d32f2f;">Loc1 (Sun):</span> {new_coords1} (Moved {final_res['drift1']:.1f} km)<br>
                            <span style="color: #1976d2;">Loc2 (Moon):</span> {new_coords2} (Moved {final_res['drift2']:.1f} km)
                        </div>
                        """

                    html = f"""
                    <div class='hit-item' style='border-left: 5px solid {color}; padding-left: 10px;'>
                        <h3>Match Found!</h3>
                        <p><strong>Angle: {final_res['sep']:.4f}°</strong> (Overlap: {final_res['overlap']:.1f}%)</p>
                        <p><strong>Loc1 (Sun):</strong>  {utc_str_1} ( {loc1_str} )</p>
                        <p><strong>Loc2 (Moon):</strong> {utc_str_2} ( {loc2_str} )</p>
                        {drift_html}
                    </div><hr>
                    """
                    eel.addHitDetail(html)

        eel.updateProgressBar(100)
        eel.showSummary(f"Search Complete. Found {hits} matches.")

    except Exception as e:
        import traceback
        eel.updateProgress(f"ERROR: {e}\n{traceback.format_exc()}")


if __name__ == '__main__':
    eel.init('web')
    try: load_skyfield_objects()
    except: pass
    eel.start('main.html', size=(700, 900))