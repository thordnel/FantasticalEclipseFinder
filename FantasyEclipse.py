import eel # For the GUI
import time # For timing and optional delays
import datetime
import numpy as np
from skyfield.api import Topos, load, wgs84
from skyfield.constants import AU_KM, DEG2RAD # DEG2RAD might not be used directly now
from scipy.optimize import minimize_scalar
# argparse is NOT needed for the Eel version as inputs come from HTML

# --- Configuration & Global Skyfield Objects (Load Once) ---
EPHEMERIS_FILE_GLOBAL = 'de421.bsp' # Default, can be changed if you add UI for it
DEFAULT_SUN_RADIUS_KM = 695700.0
DEFAULT_MOON_RADIUS_KM = 1737.4

eph_main_global = None
planets_main_dict_global = None
ts_main_global = None

# --- HELPER FUNCTIONS (from your original calculator script) ---

def format_dms(degrees_float):
    """Converts decimal degrees to a DMS string (degrees, minutes, seconds)."""
    if degrees_float is None or np.isnan(degrees_float):
        return "N/A"
    sign = '-' if degrees_float < 0 else '+'
    degrees_float_abs = abs(degrees_float)
    d = int(degrees_float_abs)
    m_float = (degrees_float_abs - d) * 60
    m = int(m_float)
    s_float = (m_float - m) * 60
    s = round(s_float, 2) 
    if s >= 59.995:
        s = 0.0
        m += 1
        if m >= 60:
            m = 0
            d += 1
    return f"{sign}{d:02d}°{m:02d}'{s:05.2f}\""

def get_angular_diameter_arcsec(physical_radius_km, distance_au):
    if physical_radius_km is None or physical_radius_km <= 0 or np.isnan(physical_radius_km): return 0
    distance_km = distance_au * AU_KM
    if distance_km <= 0: return 0
    angular_diameter_rad = 2 * physical_radius_km / distance_km
    return np.degrees(angular_diameter_rad) * 3600

def get_altaz_and_gcrs_position(ts, time_obj, observer_loc, target_body):
    astrometric = observer_loc.at(time_obj).observe(target_body)
    apparent_gcrs = astrometric.apparent()
    alt, az, _ = apparent_gcrs.altaz()
    return alt, az, apparent_gcrs

def shortest_angle_difference(a1_deg, a2_deg):
    diff = (a2_deg - a1_deg + 180) % 360 - 180
    return diff

# --- CORE CALCULATION FUNCTION (your astronomical logic) ---
def find_codirectional_fantastical_eclipse(
    lat1, lon1, alt1_m, datetime_str1_utc,
    lat2, lon2, alt2_m, datetime_str2_utc,
    search_window_hours,
    # These next two are now guaranteed to be pre-loaded global Skyfield objects
    # So, we'll use planets_main_dict_global and ts_main_global directly
    # planets_dict_arg, # No longer needed as direct arg
    # ts_arg,           # No longer needed as direct arg
    verbose_per_call=False # To control its own prints if any remain
):
    # Access global Skyfield objects
    sun_obj = planets_main_dict_global['sun']
    moon_obj = planets_main_dict_global['moon']
    earth_obj = planets_main_dict_global['earth']
    current_ts = ts_main_global

    obs1_loc = earth_obj + wgs84.latlon(lat1, lon1, elevation_m=alt1_m)
    obs2_loc = earth_obj + wgs84.latlon(lat2, lon2, elevation_m=alt2_m)

    initial_time1 = datetime.datetime.fromisoformat(datetime_str1_utc.replace('Z', '+00:00'))
    initial_time2 = datetime.datetime.fromisoformat(datetime_str2_utc.replace('Z', '+00:00'))

    if initial_time1.tzinfo is None: initial_time1 = initial_time1.replace(tzinfo=datetime.timezone.utc)
    if initial_time2.tzinfo is None: initial_time2 = initial_time2.replace(tzinfo=datetime.timezone.utc)

    # --- Radii determination ---
    sun_radius_km_val = getattr(sun_obj, 'radius', None)
    if sun_radius_km_val: sun_radius_km_val = sun_radius_km_val.km
    if sun_radius_km_val is None:
        # No eel.updateProgress here, this function should be quiet or use verbose_per_call
        if verbose_per_call: print(f"  (calc) WARNING: Using default Sun radius: {DEFAULT_SUN_RADIUS_KM} km.")
        sun_radius_km_val = DEFAULT_SUN_RADIUS_KM

    moon_radius_km_val = getattr(moon_obj, 'radius', None)
    if moon_radius_km_val: moon_radius_km_val = moon_radius_km_val.km
    if moon_radius_km_val is None:
        if verbose_per_call: print(f"  (calc) WARNING: Using default Moon radius: {DEFAULT_MOON_RADIUS_KM} km.")
        moon_radius_km_val = DEFAULT_MOON_RADIUS_KM
    
    # --- Memoization dictionaries for the optimizer ---
    memo_times = {}
    memo_altaz_sun = {}
    memo_altaz_moon = {}

    def altaz_difference_at_offset(time_offset_seconds):
        current_dt1 = initial_time1 + datetime.timedelta(seconds=time_offset_seconds)
        current_dt2 = initial_time2 + datetime.timedelta(seconds=time_offset_seconds)
        
        skyfield_current_time1 = memo_times.setdefault((current_dt1,1), current_ts.utc(current_dt1))
        skyfield_current_time2 = memo_times.setdefault((current_dt2,2), current_ts.utc(current_dt2))

        # Using .get would require a function to compute if not found, setdefault is better here
        if skyfield_current_time1 not in memo_altaz_sun:
            memo_altaz_sun[skyfield_current_time1] = get_altaz_and_gcrs_position(current_ts, skyfield_current_time1, obs1_loc, sun_obj)
        alt_s, az_s, _ = memo_altaz_sun[skyfield_current_time1]
        
        if skyfield_current_time2 not in memo_altaz_moon:
            memo_altaz_moon[skyfield_current_time2] = get_altaz_and_gcrs_position(current_ts, skyfield_current_time2, obs2_loc, moon_obj)
        alt_m, az_m, _ = memo_altaz_moon[skyfield_current_time2]


        delta_alt_deg = alt_s.degrees - alt_m.degrees
        delta_az_deg = shortest_angle_difference(az_s.degrees, az_m.degrees)
        altaz_separation = np.sqrt(delta_alt_deg**2 + delta_az_deg**2)
        return altaz_separation

    if verbose_per_call: # Optional: Print initial state if verbose
        altaz_diff_at_zero_offset = altaz_difference_at_offset(0) 
        print(f"  (calc) Alt/Az diff at zero offset: {altaz_diff_at_zero_offset:.6f} deg ({format_dms(altaz_diff_at_zero_offset)})")

    # Clear memos before optimizer runs, as it will populate them.
    memo_times.clear(); memo_altaz_sun.clear(); memo_altaz_moon.clear()

    lower_bound_sec = -search_window_hours * 3600
    upper_bound_sec = search_window_hours * 3600

    result = minimize_scalar(
        altaz_difference_at_offset,
        bounds=(lower_bound_sec, upper_bound_sec),
        method='bounded',
        options={'xatol': 1.0}
    )

    min_altaz_separation_deg_val = result.fun
    optimal_time_offset_seconds_val = result.x

    if not result.success:
        if verbose_per_call: print(f"  (calc) Warning: Minimization might not have converged: {result.message}")
        if min_altaz_separation_deg_val is None or np.isnan(min_altaz_separation_deg_val) or min_altaz_separation_deg_val > 180 :
            if verbose_per_call: print("  (calc) Minimization failed, using zero offset.")
            optimal_time_offset_seconds_val = 0.0
            memo_times.clear(); memo_altaz_sun.clear(); memo_altaz_moon.clear()
            min_altaz_separation_deg_val = altaz_difference_at_offset(optimal_time_offset_seconds_val)

    optimal_sun_obs_time_val = initial_time1 + datetime.timedelta(seconds=optimal_time_offset_seconds_val)
    optimal_moon_obs_time_val = initial_time2 + datetime.timedelta(seconds=optimal_time_offset_seconds_val)

    skyfield_optimal_sun_time = current_ts.utc(optimal_sun_obs_time_val)
    skyfield_optimal_moon_time = current_ts.utc(optimal_moon_obs_time_val)

    final_alt_s_val, final_az_s_val, sun_gcrs_apparent = get_altaz_and_gcrs_position(current_ts, skyfield_optimal_sun_time, obs1_loc, sun_obj)
    final_alt_m_val, final_az_m_val, moon_gcrs_apparent = get_altaz_and_gcrs_position(current_ts, skyfield_optimal_moon_time, obs2_loc, moon_obj)

    sun_angular_diameter_arcsec_val = get_angular_diameter_arcsec(sun_radius_km_val, sun_gcrs_apparent.distance().au)
    moon_angular_diameter_arcsec_val = get_angular_diameter_arcsec(moon_radius_km_val, moon_gcrs_apparent.distance().au)
    
    sun_radius_deg_altaz_val = (sun_angular_diameter_arcsec_val / 2) / 3600
    moon_radius_deg_altaz_val = (moon_angular_diameter_arcsec_val / 2) / 3600
    sum_of_radii_deg_altaz_val = sun_radius_deg_altaz_val + moon_radius_deg_altaz_val

    is_visible_s_val = final_alt_s_val.degrees > -0.5 
    is_visible_m_val = final_alt_m_val.degrees > -0.5

    is_hit_altaz_val = (is_visible_s_val and is_visible_m_val and
                        sun_angular_diameter_arcsec_val > 0 and moon_angular_diameter_arcsec_val > 0 and 
                        min_altaz_separation_deg_val < sum_of_radii_deg_altaz_val)
    
    overlap_metric_val = 0.0 # Default to float
    event_type_str_val = "N/A"
    if is_hit_altaz_val:
        overlap_metric_val = max(0.0, (sum_of_radii_deg_altaz_val - min_altaz_separation_deg_val) / sum_of_radii_deg_altaz_val) * 100.0
        diff_radii_deg_altaz = abs(sun_radius_deg_altaz_val - moon_radius_deg_altaz_val)
        if min_altaz_separation_deg_val <= diff_radii_deg_altaz:
            if sun_radius_deg_altaz_val >= moon_radius_deg_altaz_val: event_type_str_val = "Annular-like (Alt/Az)"
            else: event_type_str_val = "Total-like (Alt/Az)"
        else: event_type_str_val = "Partial-like (Alt/Az)"

    # Return a dictionary of results
    return {
        "is_hit": is_hit_altaz_val,
        "obs2_initial_date_utc": datetime_str2_utc, # Pass back the date tested for Obs2
        "optimal_time_offset_seconds": optimal_time_offset_seconds_val,
        "sun_obs_time_utc": optimal_sun_obs_time_val.isoformat(),
        "moon_obs_time_utc": optimal_moon_obs_time_val.isoformat(),
        "min_altaz_separation_deg": min_altaz_separation_deg_val,
        "type": event_type_str_val,
        "overlap_metric": overlap_metric_val,
        # Optionally add more raw data if needed by JS for detailed display
        "sun_final_alt_deg": final_alt_s_val.degrees,
        "sun_final_az_deg": final_az_s_val.degrees,
        "moon_final_alt_deg": final_alt_m_val.degrees,
        "moon_final_az_deg": final_az_m_val.degrees,
    }

# --- EEL SPECIFIC FUNCTIONS ---

def load_skyfield_objects(eph_file_path=EPHEMERIS_FILE_GLOBAL):
    global eph_main_global, planets_main_dict_global, ts_main_global, EPHEMERIS_FILE_GLOBAL
    EPHEMERIS_FILE_GLOBAL = eph_file_path
    
    progress_message_start = f"Loading ephemeris ({EPHEMERIS_FILE_GLOBAL})... This might take a moment."
    progress_message_end = "Ephemeris loaded."

    # Try to send progress to Eel, print to console as fallback or in addition
    try:
        eel.updateProgress(progress_message_start) # JavaScript function exposed by you
    except Exception: # Broad exception if eel or updateProgress is not ready
        print(progress_message_start + " (Console)")

    eph_main_global = load(EPHEMERIS_FILE_GLOBAL)
    planets_main_dict_global = {
        'sun': eph_main_global['sun'],
        'moon': eph_main_global['moon'],
        'earth': eph_main_global['earth']
    }
    ts_main_global = load.timescale()

    try:
        eel.updateProgress(progress_message_end)
    except Exception:
        print(progress_message_end + " (Console)")


@eel.expose # Expose this function to JavaScript
def start_calculation(params):
    try: # Wrap the whole thing in a try-except for robustness
        eel.updateProgress("Received parameters from UI. Starting main calculation loop.")
        
        if eph_main_global is None: 
            load_skyfield_objects() 

        lat1 = params['lat1']
        lon1 = params['lon1']
        alt1_m = params['alt1'] 
        datetime_str1_utc = params['time1'] 

        lat2 = params['lat2']
        lon2 = params['lon2']
        alt2_m = params['alt2'] 
        day2 = params['day2']
        hms2_str = params['hms2'] 

        start_year2_val = params['start_year2'] 
        end_year2_val = params['end_year2'] 
        
        try:
            months2_list_str = params['months2_str'].split()
            months2_list = [int(m.strip()) for m in months2_list_str if m.strip()]
            if not months2_list or not all(1 <= m <= 12 for m in months2_list):
                raise ValueError("Invalid month number detected.")
        except ValueError as e:
            eel.updateProgress(f"ERROR: Invalid format for Months: '{params['months2_str']}'. Use space-separated numbers (e.g., '8 9'). Details: {e}")
            return

        search_hours_offset_val = params['search_hours_offset'] 

        found_hits_summary_list = []
        try:
            hour2_val, minute2_val, second2_val = map(int, hms2_str.split(':'))
        except ValueError:
            eel.updateProgress(f"ERROR: Invalid format for Obs2 Time (hms2). Expected HH:MM:SS. Got: {hms2_str}")
            return

        total_combinations_to_test = (end_year2_val - start_year2_val + 1) * len(months2_list)
        if total_combinations_to_test <= 0:
            eel.updateProgress("ERROR: Invalid year range or no months selected, resulting in zero combinations.")
            return
            
        current_combination_count = 0
        script_start_time = time.time()
        
        eel.updateProgress(f"Observer in Capiz (Sun): Lat={lat1}, Lon={lon1}, Alt={alt1_m}m, Initial Time1 (UTC)={datetime_str1_utc}")
        eel.updateProgress(f"Observer in Ypern/Gent (Moon): Lat={lat2}, Lon={lon2}, Alt={alt2_m}m, Day2={day2}, Time of Day2 (UTC)={hms2_str}")
        eel.updateProgress(f"Searching for Belgium observer dates: Years {start_year2_val}-{end_year2_val}, Months: {months2_list}")
        eel.updateProgress(f"Optimizer search window for time offset: +/- {search_hours_offset_val} hours for each date.")
        eel.updateProgress("-" * 30)

        for year_iter in range(start_year2_val, end_year2_val + 1):
            for month_iter in months2_list:
                current_combination_count += 1
                progress_percent = (current_combination_count / total_combinations_to_test) * 100
                
                try:
                    datetime.date(year_iter, month_iter, day2) 
                    obs2_datetime_str_iter = f"{year_iter:04d}-{month_iter:02d}-{day2:02d}T{hour2_val:02d}:{minute2_val:02d}:{second2_val:02d}Z"
                except ValueError as e:
                    eel.updateProgress(f"[{current_combination_count}/{total_combinations_to_test} | {progress_percent:.1f}%] Skipping invalid date {year_iter}-{month_iter}-{day2}: {e}")
                    continue

                progress_msg = f"[{current_combination_count}/{total_combinations_to_test} | {progress_percent:.1f}%] Testing Date: {obs2_datetime_str_iter}"
                eel.updateProgress(progress_msg)
                
                hit_result_data = find_codirectional_fantastical_eclipse(
                    lat1=lat1, lon1=lon1, alt1_m=alt1_m, datetime_str1_utc=datetime_str1_utc,
                    lat2=lat2, lon2=lon2, alt2_m=alt2_m, datetime_str2_utc=obs2_datetime_str_iter,
                    search_window_hours=search_hours_offset_val,
                    # planets_dict_arg=planets_main_dict_global, # Not needed as arg
                    # ts_arg=ts_main_global,                     # Not needed as arg
                    verbose_per_call=False 
                )
                
                if hit_result_data and hit_result_data.get("is_hit"):
                    sun_obs_time_utc_dt = datetime.datetime.fromisoformat(hit_result_data['sun_obs_time_utc'].replace('Z', '+00:00'))
                    moon_obs_time_utc_dt = datetime.datetime.fromisoformat(hit_result_data['moon_obs_time_utc'].replace('Z', '+00:00'))

                    # --- Parse and Format Obs2 Initial UTC Date ---
                    obs2_initial_utc_str = hit_result_data['obs2_initial_date_utc']
                    obs2_initial_utc_dt = datetime.datetime.fromisoformat(obs2_initial_utc_str.replace('Z', '+00:00'))
                    # Using f-string with .day for non-padded day
                    formatted_moon_obs_time_utc_dt = moon_obs_time_utc_dt.strftime(f"%B {moon_obs_time_utc_dt.day}, %Y at %I:%M %p UTC")

                    # --- Apply YOUR FIXED UTC OFFSETS ---
                    FIXED_OFFSET_OBS1_HOURS = +8 
                    FIXED_OFFSET_OBS2_HOURS = +1 

                    # --- Calculate and Format Local Time for Observer 1 ---
                    obs1_local_time_dt = sun_obs_time_utc_dt + datetime.timedelta(hours=FIXED_OFFSET_OBS1_HOURS)
                    obs1_local_time_str = obs1_local_time_dt.strftime(f"%B {obs1_local_time_dt.day}, %Y at %I:%M %p") + f" (UTC{FIXED_OFFSET_OBS1_HOURS:+d})" # Corrected format

                    # --- Calculate and Format Local Time for Observer 2 ---
                    obs2_local_time_dt = moon_obs_time_utc_dt + datetime.timedelta(hours=FIXED_OFFSET_OBS2_HOURS)
                    obs2_local_time_str = obs2_local_time_dt.strftime(f"%B {obs2_local_time_dt.day}, %Y at %I:%M %p") + f" (UTC{FIXED_OFFSET_OBS2_HOURS:+d})" # Corrected format
                    
                    # Store these nicely formatted local times in the result dictionary
                    hit_result_data['obs1_local_time_fixed'] = obs1_local_time_str
                    hit_result_data['obs2_local_time_fixed'] = obs2_local_time_str

                    # --- Update the HTML string for hit details (this part was already correct) ---
                    alt_az_sep_dms = format_dms(hit_result_data['min_altaz_separation_deg'])
                    sun_alt_dms = format_dms(hit_result_data.get('sun_final_alt_deg', float('nan'))) 
                    sun_az_dms = format_dms(hit_result_data.get('sun_final_az_deg', float('nan')))
                    moon_alt_dms = format_dms(hit_result_data.get('moon_final_alt_deg', float('nan')))
                    moon_az_dms = format_dms(hit_result_data.get('moon_final_az_deg', float('nan')))

                    hit_detail_html = f"""
                        <div class='hit-item'>
                            <h3 style="color: #6497b1;">
                            <span style="font-weight: normal;">Fantastical Eclipse of Moon in Lagmaran, Capiz to the Sun in Somme, France:</span><br>{formatted_moon_obs_time_utc_dt}
                            </h3>
                            <p><strong>Type:</strong> {hit_result_data['type']}, <strong>Overlap:</strong> {hit_result_data['overlap_metric']:.1f}%</p>
                            <p><strong>Min Alt/Az Separation:</strong> {alt_az_sep_dms} ({hit_result_data['min_altaz_separation_deg']:.4f}°)</p>
                            <p><strong>Time Offset Applied:</strong> {hit_result_data['optimal_time_offset_seconds']:.0f}s</p>
                            <p><strong>Optimal Sun Time (UTC):</strong> {hit_result_data['sun_obs_time_utc']}</p>
                            <p style="padding-left: 15px;"><em>Local Time (Lagmaran): {obs1_local_time_str}</em></p>
                            <p style="padding-left: 15px;">Sun Alt: {sun_alt_dms}, Sun Az: {sun_az_dms}</p>
                            
                            <p><strong>Optimal Moon Time (UTC):</strong> {hit_result_data['moon_obs_time_utc']}</p>
                            <p style="padding-left: 15px;"><em>Local Time (Somme): {obs2_local_time_str}</em></p>
                            <p style="padding-left: 15px;">Moon Alt: {moon_alt_dms}, Moon Az: {moon_az_dms}</p>
                            <p><small><em>Local times based on fixed offsets UTC{FIXED_OFFSET_OBS1_HOURS:+d} for Obs1 and UTC{FIXED_OFFSET_OBS2_HOURS:+d} for Obs2. Does not account for DST changes if any.</em></small></p>
                        </div> <hr>
                    """
                    eel.addHitDetail(hit_detail_html) 
                    found_hits_summary_list.append(hit_result_data)

        script_end_time = time.time()
        total_duration_seconds = script_end_time - script_start_time
        
        summary_text = "="*10 + " SCRIPT EXECUTION COMPLETE " + "="*10 + "\n"
        summary_text += f"Total combinations tested: {current_combination_count}\n"
        summary_text += f"Total execution time: {total_duration_seconds:.2f} seconds ({datetime.timedelta(seconds=total_duration_seconds)})\n\n"
        
        summary_text += "="*10 + " SUMMARY OF ALL FOUND HITS " + "="*10 + "\n"
        if found_hits_summary_list:
            summary_text += f"Total hits found: {len(found_hits_summary_list)}\n"
            # Individual hits already displayed via addHitDetail
        else:
            summary_text += "No co-directional 'hits' found in the specified date range for Observer in Belgium.\n"
        summary_text += "="*40
        eel.showSummary(summary_text)

    except Exception as e:
        eel.updateProgress(f"An unexpected error occurred in start_calculation: {e}")
        import traceback
        eel.updateProgress(f"Traceback: {traceback.format_exc()}")


# --- MAIN EEL EXECUTION BLOCK ---
if __name__ == '__main__':
    # Setup Eel
    eel.init('web') # Point Eel to the directory with HTML/CSS/JS files

    # Pre-load Skyfield objects so it's done once when the app starts
    # Handle potential errors during initial load
    try:
        load_skyfield_objects() # Uses global EPHEMERIS_FILE_GLOBAL default
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load Skyfield objects on startup: {e}")
        print("Please ensure the ephemeris file (e.g., de421.bsp) is available.")
        # Optionally, could try to communicate this to a basic HTML error page if Eel setup allows
        exit(1) # Exit if we can't load critical data

    # Start the Eel application
    # You can specify browser options here too, e.g., ['chrome'] or ['edge']
    # To use system default browser: eel.start('main.html', size=(1000, 800))
    # To attempt Chrome specifically (if available):
    try:
        eel.start('main.html', size=(600, 900), mode='chrome') # Try Chrome first
    except (OSError, KeyError): # OSError if Chrome not found, KeyError for some internal Eel issues on some systems
        print("Chrome not found or failed to start, trying default browser...")
        try:
            eel.start('main.html', size=(600, 900), mode='default') # Fallback to system default
        except Exception as e_default:
            print(f"Failed to start GUI with default browser: {e_default}")
            print("Please ensure you have a web browser installed and accessible.")
            print("You might need to install a browser explicitly for Eel (e.g., Chrome).")