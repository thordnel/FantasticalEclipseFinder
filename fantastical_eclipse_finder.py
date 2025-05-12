import datetime
import numpy as np
from skyfield.api import Topos, load, wgs84
from skyfield.constants import AU_KM, DEG2RAD
from scipy.optimize import minimize_scalar
import argparse
import time # For optional delay

# --- Configuration ---
# Default ephemeris file. This global can be updated by command-line arg.
EPHEMERIS_FILE = 'de421.bsp'
# Default radii (IAU 2015 values where applicable, or common values)
DEFAULT_SUN_RADIUS_KM = 695700.0
DEFAULT_MOON_RADIUS_KM = 1737.4

# --- Helper Functions ---

def format_dms(degrees_float):
    """Converts decimal degrees to a DMS string (degrees, minutes, seconds)."""
    if degrees_float is None or np.isnan(degrees_float):
        return "N/A"
    
    sign = '-' if degrees_float < 0 else '+' # Use '+' for positive values for clarity
    degrees_float_abs = abs(degrees_float)
    
    d = int(degrees_float_abs)
    m_float = (degrees_float_abs - d) * 60
    m = int(m_float)
    s_float = (m_float - m) * 60
    s = round(s_float, 2) 
    
    if s >= 59.995: # Tolerance for floating point rounding
        s = 0.0
        m += 1
        if m >= 60:
            m = 0
            d += 1
            
    return f"{sign}{d:02d}°{m:02d}'{s:05.2f}\""

def get_angular_diameter_arcsec(physical_radius_km, distance_au):
    """Calculates apparent angular diameter in arcseconds."""
    if physical_radius_km is None or physical_radius_km <= 0 or np.isnan(physical_radius_km):
        return 0
    distance_km = distance_au * AU_KM
    if distance_km <= 0:
        return 0
    angular_diameter_rad = 2 * physical_radius_km / distance_km
    return np.degrees(angular_diameter_rad) * 3600

def get_altaz_and_gcrs_position(ts, time_obj, observer_loc, target_body):
    """Gets GCRS apparent position, and local Alt/Az for a celestial body."""
    astrometric = observer_loc.at(time_obj).observe(target_body)
    apparent_gcrs = astrometric.apparent() # GCRS RA/Dec
    alt, az, _ = apparent_gcrs.altaz()    # Local Alt/Az
    return alt, az, apparent_gcrs

def shortest_angle_difference(a1_deg, a2_deg):
    """Calculates the shortest angle between two azimuths (0-360)."""
    diff = (a2_deg - a1_deg + 180) % 360 - 180
    return diff

# --- Main Calculation Function (modified to return a dictionary) ---
def find_codirectional_fantastical_eclipse(
    lat1, lon1, alt1_m, datetime_str1_utc,
    lat2, lon2, alt2_m, datetime_str2_utc, # This will be the iterated date for Obs2
    search_window_hours, # Renamed from search_window_hours for clarity
    planets_dict_arg,    # Renamed from planets_dict_arg
    ts_arg,              # Renamed from ts_arg
    verbose_per_call=False # Control printing for each iteration
):
    # These are now explicitly passed Skyfield objects
    sun_obj = planets_dict_arg['sun']
    moon_obj = planets_dict_arg['moon']
    earth_obj = planets_dict_arg['earth']
    current_ts = ts_arg

    obs1_loc = earth_obj + wgs84.latlon(lat1, lon1, elevation_m=alt1_m)
    obs2_loc = earth_obj + wgs84.latlon(lat2, lon2, elevation_m=alt2_m)

    initial_time1 = datetime.datetime.fromisoformat(datetime_str1_utc.replace('Z', '+00:00'))
    initial_time2 = datetime.datetime.fromisoformat(datetime_str2_utc.replace('Z', '+00:00')) # Iterated date for Obs2

    if initial_time1.tzinfo is None: initial_time1 = initial_time1.replace(tzinfo=datetime.timezone.utc)
    if initial_time2.tzinfo is None: initial_time2 = initial_time2.replace(tzinfo=datetime.timezone.utc)

    if verbose_per_call:
        print(f"  Obs1 (Sun): Lat={lat1:.2f}, Lon={lon1:.2f}, Alt={alt1_m}m at {initial_time1.isoformat()}")
        print(f"  Obs2 (Moon): Lat={lat2:.2f}, Lon={lon2:.2f}, Alt={alt2_m}m at {initial_time2.isoformat()}")
        print(f"  Searching for co-directional alignment with +/- {search_window_hours} hours offset.")

    sun_radius_km_val = getattr(sun_obj, 'radius', None)
    if sun_radius_km_val: sun_radius_km_val = sun_radius_km_val.km
    if sun_radius_km_val is None:
        if verbose_per_call: print(f"  WARNING: Using default Sun radius: {DEFAULT_SUN_RADIUS_KM} km.")
        sun_radius_km_val = DEFAULT_SUN_RADIUS_KM

    moon_radius_km_val = getattr(moon_obj, 'radius', None)
    if moon_radius_km_val: moon_radius_km_val = moon_radius_km_val.km
    if moon_radius_km_val is None:
        if verbose_per_call: print(f"  WARNING: Using default Moon radius: {DEFAULT_MOON_RADIUS_KM} km.")
        moon_radius_km_val = DEFAULT_MOON_RADIUS_KM

    memo_times = {}
    memo_altaz_sun = {} # Separate memo for Sun AltAz
    memo_altaz_moon = {} # Separate memo for Moon AltAz


    def altaz_difference_at_offset(time_offset_seconds):
        current_dt1 = initial_time1 + datetime.timedelta(seconds=time_offset_seconds)
        current_dt2 = initial_time2 + datetime.timedelta(seconds=time_offset_seconds)
        
        skyfield_current_time1 = memo_times.setdefault((current_dt1,1), current_ts.utc(current_dt1))# Key includes observer ID
        skyfield_current_time2 = memo_times.setdefault((current_dt2,2), current_ts.utc(current_dt2))

        alt_s, az_s, _ = memo_altaz_sun.setdefault(
            skyfield_current_time1, 
            get_altaz_and_gcrs_position(current_ts, skyfield_current_time1, obs1_loc, sun_obj)
        )
        alt_m, az_m, _ = memo_altaz_moon.setdefault(
            skyfield_current_time2,
            get_altaz_and_gcrs_position(current_ts, skyfield_current_time2, obs2_loc, moon_obj)
        )

        delta_alt_deg = alt_s.degrees - alt_m.degrees
        delta_az_deg = shortest_angle_difference(az_s.degrees, az_m.degrees)
        altaz_separation = np.sqrt(delta_alt_deg**2 + delta_az_deg**2)
        return altaz_separation

    if verbose_per_call:
        altaz_diff_at_zero_offset = altaz_difference_at_offset(0) 
        print(f"  Alt/Az difference with zero offset: {altaz_diff_at_zero_offset:.6f} degrees ({format_dms(altaz_diff_at_zero_offset)})")

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
        if verbose_per_call: print(f"  Warning: Minimization might not have converged: {result.message}")
        if min_altaz_separation_deg_val is None or np.isnan(min_altaz_separation_deg_val) or min_altaz_separation_deg_val > 180 :
            if verbose_per_call: print("  Minimization failed, using zero offset for this date.")
            optimal_time_offset_seconds_val = 0.0
            # Recalculate separation at zero offset if minimizer failed completely
            memo_times.clear(); memo_altaz_sun.clear(); memo_altaz_moon.clear() # Clear before recalculating
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

    sun_diameter_deg_val = sun_angular_diameter_arcsec_val / 3600.0
    moon_diameter_deg_val = moon_angular_diameter_arcsec_val / 3600.0

    is_visible_s_val = final_alt_s_val.degrees > -0.5 
    is_visible_m_val = final_alt_m_val.degrees > -0.5

    is_hit_altaz_val = (is_visible_s_val and is_visible_m_val and
                        sun_angular_diameter_arcsec_val > 0 and moon_angular_diameter_arcsec_val > 0 and 
                        min_altaz_separation_deg_val < sum_of_radii_deg_altaz_val)
    
    overlap_metric_val = 0
    event_type_str_val = "N/A"
    if is_hit_altaz_val:
        overlap_metric_val = max(0, (sum_of_radii_deg_altaz_val - min_altaz_separation_deg_val) / sum_of_radii_deg_altaz_val) * 100
        diff_radii_deg_altaz = abs(sun_radius_deg_altaz_val - moon_radius_deg_altaz_val)
        if min_altaz_separation_deg_val <= diff_radii_deg_altaz:
            if sun_radius_deg_altaz_val >= moon_radius_deg_altaz_val: event_type_str_val = "Annular-like (Alt/Az)"
            else: event_type_str_val = "Total-like (Alt/Az)"
        else: event_type_str_val = "Partial-like (Alt/Az)"

    if verbose_per_call:
        print("\n  --- Co-Directional Alignment Details ---")
        print(f"  Optimal Time Offset: {optimal_time_offset_seconds_val:.2f} seconds")
        print(f"  Sun Obs Time (UTC): {optimal_sun_obs_time_val.isoformat()}")
        print(f"  Moon Obs Time (UTC): {optimal_moon_obs_time_val.isoformat()}")
        print(f"  Min Alt/Az Sep: {min_altaz_separation_deg_val:.6f} deg ({format_dms(min_altaz_separation_deg_val)})")
        print("  Sun (Obs1):")
        print(f"    Alt: {final_alt_s_val.degrees:.3f}° ({format_dms(final_alt_s_val.degrees)}), Az: {final_az_s_val.degrees:.3f}° ({format_dms(final_az_s_val.degrees)})")
        print(f"    GCRS: {sun_gcrs_apparent.radec(epoch=current_ts.J2000)}")
        print(f"    Ang Diam: {sun_angular_diameter_arcsec_val:.2f}\" ({format_dms(sun_diameter_deg_val)})")
        print(f"    Visible: {is_visible_s_val}")
        print("  Moon (Obs2):")
        print(f"    Alt: {final_alt_m_val.degrees:.3f}° ({format_dms(final_alt_m_val.degrees)}), Az: {final_az_m_val.degrees:.3f}° ({format_dms(final_az_m_val.degrees)})")
        print(f"    GCRS: {moon_gcrs_apparent.radec(epoch=current_ts.J2000)}")
        print(f"    Ang Diam: {moon_angular_diameter_arcsec_val:.2f}\" ({format_dms(moon_diameter_deg_val)})")
        print(f"    Visible: {is_visible_m_val}")
        print(f"  Sum of Radii (Alt/Az): {sum_of_radii_deg_altaz_val:.6f} deg ({format_dms(sum_of_radii_deg_altaz_val)})")
        print(f"  Is Hit (Alt/Az frame, both visible)? {is_hit_altaz_val}")
        if is_hit_altaz_val:
            print(f"  Type: {event_type_str_val}, Overlap: {overlap_metric_val:.1f}%")
        elif not (is_visible_s_val and is_visible_m_val):
             if not is_visible_s_val : print("    Reason no hit: Sun not visible.")
             if not is_visible_m_val : print("    Reason no hit: Moon not visible.")
        elif not (sun_angular_diameter_arcsec_val > 0 and moon_angular_diameter_arcsec_val > 0):
            print("    Reason no hit: Zero angular diameter for one/both bodies.")
        else:
            print(f"    Reason no hit: Min Alt/Az Sep ({format_dms(min_altaz_separation_deg_val)}) >= Sum of Radii ({format_dms(sum_of_radii_deg_altaz_val)}).")
        print("  ------------------------------------")


    return {
        "is_hit": is_hit_altaz_val,
        "obs2_initial_date_utc": datetime_str2_utc,
        "optimal_time_offset_seconds": optimal_time_offset_seconds_val,
        "sun_obs_time_utc": optimal_sun_obs_time_val.isoformat(),
        "moon_obs_time_utc": optimal_moon_obs_time_val.isoformat(),
        "min_altaz_separation_deg": min_altaz_separation_deg_val,
        "type": event_type_str_val,
        "overlap_metric": overlap_metric_val
    }

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Co-Directional Fantastical Eclipse Calculator - Iterative Date Search for Obs2"
    )
    # Observer 1 Args
    parser.add_argument("--lat1", type=float, required=True, help="Latitude of Observer 1 (Sun)")
    parser.add_argument("--lon1", type=float, required=True, help="Longitude of Observer 1 (Sun)")
    parser.add_argument("--alt1", type=float, default=0, help="Altitude of Observer 1 (Sun) in meters (default: 0)")
    parser.add_argument("--time1", type=str, required=True, 
                        help="FULL UTC datetime for Observer 1 (Sun) as YYYY-MM-DDTHH:MM:SSZ")

    # Observer 2 Args
    parser.add_argument("--lat2", type=float, required=True, help="Latitude of Observer 2 (Moon)")
    parser.add_argument("--lon2", type=float, required=True, help="Longitude of Observer 2 (Moon)")
    parser.add_argument("--alt2", type=float, default=0, help="Altitude of Observer 2 (Moon) in meters (default: 0)")
    parser.add_argument("--day2", type=int, required=True, help="Day of month for Observer 2 (1-31)")
    parser.add_argument("--hms2", type=str, required=True, help="HH:MM:SS for Observer 2 (UTC), e.g., '14:30:00'")
    
    # Iteration Control
    parser.add_argument("--start_year2", type=int, default=2000, help="Start Year for Obs2 search (default: 2000)")
    parser.add_argument("--end_year2", type=int, default=2010, help="End Year for Obs2 search (inclusive, default: 2010)")
    parser.add_argument("--months2", type=int, nargs='+', default=[8, 9], 
                        help="List of months for Obs2 (1-12), e.g., 8 9 for Aug & Sep (default: [8, 9])")
    
    # Optimizer Control for each iteration
    parser.add_argument("--search_hours_offset", type=float, default=1.0, 
                        help="Time offset search window in hours for the optimizer (+/-, default: 1.0 hour)")
    parser.add_argument("--eph_file", type=str, default=EPHEMERIS_FILE, 
                        help=f"Skyfield ephemeris file (default: {EPHEMERIS_FILE})")
    parser.add_argument("--verbose_per_call", action='store_true',
                        help="Print detailed output for each date combination tested.")


    args = parser.parse_args()

    # --- Use fixed search parameters or those from args ---
    START_YEAR_OBS2 = args.start_year2
    END_YEAR_OBS2 = args.end_year2
    MONTHS_OBS2_TO_SEARCH = args.months2
    # --- ---

    if args.eph_file != EPHEMERIS_FILE:
        EPHEMERIS_FILE = args.eph_file
    
    print(f"Loading ephemeris ({EPHEMERIS_FILE})... This might take a moment.")
    eph_main = load(EPHEMERIS_FILE)
    planets_main_dict_global = {'sun': eph_main['sun'], 'moon': eph_main['moon'], 'earth': eph_main['earth']}
    ts_main_global = load.timescale()
    print("Ephemeris loaded.")

    print(f"\n--- Input Configuration ---")
    print(f"Observer 1 (Sun): Lat={args.lat1}, Lon={args.lon1}, Alt={args.alt1}m, Initial Time1 (UTC)={args.time1}")
    print(f"Observer 2 (Moon): Lat={args.lat2}, Lon={args.lon2}, Alt={args.alt2}m, Day2={args.day2}, Time of Day2 (UTC)={args.hms2}")
    print(f"Searching for Obs2 dates: Years {START_YEAR_OBS2}-{END_YEAR_OBS2}, Months: {MONTHS_OBS2_TO_SEARCH}")
    print(f"Optimizer search window for time offset: +/- {args.search_hours_offset} hours for each date.")
    print(f"Verbose output per call: {args.verbose_per_call}")
    print("-" * 50)

    found_hits_summary_list = []
    try:
        hour2_val, minute2_val, second2_val = map(int, args.hms2.split(':'))
    except ValueError:
        print(f"ERROR: Invalid format for --hms2. Expected HH:MM:SS. Got: {args.hms2}")
        exit(1)

    total_combinations_to_test = (END_YEAR_OBS2 - START_YEAR_OBS2 + 1) * len(MONTHS_OBS2_TO_SEARCH)
    current_combination_count = 0
    script_start_time = time.time()

    for year_iter in range(START_YEAR_OBS2, END_YEAR_OBS2 + 1):
        for month_iter in MONTHS_OBS2_TO_SEARCH:
            current_combination_count += 1
            progress_percent = (current_combination_count / total_combinations_to_test) * 100
            
            # Construct the Obs2 datetime string for this iteration
            try:
                # Validate day for month/year (e.g. no Feb 30) before constructing string
                datetime.date(year_iter, month_iter, args.day2) 
                obs2_datetime_str_iter = f"{year_iter:04d}-{month_iter:02d}-{args.day2:02d}T{hour2_val:02d}:{minute2_val:02d}:{second2_val:02d}Z"
            except ValueError as e:
                print(f"\n[{current_combination_count}/{total_combinations_to_test} | {progress_percent:.1f}%] Skipping invalid date {year_iter}-{month_iter}-{args.day2}: {e}")
                continue # Skip to next iteration

            if not args.verbose_per_call: # Print compact progress if not verbose
                print(f"\r[{current_combination_count}/{total_combinations_to_test} | {progress_percent:.1f}%] Testing Obs2 Date: {obs2_datetime_str_iter}...", end="")
            else: # If verbose, print a header for the detailed output that will follow
                print(f"\n[{current_combination_count}/{total_combinations_to_test} | {progress_percent:.1f}%] Testing Obs2 Date: {obs2_datetime_str_iter}")


            # Call the main calculation function
            hit_result_data = find_codirectional_fantastical_eclipse(
                lat1=args.lat1, lon1=args.lon1, alt1_m=args.alt1, datetime_str1_utc=args.time1,
                lat2=args.lat2, lon2=args.lon2, alt2_m=args.alt2, datetime_str2_utc=obs2_datetime_str_iter,
                search_window_hours=args.search_hours_offset,
                planets_dict_arg=planets_main_dict_global,
                ts_arg=ts_main_global,
                verbose_per_call=args.verbose_per_call
            )
            
            if hit_result_data and hit_result_data.get("is_hit"):
                if not args.verbose_per_call: print() # Newline after progress if a hit is found
                print(f"  **** HIT FOUND for Obs2 Initial Date: {obs2_datetime_str_iter} ****")
                print(f"    Type: {hit_result_data['type']}, Overlap: {hit_result_data['overlap_metric']:.1f}%, "
                      f"Min AltAz Sep: {format_dms(hit_result_data['min_altaz_separation_deg'])}, "
                      f"Offset: {hit_result_data['optimal_time_offset_seconds']:.0f}s")
                found_hits_summary_list.append(hit_result_data)

    if not args.verbose_per_call: print() # Final newline after progress bar

    script_end_time = time.time()
    total_duration_seconds = script_end_time - script_start_time
    print(f"\n\n" + "="*10 + " SCRIPT EXECUTION COMPLETE " + "="*10)
    print(f"Total combinations tested: {current_combination_count}")
    print(f"Total execution time: {total_duration_seconds:.2f} seconds ({datetime.timedelta(seconds=total_duration_seconds)})")
    
    print("\n" + "="*10 + " SUMMARY OF ALL FOUND HITS " + "="*10)
    if found_hits_summary_list:
        for i, hit in enumerate(found_hits_summary_list):
            print(f"\nHit #{i+1}:")
            print(f"  Obs2 Initial Date (UTC): {hit['obs2_initial_date_utc']}")
            print(f"  Optimal Sun Obs Time (UTC): {hit['sun_obs_time_utc']}")
            print(f"  Optimal Moon Obs Time (UTC): {hit['moon_obs_time_utc']}")
            print(f"  Type: {hit['type']}")
            print(f"  Overlap: {hit['overlap_metric']:.1f}%")
            print(f"  Min Alt/Az Separation: {hit['min_altaz_separation_deg']:.4f}° ({format_dms(hit['min_altaz_separation_deg'])})")
            print(f"  Time Offset Applied: {hit['optimal_time_offset_seconds']:.0f} seconds")
            print("-" * 20)
        print(f"Total hits found: {len(found_hits_summary_list)}")
    else:
        print("No co-directional 'hits' found in the specified date range for Observer 2.")
    print("="*40)