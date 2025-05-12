import datetime
import numpy as np
from skyfield.api import Topos, load, wgs84
from skyfield.constants import AU_KM, DEG2RAD
#from skyfield.positionlib import ChebyshevPosition # For type checking in radius fallback
from scipy.optimize import minimize_scalar
import argparse

# --- Configuration ---
# Default ephemeris file. This global can be updated by command-line arg.
EPHEMERIS_FILE = 'de421.bsp'
# EPHEMERIS_FILE = 'de440.bsp'

# Default radii (IAU 2015 values where applicable, or common values)
# Sun (volumetric mean radius)
DEFAULT_SUN_RADIUS_KM = 695700.0
# Moon (mean radius)
DEFAULT_MOON_RADIUS_KM = 1737.4

# --- Helper Functions ---
def get_angular_diameter_arcsec(physical_radius_km, distance_au):
    """Calculates apparent angular diameter in arcseconds."""
    # Ensure radius is valid before use
    if physical_radius_km is None or physical_radius_km <= 0 or np.isnan(physical_radius_km):
        # print(f"Debug: Invalid physical_radius_km ({physical_radius_km}) in get_angular_diameter_arcsec")
        return 0
    distance_km = distance_au * AU_KM
    if distance_km <= 0:
        # print(f"Debug: Invalid distance_km ({distance_km}) in get_angular_diameter_arcsec")
        return 0
    angular_diameter_rad = 2 * physical_radius_km / distance_km
    return np.degrees(angular_diameter_rad) * 3600

def get_apparent_position(ts, time_obj, observer_loc, target_body):
    """Gets the apparent GCRS RA/Dec and Alt/Az of a celestial body."""
    astrometric = observer_loc.at(time_obj).observe(target_body)
    apparent = astrometric.apparent()
    alt, az, _ = apparent.altaz() # Add pressure & temp for refraction if needed
    return apparent, alt, az

# --- Main Calculation Function ---
def find_fantastical_eclipse(
    lat1, lon1, alt1_m, datetime_str1_utc,
    lat2, lon2, alt2_m, datetime_str2_utc,
    search_window_hours=0.1,
    planets_dict_arg=None, # Argument for passed-in planets dictionary
    ts_arg=None            # Argument for passed-in timescale
):
    # Determine Skyfield objects (Sun, Moon, Earth, Timescale)
    # Use passed objects if available, otherwise load them.
    # These will be the working objects for this function call.
    global EPHEMERIS_FILE # Access the global that might have been updated by args

    if planets_dict_arg is None or ts_arg is None:
        print(f"INFO: Loading ephemeris ({EPHEMERIS_FILE}) and timescale within function...")
        eph = load(EPHEMERIS_FILE) # EPHEMERIS_FILE is the (potentially updated) global
        sun_obj = eph['sun']
        moon_obj = eph['moon']
        earth_obj = eph['earth']
        current_ts = load.timescale()
    else:
        sun_obj = planets_dict_arg['sun']
        moon_obj = planets_dict_arg['moon']
        earth_obj = planets_dict_arg['earth']
        current_ts = ts_arg

    obs1_loc = earth_obj + wgs84.latlon(lat1, lon1, elevation_m=alt1_m)
    obs2_loc = earth_obj + wgs84.latlon(lat2, lon2, elevation_m=alt2_m)

    initial_time1 = datetime.datetime.fromisoformat(datetime_str1_utc.replace('Z', '+00:00'))
    initial_time2 = datetime.datetime.fromisoformat(datetime_str2_utc.replace('Z', '+00:00'))

    if initial_time1.tzinfo is None: initial_time1 = initial_time1.replace(tzinfo=datetime.timezone.utc)
    if initial_time2.tzinfo is None: initial_time2 = initial_time2.replace(tzinfo=datetime.timezone.utc)
        
    skyfield_initial_time1 = current_ts.utc(initial_time1)
    skyfield_initial_time2 = current_ts.utc(initial_time2)

    print(f"Observer 1 (Sun): Lat={lat1:.2f}, Lon={lon1:.2f}, Alt={alt1_m}m")
    print(f"Observer 2 (Moon): Lat={lat2:.2f}, Lon={lon2:.2f}, Alt={alt2_m}m")
    print(f"Initial Sun Obs Time (UTC): {initial_time1.isoformat()}")
    print(f"Initial Moon Obs Time (UTC): {initial_time2.isoformat()}")
    print(f"Searching for optimal time offset within +/- {search_window_hours} hours.")
    print("-" * 30)


    print("\n--- Initial Conditions (at input times) ---")
    initial_sun_apparent, initial_sun_alt, initial_sun_az = get_apparent_position(current_ts, skyfield_initial_time1, obs1_loc, sun_obj)
    print("Sun (Observer 1) at Initial Input Time:")
    print(f"  Time (UTC): {initial_time1.isoformat()}")
    print(f"  Altitude: {initial_sun_alt.degrees:.2f}°, Azimuth: {initial_sun_az.degrees:.2f}°")
    print(f"  Visibility: {'Above horizon' if initial_sun_alt.degrees > 0 else 'Below horizon'}")

    initial_moon_apparent, initial_moon_alt, initial_moon_az = get_apparent_position(current_ts, skyfield_initial_time2, obs2_loc, moon_obj)
    print("Moon (Observer 2) at Initial Input Time:")
    print(f"  Time (UTC): {initial_time2.isoformat()}")
    print(f"  Altitude: {initial_moon_alt.degrees:.2f}°, Azimuth: {initial_moon_az.degrees:.2f}°")
    print(f"  Visibility: {'Above horizon' if initial_moon_alt.degrees > 0 else 'Below horizon'}")
    print("-" * 30)
    # --- Get radii with fallbacks ---
    sun_radius_km = None
    if hasattr(sun_obj, 'radius') and sun_obj.radius is not None:
        sun_radius_km = sun_obj.radius.km
    
    if sun_radius_km is None: # Fallback if attribute missing or radius is None
        type_name = type(sun_obj).__name__
        print(f"WARNING: Sun object (type {type_name}) lacks a valid .radius attribute or it's None. "
              f"This can occur with some ephemeris files. Using default Sun radius: {DEFAULT_SUN_RADIUS_KM} km.")
        sun_radius_km = DEFAULT_SUN_RADIUS_KM

    moon_radius_km = None
    if hasattr(moon_obj, 'radius') and moon_obj.radius is not None:
        moon_radius_km = moon_obj.radius.km

    if moon_radius_km is None: # Fallback
        type_name = type(moon_obj).__name__
        print(f"WARNING: Moon object (type {type_name}) lacks a valid .radius attribute or it's None. "
              f"This can occur with some ephemeris files. Using default Moon radius: {DEFAULT_MOON_RADIUS_KM} km.")
        moon_radius_km = DEFAULT_MOON_RADIUS_KM

    # Validate radii before proceeding
    if sun_radius_km is None or np.isnan(sun_radius_km) or sun_radius_km <= 0:
        print(f"ERROR: Invalid Sun radius determined ({sun_radius_km} km). Cannot proceed.")
        return
    if moon_radius_km is None or np.isnan(moon_radius_km) or moon_radius_km <= 0:
        print(f"ERROR: Invalid Moon radius determined ({moon_radius_km} km). Cannot proceed.")
        return

    memo = {}
    def separation_at_offset(time_offset_seconds):
        current_dt1 = initial_time1 + datetime.timedelta(seconds=time_offset_seconds)
        current_dt2 = initial_time2 + datetime.timedelta(seconds=time_offset_seconds)
        
        skyfield_current_time1 = memo.setdefault((current_dt1,), current_ts.utc(current_dt1))
        skyfield_current_time2 = memo.setdefault((current_dt2,), current_ts.utc(current_dt2))

        sun_apparent, _, _ = get_apparent_position(current_ts, skyfield_current_time1, obs1_loc, sun_obj)
        moon_apparent, _, _ = get_apparent_position(current_ts, skyfield_current_time2, obs2_loc, moon_obj)

        separation_deg = sun_apparent.separation_from(moon_apparent).degrees
        return separation_deg

    lower_bound_sec = -search_window_hours * 3600
    upper_bound_sec = search_window_hours * 3600

    result = minimize_scalar(
        separation_at_offset,
        bounds=(lower_bound_sec, upper_bound_sec),
        method='bounded',
        options={'xatol': 1.0}
    )

    if not result.success:
        print(f"Warning: Minimization might not have converged: {result.message}")
        min_val_from_result = result.fun if hasattr(result, 'fun') else None # SciPy <1.6 fun, >=1.6 fun
        if min_val_from_result is None or np.isnan(min_val_from_result):
             print("Error: Minimization failed to find a valid separation.")
             print("Attempting fallback grid search...")
             best_offset_s = 0
             min_sep_fallback = float('inf')
             # Reduce range or step for fallback to avoid excessive time
             fallback_step_sec = max(60, int(search_window_hours * 3600 / 720)) # Aim for ~720 steps
             for offset_s_fallback in np.arange(lower_bound_sec, upper_bound_sec, fallback_step_sec):
                 sep = separation_at_offset(offset_s_fallback)
                 if sep < min_sep_fallback:
                     min_sep_fallback = sep
                     best_offset_s = offset_s_fallback
             if min_sep_fallback == float('inf'):
                 print("Fallback grid search also failed.")
                 return
             min_separation_deg = min_sep_fallback
             optimal_time_offset_seconds = best_offset_s
             print(f"Fallback found minimum separation: {min_separation_deg:.4f} degrees at offset {optimal_time_offset_seconds:.2f} s")
        else:
            min_separation_deg = result.fun
            optimal_time_offset_seconds = result.x
    else:
        min_separation_deg = result.fun
        optimal_time_offset_seconds = result.x

    optimal_sun_obs_time = initial_time1 + datetime.timedelta(seconds=optimal_time_offset_seconds)
    optimal_moon_obs_time = initial_time2 + datetime.timedelta(seconds=optimal_time_offset_seconds)

    skyfield_optimal_sun_time = current_ts.utc(optimal_sun_obs_time)
    skyfield_optimal_moon_time = current_ts.utc(optimal_moon_obs_time)

    sun_final_apparent, sun_alt, sun_az = get_apparent_position(current_ts, skyfield_optimal_sun_time, obs1_loc, sun_obj)
    moon_final_apparent, moon_alt, moon_az = get_apparent_position(current_ts, skyfield_optimal_moon_time, obs2_loc, moon_obj)
    
    # sun_radius_km and moon_radius_km are already determined with fallbacks
    sun_angular_diameter_arcsec = get_angular_diameter_arcsec(sun_radius_km, sun_final_apparent.distance().au)
    moon_angular_diameter_arcsec = get_angular_diameter_arcsec(moon_radius_km, moon_final_apparent.distance().au)
    
    sun_radius_deg = (sun_angular_diameter_arcsec / 2) / 3600
    moon_radius_deg = (moon_angular_diameter_arcsec / 2) / 3600
    sum_of_radii_deg = sun_radius_deg + moon_radius_deg

    print("\n--- Optimal 'Fantastical Eclipse' Found ---")
    print(f"Optimal Time Offset: {optimal_time_offset_seconds:.2f} seconds")
    print(f"Sun Observation Time (UTC): {optimal_sun_obs_time.isoformat()}")
    print(f"Moon Observation Time (UTC): {optimal_moon_obs_time.isoformat()}")
    print("-" * 20)
    print(f"Minimum Angular Separation: {min_separation_deg:.6f} degrees ({min_separation_deg*3600:.2f} arcsec)")
    print("-" * 20)
    print("Sun (as seen from Observer 1's location & time):")
    print(f"  RA/Dec (GCRS): {sun_final_apparent.radec(epoch=current_ts.J2000)}")
    print(f"  Altitude: {sun_alt.degrees:.2f}°, Azimuth: {sun_az.degrees:.2f}°")
    print(f"  Distance: {sun_final_apparent.distance().au:.4f} AU")
    print(f"  Angular Diameter: {sun_angular_diameter_arcsec:.2f} arcsec ({sun_angular_diameter_arcsec/3600:.4f} deg)")
    print(f"  Visibility: {'Above horizon' if sun_alt.degrees > 0 else 'Below horizon'}")
    print("-" * 20)
    print("Moon (as seen from Observer 2's location & time):")
    print(f"  RA/Dec (GCRS): {moon_final_apparent.radec(epoch=current_ts.J2000)}")
    print(f"  Altitude: {moon_alt.degrees:.2f}°, Azimuth: {moon_az.degrees:.2f}°")
    print(f"  Distance: {moon_final_apparent.distance().au:.4f} AU")
    print(f"  Angular Diameter: {moon_angular_diameter_arcsec:.2f} arcsec ({moon_angular_diameter_arcsec/3600:.4f} deg)")
    print(f"  Visibility: {'Above horizon' if moon_alt.degrees > 0 else 'Below horizon'}")
    print("-" * 20)
    
    print(f"Sum of angular radii: {sum_of_radii_deg:.6f} degrees")
    
    # A "hit" requires both bodies to have a non-zero angular size
    is_hit = (sun_angular_diameter_arcsec > 0 and moon_angular_diameter_arcsec > 0 and 
              min_separation_deg < sum_of_radii_deg)
              
    print(f"Is it a 'hit' (separation < sum of radii)? {'YES!' if is_hit else 'No'}")

    if is_hit:
        overlap_metric = max(0, (sum_of_radii_deg - min_separation_deg) / sum_of_radii_deg) * 100
        diff_radii_deg = abs(sun_radius_deg - moon_radius_deg)
        
        if min_separation_deg <= diff_radii_deg:
            if sun_radius_deg >= moon_radius_deg: # Moon is smaller or same size
                print("Type: Annular-like (Moon center within Sun's disk)")
            else: # Sun is smaller than Moon
                print("Type: Total-like (Sun center within Moon's disk)")
        else:
            print("Type: Partial-like (Disks overlapping)")
        print(f"  Approximate overlap metric (center-to-center based): {overlap_metric:.1f}%")
    else:
        if not (sun_angular_diameter_arcsec > 0 and moon_angular_diameter_arcsec > 0):
            print("  Reason for no hit: One or both celestial bodies have zero calculated angular diameter.")
        else:
            print(f"  Reason for no hit: Minimum separation ({min_separation_deg:.4f}°) is not less than Sum of radii ({sum_of_radii_deg:.4f}°).")


    if not (sun_alt.degrees > -1 and moon_alt.degrees > -1):
        print("\nNote: One or both objects may be below the horizon for their respective observers.")
        print("This 'fantastical eclipse' might not be visually confirmable even if coordinates align.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fantastical Eclipse Calculator")
    parser.add_argument("--lat1", type=float, required=True, help="Latitude of Observer 1 (Sun) in degrees")
    parser.add_argument("--lon1", type=float, required=True, help="Longitude of Observer 1 (Sun) in degrees")
    parser.add_argument("--alt1", type=float, default=0, help="Altitude of Observer 1 (Sun) in meters (default: 0)")
    parser.add_argument("--time1", type=str, required=True, help="Initial UTC datetime for Observer 1 (Sun) as YYYY-MM-DDTHH:MM:SSZ")

    parser.add_argument("--lat2", type=float, required=True, help="Latitude of Observer 2 (Moon) in degrees")
    parser.add_argument("--lon2", type=float, required=True, help="Longitude of Observer 2 (Moon) in degrees")
    parser.add_argument("--alt2", type=float, default=0, help="Altitude of Observer 2 (Moon) in meters (default: 0)")
    parser.add_argument("--time2", type=str, required=True, help="Initial UTC datetime for Observer 2 (Moon) as YYYY-MM-DDTHH:MM:SSZ")
    
    parser.add_argument("--search_hours", type=float, default=12, help="Search window in hours (+/- from initial times) (default: 12)")
    # Changed --eph to --eph_file to match args.eph_file used later
    parser.add_argument("--eph_file", type=str, default=EPHEMERIS_FILE, 
                        help=f"Skyfield ephemeris file (default: {EPHEMERIS_FILE})")

    args = parser.parse_args()
    
    # Update global EPHEMERIS_FILE if user provided a specific one via command line.
    # This ensures that if find_fantastical_eclipse needs to load ephemeris locally,
    # it uses the user-specified file.
    if args.eph_file != EPHEMERIS_FILE: # Check if user provided a non-default file
        print(f"INFO: Using user-specified ephemeris file: {args.eph_file}")
        EPHEMERIS_FILE = args.eph_file # Update the global
    
    print(f"Loading ephemeris ({EPHEMERIS_FILE})... This might take a moment.")
    eph_main = load(EPHEMERIS_FILE) # This uses the (potentially updated) global EPHEMERIS_FILE
    
    planets_main_dict = {
        'sun': eph_main['sun'],
        'moon': eph_main['moon'],
        'earth': eph_main['earth']
    }
    ts_main = load.timescale()
    print("Ephemeris loaded.")

    find_fantastical_eclipse(
        args.lat1, args.lon1, args.alt1, args.time1,
        args.lat2, args.lon2, args.alt2, args.time2,
        search_window_hours=args.search_hours,
        planets_dict_arg=planets_main_dict, # Pass the dictionary of Skyfield body objects
        ts_arg=ts_main                     # Pass the Skyfield timescale object
    )