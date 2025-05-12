import datetime
import numpy as np
from skyfield.api import Topos, load, wgs84
from skyfield.constants import AU_KM, DEG2RAD
#from skyfield.framelib import TargetingOps # For ICRS to AltAz conversion if needed manually
from scipy.optimize import minimize_scalar
import argparse

# --- Configuration ---
EPHEMERIS_FILE = 'de421.bsp'
DEFAULT_SUN_RADIUS_KM = 695700.0
DEFAULT_MOON_RADIUS_KM = 1737.4

# --- Helper Functions ---
def get_angular_diameter_arcsec(physical_radius_km, distance_au):
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
    # For AltAz from different observers, pressure/temp for refraction can be more important if high precision is needed
    # alt, az, _ = apparent_gcrs.altaz(temperature_C=10, pressure_mbar=1010) # Example
    return alt, az, apparent_gcrs

def shortest_angle_difference(a1_deg, a2_deg):
    """Calculates the shortest angle between two azimuths (0-360)."""
    diff = (a2_deg - a1_deg + 180) % 360 - 180
    return diff # Will be in range -180 to 180

# --- Main Calculation Function ---
def find_codirectional_fantastical_eclipse(
    lat1, lon1, alt1_m, datetime_str1_utc,
    lat2, lon2, alt2_m, datetime_str2_utc,
    search_window_hours=12,
    planets_dict_arg=None,
    ts_arg=None
):
    global EPHEMERIS_FILE

    if planets_dict_arg is None or ts_arg is None:
        print(f"INFO: Loading ephemeris ({EPHEMERIS_FILE}) and timescale within function...")
        eph = load(EPHEMERIS_FILE)
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

    print(f"Observer 1 (Sun): Lat={lat1:.2f}, Lon={lon1:.2f}, Alt={alt1_m}m at {initial_time1.isoformat()}")
    print(f"Observer 2 (Moon): Lat={lat2:.2f}, Lon={lon2:.2f}, Alt={alt2_m}m at {initial_time2.isoformat()}")
    print(f"Searching for co-directional alignment within +/- {search_window_hours} hours offset.")
    print("-" * 30)

    sun_radius_km = getattr(sun_obj, 'radius', None)
    if sun_radius_km: sun_radius_km = sun_radius_km.km
    if sun_radius_km is None:
        print(f"WARNING: Using default Sun radius: {DEFAULT_SUN_RADIUS_KM} km.")
        sun_radius_km = DEFAULT_SUN_RADIUS_KM

    moon_radius_km = getattr(moon_obj, 'radius', None)
    if moon_radius_km: moon_radius_km = moon_radius_km.km
    if moon_radius_km is None:
        print(f"WARNING: Using default Moon radius: {DEFAULT_MOON_RADIUS_KM} km.")
        moon_radius_km = DEFAULT_MOON_RADIUS_KM

    memo_times = {}
    memo_altaz = {}

    def altaz_difference_at_offset(time_offset_seconds):
        # Generate Skyfield Time objects, memoize for efficiency
        current_dt1 = initial_time1 + datetime.timedelta(seconds=time_offset_seconds)
        current_dt2 = initial_time2 + datetime.timedelta(seconds=time_offset_seconds)
        
        skyfield_current_time1 = memo_times.setdefault((current_dt1,), current_ts.utc(current_dt1))
        skyfield_current_time2 = memo_times.setdefault((current_dt2,), current_ts.utc(current_dt2))

        # Get Alt/Az for Sun from Obs1
        # Memoize the AltAz calculation too, if the time object is the same
        if skyfield_current_time1 not in memo_altaz:
             alt_s, az_s, _ = get_altaz_and_gcrs_position(current_ts, skyfield_current_time1, obs1_loc, sun_obj)
             memo_altaz[skyfield_current_time1] = (alt_s, az_s)
        else:
            alt_s, az_s = memo_altaz[skyfield_current_time1]

        # Get Alt/Az for Moon from Obs2
        if skyfield_current_time2 not in memo_altaz:
            alt_m, az_m, _ = get_altaz_and_gcrs_position(current_ts, skyfield_current_time2, obs2_loc, moon_obj)
            memo_altaz[skyfield_current_time2] = (alt_m, az_m)
        else:
            alt_m, az_m = memo_altaz[skyfield_current_time2]


        # If either object is below horizon, this alignment is less interesting for visual.
        # We can penalize it, or just let the minimizer find the smallest geometric Alt/Az diff.
        # For now, let's just find the smallest geometric Alt/Az difference.
        # The user can later check if alt_s and alt_m are positive.

        delta_alt_deg = alt_s.degrees - alt_m.degrees
        delta_az_deg = shortest_angle_difference(az_s.degrees, az_m.degrees) # Handles 0-360 wrap

        # This is the angular separation in the "overlaid local sky"
        # We want to minimize this value.
        altaz_separation = np.sqrt(delta_alt_deg**2 + delta_az_deg**2)
        
        # Heavily penalize if either is below horizon to guide optimizer
        # This is a heuristic. May need tuning or removal if it causes issues.
        # if alt_s.degrees < -2 or alt_m.degrees < -2: # Allow for slight below-horizon due to refraction etc.
        #    altaz_separation += 1000 # Large penalty
            
        return altaz_separation

    # Explicitly check zero offset
    print("\n--- Checking Alt/Az Difference at Initial Times (Offset = 0) ---")
    altaz_diff_at_zero_offset = altaz_difference_at_offset(0) 
    print(f"Alt/Az difference with zero offset: {altaz_diff_at_zero_offset:.6f} degrees")
    print("-" * 30)

    # Reset memos for the optimizer as it will call the function many times
    memo_times.clear()
    memo_altaz.clear()

    lower_bound_sec = -search_window_hours * 3600
    upper_bound_sec = search_window_hours * 3600

    result = minimize_scalar(
        altaz_difference_at_offset,
        bounds=(lower_bound_sec, upper_bound_sec),
        method='bounded',
        options={'xatol': 1.0} # Tol in seconds for the offset
    )

    min_altaz_separation_deg = result.fun
    optimal_time_offset_seconds = result.x

    if not result.success:
        print(f"Warning: Minimization might not have converged: {result.message}")
        # Simple fallback: use zero offset if minimization failed badly.
        # Or could do a coarse grid search like before.
        if min_altaz_separation_deg is None or np.isnan(min_altaz_separation_deg) or min_altaz_separation_deg > 180 : # arbitrary large val
            print("Minimization seems to have failed, using initial zero offset as fallback.")
            optimal_time_offset_seconds = 0.0
            min_altaz_separation_deg = altaz_difference_at_offset(optimal_time_offset_seconds)


    optimal_sun_obs_time = initial_time1 + datetime.timedelta(seconds=optimal_time_offset_seconds)
    optimal_moon_obs_time = initial_time2 + datetime.timedelta(seconds=optimal_time_offset_seconds)

    skyfield_optimal_sun_time = current_ts.utc(optimal_sun_obs_time)
    skyfield_optimal_moon_time = current_ts.utc(optimal_moon_obs_time)

    # Get final values at optimal offset
    final_alt_s, final_az_s, sun_gcrs_apparent = get_altaz_and_gcrs_position(current_ts, skyfield_optimal_sun_time, obs1_loc, sun_obj)
    final_alt_m, final_az_m, moon_gcrs_apparent = get_altaz_and_gcrs_position(current_ts, skyfield_optimal_moon_time, obs2_loc, moon_obj)

    sun_angular_diameter_arcsec = get_angular_diameter_arcsec(sun_radius_km, sun_gcrs_apparent.distance().au)
    moon_angular_diameter_arcsec = get_angular_diameter_arcsec(moon_radius_km, moon_gcrs_apparent.distance().au)
    
    sun_radius_deg_altaz = (sun_angular_diameter_arcsec / 2) / 3600
    moon_radius_deg_altaz = (moon_angular_diameter_arcsec / 2) / 3600
    sum_of_radii_deg_altaz = sun_radius_deg_altaz + moon_radius_deg_altaz

    print("\n--- Optimal Co-Directional 'Fantastical Eclipse' Found ---")
    print(f"Optimal Time Offset: {optimal_time_offset_seconds:.2f} seconds")
    print(f"Sun Observation Time (UTC): {optimal_sun_obs_time.isoformat()}")
    print(f"Moon Observation Time (UTC): {optimal_moon_obs_time.isoformat()}")
    print("-" * 20)
    print(f"Minimum Alt/Az Frame Separation: {min_altaz_separation_deg:.6f} degrees")
    print(f"  (This is sqrt( (delta_alt)^2 + (delta_az_shortest)^2 ) )")
    print("-" * 20)
    print("Sun (as seen from Observer 1's location & time):")
    print(f"  Altitude: {final_alt_s.degrees:.2f}°, Azimuth: {final_az_s.degrees:.2f}°")
    print(f"  GCRS RA/Dec: {sun_gcrs_apparent.radec(epoch=current_ts.J2000)}")
    print(f"  Angular Diameter: {sun_angular_diameter_arcsec:.2f} arcsec ({sun_angular_diameter_arcsec/3600:.4f} deg)")
    print(f"  Visibility: {'Above horizon' if final_alt_s.degrees > -0.5 else 'Below horizon'}") # Using -0.5 for typical horizon
    print("-" * 20)
    print("Moon (as seen from Observer 2's location & time):")
    print(f"  Altitude: {final_alt_m.degrees:.2f}°, Azimuth: {final_az_m.degrees:.2f}°")
    print(f"  GCRS RA/Dec: {moon_gcrs_apparent.radec(epoch=current_ts.J2000)}")
    print(f"  Angular Diameter: {moon_angular_diameter_arcsec:.2f} arcsec ({moon_angular_diameter_arcsec/3600:.4f} deg)")
    print(f"  Visibility: {'Above horizon' if final_alt_m.degrees > -0.5 else 'Below horizon'}")
    print("-" * 20)
    
    # "Hit" condition based on Alt/Az frame separation vs sum of radii
    # Both objects must also be visible for a meaningful "co-directional eclipse"
    is_visible_s = final_alt_s.degrees > -0.5 # Threshold for visibility
    is_visible_m = final_alt_m.degrees > -0.5

    is_hit_altaz = (is_visible_s and is_visible_m and
                    sun_angular_diameter_arcsec > 0 and moon_angular_diameter_arcsec > 0 and 
                    min_altaz_separation_deg < sum_of_radii_deg_altaz)
              
    print(f"Sum of angular radii (for Alt/Az comparison): {sum_of_radii_deg_altaz:.6f} degrees")
    print(f"Is it a co-directional 'hit' (Alt/Az sep < sum of radii AND both visible)? {'YES!' if is_hit_altaz else 'No'}")

    if not is_visible_s: print("  Sun is below horizon for Observer 1.")
    if not is_visible_m: print("  Moon is below horizon for Observer 2.")

    if is_hit_altaz:
        overlap_metric = max(0, (sum_of_radii_deg_altaz - min_altaz_separation_deg) / sum_of_radii_deg_altaz) * 100
        diff_radii_deg_altaz = abs(sun_radius_deg_altaz - moon_radius_deg_altaz)
        
        if min_altaz_separation_deg <= diff_radii_deg_altaz:
            if sun_radius_deg_altaz >= moon_radius_deg_altaz:
                print("Type: Annular-like (in Alt/Az frame)")
            else:
                print("Type: Total-like (in Alt/Az frame)")
        else:
            print("Type: Partial-like (in Alt/Az frame)")
        print(f"  Approximate overlap metric: {overlap_metric:.1f}%")
    elif (is_visible_s and is_visible_m): # If visible but not a hit
        if not (sun_angular_diameter_arcsec > 0 and moon_angular_diameter_arcsec > 0):
            print("  Reason for no hit (despite visibility): Zero calculated angular diameter for one/both bodies.")
        else:
            print(f"  Reason for no hit (despite visibility): Min Alt/Az sep ({min_altaz_separation_deg:.4f}°) >= Sum of radii ({sum_of_radii_deg_altaz:.4f}°).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Co-Directional Fantastical Eclipse Calculator")
    # Add arguments as before
    parser.add_argument("--lat1", type=float, required=True, help="Latitude of Observer 1 (Sun) in degrees")
    parser.add_argument("--lon1", type=float, required=True, help="Longitude of Observer 1 (Sun) in degrees")
    parser.add_argument("--alt1", type=float, default=0, help="Altitude of Observer 1 (Sun) in meters (default: 0)")
    parser.add_argument("--time1", type=str, required=True, help="Initial UTC datetime for Observer 1 (Sun) as YYYY-MM-DDTHH:MM:SSZ")

    parser.add_argument("--lat2", type=float, required=True, help="Latitude of Observer 2 (Moon) in degrees")
    parser.add_argument("--lon2", type=float, required=True, help="Longitude of Observer 2 (Moon) in degrees")
    parser.add_argument("--alt2", type=float, default=0, help="Altitude of Observer 2 (Moon) in meters (default: 0)")
    parser.add_argument("--time2", type=str, required=True, help="Initial UTC datetime for Observer 2 (Moon) as YYYY-MM-DDTHH:MM:SSZ")
    
    parser.add_argument("--search_hours", type=float, default=12, help="Search window in hours (+/- from initial times) (default: 12)")
    parser.add_argument("--eph_file", type=str, default=EPHEMERIS_FILE, 
                        help=f"Skyfield ephemeris file (default: {EPHEMERIS_FILE})")
    args = parser.parse_args()
    
    if args.eph_file != EPHEMERIS_FILE:
        print(f"INFO: Using user-specified ephemeris file: {args.eph_file}")
        EPHEMERIS_FILE = args.eph_file
    
    print(f"Loading ephemeris ({EPHEMERIS_FILE})...")
    eph_main = load(EPHEMERIS_FILE)
    planets_main_dict = {
        'sun': eph_main['sun'],
        'moon': eph_main['moon'],
        'earth': eph_main['earth']
    }
    ts_main = load.timescale()
    print("Ephemeris loaded.")

    find_codirectional_fantastical_eclipse(
        args.lat1, args.lon1, args.alt1, args.time1,
        args.lat2, args.lon2, args.alt2, args.time2,
        search_window_hours=args.search_hours,
        planets_dict_arg=planets_main_dict,
        ts_arg=ts_main
    )