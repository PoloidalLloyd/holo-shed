"""
User-defined derived variables for Hermes-3 GUI.

Add new quantities by decorating functions with @register_derived_variable.
Each function receives an xarray Dataset and returns a DataArray.
"""

from typing import Dict, Callable, Optional
import numpy as np

# Registry of derived variable functions
_DERIVED_VARS: Dict[str, dict] = {}

def register_derived_variable(name: str, description: str = "", units: str = ""):
    """
    Decorator to register a derived variable function.

    Parameters
    ----------
    name : str
        Variable name as it will appear in the GUI
    description : str, optional
        Description of the variable
    units : str, optional
        Units string (e.g., "W/m^2", "eV", "m^-3")
    """
    def decorator(func: Callable):
        _DERIVED_VARS[name] = {
            'func': func,
            'description': description,
            'units': units,
        }
        return func
    return decorator

def get_available_variables():
    """Return list of available derived variable names."""
    return list(_DERIVED_VARS.keys())

def get_derived_var_units(name: str) -> Optional[str]:
    """Get units for a derived variable, if registered."""
    if name in _DERIVED_VARS:
        return _DERIVED_VARS[name].get('units', None) or None
    return None

def compute_derived_variables(ds, requested_vars=None):
    """
    Compute derived variables for a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset
    requested_vars : list of str, optional
        Only compute these variables. If None, compute all registered.

    Returns
    -------
    xarray.Dataset
        Copy of input with derived variables added
    """
    ds_out = ds.copy()

    for var_name, info in _DERIVED_VARS.items():
        if requested_vars and var_name not in requested_vars:
            continue

        if var_name in ds_out:
            continue  # Already exists

        try:
            da = info['func'](ds)
            if da is not None:
                # Set units from registration if not already set
                if info.get('units') and not da.attrs.get('units'):
                    da.attrs['units'] = info['units']
                ds_out[var_name] = da
                print(f"  ✓ {var_name}")
        except Exception as e:
            print(f"  ✗ {var_name}: {e}")

    return ds_out

# ============================================================================
# Derived Variable Definitions
# ============================================================================
@register_derived_variable("eq_par_tot", description='Parallel heat flux density', units='W/m^2')
def compute_eq_par_tot(ds):
    """eq_par_tot = efe_tot_ylow / da"""
    required = ['efe_tot_ylow', 'da']
    if not all(v in ds for v in required):
        return None

    eq_par_tot = ds['efe_tot_ylow'] / ds['da']
    return eq_par_tot


@register_derived_variable('d+_particle_flux', description='d+ particle flux', units='m^-2 s^-1')
def compute_dplus_particle_flux(ds):
    """d+_particle_flux = Vd+ * Nd+"""
    required = ['Vd+', 'Nd+']
    if not all(v in ds for v in required):
        return None
    dplus_particle_flux = ds['Vd+'] * ds['Nd+']
    return dplus_particle_flux

@register_derived_variable('e_particle_flux', description='e particle flux', units='m^-2 s^-1')
def compute_e_particle_flux(ds):
    """e_particle_flux = Ve * Ne"""
    required = ['Ve', 'Ne']
    if not all(v in ds for v in required):
        return None
    e_particle_flux = ds['Ve'] * ds['Ne']
    return e_particle_flux

@register_derived_variable('d_particle_flux', description='d particle flux', units='m^-2 s^-1')
def compute_d_particle_flux(ds):
    """d_particle_flux = Vd * Nd"""
    required = ['Vd', 'Nd']
    if not all(v in ds for v in required):
        return None
    d_particle_flux = ds['Vd'] * ds['Nd']
    return d_particle_flux

@register_derived_variable('i_particle_flux', description='i particle flux', units='m^-2 s^-1')
def compute_i_particle_flux(ds):
    """i_particle_flux = Vi * Ni"""
    required = ['Vi', 'Ni']
    if not all(v in ds for v in required):
        return None
    i_particle_flux = ds['Vi'] * ds['Ni']
    return i_particle_flux

    
def _find_crossing(dist, data, threshold):
    """
    Find the interpolated location where data crosses threshold.
    Uses linear interpolation between grid points for accuracy.

    Parameters
    ----------
    dist : array-like
        Position/distance array
    data : array-like
        Data values (e.g., Te)
    threshold : float
        Threshold value to find crossing

    Returns
    -------
    float or np.nan
        Interpolated crossing location, or nan if no crossing found
    """
    # Find indices where the data crosses the threshold
    crossings = np.where(np.diff(np.signbit(data - threshold)))[0]

    if len(crossings) == 0:
        return np.nan

    # Use the last crossing (closest to target)
    final_crossing = crossings[-1]

    # Linear interpolation to find the exact crossing location
    t1, t2 = dist[final_crossing], dist[final_crossing + 1]
    data1, data2 = data[final_crossing], data[final_crossing + 1]

    # Avoid division by zero
    if data2 == data1:
        return t1

    location = t1 + (threshold - data1) * (t2 - t1) / (data2 - data1)

    return location


@register_derived_variable('detachment_front_distance (5eV)', description='5eV front parallel distance from target', units='m')
def compute_front_pardist_5eV(ds):
    """
    Find detachment front location using interpolated threshold crossing.
    Returns the parallel distance from the target to where Te crosses 5 eV.

    The result is broadcast across the spatial dimension so it can be plotted
    as a horizontal line in profile views and also used in time histories.

    For 1D: Returns array with dims matching Te (e.g., pos/y, t)
    For 2D: Returns array with dims matching Te (e.g., x, theta/y, t)
    """
    import xarray as xr

    threshold = 5.0  # eV

    if 'Te' not in ds:
        return None

    Te = ds['Te']
    Te_dims = tuple(Te.dims)

    # Check if 2D (has both x and theta/y dimensions)
    is_2d = ('x' in Te_dims and ('theta' in Te_dims or 'y' in Te_dims))

    if is_2d:
        # 2D case: find detachment along each flux tube (for each x)
        # Spar is only available after calling fline method, so skip silently if not present
        if 'Spar' not in ds:
            return None

        pol_dim = 'theta' if 'theta' in Te_dims else 'y'
        Spar = ds['Spar']

        # Get time dimension from Te's dims
        tdim = None
        for d in Te_dims:
            if d in ('t', 'time'):
                tdim = d
                break
        if tdim is None:
            # Static case - not supported for front tracking
            return None

        n_x = len(ds['x'])
        n_pol = len(ds[pol_dim])
        n_t = len(ds[tdim])

        # First compute the front distance for each (x, t) pair
        front_1d = np.full((n_x, n_t), np.nan)

        for ix in range(n_x):
            # Get Spar values for this flux surface (distance along field line)
            try:
                spar_vals = np.asarray(Spar.isel(x=ix).values)
                max_spar = np.nanmax(spar_vals)
            except Exception:
                continue

            for it in range(n_t):
                try:
                    Te_slice = np.asarray(Te.isel(x=ix, **{tdim: it}).values)
                    crossing_loc = _find_crossing(spar_vals, Te_slice, threshold)
                    if not np.isnan(crossing_loc):
                        # Distance from target (max Spar - crossing location)
                        front_1d[ix, it] = max_spar - crossing_loc
                except Exception:
                    pass

        # Broadcast across poloidal dimension so it can be plotted as a horizontal line
        # Shape: (n_x, n_pol, n_t)
        front_broadcast = np.broadcast_to(
            front_1d[:, np.newaxis, :],
            (n_x, n_pol, n_t)
        ).copy()  # copy to make it writeable

        front_dist = xr.DataArray(
            front_broadcast,
            dims=['x', pol_dim, tdim],
            coords={'x': ds['x'], pol_dim: ds[pol_dim], tdim: ds[tdim]}
        )

        front_dist.attrs['units'] = 'm'
        front_dist.attrs['long_name'] = f'{threshold:.0f}eV front parallel distance from target'
        front_dist.attrs['short_name'] = f'{threshold:.0f}eV front pol. distance from target [m]'
        print(f"      Created with dims={front_dist.dims} (Te has dims={Te_dims})")
        return front_dist

    else:
        # 1D case: find detachment along field line
        # Get spatial dimension from Te's dims (exclude time dims)
        pos_dim = None
        tdim = None
        for d in Te_dims:
            if d in ('t', 'time'):
                tdim = d
            elif d in ('pos', 'y', 'x', 's'):
                pos_dim = d

        if pos_dim is None:
            print(f"    Warning: Could not determine spatial dim from Te dims: {Te_dims}")
            return None
        if tdim is None:
            # Static case - not supported for front tracking
            return None

        # Get position values - try 'pos' coordinate first, then the dimension itself
        if 'pos' in ds.coords:
            dist = np.asarray(ds['pos'].values)
        elif pos_dim in ds.coords:
            dist = np.asarray(ds[pos_dim].values)
        else:
            dist = np.arange(len(ds[pos_dim]))

        max_dist = np.nanmax(dist)
        n_pos = len(ds[pos_dim])
        n_t = len(ds[tdim])

        # First compute the front distance for each time step
        front_1d = np.full(n_t, np.nan)

        for it in range(n_t):
            try:
                Te_slice = np.asarray(Te.isel(**{tdim: it}).values)
                crossing_loc = _find_crossing(dist, Te_slice, threshold)
                if not np.isnan(crossing_loc):
                    # Distance from target (max pos - crossing location)
                    front_1d[it] = max_dist - crossing_loc
            except Exception:
                pass

        # Broadcast across spatial dimension so it can be plotted as a horizontal line
        # Match Te's exact dimension order
        if Te_dims[0] == tdim:
            # Te is (t, pos) - time first
            front_broadcast = np.broadcast_to(
                front_1d[:, np.newaxis],
                (n_t, n_pos)
            ).copy()
            front_dist = xr.DataArray(
                front_broadcast,
                dims=[tdim, pos_dim],
                coords={tdim: ds[tdim], pos_dim: ds[pos_dim]}
            )
        else:
            # Te is (pos, t) - position first
            front_broadcast = np.broadcast_to(
                front_1d[np.newaxis, :],
                (n_pos, n_t)
            ).copy()
            front_dist = xr.DataArray(
                front_broadcast,
                dims=[pos_dim, tdim],
                coords={pos_dim: ds[pos_dim], tdim: ds[tdim]}
            )

        front_dist.attrs['units'] = 'm'
        front_dist.attrs['long_name'] = f'{threshold:.0f}eV front parallel distance from target'
        front_dist.attrs['short_name'] = f'{threshold:.0f}eV front pol. distance from target [m]'
        print(f"      Created with dims={front_dist.dims} (Te has dims={Te_dims})")
        return front_dist

# @register_derived_variable("pressure", "Total pressure [Pa]")
# def compute_pressure(ds):
#     """P = n * (Te + Ti) in Pascals"""
#     if not all(v in ds for v in ['Ne', 'Te', 'Ti']):
#         return None
    
#     eV_to_J = 1.602e-19
#     P = ds['Ne'] * (ds['Te'] + ds['Ti']) * eV_to_J
#     P.attrs['units'] = 'Pa'
#     P.attrs['long_name'] = 'Total pressure'
#     return P

# @register_derived_variable("Mach", "Parallel Mach number")
# def compute_mach(ds):
#     """M = v_|| / c_s"""
#     if not all(v in ds for v in ['Ne', 'NVi', 'Te', 'Ti']):
#         return None
    
#     eV_to_J = 1.602e-19
#     mi = 2.0 * 1.673e-27  # D+ mass
    
#     v_par = ds['NVi'] / ds['Ne']
#     c_s = np.sqrt((ds['Te'] + ds['Ti']) * eV_to_J / mi)
    
#     M = v_par / c_s
#     M.attrs['units'] = ''
#     M.attrs['long_name'] = 'Parallel Mach number'
#     return M

# @register_derived_variable("Te_Ti_ratio", "Te/Ti ratio")
# def compute_te_ti_ratio(ds):
#     if not all(v in ds for v in ['Te', 'Ti']):
#         return None
    
#     ratio = ds['Te'] / ds['Ti']
#     ratio.attrs['units'] = ''
#     return ratio

