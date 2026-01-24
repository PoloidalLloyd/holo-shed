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


@register_derived_variable('detachment_location', 'Detachment location [m]')
def compute_detachment_location(ds):
    """
    Find detachment front location (where Te < 5 eV).
    
    For 1D: Returns scalar or 1D array (vs time)
    For 2D: Returns 1D array (vs x) or 2D array (vs x, time)
    """
    import xarray as xr
    # Check if 2D (has both x and theta/y dimensions)
    is_2d = ('x' in ds.dims and ('theta' in ds.dims or 'y' in ds.dims))
    
    if is_2d:
        # 2D case: find detachment along each flux tube (for each x)
        if 'Spar' not in ds:
            print("    Warning: Spar not found for 2D detachment calculation")
            return None
        
        pol_dim = 'theta' if 'theta' in ds.dims else 'y'
        Te = ds['Te']
        Spar = ds['Spar']
        
        # For each x position, find first point where Te < 5
        detachment_threshold = 5.0  # eV
        
        # Create output array matching dimensions (x, and possibly time)
        if 't' in Te.dims or 'time' in Te.dims:
            tdim = 't' if 't' in Te.dims else 'time'
            # Time-dependent 2D case
            det_loc = xr.DataArray(
                np.full((len(ds['x']), len(ds[tdim])), np.nan),
                dims=['x', tdim],
                coords={'x': ds['x'], tdim: ds[tdim]}
            )
            
            for it in range(len(ds[tdim])):
                for ix in range(len(ds['x'])):
                    Te_slice = Te.isel(x=ix, **{tdim: it})
                    mask = Te_slice < detachment_threshold
                    if mask.any():
                        idx = int(mask.argmax(dim=pol_dim))
                        det_loc[ix, it] = float(Spar.isel(x=ix, **{pol_dim: idx}))
        else:
            # Static 2D case
            det_loc = xr.DataArray(
                np.full(len(ds['x']), np.nan),
                dims=['x'],
                coords={'x': ds['x']}
            )
            
            for ix in range(len(ds['x'])):
                Te_slice = Te.isel(x=ix)
                mask = Te_slice < detachment_threshold
                if mask.any():
                    idx = int(mask.argmax(dim=pol_dim))
                    det_loc[ix] = float(Spar.isel(x=ix, **{pol_dim: idx}))
        
        det_loc.attrs['units'] = 'm'
        det_loc.attrs['long_name'] = 'Detachment front location (Te < 5 eV)'
        return det_loc
    
    else:
        # 1D case: find detachment along field line
        pos_dim = 'pos' if 'pos' in ds.dims else ('y' if 'y' in ds.dims else None)
        if pos_dim is None or pos_dim not in ds:
            return None
        
        Te = ds['Te']
        pos = ds[pos_dim]
        detachment_threshold = 5.0  # eV
        
        # Handle time-dependent vs static
        if 't' in Te.dims or 'time' in Te.dims:
            tdim = 't' if 't' in Te.dims else 'time'
            # Time-dependent 1D case
            det_loc = xr.DataArray(
                np.full(len(ds[tdim]), np.nan),
                dims=[tdim],
                coords={tdim: ds[tdim]}
            )
            
            for it in range(len(ds[tdim])):
                Te_slice = Te.isel(**{tdim: it})
                mask = Te_slice < detachment_threshold
                if mask.any():
                    idx = int(mask.argmax(dim=pos_dim))
                    det_loc[it] = float(pos[idx])
        else:
            # Static 1D case
            mask = Te < detachment_threshold
            if mask.any():
                idx = int(mask.argmax(dim=pos_dim))
                det_loc = float(pos[idx])
            else:
                det_loc = np.nan
        
        det_loc = xr.DataArray(det_loc) if not isinstance(det_loc, xr.DataArray) else det_loc
        det_loc.attrs['units'] = 'm'
        det_loc.attrs['long_name'] = 'Detachment front location (Te < 5 eV)'
        return det_loc

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

