"""Allegheny County geographical scenario in UTM 17N, EPSG:26917 (kilometres)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from geovalue.calculus import PriceSurface, spatial_gradient, integrate_price_surface, save_price_surface
from geovalue.official.data import INPUT_COLUMNS, CATEGORICAL_INPUTS

CRS = "EPSG:26917"
_FORWARD = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
_INVERSE = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)


def project_km(longitude: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon=np.atleast_1d(np.asarray(longitude,dtype=float))
    lat=np.atleast_1d(np.asarray(latitude,dtype=float))
    if lon.shape != lat.shape:
        raise ValueError("Longitude and latitude coordinate shapes must match")
    x,y = _FORWARD.transform(lon.ravel().tolist(),lat.ravel().tolist())
    return np.asarray(x,float).reshape(lon.shape)/1000, np.asarray(y,float).reshape(lat.shape)/1000


def geographic_from_km(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    east=np.atleast_1d(np.asarray(x,dtype=float))
    north=np.atleast_1d(np.asarray(y,dtype=float))
    if east.shape != north.shape:
        raise ValueError("Projected coordinate shapes must match")
    a,b = _INVERSE.transform((east.ravel()*1000).tolist(),(north.ravel()*1000).tolist())
    return np.asarray(a,float).reshape(east.shape),np.asarray(b,float).reshape(north.shape)


def fixed_reference(data: pd.DataFrame, *, sale_year: int = 2025, sale_month: int = 6) -> dict:
    """Controlled comparison: vary only location, all property features fixed."""
    if data.empty:
        raise ValueError("Cannot build scenario from empty property dataset")
    out = {}
    for col in INPUT_COLUMNS:
        if col in CATEGORICAL_INPUTS:
            mode = data[col].dropna().astype(str).mode()
            out[col] = str(mode.iloc[0]) if len(mode) else "Unknown"
        else:
            numeric = pd.to_numeric(data[col], errors="coerce")
            if not numeric.notna().any():
                raise ValueError(f"No reference value for {col}")
            out[col] = float(numeric.median())
    out.update(sale_year=int(sale_year),sale_month=int(sale_month))
    return out


def build_surface(data: pd.DataFrame, model: object, *, grid_size: int = 86, sigma: float = 1.5, support_km: float = 4.5, sale_year: int = 2025) -> tuple[PriceSurface,dict]:
    if not 20 <= grid_size <= 200 or not 0.5 <= sigma <= 8 or not 0.25 <= support_km <= 25:
        raise ValueError("Grid/smoothing/support parameters outside allowed bounds")
    if len(data) < 20:
        raise ValueError("Need at least 20 observed properties")
    x,y = project_km(data.longitude.to_numpy(float),data.latitude.to_numpy(float))
    if not (np.isfinite(x).all() and np.isfinite(y).all()) or np.ptp(x)<1 or np.ptp(y)<1:
        raise ValueError("Valid geographical coordinate span is required")
    xs = np.linspace(x.min(),x.max(),grid_size)
    ys = np.linspace(y.min(),y.max(),grid_size)
    gx,gy = np.meshgrid(xs,ys)
    loc = np.column_stack((gx.ravel(),gy.ravel()))
    dist = cKDTree(np.column_stack((x,y))).query(loc)[0].reshape(gx.shape)
    supported = dist <= support_km
    if supported.sum() < 15:
        raise ValueError("Not enough geographically supported grid points")
    lon,lat = geographic_from_km(gx.ravel(),gy.ravel())
    ref = fixed_reference(data,sale_year=sale_year)
    rows = pd.DataFrame({key:[ref[key]]*len(lon) for key in INPUT_COLUMNS})
    rows["longitude"],rows["latitude"] = lon,lat
    values = np.asarray(model.predict(rows),float).reshape(gx.shape)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite predictions in controlled geographical scenario")
    numerator = gaussian_filter(np.where(supported,values,0.0),sigma=sigma)
    denom = gaussian_filter(supported.astype(float),sigma=sigma)
    smooth = np.divide(numerator,denom,out=np.full_like(numerator,np.nan),where=denom>1e-8)
    smooth[~supported]=np.nan
    dx,dy = spatial_gradient(xs,ys,smooth)
    return PriceSurface(xs,ys,smooth,dx,dy),ref


def gradient_query(surface: PriceSurface, longitude: float, latitude: float, bearing: float = 0, *, max_distance_km: float = 5) -> dict:
    if not np.isfinite([longitude,latitude,bearing,max_distance_km]).all() or max_distance_km <= 0:
        raise ValueError("Invalid gradient request")
    x,y=project_km(np.array([longitude]),np.array([latitude]))
    xx,yy=np.meshgrid(surface.x_km,surface.y_km)
    good=np.isfinite(surface.values)&np.isfinite(surface.d_dx)&np.isfinite(surface.d_dy)
    if not good.any():
        raise ValueError("No supported gradients")
    locations=np.column_stack((xx[good],yy[good]))
    distance,ind=cKDTree(locations).query([x[0],y[0]],k=1)
    if distance>max_distance_km:
        raise ValueError(f"No supported gradient within {max_distance_km:g} km")
    index=int(ind)
    g_x=float(surface.d_dx[good][index]);g_y=float(surface.d_dy[good][index]); magnitude=float(np.hypot(g_x,g_y))
    lon,lat=geographic_from_km(np.array([locations[index,0]]),np.array([locations[index,1]]))
    theta=np.deg2rad(bearing)
    return {
        "longitude_sampled": float(lon[0]),"latitude_sampled": float(lat[0]),
        "distance_to_gradient_km": float(distance),
        "scenario_value_usd":float(surface.values[good][index]),
        "gradient_east_usd_per_km":g_x,"gradient_north_usd_per_km":g_y,
        "gradient_magnitude_usd_per_km":magnitude,
        "steepest_ascent_grid_bearing_degrees":float((np.degrees(np.arctan2(g_x,g_y))+360)%360) if magnitude>0 else 0.0,
        "requested_bearing_degrees":float(bearing%360),
        "directional_derivative_usd_per_km":float(g_x*np.sin(theta)+g_y*np.cos(theta)),
        "coordinate_basis":"EPSG:26917 projected grid north; not true compass north",
        "limitation":"Smoothed fixed-feature model scenario; not a causal effect or house price appreciation rate",
    }
