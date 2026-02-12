import logging
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from ufs2arco.sources import Source

logger = logging.getLogger("ufs2arco")

class NOAA20CR(Source):
    """
    Source class for NOAA 20th Century Reanalysis (20CR) NetCDF datasets.
    """

    # Typical dimensions for 20CR NetCDF files
    sample_dims = ("time", "level", "lat", "lon")
    horizontal_dims = ("lat", "lon")
    
    # Common static variables in 20CR (topography, land-sea mask)
    static_vars = ("hgt_sfc", "land") 

    # Example variables; update these based on the specific files you are indexing
    available_variables = (
        "uwnd", "vwnd", "air", "shum", "hgt", "air_2m", "shum_2m", "uwnd_10m", "vwnd_10m", "pres_sfc", "skt_sfc"
    )

    # Standard 20CR v3 pressure levels (hPa)
    available_levels = (
        1000, 975, 950, 925, 900, 850, 800, 750, 700, 
        650, 600, 550, 500, 450, 400, 350, 300, 250, 
        200, 150, 100, 70, 50, 30, 20, 10, 5, 1
    )

    @property
    def rename(self) -> dict:
        """
        Maps NOAA 20CR native naming to ufs2arco standards.
        Example: Mapping 'lat' to 'latitude' and 'lon' to 'longitude'.
        """
        return {
            "lat": "latitude",
            "lon": "longitude",
        }

    def __init__(
        self,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:
        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices
        )

    def open_dataset(self, file_path: str) -> xr.Dataset:
        """
        Helper method to open the NetCDF and apply initial standardizations.
        """
        ds = xr.open_dataset(file_path)

        # 1. Select variables and levels
        # We only want to select 'level' if it exists in the dataset (some are 2D surface fields)
        if self.levels is not None and "level" in ds.coords:
            ds = ds.sel(level=list(self.levels), **self._level_sel_kwargs)

        # 2. Rename dimensions/variables to standard
        existing_rename = {k: v for k, v in self.rename.items() if k in ds.dims or k in ds.coords}
        ds = ds.rename(existing_rename)

        # 3. Apply any user-defined slices (lat/lon bounding boxes, etc.)
        ds = self.apply_slices(ds)

        return ds
    
    #override
    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:
        #caching file first happens here
        file = "GET THE FILE HERE"

        #for this design, assume that we are getting the full year file but only pulling the returning the requested time step
        #this will keep the implementation in line with the datamover calls. Will need to address special cases with mpi before finalizing
        
        #generally want to handle only deleting the cached file if a) the value requested is the last time step of the file
        #or b) the value requested is the last timestep the user requested 

        dsdict = {}
        osv = open_static_vars or self._open_static_vars(dims)
        variables = self.variables if osv else self.dynamic_vars
        #check if we got the files here, we_got_the_data
        for varname in variables:
            dslist = []
            #get the data
            try:
                thisvar = self._open_single_variable(varname, file)
            except:
                thisvar = None
            dslist.append(thisvar)
            if len(dslist) == 1:
                    dsdict[varname] = dslist[0]
            elif len(dslist) > 1:
                dsdict[varname] = xr.merge(dslist)
            else:
                logger.warning(
                    f"{self.name}: Could not find {varname}, will stop reading variables for this sample\n\t" +
                    f"dims = {dims}, file_suffixes = {self._varmeta[varname]['file_suffixes']}"
                )
                dsdict = {}
                break
        
        # the dataset is either full or completely empty if we had trouble
        xds = xr.Dataset(dsdict)
        if len(xds) > 0:
            xds = self.apply_slices(xds)
        return xds

    def _open_single_variable(
            self,
            dims: dict,
            variable: str,
            file_path: str
    ) -> xr.DataArray:
        #here we get the variable from the file

        #data array out of the dataset pulling the data 
        #this is where we will handle translating 20cr has multiple vars named, 
        ds = xr.open_dataset(file_path)
        
        #handle variable names that have single levels
        single_level_var = {
            "air_2m" : "air",
            "shum_2m": "shum",
            "uwnd_10m": "uwnd",
            "vwnd_10m": "vwnd",
            "pres_sfc": "pres",
            "skt_sfc": "skt"
        }

        target_var = single_level_var.get(variable, variable)
        #somewhere we need to limit to the necessary dimensions of time, lat and lon? 
        var_info = ds.get(target_var)

        result = var_info.sel(time=dims['time'])

        if self.levels is not None and "level" in ds.coords:
            ds = ds.sel(level=list(self.levels), **self._level_sel_kwargs)

        return result

    
    def _build_path(
            self, 
            time: pd.Timestamp,
            variable: str
    ) -> str:
        """
        Build the path for the 20CR files to pull based on desired variable 
        
        :param self: Description
        :param time: Description
        :type time: pd.Timestamp
        :param variables: Description
        :type variables: Optional[list | tuple]
        :return: Description
        :rtype: str
        """
        #ftp = "ftp2.psl.noaa.gov"

        year_postfix = "MO"
        if time.year < 1981: 
            year_postfix = "SI"

        if variable == "hgt_sfc":
            filepath = f"timeInvariant{year_postfix}/hgt_sfc.nc"

        if variable == "land":
            filepath = f"timeInvariant{year_postfix}/land.nc"

        vars_other_names = {
            "air_2m" : f"2m{year_postfix}/air.{time.year}.nc",
            "shum_2m": f"2m{year_postfix}/shum.{time.year}.nc",
            "uwnd_10m": f"10m{year_postfix}/uwnd.{time.year}.nc",
            "vwnd_10m": f"10m{year_postfix}/vwnd.{time.year}.nc",
            "pres_sfc": f"sfc{year_postfix}/pres.{time.year}.nc",
            "skt_sfc": f"sfc{year_postfix}/skt.{time.year}.nc"
        }

        if variable == "hgt_sfc":
            filepath = f"timeInvariant{year_postfix}/hgt_sfc.nc"
        elif variable == "land":
            filepath = f"timeInvariant{year_postfix}/land.nc"
        elif variable in vars_other_names.keys:
            filepath = vars_other_names[variable]
        else:
            filepath = f"prs{year_postfix}/{variable}.{time.year}.nc"

        #TODO: fullpath = f"TO_BE_ADDED/{filepath}"
        #return full path once defined

        return filepath 


