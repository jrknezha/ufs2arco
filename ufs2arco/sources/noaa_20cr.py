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
    sample_dims = ("time",)
    horizontal_dims = ("latitude", "longitude")
    
    static_vars = ("hgt_sfc", "land") 

    available_variables = (
        "uwnd", "vwnd", "air", "shum", "hgt", "air_2m", "shum_2m", 
        "uwnd_10m", "vwnd_10m", "pres_sfc", "skt_sfc", 
        "land", "hgt_sfc", "tcdc", "lhtfl", "cape", "pr_wtr",
        "soilm", "prmsl", "prate"
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
        time: dict,
        data_folder: str,
        variables: Optional[list | tuple] = None,
        levels: Optional[list | tuple] = None,
        use_nearest_levels: Optional[bool] = False,
        slices: Optional[dict] = None,
    ) -> None:
        self.time = pd.date_range(**time)
        self.data_folder = data_folder

        super().__init__(
            variables=variables,
            levels=levels,
            use_nearest_levels=use_nearest_levels,
            slices=slices
        )

    #override
    def open_sample_dataset(
        self,
        dims: dict,
        open_static_vars: bool,
        cache_dir: Optional[str] = None,
    ) -> xr.Dataset:
        #caching file first happens here

        #for this design, assume that we are getting the full year file but only pulling the returning the requested time step
        #this will keep the implementation in line with the datamover calls. Will need to address special cases with mpi before finalizing
        
        #generally want to handle only deleting the cached file if a) the value requested is the last time step of the file
        #or b) the value requested is the last timestep the user requested 

        #TODO: implement file caching handling -- this should probably be per var below

        dsdict = {}
        osv = open_static_vars or self._open_static_vars(dims)
        variables = self.variables if osv else self.dynamic_vars
        #check if we got the files here, we_got_the_data -- in other files, may need to be below
        for varname in variables:
            dslist = []
            #get the data
            try:
                filepath = self._build_path(dims['time'], varname)
                thisvar = self._open_single_variable(dims, varname, filepath)
            except Exception as e:
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
        # Rename the dimension and its coordinate simultaneously
        ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        #handle variable names that have single levels
        single_level_var = {
            "air_2m" : "air",
            "shum_2m": "shum",
            "uwnd_10m": "uwnd",
            "vwnd_10m": "vwnd",
            "pres_sfc": "pres",
            "skt_sfc": "skt",
            "hgt_sfc": "hgt"
        }

        target_var = single_level_var.get(variable, variable)
        var_info = ds.get(target_var)

        if variable in self.static_vars:
            #static vars are time invariant, set to the requested time
            var_info = var_info.drop_indexes("time", errors="ignore").reset_coords("time", drop=True).squeeze(drop=True)
        else: 
            var_info = var_info.sel(time=dims['time']).expand_dims("time") #select the requested time slice
            #select only requested levels
            if self.levels is not None and "level" in ds.coords:
                var_info = var_info.sel(level=list(self.levels), **self._level_sel_kwargs)

        return var_info

    
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
        year_postfix = "MO"
        if time.year < 1981: 
            year_postfix = "SI"

        vars_other_names = {
            "air_2m" : f"2m{year_postfix}/air.2m.{time.year}.nc",
            "shum_2m": f"2m{year_postfix}/shum.2m.{time.year}.nc",
            "uwnd_10m": f"10m{year_postfix}/uwnd.10m.{time.year}.nc",
            "vwnd_10m": f"10m{year_postfix}/vwnd.10m.{time.year}.nc",
            "pres_sfc": f"sfc{year_postfix}/pres.sfc.{time.year}.nc",
            "skt_sfc": f"sfc{year_postfix}/skt.{time.year}.nc",
            "tcdc": f"misc{year_postfix}/tcdc.eatm.{time.year}.nc",
            "lhtfl": f"sfcFlx{year_postfix}/lhtfl.{time.year}.nc",
            "cape": f"sfc{year_postfix}/cape.{time.year}.nc",
            "pr_wtr": f"misc{year_postfix}/pr_wtr.eatm.{time.year}.nc",
            "soilm": f"subsfc{year_postfix}/soilm.{time.year}.nc",
            "prmsl": f"misc{year_postfix}/prmsl.{time.year}.nc",
            "prate": f"sfc{year_postfix}/prate.{time.year}.nc"
        }

        if variable == "hgt_sfc":
            filepath = f"timeInvariant{year_postfix}/hgt.sfc.nc"
        elif variable == "land":
            filepath = f"timeInvariant{year_postfix}/land.nc"
        elif variable in vars_other_names:
            filepath = vars_other_names[variable]
        else:
            filepath = f"prs{year_postfix}/{variable}.{time.year}.nc"

        
        fullpath = f"{self.data_folder}/{filepath}"

        return fullpath 


