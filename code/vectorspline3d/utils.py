"""
Utility functions for this particular presentation
"""
import numpy as np
import xarray as xr
import pandas as pd


def sample_from_grid(grid, coordinates=None, size=None, random_state=None):
    """
    Sample values from a grid randomly or at specifies coordinates.
    """
    if coordinates is None:
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        elif random_state is None:
            random_state = np.random.RandomState()
        coords = {
            name: xr.DataArray(
                random_state.randint(0, grid[name].size, size=size), dims="dummy"
            )
            for name in grid.coords
        }
        sampled_grid = grid.isel(**coords)
    else:
        coords = {
            name: xr.DataArray(coordinates[name], dims="dummy") for name in coordinates
        }
        sampled_grid = grid.sel(**coords, method="nearest")
    columns = {name: sampled_grid[name].values for name in grid.data_vars}
    for name in coords:
        columns[name] = sampled_grid[name].values
    sample = pd.DataFrame(columns)
    return sample
