def detrend_2step(data):
    import scipy
    from scipy import stats
    import numpy as np
    try:
        x = np.arange(data.size)
        R = scipy.stats.linregress(x[np.isfinite(data)], data[np.isfinite(data)])
        return data - x*R.slope - R.intercept
    except ValueError:
        return data
        
def correlate_nino(data, nino):
    """
    Returns the Pearson correlation coefficient at each gridpoint of 'data' against 'nino'
    """
    
    import numpy as np
    import scipy.stats
    import xarray as xr
    
    # Get the nino values at the dates matching 'data'
    # nino = nino.sel(time=data.time)
    
    # Function to apply on each gridpoint
    def correlate_gridpoint(data):
        return scipy.stats.pearsonr(nino, data)[0]
    
    # Apply the function on each gridpoint
    pearsonr = np.apply_along_axis(correlate_gridpoint, data.get_axis_num('time'), data)
    
    # This is just to get the correct coordinates for the output
    sample = data.mean('time')
    
    # Convert the numpy array back into xarray
    return xr.DataArray(pearsonr, sample.coords)

def correlate_nino_by_month(data, nino):
    """
    Runs 'correlate_nino' on each month separately.
    If quarterly or seasonal averages are inputted, will run on each quarter or season seperately. 
    """
    
    return data.groupby('time.month').map(correlate_nino, nino=nino)