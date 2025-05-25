import numpy as np
import pandas as pd


def GetDataset(name, base_path):
    """ Load a dataset
    
    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"
    
    Returns
    -------
    X : features (nXp)
    y : labels (n)
    
	"""
    if name == "wind": #wind data
        data_dict = np.load(base_path +'data.npz')
        X, y = data_dict['Xfull'], data_dict['Yfull']
        X = Ｘ.astype(np.float32)
        y = y.astype(np.float32)
        
    if name == "Weather": #weather forecasting data
        data = pd.read_csv(base_path + 'Weather.csv')
        col_names = data.columns
        data = data.to_numpy()
        covariate_col = ['p (mbar)','Tpot (K)','Tdew (degC)','rh (%)','VPmax (mbar)','VPact (mbar)',
        'VPdef (mbar)','sh (g/kg)','H2OC (mmol/mol)','rho (g/m**3)','wv (m/s)','max. wv (m/s)',
        'wd (deg)','rain (mm)','raining (s)','Tlog (degC)','OT']
        response_col = 'T (degC)'

        X = data[:, [col_name in covariate_col for col_name in col_names]]
        y = data[:, col_names == response_col].flatten()

        X = X.astype(np.float32)
        y = y.astype(np.float32)
    
    if name == "amazon":
        data = pd.read_csv(base_path + 'AMZN_data.csv')
        col_names = data.columns
        data = data.to_numpy() 
        covariate_col = ['open', 'high', 'low', 'close']
        response_col = 'volume'
    
        X = data[:, [col_name in covariate_col for col_name in col_names]]
        y = data[:, col_names == response_col].flatten()

        X = X.astype(np.float32)
        y = y.astype(np.float32)

    if name == "electricity":
        data = pd.read_csv(base_path + 'electricity.csv')
        col_names = data.columns
        data = data.to_numpy()

        # remove the first stretch of time where 'transfer' does not vary
        data = data[17760:]

        # set up variables for the task (predicting 'transfer')
        covariate_col = ['nswprice', 'nswdemand', 'vicprice', 'vicdemand']
        response_col = 'transfer'
        # keep data points for 9:00am - 12:00pm
        keep_rows = np.where((data[:,2]>data[17,2])&(data[:,2]<data[24,2]))[0]

        X = data[keep_rows][:,np.where([t in covariate_col for t in col_names])[0]]
        y = data[keep_rows][:,np.where(col_names==response_col)[0]].flatten()
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    if name == "x100-y10-reg":
        num_samples = 1000
        num_features = 100
        num_outputs = 10  # Number of dimensions for y
        lag = 3
        base_noise_std_dev = 0.5
        # Seed for reproducibility
        np.random.seed(0)

        # Generate input features X with additional rows for initial lag values
        X = np.random.randn(num_samples + lag, num_features)

        # Initialize output matrix y
        y = np.zeros((num_samples, num_outputs))

        # Define coefficients for a simple model
        # Each output y[:, j] depends on three lagged values from three features of X
        coefficients = np.array([
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Coefficients for output 1
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Coefficients for output 2
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Coefficients for output 3
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # ...
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                              [[1, 1, 1], [1, 1, 1], [1, 1, 1]]])

        # Generate outputs with heteroskedastic noise
        for i in range(lag, num_samples):
            for j in range(num_outputs):
                # Linear combination of the first three features of the last three input vectors
                y[i, j] = (coefficients[j, 0, 0] * X[i - 2, j % num_features] +
                coefficients[j, 1, 0] * X[i - 1, j % num_features] +
                coefficients[j, 2, 0] * X[i, j % num_features])
                # Add heteroskedastic noise
                noise_std_dev = base_noise_std_dev + 0.01 * i
                y[i, j] += np.random.randn() * noise_std_dev

        # Trim X to remove initial lag values and cast types
        X = X[lag:, :].astype(np.float32)
        y = y.astype(np.float32)

    '''if name == "x100-y10-reg":
        # generate a random dataset
        num_samples = 10000
        X = np.random.random((num_samples, 100))
        W = np.random.random((100, 10))
        y = X.dot(W) + np.random.randn(num_samples, 10)
        X = X.astype(np.float32)
        y = y.astype(np.float32)'''

    return X, y
