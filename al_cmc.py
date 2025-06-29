import marimo

__generated_with = "0.12.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.linear_model import ARDRegression
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel
    return (
        ARDRegression,
        ConstantKernel,
        GaussianProcessRegressor,
        RBF,
        mo,
        np,
        pd,
        plt,
    )


@app.cell
def _():
    data_files = [
        "H2O-C10E8",
        "H2O-C14E6",
        "H2O-CTAB-bulk",
        "H2O-SDS-bulk",
        "H2O-C12E5",
        "H2O-C16E8",
        "H2O-OTG",
        "H2O-Tween20"
    ]
    return (data_files,)


@app.cell
def _(data_files, pd):
    def read_data(i):
        return pd.read_csv("data/" + data_files[i] + ".csv")

    data = read_data(5)
    data
    return data, read_data


@app.cell
def _(ConstantKernel, GaussianProcessRegressor, RBF, np):
    def train_gp(data):
        X = np.array(data["concentration_mol/m^3"]).reshape(-1, 1)
        y = np.array(data["surften_N/m"])
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.05, length_scale_bounds="fixed")
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, normalize_y=True)
        gp.fit(X, y)
        return gp
    return (train_gp,)


@app.cell
def _(data, train_gp):
    gp = train_gp(data)
    return (gp,)


@app.cell
def _(data, np):
    s = np.linspace(0, data["concentration_mol/m^3"].max())
    return (s,)


@app.cell
def _(gp, s):
    gp.sample_y(s.reshape(-1,1), 2)
    return


@app.cell
def _(data, gp, plt, s):
    plt.figure()
    plt.scatter(data["concentration_mol/m^3"], data["surften_N/m"])

    function_samples = gp.sample_y(s.reshape(-1, 1), 10)
    for i in range(10):
        plt.plot(s, function_samples[:, i], color="gray", alpha=0.1)
    plt.xlabel("[surfactant] (mol/m$^3$)")
    plt.ylabel("surface tension (N/m)")
    plt.show()
    return function_samples, i


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
