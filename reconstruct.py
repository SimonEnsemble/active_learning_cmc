import marimo

__generated_with = "0.23.11"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pocomc as pc
    from scipy.stats import norm, uniform
    import pandas as pd
    import marimo as mo
    import random
    import corner
    from aquarel import load_theme

    theme = load_theme("umbra_light")
    theme.set_font(size=16)
    theme.apply()
    # ... plotting code here
    theme.apply_transforms()
    return corner, mo, norm, np, pc, pd, plt, random, uniform


@app.cell
def _(plt):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::lucide:database:: data
    """)
    return


@app.cell
def _():
    n_data = 5
    return (n_data,)


@app.cell
def _(n_data, pd):
    data = pd.DataFrame({
        "[S] (mol/m³)": [0.0, 30.0, 3.0, 12.0, 7.5, 8.5, 0.75, 8.75, 13.25],
        "γ (N/m)": [71.87, 29.54, 44.2325, 29.68, 32.42, 31.06, 55.2075, 29.89333, 29.567]
    })
    data["γ (N/m)"] /= 1000.0
    s_next = data.loc[n_data+1, "[S] (mol/m³)"]
    data = data.head(n_data)
    data
    return data, s_next


@app.cell
def _(data):
    gamma_0_obs = data.loc[0, "γ (N/m)"]
    return (gamma_0_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::icon-park:soap-bubble:: model
    """)
    return


@app.cell
def _(np):
    def gamma(c, theta):
        gamma_0, a, K, cmc = theta
        if c < cmc:
            return gamma_0 - a * np.log(1 + K * c)
        else:
            return gamma_0 - a * np.log(1 + K * cmc)

    return (gamma,)


@app.cell
def _():
    sigma = 0.001 # (N/m) 
    return (sigma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # prior
    """)
    return


@app.cell
def _(gamma_0_obs, norm, pc, sigma, uniform):
    prior = pc.Prior([
        norm(loc=gamma_0_obs, scale=sigma), # gamma_0 [N/m]
        uniform(0.001, 0.1),                # a [N/m]
        uniform(0.01, 10000.0),             # K [m3 / mol]
        uniform(0.0, 30.0),                 # cmc [N/m]
    ])
    return (prior,)


@app.cell
def _(prior):
    thetas_prior = prior.rvs(500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # posterior
    """)
    return


@app.cell
def _():
    theta_names = ["$\gamma_0$ (N/m)", "a (N/m)", "K (m$^3$/mol)", "c$^*$ (mol/m$^3$)"]
    return (theta_names,)


@app.cell
def _(gamma, np):
    def log_like(theta, data, sigma):
        gamma_0, a, K, cmc = theta
    
        gamma_preds = [gamma(c, theta) for c in data["[S] (mol/m³)"]]
    
        diff = gamma_preds - data["γ (N/m)"].values
        return -0.5 * np.dot(diff, diff) / sigma**2.0

    return (log_like,)


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, log_like, pc, prior, sigma):
    sampler = pc.Sampler(
        prior=prior,
        likelihood=log_like,
        likelihood_args=[data, sigma],
        precondition=True
    )

    # Run sampler
    sampler.run()

    thetas_posterior, weights, logl, logp = sampler.posterior()
    return thetas_posterior, weights


@app.cell
def _(data, thetas_posterior, viz_belief):
    viz_belief(data, thetas_posterior)
    return


@app.cell
def _(colors, gamma, np, plt, random, s_next):
    def viz_belief(data, thetas, n_samples=50):
        fig, (ax_hist, ax_main) = plt.subplots(
            2, 1, figsize=(6, 7),
            gridspec_kw={"height_ratios": [1, 3]},
            sharex=True
        )
    
        # CMC hist
        ax_hist.hist(
            [theta[-1] for theta in thetas],
            bins=20, color=colors[3],
            histtype="step", edgecolor=colors[3],
            lw=2
        )
        ax_hist.set_ylabel("# samples")
        ax_hist.set_xlabel("CMC (mol/m$^3$)")

        # main axis
        ax_main.set_xlabel("[surfactant] (mol/m$^3$)")
        ax_main.set_ylabel("surface tension (N/m)")
    
        ax_main.scatter(
            data["[S] (mol/m³)"], data["γ (N/m)"], 
            clip_on=False, color=colors[0],
            s=65, edgecolor="k", zorder=100,
            label="data"
        )

        for i, theta in enumerate(random.sample(list(thetas), n_samples)):
            ss = np.linspace(0, 30.0, 300)
            gs = [gamma(s, theta) for s in ss]
            ax_main.plot(
                ss, gs, color=colors[1], alpha=0.3,  
                label="posterior sample" if i == 0 else None, lw=2
            )

        ax_main.legend()

        ax_main.annotate(
            "", xy=(s_next, 0.0), xytext=(s_next, 0.01),
            arrowprops=dict(arrowstyle="->", color=colors[4], lw=2),
            ha="center", color=colors[4]
        )
    
        ax_main.set_xlim(0, 30.0)
        ax_main.set_ylim(0, 0.08)
    
        plt.show()

    return (viz_belief,)


@app.cell
def _(corner, plt, theta_names, thetas_posterior, weights):
    fig = corner.corner(
        thetas_posterior, weights=weights, 
        labels=theta_names, color='C3',
        smooth=3.0
    )
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
