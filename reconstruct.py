import marimo

__generated_with = "0.23.10"
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
    from scipy.special import logsumexp
    from aquarel import load_theme

    theme = load_theme("scientific")
    theme.set_font(size=15)
    theme.apply()
    return corner, logsumexp, mo, norm, np, pc, pd, plt, uniform


@app.cell
def _(plt):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return (colors,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::icon-park:soap-bubble:: model
    """)
    return


@app.cell
def _():
    theta_names = [
        "$\gamma_0$ (N/m)", "a (N/m)", "K (m$^3$/mol)", "c$^*$ (mol/m$^3$)"
    ]
    return (theta_names,)


@app.cell
def _(np):
    def gamma(c, theta):
        # unpack parameters
        gamma_0, a, K, cmc = theta
        c = np.asarray(c)
        c_eff = np.minimum(c, cmc) # cap c at cmc
        return gamma_0 - a * np.log(1 + K * c_eff)

    return (gamma,)


@app.cell
def _():
    sigma = 0.001 # (N/m) 
    return (sigma,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::noto:brain:: prior
    """)
    return


@app.cell
def _(norm, pc, uniform):
    prior = pc.Prior(
        [
            norm(loc=72.8/1000.0, scale=0.01), # gamma_0 [N/m]
            uniform(0.001, 0.1),                # a [N/m]
            uniform(0.01, 10000.0),             # K [m3 / mol]
            uniform(0.0, 30.0),                 # cmc [N/m]
        ]
    )
    return (prior,)


@app.cell
def _(prior):
    thetas_prior = prior.rvs(500)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::icon-park:data:: data & likelihood function
    """)
    return


@app.cell
def _():
    n_data = 2
    return (n_data,)


@app.cell
def _(n_data, pd):
    data = pd.DataFrame(
        {
            "[S] (mol/m³)": [
                0.0, 30.0, 3.0, 12.0, 7.5, 8.5, 
                0.75, 8.75, 13.25
            ],
            "γ (N/m)": [
                71.87, 29.54, 44.2325, 29.68, 32.42, 31.06,
                55.2075, 29.89333, 29.567
            ]
        }
    )
    data["γ (N/m)"] /= 1000.0

    data = data.head(n_data)
    data
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::icon-park:lightning:: posterior
    """)
    return


@app.cell
def _(gamma, np):
    def log_like(theta, data, sigma):
        gamma_0, a, K, cmc = theta

        gamma_preds = gamma(data["[S] (mol/m³)"].values, theta)

        diff = gamma_preds - data["γ (N/m)"].values
        n = len(diff)
    
        return -0.5 * np.dot(diff, diff) / sigma**2 - n * np.log(sigma)

    return (log_like,)


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

    samples, weights, logl, logp = sampler.posterior()
    return samples, weights


@app.cell
def _(np):
    def draw_samples(samples, weights, n):
        """
        Draw n samples of (x_star, alpha) from the posterior,
        using pocoMC's importance weights.
        """
        idx = np.random.choice(
            len(samples),
            p=weights,
            size=n,
            replace=True
        )
        return samples[idx, :]

    return (draw_samples,)


@app.cell
def _(draw_samples, samples, weights):
    draw_samples(samples, weights, 2)
    return


@app.cell
def _(corner, n_data, plt, samples, theta_names, weights):
    fig = corner.corner(
        samples, weights=weights, 
        labels=theta_names, color='C6',
        smooth=3.0
    )
    plt.savefig(f"posterior_distn_{n_data}.pdf", format="pdf")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::emojione:thinking-face:: decision-making
    """)
    return


@app.cell
def _(np):
    def gaussian_logpdf(y, mean, sigma):
        z = (y - mean) / sigma
        return -0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * z ** 2

    return (gaussian_logpdf,)


@app.cell
def _(draw_samples, gamma, gaussian_logpdf, logsumexp, np, sigma):
    def eig_nested_mc(
        c, prior_samples, prior_weights, 
        N_outer=1000, N_inner=1001
    ):
        """
        Estimate EIG(c_candidate) using nested Monte Carlo.
        """
        thetas_outer = draw_samples(prior_samples, prior_weights, N_outer)
        gamma_outer = np.array(
            [gamma(c, theta) for theta in thetas_outer]
        )
        gamma_obs_outer = gamma_outer + sigma * np.random.randn(N_outer)
        logp_true = gaussian_logpdf(gamma_obs_outer, gamma_outer, sigma)

        # Inner samples: used to approximate the marginal p(gamma_obs|c)
        thetas_inner = draw_samples(prior_samples, prior_weights, N_inner)
        gamma_inner = np.array(
            [gamma(c, theta) for theta in thetas_inner]
        )
    
        # (N_outer, N_inner) matrix of log p(y_outer_i | theta_inner_j)
        logp_inner_matrix = gaussian_logpdf(
            gamma_obs_outer[:, None], gamma_inner[None, :], sigma
        )
        log_marginal = logsumexp(logp_inner_matrix, axis=1) - np.log(N_inner)

        eig_estimate = np.mean(logp_true - log_marginal)
        return eig_estimate

    return (eig_nested_mc,)


@app.cell
def _(eig_nested_mc, samples, weights):
    eig_nested_mc(1.0, samples, weights)
    return


@app.cell
def _(eig_nested_mc, np, pd, samples, weights):
    cs_eig = np.linspace(0, 30, 25)

    eig_data = pd.DataFrame(
        {
            "c [mol/m3]": cs_eig,
            "EIG": [eig_nested_mc(c, samples, weights) for c in cs_eig]
        }
    )
    eig_data
    return (eig_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # :eyes: for viz
    """)
    return


@app.cell
def _(data, eig_data, samples, viz_belief, weights):
    viz_belief(
        data, samples, weights, eig_data,
        n_samples=25
    )
    return


@app.cell
def _(colors, draw_samples, gamma, n_data, np, plt):
    def viz_belief(
        data, samples, weights, eig_data,
        n_samples=50, show_cmc_hist=True
    ):
        fig, (ax_hist, ax_main, ax_eig) = plt.subplots(
            3, 1, figsize=(6, 7),
            gridspec_kw={"height_ratios": [1, 3, 1]},
            sharex=True
        )
        ax_eig.set_xlabel("[surfactant] (mol/m$^3$)")

        ###
        #   CMC hist
        ###
        thetas = draw_samples(samples, weights, len(weights))
        ax_hist.hist(
            [theta[-1] for theta in thetas],
            bins=20, color=colors[0],
            histtype="step", edgecolor=colors[0],
            lw=2
        )
        ax_hist.set_ylabel("# samples")
        ax_hist.set_xlabel("CMC (mol/m$^3$)")

        ###
        #   surface tension isotherm
        ###
        ax_main.set_ylabel("surface tension (N/m)")

        ax_main.scatter(
            data["[S] (mol/m³)"], data["γ (N/m)"], 
            clip_on=False, color=colors[1],
            s=70, edgecolor="k", zorder=100,
            label="data"
        )
        for i, (x, y) in enumerate(zip(data["[S] (mol/m³)"], data["γ (N/m)"])):
            xytext = (0, -12)
            if i in [0, 2, 6]:
                xytext = (8, 4)
            if i in [1, 3, 4, 5, 8]:
                xytext = (0, 9)

            if i in [0, 1]:
                i = 0
            else:
                i = i - 1


            ax_main.annotate(
                str(i), (x, y),
                textcoords="offset points", xytext=xytext,
                fontsize=9, color="k", zorder=101,
                ha="center", va="center",
            )

        thetas = draw_samples(samples, weights, n_samples)
        ss = np.linspace(0, 30.0, 300)
        for i, theta in enumerate(thetas):
            gs = gamma(ss, theta)
            ax_main.plot(
                ss, gs, color=colors[2], alpha=0.15,  
                label="posterior sample" if i == 0 else None, lw=2
            )

        ax_main.legend()

        ax_main.set_xlim(0, 30.0)
        ax_main.set_ylim(0, 0.08)

        ###
        #  EIG
        ###
        ax_eig.set_ylabel("EIG")
        ax_eig.set_ylim(ymin=0.0)
        ax_eig.plot(
            eig_data["c [mol/m3]"], eig_data["EIG"],
            marker="s", color=colors[4], clip_on=False
        )
    
        c_next = eig_data.loc[eig_data["EIG"].argmax(), "c [mol/m3]"]
        ax_main.annotate(
            "", xy=(c_next, 0.0), xytext=(c_next, 0.01),
            arrowprops=dict(arrowstyle="->", color=colors[0], lw=2),
            ha="center", color=colors[0]
        )

        plt.tight_layout()
        plt.savefig(f"posterior_model_{n_data}.pdf", format="pdf")

        plt.show()

    return (viz_belief,)


if __name__ == "__main__":
    app.run()
