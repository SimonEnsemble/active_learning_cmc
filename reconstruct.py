import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pocomc as pc
    from scipy.stats import norm, uniform, gaussian_kde
    import pandas as pd
    import scipy.stats
    import marimo as mo
    import random
    import corner
    from scipy.special import logsumexp
    from aquarel import load_theme

    theme = load_theme("scientific")
    theme.set_font(size=15)
    theme.apply()
    return (
        corner,
        gaussian_kde,
        logsumexp,
        mo,
        norm,
        np,
        pc,
        pd,
        plt,
        scipy,
        uniform,
    )


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
            uniform(0.0, 30.0),                 # cmc [m3/mol]
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
    n_data = 3
    return (n_data,)


@app.cell
def _(n_data, pd):
    def get_data(n_data):
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
        return data

    data = get_data(n_data)
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
def _(log_like, pc, prior):
    def get_posterior(data, sigma):
        sampler = pc.Sampler(
            prior=prior,
            likelihood=log_like,
            likelihood_args=[data, sigma],
            precondition=True
        )

        # Run sampler
        sampler.run()

        samples, weights, logl, logp = sampler.posterior()
        return samples, weights, logl, logp

    return (get_posterior,)


@app.cell
def _(data, get_posterior, sigma):
    samples, weights, logl, logp = get_posterior(data, sigma)
    return samples, weights


@app.cell
def _(np):
    def draw_samples(samples, weights, n):
        """
        Draw n samples of from the posterior,
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
def _(gaussian_kde, np):
    def entropy_cmc(samples, weights):
        cmc_samples = samples[:, -1]
        kde = gaussian_kde(cmc_samples, weights=weights, bw_method=0.5)
        S = -np.sum(weights * np.log(kde(cmc_samples)))
        return S

    return (entropy_cmc,)


@app.cell
def _(entropy_cmc, samples, weights):
    entropy_cmc(samples, weights)
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
def _(data, eig_data, n_data, samples, viz_belief, weights):
    viz_belief(
        data, samples, weights, 
        eig_data=eig_data,
        n_samples=25,
        savename=f"posterior_model_{n_data}"
    )
    return


@app.cell
def _(colors, draw_samples, entropy_cmc, gamma, np, plt):
    def viz_belief(
        data, samples, weights, 
        eig_data=None,
        data_hallucinated=None,
        n_samples=50,
        savename=None
    ):
        if eig_data is not None:
            fig, (ax_hist, ax_main, ax_eig) = plt.subplots(
                3, 1, figsize=(6, 7),
                gridspec_kw={"height_ratios": [1, 3, 1]},
                sharex=True
            )
            ax_eig.set_xlabel("[surfactant] (mol/m$^3$)")
        else:
            fig, (ax_hist, ax_main) = plt.subplots(
                2, 1, figsize=(6, 7),
                gridspec_kw={"height_ratios": [1, 3]},
                sharex=True
            )
            ax_main.set_xlabel("[surfactant] (mol/m$^3$)")

        ###
        #   CMC hist
        ###
        thetas = draw_samples(samples, weights, len(weights))
        cmcs = [theta[-1] for theta in thetas]
        S = entropy_cmc(samples, weights)
        ax_hist.hist(
            cmcs,
            bins=20, color=colors[0],
            histtype="step", edgecolor=colors[0],
            lw=2
        )
        ax_hist.set_ylabel("# samples")
        ax_hist.set_xlabel("CMC (mol/m$^3$)")
        ax_hist.legend(title=f"entropy [nats]: {S:.1f}")

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
        if data_hallucinated is not None:
            c, gamma_obs = data_hallucinated
            ax_main.scatter(
                c, gamma_obs, 
                clip_on=False, color="white",
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
        if eig_data is not None:
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
        if savename is not None:
            plt.savefig(savename + ".pdf", format="pdf")

        plt.show()

    return (viz_belief,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ::streamline-emojis:crazy-face:: hallucinate data to illustrate IG
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    do_hallucination = mo.ui.checkbox(label="do hallucination?")
    do_hallucination
    return (do_hallucination,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    implement simulation-based oracle.
    """)
    return


@app.cell
def _(draw_samples, gamma, np, samples, sigma, weights):
    def orcale_surface_tension(c, samples=samples, weights=weights):
        theta = draw_samples(samples, weights, 1)[0]
        gamma_obs = gamma(c, theta) + np.random.randn() * sigma
        return gamma_obs

    return (orcale_surface_tension,)


@app.cell
def _(orcale_surface_tension):
    orcale_surface_tension(1.0)
    return


@app.cell
def _(data):
    data
    return


@app.cell
def _(data, get_posterior, orcale_surface_tension, sigma, viz_belief):
    def hallucinate_next_expt(c, gamma_obs=None, savename=None):
        # predict outcome of this experiment (stochastic)
        if gamma_obs is None:
            gamma_obs = orcale_surface_tension(c)
            print("gamma obs: ", gamma_obs)

        # augment data set
        data_new = data.copy()
        data_new.loc[len(data)] = [c, gamma_obs]

        # update the posterior
        samples_new, weights_new, _, _ = get_posterior(data_new, sigma)

        # viz updated belief
        viz_belief(
            data, samples_new, weights_new, data_hallucinated=[c, gamma_obs],
            savename=savename
        )

    return (hallucinate_next_expt,)


@app.cell
def _(do_hallucination, hallucinate_next_expt):
    if do_hallucination.value:
        hallucinate_next_expt(12.0, gamma_obs=0.03, savename="good_choice")
    return


@app.cell
def _(do_hallucination, hallucinate_next_expt):
    if do_hallucination.value:
        hallucinate_next_expt(0.75, gamma_obs=56/1000.0, savename="bad_choice")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## a wrong prior
    """)
    return


@app.cell
def _():
    true_cmc = 9.65 # mol/m3 mode
    return (true_cmc,)


@app.cell
def _(np, plt, scipy, true_cmc):
    cs = np.linspace(0.0, 100.0, 250)

    plt.figure(figsize=(5, 3))
    plt.plot(cs, scipy.stats.gamma.pdf(cs, a=10, scale=4), label=f"prior", lw=3)
    plt.title("a wrong prior")
    plt.xlabel("CMC [mol/m$^3$]")
    plt.ylabel("density")
    plt.yticks([])
    plt.xlim(0, 100)
    plt.axvline(true_cmc, color="k", label="true CMC", linestyle="--", lw=3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bad_prior.pdf", format="pdf")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
