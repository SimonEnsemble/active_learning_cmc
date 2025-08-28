### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ cd47d8d0-5513-11f0-02cf-23409fc28fbf
begin
	import Pkg; Pkg.activate("cmc")
	using CairoMakie, DataFrames, Turing, MakieThemes, Colors, CSV, StatsBase, KernelDensity, Cubature, Test, PlutoUI, Logging, ProgressLogging, Printf, Random, Optim
end

# ╔═╡ 1e324846-70da-494c-bb88-8668a0f0e526
n_chains = Threads.nthreads() # using four threads

# ╔═╡ 0801bc21-de7c-4470-ae89-8725d90812e9
begin
	# modifying the plot scheme
	# see here for other themes
	#  https://makieorg.github.io/MakieThemes.jl/dev/themes/ggthemr/
	local my_theme = :pale
	
	set_theme!(ggthemr(my_theme))
	update_theme!(
		fontsize=20, linewidth=4, #backgroundcolor=:white,
		Axis=(; bottomspinevisible=false, leftspinevisible=false, 
			titlefont=:regular)
	)
	
	colors = parse.(Colorant, MakieThemes.GGThemr.ColorTheme[my_theme][:swatch])
end

# ╔═╡ 4cb87445-d372-4957-9cdb-4cd4bcc397de
TableOfContents()

# ╔═╡ cd6147e4-9785-4ee1-9454-2f4353dcca6c
function draw_axes!(ax)
	hlines!(ax, 0.0, color="black", linewidth=1)
	vlines!(ax, 0.0, color="black", linewidth=1)
end

# ╔═╡ fe1e0cc3-59ee-4887-8c90-af2d40b81892
#surfactant = "Triton-X-100"
surfactant = "OTG"

# ╔═╡ ef9d74b4-63e9-4337-bf6c-3147e816ebd3
md"figure saving convention"

# ╔═╡ 5a1768a0-865a-46ba-b70f-0194664d9d21
md"# 🏓 experimental data

manually added to array in order of collection.
"

# ╔═╡ 49de609d-4cc3-46d6-9141-5de0395088fb
begin
	if surfactant == "OTG"
		_data = DataFrame(
			"[S] (mol/m³)" => [
				0.0, 30.0, 3.0, 12.0, 7.5, 8.5, 0.75, 8.75, 13.25
			],
			"γ (N/m)" => [
				71.87, 29.54, 44.2325, 29.68, 32.42, 31.06, 55.2075, 29.89333, 29.567
			] / 1000.0
		)
	elseif surfactant == "Triton-X-100"
		_data = DataFrame(
			"[S] (mol/m³)" => [
				0.0, 10.0, 0.001, 0.06, 0.1, 0.215, 0.289, 0.8685, 0.3393
			],
			"γ (N/m)" => [
				71.73, 31.573, 71.3866, 46.3233, 42.81, 37.296, 35.01, 31.70667, 33.50667
			] / 1000.0
		)
	end
end

# ╔═╡ 44dd2629-bcbe-4b5b-a8e0-0f7c4add3cd2
md"# 🔘 select surfactant, iteration to view

iters to include: $(@bind iteration PlutoUI.Select(0:nrow(_data)-2, default=nrow(_data)-2))
"

# ╔═╡ 42c551c8-372e-430b-a756-10260d88936c
begin
	figdir = "figs"
	mkpath(figdir)

	fig_savetag = joinpath(figdir, "$(surfactant)_iter_$(iteration)_")
end

# ╔═╡ c451c216-4f29-4cf5-b367-fd486e634506
if iteration == 0
	data = _data[1:2, :]
else
	data = _data[1:2+iteration, :]
end

# ╔═╡ 874cc30e-0d7d-4a82-a523-c0caa9da4a59
md"# surfactant adsorption isotherm model

surface tension vs surfactant concentration

Szyszkowski eqn.

"

# ╔═╡ b686a78a-fba7-41f5-b30f-621e3416ae96
function γ_model(c, γ₀, a, K, c★)
	if c < c★
		return γ₀ - a * log(1 + K * c)
	else
		return γ₀ - a * log(1 + K * c★)
	end
end

# ╔═╡ ca288f74-bc34-457f-8caa-ab1627f5c46f
md"# Bayesian inference

## set up sampler
"

# ╔═╡ c40e781e-bd35-4541-9eb3-f943df41587d
md"🔍 search space"

# ╔═╡ 67d23697-2d05-46f2-80e4-75c85c369f80
if surfactant == "OTG"
	c_max = 30.0 # mol/m³
elseif surfactant == "Triton-X-100"
	c_max = 10.0 # mol/m³
end

# ╔═╡ 9b865570-b175-4fcb-a835-b8d6278c86ac
md"📏 measurement error"

# ╔═╡ 17ab88fc-8d65-42a8-94a7-3ac643638ef7
σ = 0.001 # (N/m) 

# ╔═╡ bcd013f5-3211-4ca4-ac1d-fae758199e75
@model function cmc_model(data::DataFrame)
	# surface tension of pure water
	@assert data[1, "[S] (mol/m³)"] == 0.0
	γ₀_obs = data[1, "γ (N/m)"]
		
	#=
	prior distributions
	=#
	γ₀ ~ Normal(γ₀_obs, σ)    # N/m
	a ~ Uniform(0.001, 0.1)   # N/m
	K ~ Uniform(0.0, 10000.0) # (mol/m³)⁻¹
	if surfactant == "OTG"
		c★ ~ Uniform(0.0, 30.0)  # mol / m³
	elseif surfactant == "Triton-X-100"
		c★ ~ LogUniform(0.001, 10.0)  # mol / m³
	end
	
	#=
	show data
	=#
	for i = 2:nrow(data)
		# surfactant concentration
		cᵢ = data[i, "[S] (mol/m³)"]
		
		# predicted surface tension
		γ_pred = γ_model(cᵢ, γ₀, a, K, c★)
		
		data[i, "γ (N/m)"] ~ Normal(γ_pred, σ)
	end
	
	return nothing
end

# ╔═╡ 9d2f66ee-03aa-42d9-ae9d-6ee14f1f1f63
model = cmc_model(data)

# ╔═╡ f0b122c9-4d43-405b-a28e-ead0c79772cb
md"## sample chain"

# ╔═╡ ea27c7f7-0073-4d0b-a171-7b404af1d0d6
n_MC_samples = 25000

# ╔═╡ 948a0fe4-e8ec-47e5-92a7-a66be020f0df
begin
	Random.seed!(45345635 + 1 + 1) # for reproducibility
	@time chain = sample(model, NUTS(), MCMCThreads(), n_MC_samples, n_chains)
	posterior_samples = DataFrame(chain)
end

# ╔═╡ a4f779ba-9410-4e67-840f-7114561f23b4
params = chain.name_map.parameters

# ╔═╡ 0556cc9b-a511-45aa-b7c9-9e86bd8a610d
param_to_unit = Dict(
	"γ₀" => "N/m",
	"a" => "N/m",
	"K" => "(mol/m³)⁻¹",
	"c★" => "mol/m³"
)

# ╔═╡ 37dc8c68-2270-4226-b209-f3fab65b3b13
md"converge diagnostics"

# ╔═╡ 52080b61-1e8d-4343-b79c-b3b39861e2c8
gelmandiag(chain)

# ╔═╡ fbe04777-fe1c-4f75-8059-80abd2da17da
md"for initial guesses for chain starts when computing info gain"

# ╔═╡ 52dc4eb7-702c-4c1f-967d-34c431b74436
function grab_posterior_sample(posterior_samples::DataFrame, params::Vector{Symbol})
	i = sample(1:nrow(posterior_samples))
	return Vector(posterior_samples[i, params])
end

# ╔═╡ 34b9ba4a-5a24-48c1-9cbe-5f4084b501ed
grab_posterior_sample(posterior_samples, params)

# ╔═╡ fdd7373d-47e7-4f17-869f-03b2145c1c02
md"## viz convergence diagnostics"

# ╔═╡ 14334653-2134-4782-a2d9-ef84837b2c45
function draw_convergence_diagnostics(
	posterior_samples::DataFrame, param::String; save_the_fig::Bool=false
)
	n_chains = length(unique(posterior_samples[:, "chain"]))

	println("mean: ", mean(posterior_samples[:, param]))
	println("std: ", std(posterior_samples[:, param]))
	
	fig = Figure()
	
	# axes
	ax = Axis(
		fig[1, 1],
		xlabel="iteration",
		ylabel=param * " [" * param_to_unit[param] * "]"
	)
	ax_d = Axis(fig[1, 2], xlabel="density", xticks=[0.0])

	# axes stuff
	linkyaxes!(ax, ax_d)
	colsize!(fig.layout, 2, Relative(0.2))
	hideydecorations!(ax_d, grid=false)

	# loop over chains
	for data in groupby(posterior_samples, :chain)
		c = data[1, :chain]
		
		# caterpillar
		lines!(
			ax, data[:, param], linewidth=1, label="chain $c", 
			color=colors[c]
		)

		# histogram
		density!(
			ax_d, data[:, param], color=colors[c], direction=:y, alpha=0.5,
			strokecolor=colors[c], strokewidth=1
		)
	end
	axislegend(ax, orientation=:horizontal, labelsize=14)
	if save_the_fig 
		save(fig_savetag * "$(param)_convergence.pdf", fig)
	end
	fig
end

# ╔═╡ 0ef63054-a677-4078-8795-0c1d7df85b80
if :σ in names(posterior_samples)
	draw_convergence_diagnostics(posterior_samples, "σ")
end

# ╔═╡ 4b298c39-1506-4f07-bf61-21bb32b8d31f
draw_convergence_diagnostics(posterior_samples, "γ₀")

# ╔═╡ 6c255255-f3b4-4112-b06b-7583781eb69e
draw_convergence_diagnostics(posterior_samples, "c★")

# ╔═╡ 9a3c24dc-3e90-4008-824e-5719bd74c1c5
draw_convergence_diagnostics(posterior_samples, "K")

# ╔═╡ 9ad673c6-68ae-4b2e-bbc3-74d42bd44fd6
draw_convergence_diagnostics(posterior_samples, "a")

# ╔═╡ c9f08ffb-b44f-4be3-881d-096020f17493
md"## viz posterior distn"

# ╔═╡ a738488f-b26d-4f9c-b6a6-f120becd28cf
function posterior_c★_mode(
	posterior_samples::DataFrame
)
	the_kde = kde(posterior_samples[:, "c★"])
	xs = range(0.0, c_max, length=200)
	ρs = [pdf(the_kde, xi) for xi in xs]
	return xs[argmax(ρs)]
end

# ╔═╡ 5521e61b-7e34-4f72-882d-c7697463bef1
posterior_c★_mode(posterior_samples)

# ╔═╡ bb6f671c-b6d3-4abb-a71f-fcfc3d2a3cf5
surfactant

# ╔═╡ d39866ea-1b9c-4723-afe2-401872285f9e
thing_to_color = Dict(
	"data" => colors[6],
	"model" => colors[5],
	"distn" => colors[5],
	"text" => colors[1],
	"info gain" => colors[4]
)

# ╔═╡ 06cf608e-782e-4c67-acb2-3aead3642704
function viz(
	data::DataFrame, posterior_samples::DataFrame;
	acq_scores::Union{DataFrame, Nothing}=nothing, n_samples_plot::Int=50,
	x_pseudo_logscale::Bool=false
)
	cs = range(1e-6, c_max + 1.0, length=1000)

	if surfactant == "OTG"
		xticks = range(0.0, 30.0, length=11)
	else
		xticks = range(0.0, 10.0, length=11)
	end
	
	fig = Figure(size=(600, 500))
	ax = Axis(
		fig[1, 1], 
		xlabel="[surfactant] (mol/m³)", 
		ylabel="surface tension (N/m)",
		xticks=xticks
	)
	ax_t = Axis(
		fig[0, 1], 
		ylabel=rich("posterior\ndensity\nof c", superscript("★")), 
		title=surfactant, 
		xticks=xticks,
		yticks=[0.0]
	)

	draw_axes!(ax)
	draw_axes!(ax_t)
	
	linkxaxes!(ax, ax_t)
	rowsize!(fig.layout, 1, Relative(isnothing(acq_scores) ? 0.8 : 0.7))
	
	# posterior over c★
	density!(
		ax_t, posterior_samples[:, "c★"], 
		color=(thing_to_color["distn"], 0.1), strokewidth=3, 
		strokecolor=thing_to_color["distn"], boundary=(0.0, c_max + 0.5)
	)

	# posterior surface tension vs. surfactant conc. samples
	for s = 1:n_samples_plot
		i = sample(1:nrow(posterior_samples))
		γ₀, a, K, c★ = posterior_samples[i, ["γ₀", "a", "K", "c★"]]
				
		lines!(
			ax, cs, γ_model.(cs, γ₀, a, K, c★), 
			color=(thing_to_color["model"], 0.1), label="posterior sample"
		)
	end
	
	# data
	scatter!(
		ax, data[:, "[S] (mol/m³)"], data[:, "γ (N/m)"], label="data",
		color=thing_to_color["data"], markersize=16,
		strokewidth=2, strokecolor="black"
	)
	# errorbars!(
	# 	ax, data[:, "[S] (mol/m³)"], data[:, "γ (N/m)"], σ * ones(nrow(data)),
	# 	color=thing_to_color["data"]
	# )
	annotation!(
		ax, 
		[(row["[S] (mol/m³)"], row["γ (N/m)"]) for row in eachrow(data)], 
		text=vcat([" 0", " 0 "], [" $i" for i = 1:(nrow(data)-2)]),
		color=thing_to_color["text"],
		fontsize=14,
	)

	# credible interval and mode
	lo, hi = quantile(posterior_samples[:, "c★"], [0.05, 0.95])
	lines!(ax_t, [lo, hi], [0, 0], color="gray")
	c★_mode = posterior_c★_mode(posterior_samples)
	
	ci_string = rich(
		rich("posterior of c", font=:bold), 
		superscript("★"), 
		":\n",
		"\t 90% " * @sprintf("CI: [%.2f, %.2f] mol/m³", lo, hi) * "\n" * @sprintf("\t mode: %.2f mol/m³", c★_mode)
	)
	
	println("\tposterior mode: ", c★_mode)
	println("\tCI width / posterior mode: ", (hi - lo) / c★_mode)
	
	hidexdecorations!(ax_t, grid=false)
	axislegend(
		ax, unique=true, titlefont=:regular, position=(0.9, 0.1), 
		framevisible=true, bgcolor="white"
	)
	# Label(
	# 	fig[1, 1], ci_string, tellwidth=false, tellheight=false,
	# 	halign=0.9, valign=0.9, justification=:left,
	# 	framevisible=true, bgcolor="white"
	# )
	textlabel!(
	    ax, [surfactant == "OTG" ? 12 : 4], [0.06], text=ci_string,
	    text_align = (:left, :top)
	)

	if ! isnothing(acq_scores)
		ax_b = Axis(
			fig[2, 1], ylabel="information\ngain", xlabel="[surfactant] (mol/m³)"
		)
		hidexdecorations!(ax, grid=false)
		linkxaxes!(ax_b, ax_t, ax)
		scatterlines!(
			acq_scores[:, "c [mol/m³]"], acq_scores[:, "info gain"], color=colors[4]
		)
	end
	
	if x_pseudo_logscale
		xlims!(ax, 0.001, 10.0)
		ax.xscale = Makie.pseudolog10
		ax.xticks = [0, 0.001, 0.01, 0.1, 1, 10]
		ax_t.xscale = Makie.pseudolog10
		if ! isnothing(acq_scores)
			ax_b.xscale = Makie.pseudolog10
			ax_b.xticks = [0, 0.001, 0.01, 0.1, 1, 10]
		end
	end
		
	xlims!(-0.5, c_max + 0.5)

	savename = surfactant
	if ! isnothing(acq_scores)
		save(fig_savetag * "fit.pdf", fig)
	else
		save(fig_savetag * "fit_w_info_gain.pdf", fig)
	end
	fig
end

# ╔═╡ 947e44ff-e2e0-495a-a7a6-7632d18733fb
colors

# ╔═╡ e6ea645f-282c-4598-8755-be568d7b3d2e
viz(data, posterior_samples, n_samples_plot=25, x_pseudo_logscale=false)

# ╔═╡ 49199459-f93c-4a23-8bed-1ea6b2fa2c94
md"# entropy calculations

computing the entropy of a probability distribution from samples.

💡 integrate a kernel density estimate of the pdf.
"

# ╔═╡ f571f7f7-928a-4908-9a18-9cf90b3466d6
if surfactant == "Triton-X-100"
	entropy_of_log10 = true
else
	entropy_of_log10 = false
end

# ╔═╡ 192b5353-c0d5-457a-bf59-579709d8f2ec
function entropy(_xs::Vector{Float64}, log_transform_first::Bool=entropy_of_log10)
	if log_transform_first
		xs = log10.(_xs)
	else
		xs = _xs
	end
	# integration bounds
	xmin = minimum(xs) - std(xs)
	xmax = maximum(xs) + std(xs)

	# kernel density estimation
	the_kde = kde(xs)
	ρ = x -> pdf(the_kde, x)

	# integrate density to get entropy
	function S_integrand(x)
		the_ρ = ρ(x)
		if the_ρ > 0.0
			return - the_ρ * log(the_ρ)
		else
			return 0.0
		end
	end
	
	S = hquadrature(S_integrand, xmin, xmax, reltol=1e-4, maxevals=250)[1]

	return S
end

# ╔═╡ 085d09d1-375f-4d97-92c1-73161383c0cf
begin
	# test with entropy of a Gaussian
	local σ = 2.0
	local H̃ = entropy(σ * randn(100000), false)
	local H = 1/2 * (1 + log(2 * π * σ ^ 2))
	@test isapprox(H, H̃, atol=0.01)
end

# ╔═╡ aeaac1d5-d5f4-4993-ae95-e8b9a5c82e77
md"entropy of c★ over the multiple chains"

# ╔═╡ 64b3b08d-733d-4cbb-b488-7a54778a4980
hist(
	posterior_samples[:, "c★"],
	axis=(; 
		  title="posterior c★", 
		  xlabel="c★ [mol/m³]",   
		  ylabel="density",
		  xscale=entropy_of_log10 ? log10 : identity
	)
)

# ╔═╡ fa9012a4-24f4-4358-92b3-74cb37270d31
[entropy(Vector(chain[:c★][:, c])) for c = 1:n_chains]

# ╔═╡ 64ebafed-7692-4fa1-bbed-fc2cde90af6b
md"# acquisition: information gain

calculate information gain about the CMC
"

# ╔═╡ 97e4a572-0bfe-4b0c-b3a6-36201ae36701
params

# ╔═╡ f1ec7091-d47e-475d-885a-fcc96ceab663
function α_ig(
	c, data::DataFrame, posterior_samples::DataFrame; 
	n_samples::Int=100, n_MC_samples::Int=100
)
	Random.seed!(45345635)
	Logging.disable_logging(Logging.Info)  # Disables info-level messages
	S_news = zeros(n_samples)
	for s = 1:n_samples
		#=
		sample from posterior
		=#
		i = sample(1:nrow(posterior_samples))
		γ₀, a, K, c★ = posterior_samples[i, ["γ₀", "a", "K", "c★"]]
	
		#=
		fantasize a measurement at this c
		=#
		γ_obs = γ_model(c, γ₀, a, K, c★) + randn() * σ
	
		# augment data with fantasized data
		new_data = deepcopy(data)
		push!(new_data, Dict("[S] (mol/m³)" => c, "γ (N/m)" => γ_obs))
	
		#=
		update posterior with fantasized data point
		=#
		new_model = cmc_model(new_data)
		initial_params = [
			grab_posterior_sample(posterior_samples, params) for c = 1:n_chains
		]
		new_chain = DataFrame(
			sample(
				new_model, NUTS(), MCMCThreads(), 
				round(Int, n_MC_samples / n_chains), n_chains, 
				progress=false,
				initial_params=initial_params
			)
		)

		#=
		compute entropy of c★ in new posterior
		=#
		S_news[s] = entropy(new_chain[:, "c★"])
	end
	
	#=
	compute current and average of new entropies of c★
	=#
	S_now = entropy(posterior_samples[:, "c★"])
	𝔼_S_next = mean(S_news)
	
	Logging.disable_logging(Logging.BelowMinLevel) # don't wanna disable logging
	
	return S_now - 𝔼_S_next
end

# ╔═╡ e759e6f4-3366-4d94-93fc-1f6f5cb59e2b
md"time a single run"

# ╔═╡ fc333a63-86f1-43d6-9f7e-1f43bd926caf
# @time α_ig(1.0, data, posterior_samples, n_samples=100, n_MC_samples=100)

# ╔═╡ 1b92732c-e918-41d1-b422-822794f850e5
md"
🧪 check if the estimate of the information gradient via sampling is consisistent over multiple runs, so we can assess if we have a sufficient number of samples? check here. 👇

$(@bind check_sampling CheckBox(default=false))"

# ╔═╡ 48e51f57-3d7e-4096-b5c2-67a2244ba2e9
if check_sampling
	[α_ig(1.0, data, posterior_samples, n_samples=200, n_MC_samples=50) for i = 1:4]
end

# ╔═╡ 3dd13aca-090d-4ba4-8086-85c56f7d0065
md"
## calculate info gain
🔨 actually compute the information gradient acquisition function at each next surface concentration? check here. 👇

$(@bind compute_α CheckBox(default=false))"

# ╔═╡ ed12167e-0ee3-472c-93d5-3424453019c4
begin
	#=
	candidate experiments
	i.e. surfactant concentrations [mol/m³]
	=#
	if surfactant == "OTG"
		if iteration in [0, 1]
			cs = 0:1.0:c_max
		elseif iteration == [2, 3]
			cs = 0:0.5:c_max
		else
			cs = 0:0.25:c_max
		end
	elseif surfactant == "Triton-X-100"
		if iteration in [0, 1]
			cs = 10.0 .^ range(-3.0, 1.0, length=10)
		elseif iteration in [2]
			cs = 10.0 .^ range(-3.0, 1.0, length=15)
		elseif iteration in [3]
			cs = 10.0 .^ range(-3.0, 1.0, length=25)
		elseif iteration in [4]
			cs = 10.0 .^ range(-3.0, 1.0, length=40)
		elseif iteration in [5, 6]
			cs = 10.0 .^ range(-3.0, 1.0, length=50)
		end
	end
	# cs = collect(range(0.0, c_max, length=10)) # toy
	
	#=
	info gains from each candidate expt
	=#
	αs = zeros(length(cs))
	if compute_α
		@progress for i = 1:length(cs)
			αs[i] = α_ig(
				cs[i], data, posterior_samples, 
				n_samples=250, n_MC_samples=100
				# n_samples=300, n_MC_samples=150,
				# n_samples=300, n_MC_samples=200,
				#n_samples=300, n_MC_samples=200
			)
		end
	end
end

# ╔═╡ a17064d4-38ce-49b6-a34a-1f1de50f63b6
begin
	acq_scores = DataFrame("c [mol/m³]" => cs, "info gain" => αs) 
			
	sort(acq_scores, "info gain")
end

# ╔═╡ e42e86a9-8b9a-432a-8c5a-f463d97ce1f2
if compute_α
	viz(data, posterior_samples, acq_scores=acq_scores, x_pseudo_logscale=true)
end

# ╔═╡ 78f08666-d2a3-4bd0-9c92-ecb383eebb07
md"## pick experiment"

# ╔═╡ 5da23800-29e3-4323-905c-cb3a31a03e7f
if compute_α
	# pick c with largest info gain about CMC, that hasn't been done.
	picked_c = 0.0
	for i in sortperm(αs, rev=true)
		c = cs[i]
		if ! (c in data[:, "[S] (mol/m³)"])
			picked_c = c
			break
		end
	end
	
	println(
		"design: choose [S] (mol/m³) = ",
		picked_c
	)
end

# ╔═╡ 630c2ab7-fa68-4a74-8c31-48b68b70b37b
md"
## stock solution calculation

$m V = m_s V_s$"

# ╔═╡ c24c3e26-f940-4a8c-a88d-26a619415427
c_stock = 10.0 # mol/L

# ╔═╡ a6d7623e-350d-4e36-88db-89adf99043a9
V_sample = 25 # mL

# ╔═╡ e62a4099-9f49-4636-a828-76918a437170
if compute_α
	println("stock solution needed: ", V_sample * picked_c / c_stock, " mL")
end

# ╔═╡ faef0439-9571-463a-adfa-714b6294d6c4
md"# post-AL analysis: info dynamics

"

# ╔═╡ dd46e6a0-4cf3-4b19-9137-df7c9e86fc14
md"$(@bind run_info_dynamics PlutoUI.CheckBox(default=false))"

# ╔═╡ e0d9f20d-7d0a-48c2-b10c-f0c251280a66
function entropy_dynamics(data::DataFrame)
	nb_iters = nrow(data) - 2

	S = zeros(nb_iters+1)
	lo = zeros(nb_iters+1)
	hi = zeros(nb_iters+1)
	
	c★_posterior_samples = DataFrame(
		"iteration" => Int[],
		"c★" => Float64[]
	)
	
	for i = 0:nb_iters
		#=
		Bayesian inference with only this data
		=#
		model = cmc_model(data[1:(2+i), :])

		chain = sample(model, NUTS(), MCMCThreads(), n_MC_samples, n_chains)
		
		if ! all(gelmandiag(chain)[:, :psrfci] .< 1.1)
			println("chain not converged.")
		end
		
		posterior_samples = DataFrame(chain)

		# store
		c★_posterior_samples = vcat(
			c★_posterior_samples,
			DataFrame(
				"iteration" => [i for j = 1:nrow(posterior_samples)],
				"c★" => posterior_samples[:, "c★"]
			)
		)

		# compute entry and quantile of posterior of CMC
		S[i+1] = entropy(posterior_samples[:, "c★"])

		lo[i+1], hi[i+1] = quantile(posterior_samples[:, "c★"], [0.05, 0.95])
	end
	
	info_dynamics = DataFrame(
		"iteration" => [i for i = 0:nb_iters],
		"entropy c★" => S,
		"CI lo" => lo,
		"CI hi" => hi
	)
	return info_dynamics, c★_posterior_samples
end

# ╔═╡ 6d2ff265-8014-462e-982a-19bc1c19cef2
if run_info_dynamics
	info_dynamics, c★_posterior_samples = entropy_dynamics(data)
end

# ╔═╡ 348a7004-6a30-4586-9f0d-6fa25827102f
function viz_entropy_over_iters(info_dynamics::DataFrame)
	fig = Figure()
	ax = Axis(
		fig[1, 1], xlabel="iteration", ylabel="entropy, S(C★) [nat]", xticks=0:nrow(data),
		title="entropy of posterior of CMC"
	)
	scatterlines!(
		info_dynamics[:, "iteration"], info_dynamics[:, "entropy c★"],
		markersize=20
	)
	ylims!(0, nothing)
	save(joinpath(figdir, surfactant * "_entropy_over_iters.pdf"), fig)
	fig
end

# ╔═╡ 5e9b76da-4f51-420c-badf-8b29c33e5a58
if run_info_dynamics
	viz_entropy_over_iters(info_dynamics)
end

# ╔═╡ 8fe4882b-0ffe-4b12-aee3-1e1d02dfd368
function viz_posterior_cmc_over_iters(c★_posterior_samples)
	fig = Figure()
	ax = Axis(
		fig[1, 1], xlabel="iteration", ylabel="CMC, c★ [mol/m³]", xticks=0:nrow(data),
		title="posterior and credible interval for CMC"
	)
	hlines!(
		[9.0], label="literature-reported CMC", color=colors[3], linewidth=1
	)
	violin!(
		c★_posterior_samples[:, "iteration"], c★_posterior_samples[:, "c★"],
		side=:right, label="posterior density"
	)
	for (i, row) in enumerate(eachrow(info_dynamics))
		lines!(
			[row["iteration"], row["iteration"]], 
			[row["CI lo"], row["CI hi"]], 
			color="black", linewidth=2, label="90% credible interval"
		)
	end
	axislegend(unique=true)
	ylims!(0, nothing)
	save(joinpath(figdir, surfactant * "posterior_over_iters.pdf"), fig)
	fig
end

# ╔═╡ cf20b7c9-85bc-4f57-a74e-edcf77e3033d
if run_info_dynamics
	viz_posterior_cmc_over_iters(c★_posterior_samples)
end

# ╔═╡ f2f70823-5990-43a2-a31e-60de32cee6d3
md"# into figure (traditional fitting routine with tons of data)"

# ╔═╡ 6639dcc9-8e98-4746-b4be-93f1f4704859
trad_surfactant = "OTG"

# ╔═╡ 12e7cf6b-3685-4bb2-814c-ace95fcb5142
trad_data = CSV.read("data/$(trad_surfactant)_trad.csv", DataFrame)

# ╔═╡ e117d5b7-331c-4c36-8a3c-eb37f9dfc799
posterior_samples[:, "a"]

# ╔═╡ 1b8c75e8-d814-4674-a957-6507ededeea2
function ls_fit(
	data::DataFrame, 
	θ₀::Vector{Float64}=[
		mean(posterior_samples[:, "γ₀"]),
		mean(posterior_samples[:, "a"]),
		mean(posterior_samples[:, "K"]),
		mean(posterior_samples[:, "c★"]),
	]
)
	function loss(θ)
		γ₀, a, K, c★ = θ

		ℓ = 0.0
		for i = 1:nrow(data)
			cᵢ = data[i, "[S] (mol/m³)"]
			γᵢ = data[i, "γ (N/m)"]

			γ̂ᵢ = γ_model(cᵢ, γ₀, a, K, c★)

			ℓ += (γᵢ - γ̂ᵢ) ^ 2
		end
		return ℓ
	end

	θ = res = optimize(loss, θ₀).minimizer
	return θ
end

# ╔═╡ 1953f157-ae09-47a7-854c-2352f8b5f131
function viz_ls_fit(data::DataFrame)
	# fit model to data
	γ₀, a, K, c★ = ls_fit(data)

	fig = Figure(size=(450, 450))
	ax = Axis(
		fig[1, 1], 
		xlabel="[surfactant] (mol/m³)", 
		ylabel="surface tension (N/m)",
		title="surfactant: $trad_surfactant"
	)

	# model
	cs = range(0.0, 35.0, length=150)
	lines!(
		ax, cs, γ_model.(cs, γ₀, a, K, c★), 
		color=thing_to_color["model"], label="fitted model"
	)

	# data
	scatter!(
		ax, data[:, "[S] (mol/m³)"], data[:, "γ (N/m)"], label="data",
		color=thing_to_color["data"], markersize=16,
		strokewidth=2, strokecolor="black"
	)

	# CMC
	lines!(
		[c★, c★], [0.0, γ_model.(30.0, γ₀, a, K, c★)],
		color="gray", linewidth=1
	)

	annotation!(
		17.0, 0.0125, c★, 0.0,
	    text = "critical\nmicelle\nconcentration",
	    path = Ann.Paths.Arc(0.2),
	    style = Ann.Styles.LineArrow(),
		labelspace=:data,
		fontsize=16
	)

	xlims!(-1, 31)
	axislegend()
	draw_axes!(ax)
	save("trad_approach_$(trad_surfactant).pdf", fig)
	fig
	# return γ_model.(cs, γ₀, a, K, c★)
end

# ╔═╡ a77f2620-c2fe-4d5d-a42c-6154e92195ea
viz_ls_fit(trad_data)

# ╔═╡ Cell order:
# ╠═cd47d8d0-5513-11f0-02cf-23409fc28fbf
# ╠═1e324846-70da-494c-bb88-8668a0f0e526
# ╠═0801bc21-de7c-4470-ae89-8725d90812e9
# ╠═4cb87445-d372-4957-9cdb-4cd4bcc397de
# ╠═cd6147e4-9785-4ee1-9454-2f4353dcca6c
# ╟─44dd2629-bcbe-4b5b-a8e0-0f7c4add3cd2
# ╠═fe1e0cc3-59ee-4887-8c90-af2d40b81892
# ╟─ef9d74b4-63e9-4337-bf6c-3147e816ebd3
# ╠═42c551c8-372e-430b-a756-10260d88936c
# ╟─5a1768a0-865a-46ba-b70f-0194664d9d21
# ╠═49de609d-4cc3-46d6-9141-5de0395088fb
# ╠═c451c216-4f29-4cf5-b367-fd486e634506
# ╟─874cc30e-0d7d-4a82-a523-c0caa9da4a59
# ╠═b686a78a-fba7-41f5-b30f-621e3416ae96
# ╟─ca288f74-bc34-457f-8caa-ab1627f5c46f
# ╟─c40e781e-bd35-4541-9eb3-f943df41587d
# ╠═67d23697-2d05-46f2-80e4-75c85c369f80
# ╟─9b865570-b175-4fcb-a835-b8d6278c86ac
# ╠═17ab88fc-8d65-42a8-94a7-3ac643638ef7
# ╠═bcd013f5-3211-4ca4-ac1d-fae758199e75
# ╠═9d2f66ee-03aa-42d9-ae9d-6ee14f1f1f63
# ╟─f0b122c9-4d43-405b-a28e-ead0c79772cb
# ╠═ea27c7f7-0073-4d0b-a171-7b404af1d0d6
# ╠═948a0fe4-e8ec-47e5-92a7-a66be020f0df
# ╠═a4f779ba-9410-4e67-840f-7114561f23b4
# ╠═0556cc9b-a511-45aa-b7c9-9e86bd8a610d
# ╟─37dc8c68-2270-4226-b209-f3fab65b3b13
# ╠═52080b61-1e8d-4343-b79c-b3b39861e2c8
# ╟─fbe04777-fe1c-4f75-8059-80abd2da17da
# ╠═52dc4eb7-702c-4c1f-967d-34c431b74436
# ╠═34b9ba4a-5a24-48c1-9cbe-5f4084b501ed
# ╟─fdd7373d-47e7-4f17-869f-03b2145c1c02
# ╠═14334653-2134-4782-a2d9-ef84837b2c45
# ╠═0ef63054-a677-4078-8795-0c1d7df85b80
# ╠═4b298c39-1506-4f07-bf61-21bb32b8d31f
# ╠═6c255255-f3b4-4112-b06b-7583781eb69e
# ╠═9a3c24dc-3e90-4008-824e-5719bd74c1c5
# ╠═9ad673c6-68ae-4b2e-bbc3-74d42bd44fd6
# ╟─c9f08ffb-b44f-4be3-881d-096020f17493
# ╠═a738488f-b26d-4f9c-b6a6-f120becd28cf
# ╠═5521e61b-7e34-4f72-882d-c7697463bef1
# ╠═bb6f671c-b6d3-4abb-a71f-fcfc3d2a3cf5
# ╠═06cf608e-782e-4c67-acb2-3aead3642704
# ╠═d39866ea-1b9c-4723-afe2-401872285f9e
# ╠═947e44ff-e2e0-495a-a7a6-7632d18733fb
# ╠═e6ea645f-282c-4598-8755-be568d7b3d2e
# ╟─49199459-f93c-4a23-8bed-1ea6b2fa2c94
# ╠═f571f7f7-928a-4908-9a18-9cf90b3466d6
# ╠═192b5353-c0d5-457a-bf59-579709d8f2ec
# ╠═085d09d1-375f-4d97-92c1-73161383c0cf
# ╟─aeaac1d5-d5f4-4993-ae95-e8b9a5c82e77
# ╠═64b3b08d-733d-4cbb-b488-7a54778a4980
# ╠═fa9012a4-24f4-4358-92b3-74cb37270d31
# ╟─64ebafed-7692-4fa1-bbed-fc2cde90af6b
# ╠═97e4a572-0bfe-4b0c-b3a6-36201ae36701
# ╠═f1ec7091-d47e-475d-885a-fcc96ceab663
# ╟─e759e6f4-3366-4d94-93fc-1f6f5cb59e2b
# ╠═fc333a63-86f1-43d6-9f7e-1f43bd926caf
# ╟─1b92732c-e918-41d1-b422-822794f850e5
# ╠═48e51f57-3d7e-4096-b5c2-67a2244ba2e9
# ╟─3dd13aca-090d-4ba4-8086-85c56f7d0065
# ╠═ed12167e-0ee3-472c-93d5-3424453019c4
# ╠═a17064d4-38ce-49b6-a34a-1f1de50f63b6
# ╠═e42e86a9-8b9a-432a-8c5a-f463d97ce1f2
# ╟─78f08666-d2a3-4bd0-9c92-ecb383eebb07
# ╠═5da23800-29e3-4323-905c-cb3a31a03e7f
# ╟─630c2ab7-fa68-4a74-8c31-48b68b70b37b
# ╠═c24c3e26-f940-4a8c-a88d-26a619415427
# ╠═a6d7623e-350d-4e36-88db-89adf99043a9
# ╠═e62a4099-9f49-4636-a828-76918a437170
# ╟─faef0439-9571-463a-adfa-714b6294d6c4
# ╟─dd46e6a0-4cf3-4b19-9137-df7c9e86fc14
# ╠═e0d9f20d-7d0a-48c2-b10c-f0c251280a66
# ╠═6d2ff265-8014-462e-982a-19bc1c19cef2
# ╠═348a7004-6a30-4586-9f0d-6fa25827102f
# ╠═5e9b76da-4f51-420c-badf-8b29c33e5a58
# ╠═8fe4882b-0ffe-4b12-aee3-1e1d02dfd368
# ╠═cf20b7c9-85bc-4f57-a74e-edcf77e3033d
# ╟─f2f70823-5990-43a2-a31e-60de32cee6d3
# ╠═6639dcc9-8e98-4746-b4be-93f1f4704859
# ╠═12e7cf6b-3685-4bb2-814c-ace95fcb5142
# ╠═e117d5b7-331c-4c36-8a3c-eb37f9dfc799
# ╠═1b8c75e8-d814-4674-a957-6507ededeea2
# ╠═1953f157-ae09-47a7-854c-2352f8b5f131
# ╠═a77f2620-c2fe-4d5d-a42c-6154e92195ea
