### A Pluto.jl notebook ###
# v0.20.11

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
	using CairoMakie, DataFrames, Turing, MakieThemes, Colors, CSV, StatsBase, KernelDensity, Cubature, Test, PlutoUI, Logging, ProgressLogging, Printf
end

# ╔═╡ 1e324846-70da-494c-bb88-8668a0f0e526
n_chains = Threads.nthreads() # using four threads

# ╔═╡ 0801bc21-de7c-4470-ae89-8725d90812e9
begin
	# modifying the plot scheme
	# see here for other themes
	#  https://makieorg.github.io/MakieThemes.jl/dev/themes/ggthemr/
	local my_theme = :flat
	
	set_theme!(ggthemr(my_theme))
	update_theme!(
		fontsize=20, linewidth=4, 
		Axis=(bottomspinevisible=false, leftspinevisible=false, titlefont=:regular)
	)
	
	colors = parse.(Colorant, MakieThemes.GGThemr.ColorTheme[my_theme][:swatch])
end

# ╔═╡ 4cb87445-d372-4957-9cdb-4cd4bcc397de
TableOfContents()

# ╔═╡ 5a1768a0-865a-46ba-b70f-0194664d9d21
md"# read data"

# ╔═╡ d506f9e7-0d4d-40eb-a938-e1465c836222
datafiles = [
	"H2O-C10E8", "H2O-C14E6", "H2O-CTAB-bulk",	"H2O-SDS-bulk",
	"H2O-C12E5", "H2O-C16E8", "H2O-OTG", "H2O-Tween20"
]

# ╔═╡ 5da8817c-3460-4c04-b408-aff71e4576d4
md"
🔨 subsample the data? check here. 👇

$(@bind subsample_the_data CheckBox(default=false))"

# ╔═╡ 17d43d3c-ad92-4447-adc1-dbd23001c45e
md"❓ wut experiment?

$(@bind i_expt Select(1:length(datafiles), default=3))"

# ╔═╡ 8511592a-9059-43a7-b14c-941b66641cb0
expt = datafiles[i_expt]

# ╔═╡ 49de609d-4cc3-46d6-9141-5de0395088fb
begin
	function read_data(i::Int)
		data = CSV.read("data/" * datafiles[i] * ".csv", DataFrame)
		
		rename!(data, "concentration_mol/m^3" => "[S] (mol/m³)")
		rename!(data, "surften_N/m" => "γ (N/m)")
		select!(data, ["[S] (mol/m³)", "γ (N/m)"])

		@warn "assuming solvent is pure water."
		γ₀ = 72.8 / 1000.0 # N/m
		pushfirst!(data, Dict("[S] (mol/m³)" => 0.0, "γ (N/m)" => γ₀))
		
		return data
	end

	data = read_data(i_expt)
	if subsample_the_data
		data = data[[1, 5, 18, end], :] # always include 1
	end
end

# ╔═╡ 842d16b1-26e3-4cd2-81ac-83ed6cd3b6b3
md"surface tension of pure solvent (water)."

# ╔═╡ 533a51d0-3b42-42e0-b8db-1ab7c672f3df
γ₀ = data[1, "γ (N/m)"]

# ╔═╡ 874cc30e-0d7d-4a82-a523-c0caa9da4a59
md"# surface tension vs surfactant concentration model"

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
c_max = maximum(read_data(i_expt)[:, "[S] (mol/m³)"]) * 1.1

# ╔═╡ 9b865570-b175-4fcb-a835-b8d6278c86ac
md"📏 measurement error"

# ╔═╡ 17ab88fc-8d65-42a8-94a7-3ac643638ef7
σ = 0.001 # (N/m) 

# ╔═╡ bcd013f5-3211-4ca4-ac1d-fae758199e75
@model function cmc_model(data::DataFrame)
	# begin with pure solvent so γ₀ doesn't need inferred
	@assert data[1, "[S] (mol/m³)"] == 0.0
	γ₀ = data[1, "γ (N/m)"]
		
	#=
	prior distributions
	=#
	# σ ~ Uniform(0.0, 0.01)
	a ~ Uniform(0.0, 0.25)
	K ~ Uniform(0.0, 20.0)
	c★ ~ Uniform(0.0, c_max)

	#=
	show data
	=#
	for i = 1:nrow(data)
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
n_MC_samples = 2500

# ╔═╡ 948a0fe4-e8ec-47e5-92a7-a66be020f0df
@time chain = sample(model, NUTS(), MCMCThreads(), n_MC_samples, n_chains)

# ╔═╡ a4f779ba-9410-4e67-840f-7114561f23b4
params = chain.name_map.parameters

# ╔═╡ 37dc8c68-2270-4226-b209-f3fab65b3b13
md"converge diagnostics"

# ╔═╡ 52080b61-1e8d-4343-b79c-b3b39861e2c8
gelmandiag(chain)

# ╔═╡ bfa76e5b-e2c3-449b-8de0-cfe5df15330d
posterior_samples = DataFrame(chain)

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
md"## viz convergence"

# ╔═╡ 14334653-2134-4782-a2d9-ef84837b2c45
function draw_convergence_diagnostics(posterior_samples::DataFrame, param::String)
	n_chains = length(unique(posterior_samples[:, "chain"]))
	
	fig = Figure()
	
	# axes
	ax = Axis(fig[1, 1], xlabel="iteration", ylabel=param)
	ax_d = Axis(fig[1, 2], xlabel="density")

	# axes stuff
	linkyaxes!(ax, ax_d)
	colsize!(fig.layout, 2, Relative(0.2))
	hideydecorations!(ax_d, grid=false)

	# loop over chains
	for data in groupby(posterior_samples, :chain)
		c = data[1, :chain]
		
		# caterpillar
		lines!(ax, data[:, param], linewidth=1, label="chain $c", color=colors[c])

		# histogram
		density!(
			ax_d, data[:, param], color=colors[c], direction=:y, alpha=0.5,
			strokecolor=colors[c], strokewidth=1
		)
	end
	axislegend(ax)
	
	fig
end

# ╔═╡ 6c255255-f3b4-4112-b06b-7583781eb69e
draw_convergence_diagnostics(posterior_samples, "c★")

# ╔═╡ c9f08ffb-b44f-4be3-881d-096020f17493
md"## viz posterior distn"

# ╔═╡ 06cf608e-782e-4c67-acb2-3aead3642704
function viz(
	data::DataFrame, posterior_samples::DataFrame;
	αs::Union{Vector{Float64}, Nothing}=nothing
)
	cs = range(0.0, c_max, length=100)
	
	fig = Figure(size=(500, isnothing(αs) ? 500 : 700))
	ax = Axis(
		fig[1, 1], xlabel="[surfactant] (mol/m³)", ylabel="surface tension (N/m)"
	)
	ax_t = Axis(
		fig[0, 1], ylabel="posterior\ndensity\nof c★", title=expt
	)
	
	linkxaxes!(ax, ax_t)
	rowsize!(fig.layout, 1, Relative(isnothing(αs) ? 0.8 : 0.7))
	
	# posterior over c★
	density!(ax_t, posterior_samples[:, "c★"], color=colors[3])

	# posterior surface tension vs. surfactant conc. samples
	for s = 1:25
		i = sample(1:nrow(posterior_samples))
		a, K, c★ = posterior_samples[i, ["a", "K", "c★"]]
				
		lines!(
			ax, cs, γ_model.(cs, γ₀, a, K, c★), 
			color=(colors[2], 0.1), label="posterior sample")
	end
	
	# data
	scatter!(
		ax, data[:, "[S] (mol/m³)"], data[:, "γ (N/m)"], label="data",
		color=colors[1]
	)

	# credible interval
	lo, hi = quantile(posterior_samples[:, "c★"], [0.1, 0.9])
	ci_string = "80%" * @sprintf(" CI for c★:\n[%.2f, %.2f] mol/m³", lo, hi)
	
	hidexdecorations!(ax_t, grid=false)
	axislegend(ax, ci_string, unique=true, titlefont=:regular)

	if ! isnothing(αs)
		ax_b = Axis(
			fig[2, 1], ylabel="information\ngain", xlabel="[surfactant] (mmol/m³)"
		)
		hidexdecorations!(ax, grid=false)
		linkxaxes!(ax_b, ax_t, ax)
		scatterlines!(range(0.0, c_max, length=length(αs)), αs, color=colors[4])
	end
	xlims!(0, c_max)
	if ! subsample_the_data
		save(expt * "_fit.pdf", fig)
	else
		save(expt * "_w_info_gain2.pdf", fig)
	end
	fig
end

# ╔═╡ e6ea645f-282c-4598-8755-be568d7b3d2e
viz(data, posterior_samples)

# ╔═╡ 49199459-f93c-4a23-8bed-1ea6b2fa2c94
md"# entropy

computing the entropy of a distribution from samples.

💡 integrate a kernel density estimate of the pdf.
"

# ╔═╡ 192b5353-c0d5-457a-bf59-579709d8f2ec
function entropy(xs::Vector{Float64})
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
	local H̃ = entropy(σ * randn(100000))
	local H = 1/2 * (1 + log(2 * π * σ ^ 2))
	@test isapprox(H, H̃, atol=0.01)
end

# ╔═╡ aeaac1d5-d5f4-4993-ae95-e8b9a5c82e77
md"entropy of c★ over the multiple chains"

# ╔═╡ fa9012a4-24f4-4358-92b3-74cb37270d31
[entropy(Vector(chain[:c★][:, c])) for c = 1:n_chains]

# ╔═╡ 64ebafed-7692-4fa1-bbed-fc2cde90af6b
md"# acquisition"

# ╔═╡ f1ec7091-d47e-475d-885a-fcc96ceab663
function α_ig(
	c, data::DataFrame, posterior_samples::DataFrame; 
	n_samples::Int=100, n_MC_samples::Int=100
)
	Logging.disable_logging(Logging.Info)  # Disables info-level messages
	S_news = zeros(n_samples)
	for s = 1:n_samples
		#=
		sample from posterior
		=#
		i = sample(1:nrow(posterior_samples))
		a, K, c★ = posterior_samples[i, ["a", "K", "c★"]]
	
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
@time α_ig(1.0, data, posterior_samples, n_samples=200, n_MC_samples=50)

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
🔨 actually compute the information gradient acquisition function at each next surface concentration? check here. 👇

$(@bind compute_α CheckBox(default=false))"

# ╔═╡ ed12167e-0ee3-472c-93d5-3424453019c4
begin
	cs = collect(range(0.0, c_max, length=50))
	αs = zeros(length(cs))
	if compute_α
		@progress for i = 1:length(cs)
			αs[i] = α_ig(
				cs[i], data, posterior_samples, 
				n_samples=250, n_MC_samples=50
			)
		end
	end
end

# ╔═╡ e42e86a9-8b9a-432a-8c5a-f463d97ce1f2
if compute_α
	viz(data, posterior_samples, αs=αs)
end

# ╔═╡ Cell order:
# ╠═cd47d8d0-5513-11f0-02cf-23409fc28fbf
# ╠═1e324846-70da-494c-bb88-8668a0f0e526
# ╠═0801bc21-de7c-4470-ae89-8725d90812e9
# ╠═4cb87445-d372-4957-9cdb-4cd4bcc397de
# ╟─5a1768a0-865a-46ba-b70f-0194664d9d21
# ╠═d506f9e7-0d4d-40eb-a938-e1465c836222
# ╟─5da8817c-3460-4c04-b408-aff71e4576d4
# ╟─17d43d3c-ad92-4447-adc1-dbd23001c45e
# ╠═8511592a-9059-43a7-b14c-941b66641cb0
# ╠═49de609d-4cc3-46d6-9141-5de0395088fb
# ╟─842d16b1-26e3-4cd2-81ac-83ed6cd3b6b3
# ╠═533a51d0-3b42-42e0-b8db-1ab7c672f3df
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
# ╟─37dc8c68-2270-4226-b209-f3fab65b3b13
# ╠═52080b61-1e8d-4343-b79c-b3b39861e2c8
# ╠═bfa76e5b-e2c3-449b-8de0-cfe5df15330d
# ╟─fbe04777-fe1c-4f75-8059-80abd2da17da
# ╠═52dc4eb7-702c-4c1f-967d-34c431b74436
# ╠═34b9ba4a-5a24-48c1-9cbe-5f4084b501ed
# ╟─fdd7373d-47e7-4f17-869f-03b2145c1c02
# ╠═14334653-2134-4782-a2d9-ef84837b2c45
# ╠═6c255255-f3b4-4112-b06b-7583781eb69e
# ╟─c9f08ffb-b44f-4be3-881d-096020f17493
# ╠═06cf608e-782e-4c67-acb2-3aead3642704
# ╠═e6ea645f-282c-4598-8755-be568d7b3d2e
# ╟─49199459-f93c-4a23-8bed-1ea6b2fa2c94
# ╠═192b5353-c0d5-457a-bf59-579709d8f2ec
# ╠═085d09d1-375f-4d97-92c1-73161383c0cf
# ╟─aeaac1d5-d5f4-4993-ae95-e8b9a5c82e77
# ╠═fa9012a4-24f4-4358-92b3-74cb37270d31
# ╟─64ebafed-7692-4fa1-bbed-fc2cde90af6b
# ╠═f1ec7091-d47e-475d-885a-fcc96ceab663
# ╟─e759e6f4-3366-4d94-93fc-1f6f5cb59e2b
# ╠═fc333a63-86f1-43d6-9f7e-1f43bd926caf
# ╟─1b92732c-e918-41d1-b422-822794f850e5
# ╠═48e51f57-3d7e-4096-b5c2-67a2244ba2e9
# ╟─3dd13aca-090d-4ba4-8086-85c56f7d0065
# ╠═ed12167e-0ee3-472c-93d5-3424453019c4
# ╠═e42e86a9-8b9a-432a-8c5a-f463d97ce1f2
