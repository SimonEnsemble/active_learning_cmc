### A Pluto.jl notebook ###
# v0.20.11

using Markdown
using InteractiveUtils

# ╔═╡ cd47d8d0-5513-11f0-02cf-23409fc28fbf
begin
	import Pkg; Pkg.activate("cmc")
	using CairoMakie, DataFrames, Turing, MakieThemes, Colors, CSV, StatsBase, KernelDensity, Cubature, Test, PlutoUI, Logging
end

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

# ╔═╡ 49de609d-4cc3-46d6-9141-5de0395088fb
begin
	function read_data(i::Int)
		data = CSV.read("data/" * datafiles[3] * ".csv", DataFrame)
		
		rename!(data, "concentration_mol/m^3" => "[S] (mol/m³)")
		rename!(data, "surften_N/m" => "γ (N/m)")
		select!(data, ["[S] (mol/m³)", "γ (N/m)"])
		return data
	end
	
	data = read_data(2)[[2, 12, 18], :]
end

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
md"# Bayesian inference"

# ╔═╡ bcd013f5-3211-4ca4-ac1d-fae758199e75
@model function cmc_model(
	data::DataFrame
)
	#=
	prior distributions
	=#
	σ ~ Uniform(0.0, 0.001)
	γ₀ ~ Uniform(0.0, 0.1)
	a ~ Uniform(0.0, 1.0)
	K ~ Uniform(0.0, 10.0)
	c★ ~ Uniform(0.0, 5.0)

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

# ╔═╡ ea27c7f7-0073-4d0b-a171-7b404af1d0d6
n_MC_samples = 1000

# ╔═╡ 948a0fe4-e8ec-47e5-92a7-a66be020f0df
chain = sample(model, NUTS(), MCMCThreads(), n_MC_samples, 3)

# ╔═╡ a4f779ba-9410-4e67-840f-7114561f23b4
params = chain.name_map.parameters

# ╔═╡ bfa76e5b-e2c3-449b-8de0-cfe5df15330d
posterior_samples = DataFrame(chain)

# ╔═╡ fa842b5f-084d-401e-8f27-8831aae06b18
θ₀ = [mean(posterior_samples[:, θ]) for θ in params]

# ╔═╡ 70171568-f20e-46d5-b2fa-69f9354a6db2
chain2 = sample(model, NUTS(), MCMCThreads(), 10, 1, initial_params=[θ₀])

# ╔═╡ 3f4237e8-9c04-4ed1-8d45-aa5e9953a15b
K̄ = mean(posterior_samples[:, "K"])

# ╔═╡ 6f2b7b80-db66-43f2-85a6-d8935a3206fc
ā = mean(posterior_samples[:, "a"])

# ╔═╡ 3d27c024-dd14-4b15-b558-874457240ef6
γ̄₀ = mean(posterior_samples[:, "γ₀"])

# ╔═╡ 6d3d04a4-09fe-474b-af5e-714eaf27e968
c̄★ = mean(posterior_samples[:, "c★"])

# ╔═╡ 48c3a89f-8ce7-4dbe-8f8c-913c187b7409
σ̄ = mean(posterior_samples[:, "σ"])

# ╔═╡ c9f08ffb-b44f-4be3-881d-096020f17493
md"## viz"

# ╔═╡ 06cf608e-782e-4c67-acb2-3aead3642704
function viz(data::DataFrame, posterior_samples::DataFrame)
	cs = range(0.0, maximum(data[:, "[S] (mol/m³)"]), length=100)
	
	fig = Figure()
	ax = Axis(
		fig[1, 1], xlabel="[surfactant] (mmol/m³)", ylabel="surface tension (N/m)"
	)
	ax_t = Axis(
		fig[0, 1], ylabel="density"
	)
	linkxaxes!(ax, ax_t)
	rowsize!(fig.layout, 1, Relative(0.8))
	density!(ax_t, posterior_samples[:, "c★"], color=colors[3])
	for s = 1:15
		i = sample(1:nrow(posterior_samples))
		
		a = posterior_samples[i, "a"]
		K = posterior_samples[i, "K"]
		γ₀ = posterior_samples[i, "γ₀"]
		c★ = posterior_samples[i, "c★"]
		
		lines!(
			ax, cs, γ_model.(cs, γ₀, a, K, c★), 
			color=(colors[2], 0.2), label="posterior sample")
	end
	scatter!(
		ax, data[:, "[S] (mol/m³)"], data[:, "γ (N/m)"], label="data",
		color=colors[1]
	)
	axislegend(ax, unique=true)
	fig
end

# ╔═╡ e6ea645f-282c-4598-8755-be568d7b3d2e
viz(data, posterior_samples)

# ╔═╡ 64ebafed-7692-4fa1-bbed-fc2cde90af6b
md"# acquisition"

# ╔═╡ 7ad08f7c-ce02-4685-9591-27ba3a42ae52
data

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
	
	S = hquadrature(S_integrand, xmin, xmax)[1]

	return S
end

# ╔═╡ f1ec7091-d47e-475d-885a-fcc96ceab663
function α_ig(
	c, data::DataFrame, chain::DataFrame; 
	n_samples::Int=100, n_MC_samples::Int=100
)
	Logging.disable_logging(Logging.Info)  # Disables info-level messages
	cmcs_new = zeros(n_samples)

	initial_state = Dict(θ => mean(chain[:, θ]) for θ in params)
	for s = 1:n_samples
		#=
		sample from posterior
		=#
		i = sample(1:nrow(chain))
	
		a = chain[i, "a"]
		K = chain[i, "K"]
		γ₀ = chain[i, "γ₀"]
		c★ = chain[i, "c★"]
		σ = chain[i, "σ"]
	
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
		new_chain = DataFrame(
			sample(
				new_model, NUTS(), MCMCThreads(), 
				n_MC_samples, 1, progress=false,
				initial_params=[θ₀]
			)
		)

		#=
		sample c★ from this posterior
		=#
		cmcs_new[s] = new_chain[end, "c★"]
	end
	S_now = entropy(chain[:, "c★"])
	S_next = entropy(cmcs_new)
	Logging.disable_logging(Logging.BelowMinLevel)
	return S_now - S_next
end

# ╔═╡ 085d09d1-375f-4d97-92c1-73161383c0cf
begin
	# test with entropy of a Gaussian
	local σ = 2.0
	local H̃ = entropy(σ * randn(100000))
	local H = 1/2 * (1 + log(2 * π * σ ^ 2))
	@test isapprox(H, H̃, atol=0.01)
end

# ╔═╡ 740885bd-c7f7-4598-bf7a-06f640ad12da
cs = collect(range(0.0, 2.0, length=12))

# ╔═╡ 48e51f57-3d7e-4096-b5c2-67a2244ba2e9
[α_ig(1.0, data, posterior_samples, n_samples=1000, n_MC_samples=1000) for i = 1:3]

# ╔═╡ ed12167e-0ee3-472c-93d5-3424453019c4
# αs = [α_ig(cᵢ, data, chain) for cᵢ in cs]

# ╔═╡ e42e86a9-8b9a-432a-8c5a-f463d97ce1f2
lines(cs, αs)

# ╔═╡ Cell order:
# ╠═cd47d8d0-5513-11f0-02cf-23409fc28fbf
# ╠═0801bc21-de7c-4470-ae89-8725d90812e9
# ╠═4cb87445-d372-4957-9cdb-4cd4bcc397de
# ╟─5a1768a0-865a-46ba-b70f-0194664d9d21
# ╠═d506f9e7-0d4d-40eb-a938-e1465c836222
# ╠═49de609d-4cc3-46d6-9141-5de0395088fb
# ╟─874cc30e-0d7d-4a82-a523-c0caa9da4a59
# ╠═b686a78a-fba7-41f5-b30f-621e3416ae96
# ╟─ca288f74-bc34-457f-8caa-ab1627f5c46f
# ╠═bcd013f5-3211-4ca4-ac1d-fae758199e75
# ╠═9d2f66ee-03aa-42d9-ae9d-6ee14f1f1f63
# ╠═ea27c7f7-0073-4d0b-a171-7b404af1d0d6
# ╠═948a0fe4-e8ec-47e5-92a7-a66be020f0df
# ╠═a4f779ba-9410-4e67-840f-7114561f23b4
# ╠═fa842b5f-084d-401e-8f27-8831aae06b18
# ╠═70171568-f20e-46d5-b2fa-69f9354a6db2
# ╠═bfa76e5b-e2c3-449b-8de0-cfe5df15330d
# ╠═3f4237e8-9c04-4ed1-8d45-aa5e9953a15b
# ╠═6f2b7b80-db66-43f2-85a6-d8935a3206fc
# ╠═3d27c024-dd14-4b15-b558-874457240ef6
# ╠═6d3d04a4-09fe-474b-af5e-714eaf27e968
# ╠═48c3a89f-8ce7-4dbe-8f8c-913c187b7409
# ╟─c9f08ffb-b44f-4be3-881d-096020f17493
# ╠═06cf608e-782e-4c67-acb2-3aead3642704
# ╠═e6ea645f-282c-4598-8755-be568d7b3d2e
# ╟─64ebafed-7692-4fa1-bbed-fc2cde90af6b
# ╠═f1ec7091-d47e-475d-885a-fcc96ceab663
# ╠═7ad08f7c-ce02-4685-9591-27ba3a42ae52
# ╠═192b5353-c0d5-457a-bf59-579709d8f2ec
# ╠═085d09d1-375f-4d97-92c1-73161383c0cf
# ╠═740885bd-c7f7-4598-bf7a-06f640ad12da
# ╠═48e51f57-3d7e-4096-b5c2-67a2244ba2e9
# ╠═ed12167e-0ee3-472c-93d5-3424453019c4
# ╠═e42e86a9-8b9a-432a-8c5a-f463d97ce1f2
