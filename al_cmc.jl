### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ cd47d8d0-5513-11f0-02cf-23409fc28fbf
begin
	import Pkg; Pkg.activate("cmc")
	using CairoMakie, DataFrames, Turing, MakieThemes, Colors, CSV
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
	
	data = read_data(2)
end

# ╔═╡ b686a78a-fba7-41f5-b30f-621e3416ae96
function γ_model(c, γ₀, a, K, c★)
	if c < c★
		return γ₀ - a * log(1 + K * c)
	else
		return γ₀ - a * log(1 + K * c★)
	end
end

# ╔═╡ bcd013f5-3211-4ca4-ac1d-fae758199e75
@model function cmc_model(
	data::DataFrame
)
	#=
	prior distributions
	=#
	σ ~ Uniform(0.0, 0.01)
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
chain = DataFrame(sample(model, NUTS(), MCMCThreads(), n_MC_samples, 3))

# ╔═╡ 3f4237e8-9c04-4ed1-8d45-aa5e9953a15b
K̄ = mean(chain[:, "K"])

# ╔═╡ 6f2b7b80-db66-43f2-85a6-d8935a3206fc
ā = mean(chain[:, "a"])

# ╔═╡ 3d27c024-dd14-4b15-b558-874457240ef6
γ̄₀ = mean(chain[:, "γ₀"])

# ╔═╡ 6d3d04a4-09fe-474b-af5e-714eaf27e968
c̄★ = mean(chain[:, "c★"])

# ╔═╡ 48c3a89f-8ce7-4dbe-8f8c-913c187b7409
σ̄ = mean(chain[:, "σ"])

# ╔═╡ 06cf608e-782e-4c67-acb2-3aead3642704
begin
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
	density!(ax_t, chain[:, "c★"], color=colors[3])
	for s = 1:15
		i = sample(1:nrow(chain))
		
		a = chain[i, "a"]
		K = chain[i, "K"]
		γ₀ = chain[i, "γ₀"]
		c★ = chain[i, "c★"]
		
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

# ╔═╡ f1ec7091-d47e-475d-885a-fcc96ceab663
function α_ig(
	c, data::DataFrame, chain::DataFrame; 
	n_samples::Int=10, n_MC_samples::Int=1000
)
	cmcs_new = zeros(n_samples)
	cmcs_updated = zeros(n_samples)
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
	
		#= 
		augment data with fantasized data
		=#
		new_data = deepcopy(data)
		push!(new_data, Dict("[S] (mol/m³)" => c, "γ (N/m)" => γ_obs))
	
		#=
		update posterior
		=#
		new_model = cmc_model(new_data)
		new_chain = DataFrame(
			sample(new_model, NUTS(), MCMCThreads(), n_MC_samples, 3)
		)
	end
	return new_data
end

# ╔═╡ 7ad08f7c-ce02-4685-9591-27ba3a42ae52
data

# ╔═╡ cbb41a00-7c57-4f7c-b385-3cda0657fb4f
α_ig(1.0, data, chain)

# ╔═╡ Cell order:
# ╠═cd47d8d0-5513-11f0-02cf-23409fc28fbf
# ╠═0801bc21-de7c-4470-ae89-8725d90812e9
# ╠═d506f9e7-0d4d-40eb-a938-e1465c836222
# ╠═49de609d-4cc3-46d6-9141-5de0395088fb
# ╠═b686a78a-fba7-41f5-b30f-621e3416ae96
# ╠═bcd013f5-3211-4ca4-ac1d-fae758199e75
# ╠═9d2f66ee-03aa-42d9-ae9d-6ee14f1f1f63
# ╠═ea27c7f7-0073-4d0b-a171-7b404af1d0d6
# ╠═948a0fe4-e8ec-47e5-92a7-a66be020f0df
# ╠═3f4237e8-9c04-4ed1-8d45-aa5e9953a15b
# ╠═6f2b7b80-db66-43f2-85a6-d8935a3206fc
# ╠═3d27c024-dd14-4b15-b558-874457240ef6
# ╠═6d3d04a4-09fe-474b-af5e-714eaf27e968
# ╠═48c3a89f-8ce7-4dbe-8f8c-913c187b7409
# ╠═06cf608e-782e-4c67-acb2-3aead3642704
# ╠═f1ec7091-d47e-475d-885a-fcc96ceab663
# ╠═7ad08f7c-ce02-4685-9591-27ba3a42ae52
# ╠═cbb41a00-7c57-4f7c-b385-3cda0657fb4f
