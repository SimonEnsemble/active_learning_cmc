### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ ad99ef74-0903-11f1-9eca-2b8ebc136af4
begin
	import Pkg; Pkg.activate("cmc")
	using CairoMakie, DataFrames, Turing,  Random, Turing, AdvancedVI
	using Turing: Variational
end

# ╔═╡ 1c70366d-cb19-45a1-8235-b2d343c09c0f
data = DataFrame(
	"[S] (mol/m³)" => [
		0.0, 30.0, 3.0, 12.0, 7.5, 8.5, 0.75, 8.75, 13.25
	],
	"γ (N/m)" => [
		71.87, 29.54, 44.2325, 29.68, 32.42, 31.06, 55.2075, 29.89333, 29.567
	] / 1000.0
)

# ╔═╡ 1321b6e9-7978-44b4-860c-256385843fbd
function γ_model(c, γ₀, a, K, c★)
	if c < c★
		return γ₀ - a * log(1 + K * c)
	else
		return γ₀ - a * log(1 + K * c★)
	end
end

# ╔═╡ 54f024f8-5a41-4812-8fb0-11c7a57e1344
σ = 0.001 # (N/m) 

# ╔═╡ bc5cc4a5-954b-4bb9-94a1-66e13fce6991
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
	c★ ~ Uniform(0.0, 30.0)  # mol / m³
	
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

# ╔═╡ 743e2d18-89b9-4137-b4e2-c57a81b67911
model = cmc_model(data)

# ╔═╡ c59eba08-6a61-4a30-a6f2-570341c1fd41
params = begin
	ch = sample(model, Prior(), 1)
	ch.name_map.parameters
end 

# ╔═╡ b18e2156-1396-47f9-86c7-b8abb9f93f5c
q_init = q_fullrank_gaussian(
	model#, location=[0.07, 0.01, 100.0, 15.0]#, scale=10.0*collect(I(4))
)

# ╔═╡ 4362bfc9-afdb-43b1-a3e1-87517f2ab7ff
q_last, _, _ = vi(model, q_init, 10000)

# ╔═╡ 77f1567b-c386-484c-816a-ede36c0e1412
q_last

# ╔═╡ Cell order:
# ╠═ad99ef74-0903-11f1-9eca-2b8ebc136af4
# ╠═1c70366d-cb19-45a1-8235-b2d343c09c0f
# ╠═1321b6e9-7978-44b4-860c-256385843fbd
# ╠═54f024f8-5a41-4812-8fb0-11c7a57e1344
# ╠═bc5cc4a5-954b-4bb9-94a1-66e13fce6991
# ╠═743e2d18-89b9-4137-b4e2-c57a81b67911
# ╠═c59eba08-6a61-4a30-a6f2-570341c1fd41
# ╠═b18e2156-1396-47f9-86c7-b8abb9f93f5c
# ╠═4362bfc9-afdb-43b1-a3e1-87517f2ab7ff
# ╠═77f1567b-c386-484c-816a-ede36c0e1412
