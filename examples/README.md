# Melinoe examples

A guided tour of the package, built as a series. Each notebook uses the *same*
assembled operator (`assemble_planet`).

**Start here:** [`0_quickstart.ipynb`](0_quickstart.ipynb) — build a model, read Love numbers, free modes and compliances, and look at the assembled matrices. 

First run, from the repository root (needs Julia ≥ 1.11 and a Python with matplotlib,
for `PyPlot`):

```julia
using Pkg; Pkg.activate("examples"); Pkg.instantiate()
```

On Julia 1.10 the `[sources]` entry is ignored, so add the checkout by hand instead:
`Pkg.develop(path = ".")` before `Pkg.instantiate()`.

## Tutorials (read in order)

| | notebook | what it covers |
|---|---|---|
| 1 | [`1_love_numbers_tutorial.ipynb`](1_love_numbers_tutorial.ipynb) | A single self-gravitating sphere → the tidal Love numbers `h, k, l`, step by step. |
| 2 | [`2_dahlen_fluid_tutorial.ipynb`](2_dahlen_fluid_tutorial.ipynb) | A **static fluid** layer and the Longman gauge problem → the `dahlenize` closure that makes a fluid core well-posed. |
| 3 | [`3_three_layer_forcing_tutorial.ipynb`](3_three_layer_forcing_tutorial.ipynb) | A three-layer Earth: `potential_forcing`, layer masks, and the `3×3` **compliance matrix** (κ, ξ, γ, …). |
| 4 | [`4_prem_tutorial.ipynb`](4_prem_tutorial.ipynb) | The real **PREM**: Love numbers and Sasao/Mathews compliances vs reference values. |
| 5 | [`5_free_modes_tutorial.ipynb`](5_free_modes_tutorial.ipynb) | Free oscillations: the same matrices as `A x = ω² B x` → the spheroidal normal modes vs MINEOS. |

**Companion:** [`betti_reciprocity.ipynb`](betti_reciprocity.ipynb) — the field-by-field proof
that the compliance matrix is reciprocal, `Âᵢ Sᵢⱼ = Âⱼ Sⱼᵢ` (referenced from notebooks 3 and 5).

## Reading and writing tabulated models

[`PREM_1s.csv`](PREM_1s.csv) is real PREM as distributed — radius *and* depth in km,
density in g/cm³, transversely isotropic `Vpv/Vph/Vsv/Vsh` with `eta`, `Q-mu`/`Q-kappa`
columns, rows running surface inward, and no `layer` column (regions are delimited by
**repeated radii**):

```
radius[unit="km"],depth[unit="km"],density[unit="g/cm^3"],Vpv[unit="km/s"],…,Q-mu,Q-kappa
6371.0,0.0,1.02000,1.45000,1.45000,0.00000,0.00000,1.00000,0,57823
…
3480.0,2891.0,5.56646,13.71662,13.71662,7.26465,7.26465,1.00000,312,57823   ← CMB, and
3480.0,2891.0,9.90344,8.06479,8.06479,0.00000,0.00000,1.00000,0,57823      ← again
```

`load_planet_csv` matches columns by name, honours the `[unit="…"]` annotations,
ignores what it doesn't recognise, and splits layers at the repeated radii:

```julia
julia> m = load_planet_csv("PREM_1s.csv")
loaded: 13 layers, R = 6371.0 km, ρ̄ = 5514.2 kg/m³

julia> love_numbers(m).k
0.3639
```

`write_planet_csv` emits either of the two layouts it can read, and both round-trip:

```julia
write_planet_csv(Melinoe.Models.Mars(), "mars.csv")                     # :moduli
write_planet_csv(Melinoe.Models.Mars(), "mars.csv"; format = :velocity)
```

```
# :moduli — explicit layer column, SI, moduli as stored
# Melinoe planet model  name=Mars  Omega_SI=7.088e-05
layer,radius[unit="m"],density[unit="kg/m^3"],K[unit="Pa"],mu[unit="Pa"]
layer1,0.0,6582.201563324427,2.1199076079174863e11,0.0

# :velocity — the published layout: no layer column, km / g·cm⁻³ / km·s⁻¹, surface inward
# Melinoe planet model  name=Mars  Omega_SI=7.088e-05
radius[unit="km"],depth[unit="km"],density[unit="g/cm^3"],vp[unit="km/s"],vs[unit="km/s"]
3389.500000,0.000000,2.90000000,6.30635512,3.71390676
```

Recognised column names: `radius`|`depth`, `density`|`rho`, then either `K`|`kappa`|`bulk`
and `mu`|`shear`, or `vp`/`vs`, or `vpv`/`vph`/`vsv`/`vsh` with optional `eta`. A `layer`
column is used if present. An incompressible layer is written as a literal `Inf` and read back as one.
