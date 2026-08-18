# Melinoë (Melinoe.jl)

Internal geodynamics of planetary bodies, on an
[ApproxFun](https://github.com/JuliaApproximation/ApproxFun.jl) spectral
discretisation. Build one `PlanetModel`, then ask it different questions — all
from the same assembled elastic–gravitational operator.

| | | validated against |
|---|---|---|
| tidal Love numbers | `love_numbers` | Kelvin analytic; PREM published values |
| bulk properties | — | every model matches observed mass and moment of inertia |
| free spheroidal modes | `free_modes` | MINEOS on PREM (ℓ=2, n=0–7, < 2%) |
| pressure Love numbers | `pressure_love` | Dumberry & Bloxham (2004) PREM, ℓ=2–8 |
| Sasao compliances | `compliances` | PREM references; Le Maistre Mars S8 |
| forcing → open fields | `forced`, `radial`/`tangential`/`potential`/`moment` | the above, field by field |

## Install

```julia
using Pkg; Pkg.add(url = "https://github.com/jrekier/Melinoe.jl")
```

## Use

```julia
using Melinoe

m  = Melinoe.Models.PREM()            # or .Mars(), .Ganymede(), .Kelvin()
ln = love_numbers(m)                  # (; h, l, k) at ℓ=2
fm = free_modes(m)                    # spheroidal eigenmodes
c  = compliances(m; core_layer = 2)   # Sasao κ, ξ, γ, β + Betti check

d  = forced(m, Tide())                # assemble → force → solve, keeping the fields
h  = -radial(d, 1.0)
```

Custom models: `build_model` from layer polynomials, or `load_planet_csv` from a
tabulated model —

```julia
m = load_planet_csv("examples/PREM_1s.csv")   # real PREM, as published
```

which reads columns by name (`radius` or `depth`; `density`; then either `K`/`mu`
or velocities `vp`/`vs`, or the anisotropic `vpv`/`vph`/`vsv`/`vsh` with `eta`),
honours `[unit="km"]`-style annotations, ignores what it doesn't know (`Q-mu`, …),
takes repeated radii as layer boundaries, and carries incompressible layers
(`κ = Inf`) through in both directions. For static solves with stratified
fluid layers, `dahlenize(m)` switches them to the Dahlen (1974) reduced
formulation.

Two gotchas: `l` is `NaN` for a static solve on a fluid-surfaced body
(`Kelvin()`, `PREM(ocean=true)`), where it is undetermined; and the moment reader
is `moment`, not `inertia`, which `LinearAlgebra` claims from Julia 1.12.

The [`examples/`](examples/) notebooks are the guided tour and the theory
reference — they derive the Love numbers, the static-fluid (Longman–Dahlen)
gauge, the Sasao–Okubo–Saito compliances and the free modes as they go.

## Tests

```julia
using Pkg; Pkg.test()
```

## References

Dahlen (1974); Sasao–Okubo–Saito; Mathews–Herring–Buffett (2002);
Dumberry & Bloxham (2004), *GJI* **159**(2), 417–434; Dziewonski & Anderson (1981).

MIT licensed — see [LICENSE](LICENSE).
