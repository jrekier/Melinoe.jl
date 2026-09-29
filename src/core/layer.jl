# ── Layer definition ─────────────────────────────────────────────────────────

"""
    Layer(; domain, n, κ, μ, ρ₀, g₀, aw=true, dahlen=false, ν_SI=0.0)

One spherical shell of the body, on `domain = a..b`, discretised with `n` spectral
modes. The four profiles are callables of radius in normalised units:

| field | |
|---|---|
| `κ` | bulk modulus; `Inf` marks an incompressible layer |
| `μ` | shear modulus; `0` marks a fluid layer |
| `ρ₀` | background density |
| `g₀` | background gravity, `g₀(r) = dφ₀/dr` |

`g₀` comes from `self_gravity`, which integrates the density analytically.
Differentiating a Chebyshev-expanded `φ₀` instead introduces a 1/r² singularity.
`build_model` fills all four.

`aw` selects the fluid Poisson closure. `true` is Adams-Williamson, `ρ₀′ = −ρ₀²g/κ`,
an adiabatic shell. `false` keeps the actual `ρ₀′`, which a core reaching the centre
or a subadiabatic tabulated `κ` requires.

`dahlen` switches a fluid layer to the Dahlen (1974) reduced static closure; see
`dahlenize`. Static solves only, and ignored for solids. `ν_SI` is the kinematic
viscosity [m²/s], carried for the rotating solvers and unused by the
gravito-elastic ones.
"""
@kwdef struct Layer
    domain
    n   :: Int
    κ   :: Function
    μ   :: Function
    ρ₀  :: Function
    g₀  :: Function
    aw  :: Bool = true       # fluid Poisson closure: Adams-Williamson vs raw ρ₀′
    dahlen :: Bool = false   # Dahlen reduced static closure; static solves only
    ν_SI :: Float64 = 0.0    # kinematic viscosity [m²/s]; 0 ⇒ inviscid
end

"""
    is_fluid(l::Layer) -> Bool

`true` when the shear modulus vanishes at the layer's midpoint. A fluid layer
carries `(U, P, δφ)` where a solid carries `(U, V, δφ)`; see `gravitoelastic_blocks`.
"""
is_fluid(l::Layer) = l.μ((leftendpoint(l.domain) + rightendpoint(l.domain)) / 2) == 0.0

"""
    is_incompressible(l::Layer) -> Bool

`true` when the bulk modulus is `Inf` at the layer's midpoint, which drops every
1/κ term from the operators.
"""
is_incompressible(l::Layer) = isinf(l.κ((leftendpoint(l.domain) + rightendpoint(l.domain)) / 2))
