# ── PlanetModel: model construction + non-dimensional metadata ────────────────
#
# `build_model` handles the non-dimensionalisation (modulus unit ρ̄gR =
# (4πG/3)ρ̄²R²), the self-consistent gravity, and the Layer boilerplate. `g₀`
# always comes from `self_gravity`, never hand-written per model.

const G_SI = 6.674e-11
const Ω_EARTH = 7.2921e-5   # rad/s

"""
    PlanetModel

Holds the `Vector{Layer}` plus the non-dimensional metadata needed to convert
back to SI: `R_SI` (surface radius, m), `ρ̄_SI` (mean density, kg/m³),
`ω_unit = √(4πGρ̄/3)` (rad/s per dimensionless ω), `p_unit = (4πG/3)ρ̄²R²`
(Pa per dimensionless modulus), and the body rotation rate `Ω_SI` (rad/s;
0 ⇒ non-rotating). Rotation is a body property, viscosity a per-layer
material property (`Layer.ν_SI`).
"""
struct PlanetModel
    name   :: String
    layers :: Vector{Layer}
    R_SI   :: Float64
    ρ̄_SI   :: Float64
    ω_unit :: Float64
    p_unit :: Float64
    Ω_SI   :: Float64
end

"""
    rotation_rate(m, override=nothing) -> Float64

Rotation rate for a solve: `override` wins, else the model's `Ω_SI`, else Earth's.
"""
rotation_rate(m::PlanetModel, override = nothing) =
    override !== nothing ? Float64(override) :
    m.Ω_SI > 0            ? m.Ω_SI            : Ω_EARTH

layers_of(m::PlanetModel) = m.layers

"""
    radial_profile(m::PlanetModel, field::Symbol) -> Function

One callable `f(r)` for a per-layer `field` (`:ρ₀`, `:g₀`, `:κ`, `:μ`) across the
whole model, e.g. `plot(radial_profile(m, :ρ₀), 0, 1)`.
"""
radial_profile(m::PlanetModel, field::Symbol) =
    r -> getfield(m.layers[layer_at(m, r)], field)(r)

"""
    layer_at(m::PlanetModel, r) -> Int

Index of the layer containing radius `r`; interfaces resolve to the layer below.
`DomainError` outside the body, since the layer profiles extrapolate silently.
"""
function layer_at(m::PlanetModel, r)
    r0 = leftendpoint(m.layers[1].domain)
    rN = rightendpoint(m.layers[end].domain)
    (isfinite(r) && r0 - 1e-9 ≤ r ≤ rN + 1e-9) ||
        throw(DomainError(r, "radius outside the model [$r0, $rN]"))
    for (i, l) in enumerate(m.layers)
        r ≤ rightendpoint(l.domain) + 1e-9 && return i
    end
    return length(m.layers)
end
ω_unit(m::PlanetModel)    = m.ω_unit
"Period in minutes for a dimensionless frequency `ω` in model `m`."
T_minutes(m::PlanetModel, ω) = 2π / (ω * m.ω_unit) / 60

Base.show(io::IO, m::PlanetModel) =
    print(io, "PlanetModel(\"", m.name, "\", ", length(m.layers), " layers, R=",
          round(m.R_SI/1e3, digits=1), " km, ρ̄=", round(m.ρ̄_SI, digits=1), " kg/m³)")

"""
    dahlenize(m::PlanetModel) -> PlanetModel

Copy of `m` with every fluid layer switched to the Dahlen (1974) reduced static
formulation: the fluid interior carries only the closed δφ equation (source
`ρ₀′(δφ+Φ)/g`), with `U` and `P` as slave fields. Removes the Longman boundary
inconsistency of a stratified fluid at a solid wall
(`examples/2_dahlen_fluid_tutorial.ipynb`).

**Static solves only** — fluid inertia is dropped, so `free_modes` rejects a
dahlenized model.
"""
dahlenize(m::PlanetModel) = PlanetModel(m.name * "+dahlen",
    [is_fluid(l) ?
        Layer(domain=l.domain, n=l.n, κ=l.κ, μ=l.μ, ρ₀=l.ρ₀, g₀=l.g₀, aw=l.aw, dahlen=true,
              ν_SI=l.ν_SI) : l
     for l in m.layers],
    m.R_SI, m.ρ̄_SI, m.ω_unit, m.p_unit, m.Ω_SI)

"""
    build_model(name, domains, density_coeffs, κ, μ, n; R_SI, ρ̄_SI) -> PlanetModel

Construct a model from mean-normalised density polynomials `density_coeffs[i]`
(`3·Σᵢ∫ρᵢx²dx = 1`, giving `g(1)=1`) and normalised moduli `κ[i], μ[i]` on
`domains[i]`, with `n[i]` spectral modes. Gravity comes from `self_gravity`.

`R_SI` and `ρ̄_SI` set the SI metadata only; the solver is dimensionless.
"""
function build_model(name::AbstractString, domains, density_coeffs::Vector{<:Vector},
                     κ::Vector, μ::Vector, n::Vector{Int}; R_SI, ρ̄_SI,
                     aw::Vector{Bool} = fill(true, length(domains)),
                     Ω_SI::Real = 0.0,
                     ν_SI::Vector{<:Real} = zeros(length(domains)))
    gs = self_gravity(density_coeffs, domains)
    layers = [Layer(domain = domains[i], n = n[i],
                    κ = κ[i], μ = μ[i],
                    ρ₀ = let c = density_coeffs[i]; r -> evalpoly(r, c) end,
                    g₀ = gs[i], aw = aw[i], ν_SI = Float64(ν_SI[i]))
              for i in eachindex(domains)]
    ω_u = sqrt(4π * G_SI * ρ̄_SI / 3)
    p_u = (4π * G_SI / 3) * ρ̄_SI^2 * R_SI^2
    return PlanetModel(name, layers, R_SI, ρ̄_SI, ω_u, p_u, Float64(Ω_SI))
end

# ── Model-based solver API ────────────────────────────────────────────────────

love_numbers(m::PlanetModel, ℓ::Int = 2; ω² = 0.0) = read_love(forced(m, Tide(ℓ); ω², ℓ))

"""
    free_modes(m; ℓ=2) -> (; ωs, T_min, sorted_idx, vs, recover)

Free spheroidal frequencies at degree `ℓ`, ascending. `T_min` are periods in
minutes and `vs[:, sorted_idx[k]]` is the k-th eigenvector, in the full
`[U; V/P; δφ]` basis. `recover` is `identity`, kept so callers need not care.

`B` is singular — Poisson carries no time derivative — so the pencil is solved as
it stands and `solve_modes` discards what the singularity produces. Eliminating
`δφ` first by Schur complement would halve the work but needs `A_zz⁻¹`, whose
condition number reaches 1e13 on PREM; the direct route is better behaved and
matches Kelvin's analytic f-mode to 2e-13.

Errors on a `dahlenize`d model, which has no fluid inertia.
"""
function free_modes(m::PlanetModel; ℓ::Int = 2)
    any(l -> l.dahlen, m.layers) && throw(ArgumentError(
        "free_modes needs fluid inertia; pass the undahlenized model"))
    Amat, Bmat, _, ns = assemble_planet(m.layers, ℓ)
    ωs, sorted_idx, vs, _ = solve_modes(Amat, Bmat)
    T_min = [T_minutes(m, ω) for ω in ωs]
    return (; ωs, T_min, sorted_idx, vs, recover = identity)
end
