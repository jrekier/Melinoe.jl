# ── Composable forcing → forced response → field access ───────────────────────
#
# Assemble, force, read. `love_numbers` and `compliances` are thin recipes over
# these three moves; `radial`/`tangential`/`potential` give the raw fields.

abstract type Forcing end

"Tidal potential `rℓ` over the whole body (unit amplitude)."
struct Tide <: Forcing; ℓ::Int; end
Tide(; ℓ = 2) = Tide(ℓ)

"Centrifugal (wobble) potential `r²`, amplitude `Ω̂²/3`, over `layers` (`:all`, `:oc`, `:ic`, or indices)."
struct Centrifugal <: Forcing; layers; end
Centrifugal(; layers = :all) = Centrifugal(layers)

"Inner-core tilt potential `−(2/3) ε r g(r)` over `region`."
struct Tilt <: Forcing; region; ε::Float64; end
Tilt(region; ε) = Tilt(region, ε)

"Arbitrary potential `Φ(r)` with `amplitude`, restricted to `layers`."
struct Potential <: Forcing; Φ::Function; amplitude::Float64; layers; end
Potential(Φ; amplitude = 1.0, layers = :all) = Potential(Φ, amplitude, layers)

"Superposition of forcings — solved in a single linear solve."
struct Sum <: Forcing; parts::Vector{Forcing}; end
Base.:+(a::Forcing, b::Forcing) = Sum(Forcing[a, b])
Base.:+(a::Sum,     b::Forcing) = Sum([a.parts; b])
Base.:+(a::Forcing, b::Sum)     = Sum([a; b.parts])

# region spec → layer indices
_resolve(m, s::Symbol)         = s === :oc ? findall(is_fluid, m.layers) :
                                 s === :ic ? findall(l -> !is_fluid(l), m.layers)[1:1] :
                                 error("unknown region :$s (use :oc, :ic, indices, or :all)")
_resolve(_, i::Integer)        = [i]
_resolve(m, v::AbstractVector) = reduce(vcat, _resolve(m, x) for x in v)
_mask(m, ls) = ls === :all ? trues(length(m.layers)) :
               [i in _resolve(m, ls) for i in eachindex(m.layers)]

_Ω̂²(m)   = (rotation_rate(m) / m.ω_unit)^2
_g(m, r) = m.layers[layer_at(m, r)].g₀(r)

# each forcing → (potential Φ(r), amplitude, layer mask), resolved against the model
_spec(f::Tide, m)        = (r -> r^f.ℓ,                       1.0,         trues(length(m.layers)))
_spec(f::Centrifugal, m) = (r -> r^2,                         _Ω̂²(m)/3,    _mask(m, f.layers))
_spec(f::Tilt, m)        = (r -> -(2/3) * f.ε * r * _g(m, r), 1.0,         _mask(m, f.region))
_spec(f::Potential, m)   = (f.Φ,                              f.amplitude, _mask(m, f.layers))

"""
A solved forced state, holding the spectral field `x = [U; V/P; δφ]`. Read it
with `radial` / `tangential` / `potential` / `moment`.
"""
struct Forced
    x     :: Vector{Float64}
    model :: PlanetModel
    ops
    ns    :: Vector{Int}
    ℓ     :: Int
    ω²    :: Float64
end

"""
    forced(model, forcing; ω²=0.0, ℓ=2) -> Forced

Solve `(A − ω²B) y = f` for the body force of `forcing` on `model`, keeping the
field. Forcings compose: `Tide() + Centrifugal(layers=:oc)` is one solve.
"""
function forced(model::PlanetModel, fc::Forcing; ω² = 0.0, ℓ::Int = 2)
    A, B, ops, ns, jr = assemble_planet(model.layers, ℓ)
    function _rhs(g)
        Φ, amp, msk = _spec(g, model)
        potential_forcing(ops, ns; ℓ, potential = Φ, amplitude = amp, layer_mask = msk, jrhs = jr)
    end
    f = fc isa Sum ? sum(_rhs, fc.parts) : _rhs(fc)
    Forced((A - ω² * B) \ f, model, ops, ns, ℓ, Float64(ω²))
end

# ── field access ──────────────────────────────────────────────────────────────

# spectral field `f` (0:U, 1:V/P, 2:δφ) on layer `i`, as an evaluable `Fun`
function _layerfun(d::Forced, f::Int, i::Int)
    Ntot = sum(d.ns); cumN = cumsum([0; d.ns])
    Fun(d.ops[i].S, d.x[f*Ntot + cumN[i]+1 : f*Ntot + cumN[i+1]])
end

# Stitch per-layer callables into one piecewise function of radius. `layer_at`
# throws out of range, since the per-layer `Fun`s extrapolate silently.
_piecewise(d::Forced, funs) = r -> funs[layer_at(d.model, r)](r)

"Radial displacement `U(r)`. `radial(d)` is a callable; `radial(d, r)` samples it."
radial(d::Forced)    = _piecewise(d, [_layerfun(d, 0, i) for i in eachindex(d.ns)])
radial(d::Forced, r) = radial(d)(r)

"Perturbed gravitational potential `δφ(r)`."
potential(d::Forced)    = _piecewise(d, [_layerfun(d, 2, i) for i in eachindex(d.ns)])
potential(d::Forced, r) = potential(d)(r)

"""
Horizontal displacement `V(r)` (derived from the pressure field inside fluid
layers). Undetermined at a static fluid surface — see `love_numbers`.
"""
function tangential(d::Forced)
    ℓ = d.ℓ
    funs = map(eachindex(d.ns)) do i
        L = d.model.layers[i]
        if is_fluid(L)
            U = _layerfun(d, 0, i); P = _layerfun(d, 1, i); DU = U'
            inc = is_incompressible(L)
            r -> r / (ℓ*(ℓ+1)) * (DU(r) + 2U(r)/r + (inc ? 0.0 : P(r)/L.κ(r)))
        else
            _layerfun(d, 1, i)
        end
    end
    _piecewise(d, funs)
end
tangential(d::Forced, r) = tangential(d)(r)

# ── named readers ─────────────────────────────────────────────────────────────

"""
Surface Love numbers `(; h, l, k)` — the three fields evaluated at `R`.
`l` is `NaN` for a static solve on a fluid-surfaced body (see `love_numbers`).
"""
function read_love(d::Forced)
    R = rightendpoint(d.model.layers[end].domain)
    l = _static_fluid_surface(d.ops, d.ω²) ? NaN : tangential(d, R)
    (; h = radial(d, R), l, k = potential(d, R))
end

"Induced surface gravitational potential `δφ(R)` (equals `k₂` for a unit tide)."
surface_potential(d::Forced) = potential(d, rightendpoint(d.model.layers[end].domain))

"""
    moment(d::Forced, region) -> c̃₃

Equatorial moment-of-inertia increment `δI₁₃ + i δI₂₃` of a region — `:whole`, or
a fluid-layer index / `:oc`. From the telescoping field
`F = r⁴δφ′ + 3ρ₀r⁴U − 2r³δφ`; same `ℓ=2, m=1` normalization as `compliances`.
"""
function moment(d::Forced, region)
    region === :whole && return (4π/3) * surface_potential(d)
    i = only(_resolve(d.model, region))
    U = _layerfun(d, 0, i); φ = _layerfun(d, 2, i); Dφ = φ'; ρ = d.model.layers[i].ρ₀
    F(r) = r^4*Dφ(r) + 3*ρ(r)*r^4*U(r) - 2*r^3*φ(r)
    dom = d.model.layers[i].domain
    -(4π/15) * (F(rightendpoint(dom)) - F(leftendpoint(dom)))
end
