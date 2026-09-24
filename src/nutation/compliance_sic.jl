# ── Solid-inner-core (SIC) compliances ────────────────────────────────────────
#
# The 3×3 deformation-compliance matrix for a body with a solid inner core:
# three regions (whole Earth, fluid outer core, solid inner core) × three
# centrifugal forcing channels (all layers = tidal, OC only, IC only), read with
# the same F-integral machinery as `compliances`. Reproduces κ, ξ, γ, β and adds
# the inner-core columns/rows ζ, δ, θ, χ, ν — the MHB (Mathews–Herring–Buffett
# 2002) SIC compliance set:
#
#         │ tidal (m̃)   OC (m̃_f)    IC (m̃_s)
#   whole │ κ            ξ            ζ
#   fluid │ γ            β            δ
#   IC    │ θ            χ            ν
#
# Betti reciprocity (Saito eq. 58, three ties) is returned as a self-consistency
# check: Â_tot·ξ = Â_f·γ, Â_tot·ζ = Â_s·θ, Â_f·δ = Â_s·χ.

"""
    sic_compliances(model; ic_layer=1, oc_layer=2, Ω_SI=nothing, ℓ=2, diurnal=false, ω²=nothing)

Deformation compliances for a model with a **solid inner core** (`ic_layer`,
`[0,d]`) inside a **fluid outer core** (`oc_layer`, `[d,b]`). Returns a
NamedTuple with the nine MHB SIC compliances `κ,ξ,ζ` (whole), `γ,β,δ` (outer
core), `θ,χ,ν` (inner core), the reduced moments `Ât,Âf,Âs`, and `recip` (three
Betti residuals — should be ~0). `κ,ξ,γ,β` match `compliances`; the inner-core
columns/rows `ζ,δ,θ,χ,ν` are the new SIC terms. `Ω_SI` defaults to the model's
own rotation rate.

By default the solve is **static** (`ω² = 0`). `diurnal=true` retains inertia at
`ω² = Ω̂²`: a forced nutation at celestial frequency `ν ≈ 0` is a wobble at
`σ = ν−1 ≈ −1` in the terrestrial frame — retrograde diurnal — so the
wobble-driven centrifugal deformation occurs at `|ω| = Ω`. Pass `ω²` to override.
Compare against a *static* tabulation only when `diurnal=false`. Note that with
inertia the static fluid gauge freedom is lifted, so `dahlenize` is neither needed
nor permitted here (see `docs/how_elastic_computations_work.md` §6).
"""
function sic_compliances(m::PlanetModel; ic_layer::Int = 1, oc_layer::Int = 2,
                         Ω_SI = nothing, ℓ::Int = 2, diurnal::Bool = false, ω² = nothing)
    Ω_SI = rotation_rate(m, Ω_SI)
    layers = m.layers
    @assert !is_fluid(layers[ic_layer]) "ic_layer must be the solid inner core"
    @assert is_fluid(layers[oc_layer])  "oc_layer must be the fluid outer core"
    Ω̂² = (Ω_SI / m.ω_unit)^2

    # Inertia. A forced nutation at celestial frequency ν ≈ 0 is a wobble at σ = ν−1 ≈ −1
    # in the terrestrial frame — i.e. RETROGRADE DIURNAL — so the wobble-driven centrifugal
    # deformation happens at |ω| = Ω and `ω² = Ω̂²`. `diurnal=true` is that; `ω²` overrides.
    ω²_use = ω² !== nothing ? ω² : (diurnal ? Ω̂² : 0.0)
    # Three centrifugal channels, one forced solve each; regions read with `moment`
    # (S_ij = inertia_i / A_i, A_i = (8π/3)Âᵢ — same F-integral as `compliances`).
    cf(ls) = Potential(r -> r^ℓ; amplitude = Ω̂²/3, layers = ls)
    ft = forced(m, cf(:all);     ω² = ω²_use, ℓ)     # tidal (all layers)
    ff = forced(m, cf(oc_layer); ω² = ω²_use, ℓ)     # outer core only
    fs = forced(m, cf(ic_layer); ω² = ω²_use, ℓ)     # inner core only

    Ât = sum(_Âintegral(l.ρ₀, l.domain) for l in layers)
    Âf = _Âintegral(layers[oc_layer].ρ₀, layers[oc_layer].domain)
    Âs = _Âintegral(layers[ic_layer].ρ₀, layers[ic_layer].domain)
    A_t = (8π/3)*Ât;  A_f = (8π/3)*Âf;  A_s = (8π/3)*Âs

    whole(f) = moment(f, :whole)   / A_t
    ocmp(f)  = moment(f, oc_layer) / A_f
    scmp(f)  = moment(f, ic_layer) / A_s

    κ, ξ, ζ = whole(ft), whole(ff), whole(fs)
    γ, β, δ = ocmp(ft),  ocmp(ff),  ocmp(fs)
    θ, χ, ν = scmp(ft),  scmp(ff),  scmp(fs)
    recip = (ξγ = (Ât*ξ - Âf*γ)/(Âf*γ),      # whole↔OC
             ζθ = (Ât*ζ - Âs*θ)/(Âs*θ),      # whole↔IC
             δχ = (Âf*δ - Âs*χ)/(Âs*χ))       # OC↔IC
    return (; κ, ξ, ζ, γ, β, δ, θ, χ, ν, Ât, Âf, Âs, recip)
end
