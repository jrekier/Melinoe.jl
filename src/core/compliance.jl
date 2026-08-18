# ── Sasao–Okubo–Saito compliances κ, ξ, γ, β ──────────────────────────────────
#
# Two static (ω=0) solves of the same operator with the ℓ=2 centrifugal potential
# Φ = (Ω̂²/3) r²: forced in every layer (→ κ, γ) and in the core only (→ ξ, β).
#
#   κ = δφ_s(1)/(2 Â_tot)              ξ = δφ_d(1)/(2 Â_tot)
#   γ = −(F_s(b) − F_s(c))/(10 Â_f)    β = −(F_d(b) − F_d(c))/(10 Â_f)
#   F(r) = r⁴ ( δφ′ + 3 ρ₀ U − 2 δφ/r ),  b = r_CMB, c = r_ICB,
#   Â = ∫ ρ₀ r⁴ dr   (whole Earth / fluid outer core)
# Betti reciprocity  Â_tot·ξ = Â_f·γ  (Saito eq. 58) is returned as a check.

# F(r) on layer i, from a solution vector. Written as r⁴δφ′ + 3ρ₀r⁴U − 2r³δφ (no
# 0·∞) so a core reaching the centre, as on Mars, evaluates F(0) = 0 cleanly.
function _F_at(x, layers, ops, ns, i, r)
    Ntot = sum(ns)
    cumN = cumsum([0; ns])
    S    = ops[i].S
    Uf   = Fun(S, x[cumN[i]+1        : cumN[i+1]])
    φf   = Fun(S, x[2Ntot+cumN[i]+1  : 2Ntot+cumN[i+1]])
    Dφf  = φf'
    ρ    = layers[i].ρ₀(r)
    return r^4 * Dφf(r) + 3ρ * r^4 * Uf(r) - 2 * r^3 * φf(r)
end

function _δφ_surface(x, layers, ops, ns)
    L    = length(layers)
    Ntot = sum(ns)
    cumN = cumsum([0; ns])
    r_s  = rightendpoint(layers[L].domain)
    return Fun(ops[L].S, x[2Ntot+cumN[L]+1 : 2Ntot+cumN[L+1]])(r_s)
end

# Reduced equatorial moment  Â = ∫ ρ₀(r) r⁴ dr  over a domain (ApproxFun quadrature).
_Âintegral(ρ₀, dom) = sum(Fun(r -> ρ₀(r) * r^4, Chebyshev(dom)))

function _compliances(layers::Vector{Layer}, ℓ::Int; Ω̂², core_layer::Int)
    @assert is_fluid(layers[core_layer]) "core_layer must be the fluid outer core"
    L  = length(layers)
    ns = [l.n for l in layers]

    Amat, _, ops, _, jrhs = assemble_planet(layers, ℓ)
    Alu = lu(Amat)

    # two RHS, same operator (Φ = (Ω̂²/3) r^ℓ; ℓ=2 ⇒ r²)
    mask_all  = fill(true, L)
    mask_core = [i == core_layer for i in 1:L]
    x_s = Alu \ potential_forcing(ops, ns; ℓ, amplitude = Ω̂²/3, layer_mask = mask_all, jrhs)
    x_d = Alu \ potential_forcing(ops, ns; ℓ, amplitude = Ω̂²/3, layer_mask = mask_core, jrhs)

    Â_tot = sum(_Âintegral(l.ρ₀, l.domain) for l in layers)
    Â_f   = _Âintegral(layers[core_layer].ρ₀, layers[core_layer].domain)
    b = rightendpoint(layers[core_layer].domain)   # CMB
    c = leftendpoint(layers[core_layer].domain)     # ICB

    δφ_s1 = _δφ_surface(x_s, layers, ops, ns)
    δφ_d1 = _δφ_surface(x_d, layers, ops, ns)
    κ = δφ_s1 / (2 * Â_tot)
    ξ = δφ_d1 / (2 * Â_tot)     # + sign (matches κ); pinned by reciprocity below
    k₂ = 3 * δφ_s1 / Ω̂²

    Fs_b = _F_at(x_s, layers, ops, ns, core_layer, b)
    Fs_c = _F_at(x_s, layers, ops, ns, core_layer, c)
    Fd_b = _F_at(x_d, layers, ops, ns, core_layer, b)
    Fd_c = _F_at(x_d, layers, ops, ns, core_layer, c)
    γ = -(Fs_b - Fs_c) / (10 * Â_f)   # 10 = (2ℓ+1)·2 at ℓ=2
    β = -(Fd_b - Fd_c) / (10 * Â_f)

    betti = (Â_tot * ξ - Â_f * γ) / (Â_f * γ)   # reciprocity residual (Saito eq. 58)
    return (; κ, ξ, γ, β, k₂, Â_tot, Â_f, δφ_s1, δφ_d1, betti)
end

"""
    compliances(layers, ℓ=2; Ω̂², core_layer)
    compliances(m::PlanetModel; ℓ=2, Ω_SI=nothing, core_layer)

Sasao–Okubo–Saito compliances `(; κ, ξ, γ, β, k₂, Â_tot, Â_f, δφ_s1, δφ_d1, betti)`.
The model form takes `Ω̂² = (Ω_SI/ω_unit)²`, with `Ω_SI` defaulting to the model's
own rotation rate (see `rotation_rate`). `betti` is the reciprocity residual.
"""
compliances(layers::Vector{Layer}, ℓ::Int = 2; Ω̂², core_layer) =
    _compliances(layers, ℓ; Ω̂², core_layer)

function compliances(m::PlanetModel; ℓ::Int = 2, Ω_SI = nothing, core_layer)
    @assert is_fluid(m.layers[core_layer]) "core_layer must be the fluid outer core"
    Ω̂² = (rotation_rate(m, Ω_SI) / m.ω_unit)^2
    cf(ls) = Potential(r -> r^ℓ; amplitude = Ω̂²/3, layers = ls)   # centrifugal (Ω̂²/3)·rℓ
    fs = forced(m, cf(:all); ℓ)               # static:    all layers → κ, γ
    fd = forced(m, cf(core_layer); ℓ)         # dynamical: core only  → ξ, β
    Â_tot = sum(_Âintegral(l.ρ₀, l.domain) for l in m.layers)
    Â_f   = _Âintegral(m.layers[core_layer].ρ₀, m.layers[core_layer].domain)
    A_w = (8π/3) * Â_tot;  A_c = (8π/3) * Â_f          # S_ij = inertia_i / A_i
    κ = moment(fs, :whole)/A_w;      ξ = moment(fd, :whole)/A_w
    γ = moment(fs, core_layer)/A_c;  β = moment(fd, core_layer)/A_c
    δφ_s1 = surface_potential(fs);    δφ_d1 = surface_potential(fd)
    k₂    = 3 * δφ_s1 / Ω̂²
    betti = (Â_tot * ξ - Â_f * γ) / (Â_f * γ)         # reciprocity residual (Saito eq. 58)
    (; κ, ξ, γ, β, k₂, Â_tot, Â_f, δφ_s1, δφ_d1, betti)
end
