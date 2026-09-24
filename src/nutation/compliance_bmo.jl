# ── Compliances with a basal magma ocean (BMO) ────────────────────────────────
#
# Sasao/Mathews compliances for a body with a BMO between the fluid core and the
# mantle (interface Σ_bc at r=d, CMB at r=b). Three deformation channels, each a
# static body-force solve with the wobble potential Φ=(Ω̂²/3)r²: all layers, the core
# alone, the BMO alone. Forcing one layer at a time makes each compliance a
# primitive and both Betti ties two-term, with m̃_c and m̃_b relative to the mantle.
#
# assemble_planet handles the Σ_bc fluid–fluid density jump; the body forces
# generate the interface centrifugal sources. Sign of ξ is + (matches κ), pinned by
# the reciprocities. F(r) = r⁴(δφ′ + 3ρ₀U − 2δφ/r), validated on Earth and Le Maistre.

"""
    bmo_compliances(model; core_layer=1, bmo_layer=2, Ω_SI=Ω_EARTH, ℓ=2)

Compliances for a model with a fluid core (`core_layer`, `[ICB,d]`) under a fluid
BMO (`bmo_layer`, `[d,b]`). Returns `κ, ξc, ξb` (whole-body), `γc, βc, δc` (core),
`γb, δb, βb` (BMO), the reduced moments `Â_c, Â_b, Â_tot`, the interface radii `d`
and `b`, and `recip`, the residuals of the Betti ties `Â_tot·ξc = Â_c·γc`,
`Â_tot·ξb = Â_b·γb` and `Â_c·δc = Â_b·δb`, which should be ~0.

There is no Σ_bc buoyancy compliance: at this static closure the interface torque
reduces to the figure coupling already carried by `(1+e_c)`, `(1+e_b)`.
"""
function bmo_compliances(model::PlanetModel; core_layer::Int = 1, bmo_layer::Int = 2,
                         Ω_SI = nothing, ℓ::Int = 2)
    Ω_SI = rotation_rate(model, Ω_SI)
    layers = model.layers
    @assert is_fluid(layers[core_layer]) && is_fluid(layers[bmo_layer]) "core & BMO must be fluid"
    Ω̂² = (Ω_SI / model.ω_unit)^2

    # One forced solve per channel; regions read with `moment`,
    # S_ij = inertia_i / A_i with A_i = (8π/3)Âᵢ.
    cf(ls) = Potential(r -> r^ℓ; amplitude = Ω̂²/3, layers = ls)
    ft = forced(model, cf(:all);       ℓ)                # tidal: all layers
    fc = forced(model, cf(core_layer); ℓ)                # core alone
    fb = forced(model, cf(bmo_layer);  ℓ)                # BMO alone

    Â_tot = sum(_Âintegral(l.ρ₀, l.domain) for l in layers)
    Â_c   = _Âintegral(layers[core_layer].ρ₀, layers[core_layer].domain)
    Â_b   = _Âintegral(layers[bmo_layer].ρ₀,  layers[bmo_layer].domain)
    A_tot = (8π/3)*Â_tot;  A_c = (8π/3)*Â_c;  A_b = (8π/3)*Â_b

    whole(f)    = moment(f, :whole)     / A_tot
    core_cmp(f) = moment(f, core_layer) / A_c
    bmo_cmp(f)  = moment(f, bmo_layer)  / A_b

    κ,  ξc, ξb = whole(ft),    whole(fc),    whole(fb)
    γc, βc, δc = core_cmp(ft), core_cmp(fc), core_cmp(fb)
    γb, δb, βb = bmo_cmp(ft),  bmo_cmp(fc),  bmo_cmp(fb)

    recip = (ξc_γ    = (Â_tot*ξc - Â_c*γc) / (Â_c*γc),
             ξb_γ    = (Â_tot*ξb - Â_b*γb) / (Â_b*γb),
             δ_cross = (Â_c*δc   - Â_b*δb) / (Â_b*δb))
    return (; κ, ξc, ξb, γc, βc, δc, γb, δb, βb, Â_c, Â_b, Â_tot,
            d = rightendpoint(layers[core_layer].domain),
            b = rightendpoint(layers[bmo_layer].domain), recip)
end
