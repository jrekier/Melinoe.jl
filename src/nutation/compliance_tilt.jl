# ── Inner-core TILT compliances (Buffett/Dumberry S^g, S^p) ───────────────────
#
# A tilted inner-core figure (ñ_s = 1, rotation vectors fixed) forces the Earth
# through Dumberry (2009) eq. (27):  f_t = −∇(ρ₀g₀Δr_s) − ρ₀∇φ_s − ρ_s∇φ₀, with
# Δr_s = −(2/3) r ε(r) Y the tilted-figure level-surface displacement inside the
# inner core, ρ_s = −ρ₀′Δr_s the advected density, and φ_s its potential (bulk +
# ICB sheet Δρ·Δr_s). One forced solve per channel, read with `moment`, gives the
# tilt compliances of eq. (23)–(25):
#
#   gravitational  S^g_i4  ←  −ρ₀∇φ_s − ρ_s∇φ₀   multiplies ñ_s
#   pressure       S^p_i4  ←  −∇(ρ₀g₀Δr_s)       multiplies ñ_s − m̃ − m̃_s
#
# The second combination is why the two are kept apart: it vanishes when the inner
# core's figure and rotation axes move together. `_add_tilt!` places them in the
# nutation matrix per eqs (28)–(37).
#
# Validated on elastic PREM (`anelastic=false`, as Table 1 assumes):
#
#           S^g14   S^g24   S^g34   S^p14   S^p24   S^p34   lumped S14
#   error   0.10%   0.03%   0.07%   0.16%   3.63%   3.42%     0.15%
#
# and against Table 2, where the tilt column lengthens the ICW from 2435 d to
# 2730 d (his 2410 → 2715) and leaves the CW, RFCN and PFCN unmoved.

"""
    sic_tilt_compliances(m; ic_layer=1, oc_layer=2, Ω_SI=nothing, ℓ=2,
                         diurnal=false, ω²=nothing)

Tilt compliances `S^g_i4`, `S^p_i4` (i = 1 whole, 2 fluid outer core, 3 inner core)
for a solid inner core whose figure is tilted by `ñ_s = 1`. Returns
`(; Sg, Sp, lumped)`, each a 3-tuple over the three regions.

Pass the whole result to `sic_nutation_modes` or `sic_nutation_transfer` as `tilt`,
which places the two channels separately; `lumped = Sg .+ Sp` is correct only when
`m̃ + m̃_s = 0`, as in the ICW and in Dumberry (2008).

`diurnal=true` retains inertia at `ω² = Ω̂²`. Dumberry's diurnal *tilt* values also
carry Coriolis dynamics this operator omits, so only the static column is
reproducible.
"""
function sic_tilt_compliances(m::PlanetModel; ic_layer::Int = 1, oc_layer::Int = 2,
                              Ω_SI = nothing, ℓ::Int = 2,
                              diurnal::Bool = false, ω² = nothing)
    @assert ℓ == 2 "tilt compliances are a degree-2 (tesseral) problem"
    Ω_SI = rotation_rate(m, Ω_SI)
    layers = m.layers; L = length(layers); ns = [l.n for l in layers]
    @assert !is_fluid(layers[ic_layer]) "ic_layer must be the solid inner core"
    @assert is_fluid(layers[oc_layer])  "oc_layer must be the fluid outer core"
    Ω̂² = (Ω_SI / m.ω_unit)^2
    ω²_use = ω² !== nothing ? ω² : (diurnal ? Ω̂² : 0.0)

    flt = flattening(m; Ω_SI)
    xI = rightendpoint(layers[ic_layer].domain)
    # smooth ε inside the IC (flt.ε is a dense linear interpolant; adaptive Fun does
    # not converge on it, so a fixed-degree LSQ polynomial is used, rel err ~1e-6)
    xs = range(0.0, xI, length = 120); ys = [flt.ε(x) for x in xs]
    deg = min(ns[ic_layer] - 1, 15)
    cf = ([x^j for x in xs, j in 0:deg]) \ ys
    ε(x) = evalpoly(x, cf)
    d(x) = -(2/3) * x * ε(x)                       # imposed tilt displacement
    ρ0(x)  = layers[ic_layer].ρ₀(x)
    ρ0p(x) = (ρ0(x + 1e-7) - ρ0(x - 1e-7)) / 2e-7
    g0(x)  = layers[ic_layer].g₀(x)

    Amat, Bmat, ops, _, _ = assemble_planet(layers, ℓ)
    Ntot = sum(ns); cumN = cumsum([0; ns])
    padv(ψ, i) = (c = Fun(ψ, ops[i].S, ns[i]).coefficients;
                  length(c) >= ns[i] ? c[1:ns[i]] : [c; zeros(ns[i] - length(c))])
    rgI = interior_ranges(ic_layer, ns[ic_layer], false, Ntot, cumN)

    # pressure channel: T1 = −∇(ρ₀ g₀ d Y), IC bulk. Rows are built by applying
    # the layer's own operators so they land in the assembled rangespace.
    pgd(x)  = ρ0(x) * g0(x) * d(x)
    pgdp(x) = (pgd(x + 1e-7) - pgd(x - 1e-7)) / 2e-7
    fP = zeros(3Ntot)
    fP[rgI.U_int] = ops[ic_layer].A_Vφ[rgI.U_op, :] * padv(x -> -x * pgdp(x) / ρ0(x), ic_layer)
    fP[rgI.V_int] = ops[ic_layer].A_Vφ[rgI.V_op, :] * padv(x -> -g0(x) * d(x), ic_layer)
    # gravitational channel: T3 = −ρ_s∇φ₀ (IC bulk, radial), plus T2 = −ρ₀∇φ_s
    # with φ_s from the ℓ=2 Green's function of ρ_s = −ρ₀′d + ICB sheet Δρ·d.
    fG = zeros(3Ntot)
    fG[rgI.U_int] = ops[ic_layer].A_Vφ[rgI.U_op, :] * padv(x -> x * (ρ0p(x)/ρ0(x)) * g0(x) * d(x), ic_layer)
    σsheet = (layers[ic_layer].ρ₀(xI) - layers[oc_layer].ρ₀(xI)) * d(xI)
    ρsb(x) = -ρ0p(x) * d(x)
    NQ = 2000; qs = range(1e-8, xI, length = NQ); hq = step(qs)
    c4 = zeros(NQ); t1 = zeros(NQ)
    for i in 2:NQ
        c4[i] = c4[i-1] + (ρsb(qs[i])*qs[i]^4 + ρsb(qs[i-1])*qs[i-1]^4)/2 * hq
    end
    for i in NQ-1:-1:1
        t1[i] = t1[i+1] + (ρsb(qs[i])/qs[i] + ρsb(qs[i+1])/qs[i+1])/2 * hq
    end
    tot4 = c4[end]
    lin(v, x) = x <= qs[1] ? v[1] : x >= qs[end] ? v[end] :
        (j = clamp(Int(floor((x - qs[1])/hq)) + 1, 1, NQ - 1); t = (x - qs[j])/hq;
         v[j]*(1 - t) + v[j+1]*t)
    φs(r) = r < 1e-8 ? 0.0 :
            r < xI   ? -(3/5) * (lin(c4, r)/r^3 + r^2 * (lin(t1, r) + σsheet/xI)) :
                       -(3/5) * (tot4 + σsheet*xI^4) / r^3
    for i in 1:L
        vc = padv(φs, i)
        rg = interior_ranges(i, ns[i], ops[i].fluid, Ntot, cumN)
        fG[rg.U_int] -= ops[i].A_Uφ[rg.U_op, :] * vc
        fG[rg.V_int] -= ops[i].A_Vφ[rg.V_op, :] * vc
    end

    Alu = lu(ω²_use == 0 ? Amat : Amat - ω²_use .* Bmat)
    Ât = sum(_Âintegral(l.ρ₀, l.domain) for l in layers)
    Âf = _Âintegral(layers[oc_layer].ρ₀, layers[oc_layer].domain)
    Âs = _Âintegral(layers[ic_layer].ρ₀, layers[ic_layer].domain)
    A_t = (8π/3)*Ât;  A_f = (8π/3)*Âf;  A_s = (8π/3)*Âs
    # bespoke Dumberry forcing (fP, fG) above; read-out is the shared `moment` reader.
    function read3(rhs)
        f = Forced(Alu \ rhs, m, ops, ns, ℓ, Float64(ω²_use))
        (moment(f, :whole)/A_t, moment(f, oc_layer)/A_f, moment(f, ic_layer)/A_s)
    end
    Sp = read3(fP)
    Sg = read3(fG)
    return (; Sg, Sp, lumped = Sg .+ Sp)
end
