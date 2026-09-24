# ── Nutation with a solid inner core (MHB 4-variable theory) ──────────────────
#
# The wobble/nutation free modes of a body with a solid inner core: mantle m̃,
# fluid outer core m̃_f, inner core m̃_s, and the inner-core TILT ñ_s. The 4×4
# matrix M(ω)=M0+ωM1 is Mathews–Herring–Buffett (Dehant & Mathews eq. 7.151); the
# four roots are the FCN, FICN, Chandler wobble and inner-core wobble (ICW).
#
# Inputs are all computed from the model: the nine deformation compliances
# (`sic_compliances`) and the gravitational coupling constants α1, α2, α3
# (`sic_alphas`, Dehant & Mathews eq. 7.152). Validated on PREM: α's match MHB to
# sub-percent, FICN = +474 d (observed +475), hydrostatic FCN = −459 d.

"""
    sic_alphas(m; ic_layer=1, oc_layer=2, Ω_SI=nothing) -> (; α1, α2, α3, α_g)

Inner-core gravitational coupling constants (Dehant & Mathews eq. 7.152):
`α1 = (C'−A')/(C_s−A_s)` with `C'−A' = (8π/15)ρ_f a_s⁵ ε_s` (a body with the
inner-core shape but uniform fluid density `ρ_f` near the ICB); `α3 = 1−α1`;
`α_g = (8πG/5Ω²)[∫_{a_s}^{a} ρ₀(dε/ds)ds + ρ_f ε_s]`; `α2 = α1 − α3 α_g`.
PREM: 0.947, 0.831, 0.053, 2.175 (MHB: 0.946, 0.829, 0.054, 2.175).
"""
function sic_alphas(m::PlanetModel; ic_layer::Int = 1, oc_layer::Int = 2, Ω_SI = nothing)
    Ω_SI = rotation_rate(m, Ω_SI)
    R = m.R_SI; ρ̄ = m.ρ̄_SI; Gsi = 6.674e-11
    flt = flattening(m; Ω_SI)
    xICB = rightendpoint(m.layers[ic_layer].domain)
    A_s, e_s = _region_moment_e(m, flt, 0.0, xICB)
    a_s = xICB * R
    ρ_f = m.layers[oc_layer].ρ₀(xICB) * ρ̄          # fluid density just above the ICB
    ε_s = flt.ε(xICB)                              # geometric flattening of the ICB
    α1 = (8π/15) * ρ_f * a_s^5 * ε_s / (A_s * e_s)
    α3 = 1 - α1
    ρ0(x) = (for l in m.layers; x ≤ rightendpoint(l.domain)+1e-12 && return l.ρ₀(x)*ρ̄; end;
             m.layers[end].ρ₀(1.0)*ρ̄)
    dεdx(x) = (flt.ε(x+1e-6) - flt.ε(x-1e-6)) / 2e-6
    xs = range(xICB, 1.0, length = 4000); integ = 0.0
    for i in 1:length(xs)-1
        xm = (xs[i]+xs[i+1])/2
        integ += ρ0(xm) * dεdx(xm) * (xs[i+1]-xs[i])   # ∫ρ₀(dε/ds)ds = ∫ρ₀(dε/dx)dx
    end
    α_g = (8π*Gsi/(5*Ω_SI^2)) * (integ + ρ_f*ε_s)
    α2 = α1 - α3*α_g
    return (; α1, α2, α3, α_g)
end

# 4×4 MHB matrix M(ω)=M0+ωM1, variables (m̃, m̃_f, m̃_s, ñ_s).
function _sic_matrices(s)
    c = s.c; α = s.α
    e, ef, es = s.e, s.ef, s.es; AfA, AsA = s.AfA, s.AsA
    M0 = zeros(4,4); M1 = zeros(4,4)
    M0[1,1] = -e + c.κ;  M1[1,1] = 1 + c.κ
    M0[1,2] = AfA + c.ξ; M1[1,2] = AfA + c.ξ
    M0[1,3] = AsA + c.ζ; M1[1,3] = AsA + c.ζ
    M0[1,4] = AsA*es*α.α3; M1[1,4] = AsA*es*α.α3
    M1[2,1] = 1 + c.γ;   M0[2,2] = 1 + ef; M1[2,2] = 1 + c.β
    M1[2,3] = c.δ;       M1[2,4] = -AsA*(A_over_Af(s))*es*α.α1  # = -(As/Af)es α1
    M0[3,1] = -es*α.α3;  M1[3,1] = 1 + c.θ
    M0[3,2] = es*α.α1;   M1[3,2] = c.χ
    M0[3,3] = 1;         M1[3,3] = 1 + c.ν
    M0[3,4] = es*(1 - α.α2); M1[3,4] = es
    M0[4,3] = 1;         M1[4,4] = 1
    return M0, M1
end
A_over_Af(s) = 1 / s.AfA          # As/Af = (As/A)/(Af/A); here M1[2,4] wants (As/Af)

"""
    sic_nutation_modes(m; ic_layer=1, oc_layer=2, Ω_SI=nothing)

Free wobble/nutation modes of a mantle · fluid-outer-core · solid-inner-core
body, from the MHB 4×4 (Dehant & Mathews eq. 7.151), with all inputs computed
from the model. Returns `(; ω, T_cel, T_wob, FCN, FICN, CW, ICW, c, α)`:
`T_cel = 2π/(Ω(1+ω))` celestial periods (FCN, FICN), `T_wob = 2π/(Ωω)` wobble
periods (Chandler, ICW), in days. Validated on PREM: FICN = +474 d (obs +475),
hydrostatic FCN = −459 d.
"""
# Gather compliances, α's, moments and ellipticities for the SIC nutation matrix.
function _sic_setup(m::PlanetModel; ic_layer::Int = 1, oc_layer::Int = 2, Ω_SI = nothing)
    Ω_SI = rotation_rate(m, Ω_SI)
    flt = flattening(m; Ω_SI)
    xICB = rightendpoint(m.layers[ic_layer].domain)
    xCMB = rightendpoint(m.layers[oc_layer].domain)
    A, e   = _region_moment_e(m, flt, 0.0, 1.0)
    Af, ef = _region_moment_e(m, flt, xICB, xCMB)
    As, es = _region_moment_e(m, flt, 0.0, xICB)
    c = sic_compliances(m; ic_layer, oc_layer, Ω_SI)
    α = sic_alphas(m; ic_layer, oc_layer, Ω_SI)
    return (; c, α, e, ef, es, AfA = Af/A, AsA = As/A, Ω_SI)
end

function sic_nutation_modes(m::PlanetModel; ic_layer::Int = 1, oc_layer::Int = 2, Ω_SI = nothing)
    s = _sic_setup(m; ic_layer, oc_layer, Ω_SI); Ω_SI = s.Ω_SI
    M0, M1 = _sic_matrices(s)
    ω = eigvals(M0, -M1); day = 86400.0
    T_cel = [2π/(Ω_SI*(1+ωi))/day for ωi in ω]
    T_wob = [2π/(Ω_SI*ωi)/day     for ωi in ω]
    ωr = real.(ω)
    dia = findall(x -> abs(x+1) < 0.1, ωr)      # nearly-diurnal: FCN, FICN
    zer = findall(x -> abs(x)   < 0.1, ωr)      # near-zero: Chandler, ICW
    FCN  = T_cel[dia[argmin(ωr[dia])]]           # most retrograde
    FICN = T_cel[dia[argmax(ωr[dia])]]           # prograde
    CW   = T_wob[zer[argmax(abs.(ωr[zer]))]]      # shorter wobble period
    ICW  = T_wob[zer[argmin(abs.(ωr[zer]))]]      # longer wobble period
    return (; ω, T_cel, T_wob, FCN, FICN, CW, ICW, c = s.c, α = s.α)
end

"""
    sic_nutation_transfer(m, ω; ic_layer=1, oc_layer=2, Ω_SI=nothing, S14=0, S24=0, S34=0)

Forced-nutation transfer function `T(ω) = m̃/m̃_rigid` for the mantle · fluid ·
solid-inner-core body at (terrestrial-frame) tidal frequency `ω` (units of Ω;
celestial frequency `1+ω`). Solves `M(ω)·x = y·φ` with the luni-solar tidal
forcing vector (Dehant & Mathews)

    y = [ (1+ω)κ − e ,  γω ,  ωθ − α₃e_s ,  0 ]

— the external torque on the whole figure (`−e`) and on the inner-core figure
(`−α₃e_s`) plus the tidal-deformation couplings — and returns
`x[1]·(ω−e)/e = m̃/m̃_rigid`. Poles sit at the free modes (FCN, FICN). `ω` may be
a scalar or vector.

The Dumberry tilt-deformation compliances enter the tilt (4th) column: `S34` in
the inner-core row `M1[3,4]` (validated; shifts the FICN 476 d → 546 d), `S14`,
`S24` in the whole/fluid rows (enter at O(e_s), negligible for the observable).
Note the FICN residue is `∼A_s/A ≈ 7×10⁻⁴` of the FCN: the inner core torques the
mantle only through its own tiny moment, an inertia ceiling no forcing can beat.
"""
function sic_nutation_transfer(m::PlanetModel, ω; ic_layer::Int = 1, oc_layer::Int = 2,
                               Ω_SI = nothing, S14 = 0.0, S24 = 0.0, S34 = 0.0)
    s = _sic_setup(m; ic_layer, oc_layer, Ω_SI)
    c = s.c; α = s.α; e = s.e; es = s.es
    M0, M1 = _sic_matrices(s)
    M1[1,4] += S14; M1[2,4] += S24; M1[3,4] += S34   # tilt-deformation (Dumberry)
    _T(w) = begin
        y = [(1 + w)*c.κ - e, c.γ*w, w*c.θ - α.α3*es, 0.0]
        ((M0 + w*M1) \ y)[1] * (w - e)/e
    end
    return ω isa Number ? _T(ω) : _T.(ω)
end
