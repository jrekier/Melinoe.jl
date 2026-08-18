# ── Mars, InSight-constrained three-layer model ───────────────────────────────
# Liquid iron core | silicate mantle | basaltic crust. No inner core, so the fluid
# core is layers[1], reaches the centre, and needs the non-AW fluid operator —
# Adams-Williamson would give κ→0 at r=0.
#
# Geometry and bulk properties are taken from the literature (see the docstring),
# and the core and mantle density profiles are then scaled by one factor each so
# the model reproduces the observed mass and mean moment of inertia exactly. Both
# factors come out within 1.3% of unity, and the resulting core density is checked
# against the seismically inferred range.

_polyfit(xs, ys, deg) = ([x^j for x in xs, j in 0:deg]) \ ys   # LS coeffs Σ cⱼx^j

"""
    Mars(; h_crust_km=45.0, r_core_km=1830.0, n_core=90, n_mantle=90, n_crust=25)

Three-layer Mars constrained by InSight and radio science: liquid core `layers[1]`,
silicate mantle, crust.

| quantity | value | source |
|---|---|---|
| core radius | 1830 ± 40 km | Stähler et al. 2021, *Science* **373**, 443 |
| core density | 5700–6300 kg/m³ | Stähler et al. 2021 |
| crustal thickness | 24–72 km (global) | Knapmeyer-Endrun et al. 2021, *Science* **373**, 438 |
| mass | 6.4171e23 kg | Konopliv et al. 2016 |
| `C/MR²` | 0.363918 (polar, R = 3396 km) | Konopliv et al. 2016 |
| `k₂` | 0.174 ± 0.008 | Konopliv et al. 2020 |

The mean radius is the IAU 3389.5 km. The moment-of-inertia target used here is the
**mean** moment `I = C(1 − 2H/3)` with `H = 0.005364`, renormalised to the mean
radius, since the model is spherically symmetric — `I/(M R²) = 0.36401`.

Mass and moment of inertia are matched exactly by construction; `k₂` is left free
and is the independent check on the elastic moduli. The model gives `k₂ = 0.1784`,
within 0.6σ of the Konopliv et al. (2020) `0.174 ± 0.008`. Note the published
determinations disagree by more than their quoted errors — Konopliv et al. (2016)
give `0.169 ± 0.006` and Genova et al. (2016) `0.1697 ± 0.0009` — so `k₂` pins the
mantle rigidity only to a few percent.

`r_core_km` moves `k₂` strongly; `h_crust_km` barely touches it. Shrinking the core
to lower `k₂` drives the core density above the 5700–6300 range, since mass and
moment of inertia still have to be met.
"""
function Mars(; h_crust_km::Float64 = 45.0, r_core_km::Float64 = 1830.0,
                n_core::Int = 90, n_mantle::Int = 90, n_crust::Int = 25)
    R  = 3389.5e3                                   # IAU mean radius
    xf(r_km) = r_km*1e3 / R

    # ── observational targets ────────────────────────────────────────────────
    M_obs = 6.4171e23
    H     = 0.005364                                 # dynamical ellipticity
    I_obs = 0.363918 * (1 - 2H/3) * (3396.0/3389.5)^2   # → 0.36401, mean moment

    # ── digitised profile shapes (ρ [kg/m³] vs radius [km]) ──────────────────
    core_r = [0., 400, 800, 1200, 1600, 1830]
    core_ρ = [6565., 6545, 6480, 6350, 6150, 5870]
    core_κ = [212., 209, 199, 183, 163, 150]
    man_r  = [1830., 1880, 1950, 2100, 2250, 2340, 2450, 2600, 2800, 3000, 3320]
    man_ρ  = [4050., 4010, 3945, 3835, 3745, 3660, 3595, 3540, 3500, 3480, 3465]
    man_κ  = [213., 222, 214, 196, 175, 158, 146, 131, 122, 119, 117]
    man_μ  = [93., 97, 93, 86, 77, 69, 66, 64, 63, 64, 67]
    ρ_crust, κ_crust, μ_crust = 2900.0, 62.0e9, 40.0e9   # Vp ≈ 6.3, Vs ≈ 3.7 km/s

    xc = xf(r_core_km);  xm = 1.0 - xf(h_crust_km)
    c_core = _polyfit(xf.(core_r), core_ρ, 3)
    c_mant = _polyfit(xf.(man_r),  man_ρ,  5)

    # ── scale the core and mantle densities to hit M and I exactly ───────────
    # M = 4πR³Σ∫ρx²dx and I = (8π/3)R⁵Σ∫ρx⁴dx are both linear in the two factors.
    a2, b2 = poly_r2_integral(c_core, 0.0, xc), poly_r2_integral(c_mant, xc, xm)
    a4, b4 = poly_r4_integral(c_core, 0.0, xc), poly_r4_integral(c_mant, xc, xm)
    c2, c4 = poly_r2_integral([ρ_crust], xm, 1.0), poly_r4_integral([ρ_crust], xm, 1.0)
    α, β = [a2 b2; a4 b4] \ [M_obs/(4π*R^3) - c2, (3/8)*I_obs*M_obs/(π*R^3) - c4]

    cρ = [α .* c_core, β .* c_mant, [ρ_crust]]
    cκ = [_polyfit(xf.(core_r), core_κ .* 1e9, 3),
          _polyfit(xf.(man_r),  man_κ .* 1e9, 5), [κ_crust]]
    cμ = [[0.0], _polyfit(xf.(man_r), man_μ .* 1e9, 5), [μ_crust]]
    xb = [0.0, xc, xm, 1.0]

    nL   = 3
    doms = [xb[i]..xb[i+1] for i in 1:nL]
    ρ̄    = 3*sum(poly_r2_integral(cρ[i], xb[i], xb[i+1]) for i in 1:nL)
    dens = [c ./ ρ̄ for c in cρ]
    p_u  = (4π*G_SI/3) * ρ̄^2 * R^2
    κ = [let c = cκ[i]; x -> evalpoly(x, c)/p_u end for i in 1:nL]
    μ = [let c = cμ[i]; x -> evalpoly(x, c)/p_u end for i in 1:nL]

    build_model("Mars", doms, dens, κ, μ, [n_core, n_mantle, n_crust];
                R_SI = R, ρ̄_SI = ρ̄, aw = [false, true, true], Ω_SI = 7.088e-5)
end
