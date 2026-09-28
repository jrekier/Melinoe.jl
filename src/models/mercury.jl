# ── Mercury, MESSENGER-constrained ────────────────────────────────────────────
# Liquid Fe-S core | silicate mantle | crust, with an optional solid inner core.
#
# This model solves for the *radii*: mantle and crust densities are inputs, and
# the core radius and density are the two unknowns closed by the
# observed mass and mean moment of inertia. `R_core` is therefore an output.
#
# Without an inner core the fluid core reaches r=0, so it takes the non-AW
# operator: the layer densities are constant (ρ₀′ = 0) and Adams-Williamson
# would send κ→0 at the centre.

"""
    Mercury(; h_crust_km=26.0, r_inner_km=0.0, Cnd=0.346, ρ_mantle=3250.0,
              Δρ_ic=500.0, μ_mantle=67.0e9, n_core=70, n_mantle=60,
              n_crust=25) -> PlanetModel

Mercury from MESSENGER radio science and Earth-based radar: liquid Fe-S core
`layers[1]`, silicate mantle, crust. `r_inner_km > 0` seats a solid inner core
below the liquid, which shifts every layer index up by one.

| quantity | value | source |
|---|---|---|
| mean radius | 2439.36 km | Perry et al. 2015, *GRL* **42**, 6951 |
| mass | 3.3011e23 kg | Mazarico et al. 2014, *JGR Planets* **119**, 2417 |
| `C/MR²` | 0.346 ± 0.014 | Margot et al. 2012, *JGR* **117**, E00L09 |
| `J₂` | 5.0323e-5 | Mazarico et al. 2014 |
| crustal thickness | 26 ± 11 km | Sori 2018, *EPSL* **489**, 92 |
| crustal density | 2974 ± 89 kg/m³ | Sori 2018 |
| `k₂` | 0.464 ± 0.023 | Verma & Margot 2016, *JGR Planets* **121**, 1627 |
| rotation period | 58.646 d (3:2 resonance) | Margot et al. 2012 |

The moment target is the **mean** moment `I = C(1 − 2H/3)` with `H = J₂/(C/MR²)`,
renormalised from the 2440 km reference radius of the gravity field to the mean
radius, since the model is spherically symmetric.

Mass and mean moment are matched by construction; two outputs are left free and
serve as the checks.

- `R_core` = 1994.5 km, against the 2020 ± 30 km of Hauck et al. (2013), *JGR
  Planets* **118**, 1204, at 0.9σ and with nothing tuned against it. Core density
  7253 kg/m³.
- `k₂` = 0.489, which depends only on the elastic moduli and is not seen by the
  fit, against 0.464 ± 0.023.

Knobs. `Cnd` is the moment target: lowering it raises the required core density
(8105 kg/m³ at 0.333), and a value outside 6000–8000 kg/m³ warns. `ρ_mantle`
moves `R_core` by about −20 km per +100 kg/m³, `h_crust_km` barely at all.
`μ_mantle` (Pa) never enters the mass/moment solve, so it moves `k₂` without
touching the structure.
"""
function Mercury(; h_crust_km::Float64 = 26.0, r_inner_km::Float64 = 0.0,
                   Cnd::Float64 = 0.346, ρ_mantle::Float64 = 3250.0,
                   Δρ_ic::Float64 = 500.0, ρ_crust::Float64 = 2974.0,
                   μ_mantle::Float64 = 67.0e9,
                   n_core::Int = 70, n_mantle::Int = 60, n_crust::Int = 25)
    R = 2439.36e3                                    # Perry et al. 2015

    # ── observational targets ────────────────────────────────────────────────
    M_obs = 3.3011e23                                # Mazarico et al. 2014
    J₂    = 5.0323e-5
    H     = J₂ / Cnd                                 # dynamical ellipticity
    I_obs = Cnd * (1 - 2H/3) * (2440.0/2439.36)^2    # mean moment, at mean radius
    ρ̄     = M_obs / (4π/3 * R^3)                     # 5429.3 kg/m³

    # ── solve (R_core, ρ_core) from M and I ──────────────────────────────────
    # Both constraints are linear in ρ_core at fixed geometry, so eliminating it
    # leaves x_c² = B/A in closed form. An inner core of *fixed* radius enters
    # only through its density excess Δρ_ic, which keeps that closed form.
    xm  = 1 - h_crust_km*1e3/R                       # mantle top
    xic = r_inner_km*1e3/R
    A = ρ̄          - Δρ_ic*xic^3 - ρ_mantle*xm^3 - ρ_crust*(1 - xm^3)   # = x_c³(ρ_f − ρ_m)
    B = 2.5*I_obs*ρ̄ - Δρ_ic*xic^5 - ρ_mantle*xm^5 - ρ_crust*(1 - xm^5)  # = x_c⁵(ρ_f − ρ_m)
    (A > 0 && B > 0) || throw(ArgumentError(
        "Mercury: no core denser than the mantle satisfies M and C/MR² = $Cnd " *
        "with ρ_mantle = $ρ_mantle kg/m³."))
    xc = sqrt(B/A)
    ρ_f = ρ_mantle + A/xc^3                          # liquid core density

    # The closed form returns a core above the mantle top for a moment target near
    # the uniform-sphere 0.4, so the geometry is checked here.
    xc < xm || throw(ArgumentError(
        "Mercury: C/MR² = $Cnd with ρ_mantle = $ρ_mantle needs a core of radius " *
        "$(round(xc*R/1e3, digits=1)) km, at or above the base of the crust " *
        "($(round(xm*R/1e3, digits=1)) km). The body is not that centrally condensed."))
    xic < xc || throw(ArgumentError(
        "Mercury: r_inner_km = $r_inner_km lies at or above the solved core radius " *
        "$(round(xc*R/1e3, digits=1)) km."))
    ρ_mantle < ρ_f || throw(ArgumentError(
        "Mercury: C/MR² = $Cnd with ρ_mantle = $ρ_mantle gives a core lighter than " *
        "the mantle (ρ_core = $(round(ρ_f, digits=0)) kg/m³), a density inversion."))
    if !(6000 ≤ ρ_f ≤ 8000)
        @warn "Mercury: C/MR² = $Cnd requires ρ_core = $(round(ρ_f, digits=0)) kg/m³, \
outside the 6000–8000 range liquid Fe-S can span at Mercury's core pressures. Pass \
r_inner_km to seat a solid inner core instead." maxlog = 1
    end

    # ── layers ───────────────────────────────────────────────────────────────
    # κ: liquid Fe-S core and solid inner core from Fe-alloy equations of state at
    # 5–40 GPa; mantle and crust from silicate seismic velocities.
    if xic > 0
        xa   = (0.0,  xic, xc,  xm,  1.0)
        ρ_SI = (ρ_f + Δρ_ic, ρ_f, ρ_mantle, ρ_crust)
        κ_SI = (170.0e9, 130.0e9, 125.0e9,  55.0e9)
        μ_SI = ( 90.0e9,   0.0,   μ_mantle, 30.0e9)
        ns   = [n_core ÷ 2, n_core, n_mantle, n_crust]
    else
        xa   = (0.0,  xc,  xm,  1.0)
        ρ_SI = (ρ_f, ρ_mantle, ρ_crust)
        κ_SI = (130.0e9, 125.0e9, 55.0e9)
        μ_SI = (  0.0,   μ_mantle, 30.0e9)
        ns   = [n_core, n_mantle, n_crust]
    end
    nL     = length(ρ_SI)
    doms   = [xa[i]..xa[i+1] for i in 1:nL]
    ρ̄_chk  = sum(ρ_SI[i] * (xa[i+1]^3 - xa[i]^3) for i in 1:nL)
    p_unit = (4π * G_SI / 3) * ρ̄_chk^2 * R^2

    build_model("Mercury", doms, [[ρ_SI[i] / ρ̄_chk] for i in 1:nL],
                [let v = κ_SI[i]/p_unit; r -> v end for i in 1:nL],
                [let v = μ_SI[i]/p_unit; r -> v end for i in 1:nL], ns;
                R_SI = R, ρ̄_SI = ρ̄_chk,
                # constant ρ per layer ⇒ ρ₀′ = 0, so Adams-Williamson is wrong in
                # the fluid core (and would give κ→0 at r=0 when it reaches the centre)
                aw = [μ_SI[i] != 0.0 for i in 1:nL],
                Ω_SI = 2π / (58.646 * 86400))        # 3:2 spin-orbit resonance
end
