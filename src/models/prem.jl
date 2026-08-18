# ── PREM Earth (Dziewonski & Anderson 1981, Table I) ──────────────────────────
# ρ in g/cm³, Vp/Vs in km/s, x = r/a, a = 6371 km. Moduli come from the velocities,
# κ = ρ(Vp²−4Vs²/3) and μ = ρVs², each ÷ p_unit (≈345 GPa) into solver units.
# Density is the single source (the coefficient vectors); velocities feed only κ, μ.

"""
    PREM(; use_aw_oc=false, anelastic=false, ocean=false) -> PlanetModel

Earth from PREM (Dziewonski & Anderson 1981): inner core `layers[1]`, fluid outer
core `layers[2]`, the mantle regions, and the crust. Defaults to the raw published
model — tabulated moduli, 1 s elastic reference, solid (continentalized) surface.

- `use_aw_oc` — replace the tabulated outer-core `κ` by the Adams-Williamson
  closure `κ = −ρ²g/Dρ`, enforcing neutral stratification `N² = 0` (no spurious
  g-modes). Use it for free-mode spectra.
- `anelastic` — apply PREM's `Q_μ` dispersion from the 1 s reference to the
  semi-diurnal tidal band, softening `μ` (matches MINEOS-PREM periods).
- `ocean` — keep PREM's 3 km fluid ocean as the top layer (`Vs = 0`). The surface
  is then fluid, so tangential `l` is ill-posed (see `love_numbers`); the default
  continentalizes it into solid crust.
"""
function PREM(; use_aw_oc::Bool = false, anelastic::Bool = false, ocean::Bool = false)
    a = 6371.0
    r_ICB  = 1221.5/a; r_CMB = 3480.0/a; r_3630 = 3630.0/a; r_5600 = 5600.0/a
    r_5701 = 5701.0/a; r_5771 = 5771.0/a; r_5971 = 5971.0/a; r_6151 = 6151.0/a
    r_6291 = 6291.0/a; r_Moho = 6346.6/a; r_LC = 6356.0/a; r_UC = 6368.0/a

    # ── region density coefficients (g/cm³, polynomials in x) ─────────────────
    c_IC  = [13.0885, 0.0, -8.8381]
    c_OC  = [12.5815, -1.2638, -3.6426, -5.5281]
    c_Dpp = [7.9565, -6.4761, 5.5283, -3.0807]           # shared by all three lower-mantle shells
    c_TZd = [5.3197, -1.4836]
    c_TZs = [11.2494, -8.0298]
    c_UM  = [7.1089, -3.8045]
    c_Lid = [2.6910, 0.6924]
    ρ(c) = r -> evalpoly(r, c)                           # density polynomial → callable

    # ── region velocities (km/s, polynomials in x) ───────────────────────────
    Vp_IC(r)  = 11.2622 - 6.3640r^2;   Vs_IC(r)  = 3.6678 - 4.4475r^2
    Vp_OC(r)  = 11.0487 - 4.0362r + 4.8023r^2 - 13.5732r^3          # Vs = 0 (fluid)
    Vp_Dpp(r) = 15.3891 - 5.3181r + 5.5242r^2 - 2.5514r^3
    Vs_Dpp(r) = 6.9254 + 1.4672r - 2.0834r^2 + 0.9783r^3
    Vp_LM(r)  = 24.9520 - 40.4673r + 51.4832r^2 - 26.6419r^3
    Vs_LM(r)  = 11.1671 - 13.7818r + 17.4575r^2 - 9.2777r^3
    Vp_LMb(r) = 29.2766 - 23.6027r + 5.5242r^2 - 2.5514r^3
    Vs_LMb(r) = 22.3459 - 17.2473r - 2.0834r^2 + 0.9783r^3
    Vp_TZd(r) = 19.0957 - 9.8672r;     Vs_TZd(r) = 9.9839 - 4.9324r
    Vp_TZs(r) = 39.7027 - 32.6166r;    Vs_TZs(r) = 22.3512 - 18.5856r
    Vp_UM(r)  = 20.3926 - 12.2569r;    Vs_UM(r)  = 8.9496 - 4.4597r
    # 6151–6346.6 km (24.4–220 km depth) is transversely isotropic in PREM; these are
    # Table I's effective-isotropic approximation (the footnote). Split below into
    # LVZ (Q_μ=80) and LID (Q_μ=600) — identical velocities, different attenuation.
    Vp_Lid(r) = 4.1875 + 3.9382r;      Vs_Lid(r) = 2.1519 + 2.3481r

    # ── moduli in solver units (÷345 GPa); anelastic softens μ via Q_μ ────────
    ω_tide = 2π/(12*3600); ω_ref = 2π
    fμ(Q) = anelastic ? 1 + (2/π)*log(ω_tide/ω_ref)/Q : 1.0        # 1 s → tidal-band
    κ_of(ρc, Vp, Vs) = r -> ρc(r)*(Vp(r)^2 - 4Vs(r)^2/3)/345
    μ_of(ρc, Vs, Q)  = (f = fμ(Q); r -> ρc(r)*Vs(r)^2/345 * f)

    # ── regions inner core → LID.  PREM Q_μ: IC 84.6; lower mantle (all of it, to
    #    5701 km) 312; transition zone / upper mantle 143; LVZ 80, LID 600; crust 600.
    domains = [0.0..r_ICB, r_ICB..r_CMB, r_CMB..r_3630, r_3630..r_5600, r_5600..r_5701,
               r_5701..r_5771, r_5771..r_5971, r_5971..r_6151, r_6151..r_6291, r_6291..r_Moho]
    cphys   = [c_IC, c_OC, c_Dpp, c_Dpp, c_Dpp, c_TZd, c_TZs, c_UM, c_Lid, c_Lid]
    # n_IC = 64: the standard fluid formulation's error is erratic, not convergent,
    # in the core pair (n_IC, n_OC). Measured against `dahlenize`, 64 leaves ~1e-12
    # in h where 60 leaves ~3e-7.
    n       = [64, 120, 30, 60, 30, 20, 30, 30, 20, 20]
    κ = Function[κ_of(ρ(c_IC), Vp_IC, Vs_IC), κ_of(ρ(c_OC), Vp_OC, _ -> 0.0),  # OC: tabulated; AW below
                 κ_of(ρ(c_Dpp), Vp_Dpp, Vs_Dpp), κ_of(ρ(c_Dpp), Vp_LM, Vs_LM),
                 κ_of(ρ(c_Dpp), Vp_LMb, Vs_LMb), κ_of(ρ(c_TZd), Vp_TZd, Vs_TZd),
                 κ_of(ρ(c_TZs), Vp_TZs, Vs_TZs), κ_of(ρ(c_UM), Vp_UM, Vs_UM),
                 κ_of(ρ(c_Lid), Vp_Lid, Vs_Lid), κ_of(ρ(c_Lid), Vp_Lid, Vs_Lid)]   # LVZ, LID (same κ)
    μ = Function[μ_of(ρ(c_IC), Vs_IC, 84.6), _ -> 0.0,
                 μ_of(ρ(c_Dpp), Vs_Dpp, 312.0), μ_of(ρ(c_Dpp), Vs_LM, 312.0),
                 μ_of(ρ(c_Dpp), Vs_LMb, 312.0), μ_of(ρ(c_TZd), Vs_TZd, 143.0),
                 μ_of(ρ(c_TZs), Vs_TZs, 143.0), μ_of(ρ(c_UM), Vs_UM, 143.0),
                 μ_of(ρ(c_Lid), Vs_Lid, 80.0), μ_of(ρ(c_Lid), Vs_Lid, 600.0)]      # LVZ 80, LID 600

    # ── crust: two constant shells (Table I), + optional fluid ocean ──────────
    shell!(dom, ρ0, Vp0, Vs0) = (push!(domains, dom); push!(cphys, [ρ0]); push!(n, 15);
                                 push!(κ, κ_of(_ -> ρ0, _ -> Vp0, _ -> Vs0));
                                 push!(μ, μ_of(_ -> ρ0, _ -> Vs0, 600.0)))
    shell!(r_Moho..r_LC, 2.900, 6.800, 3.900)                       # lower crust
    if ocean
        shell!(r_LC..r_UC, 2.600, 5.800, 3.200)                     # upper crust
        push!(domains, r_UC..1.0); push!(cphys, [1.020]); push!(n, 15)          # ocean (fluid, Vs=0)
        push!(κ, κ_of(_ -> 1.020, _ -> 1.450, _ -> 0.0)); push!(μ, _ -> 0.0)
    else
        shell!(r_LC..1.0, 2.600, 5.800, 3.200)                      # upper crust → surface
    end

    # ── mean-normalise density (3∑∫ρr²dr = 1 ⇒ g(1)=1), then self-gravity ─────
    ρ_mean = 5.515
    c = [cp ./ ρ_mean for cp in cphys]
    ρ_norm = 3 * sum(poly_r2_integral(c[i], leftendpoint(domains[i]), rightendpoint(domains[i]))
                     for i in eachindex(domains))
    dens = [ci ./ ρ_norm for ci in c]
    gs   = self_gravity(dens, domains)

    if use_aw_oc                                                    # κ = −ρ²g/Dρ, already in solver units
        dc_OC = [j * c[2][j+1] for j in 1:length(c[2])-1]           # d(ρ/ρ_mean)/dr coefficients
        ρ_OC(r)  = evalpoly(r, dens[2])
        Dρ_OC(r) = evalpoly(r, dc_OC) / ρ_norm
        κ[2] = r -> -ρ_OC(r)^2 * gs[2](r) / Dρ_OC(r)
    end

    # Fluid Poisson closure. Adams-Williamson presumes ρ₀′ = −ρ₀²g/κ, which holds for
    # the outer core only when `use_aw_oc` actually installed that κ; the tabulated
    # core is stratified, and the ocean has constant ρ (ρ₀′ = 0) against a soft
    # κ ≈ 2.1 GPa, where AW would invent a large spurious stratification.
    aw = fill(true, length(domains))
    aw[2] = use_aw_oc
    ocean && (aw[end] = false)

    build_model("PREM", domains, dens, κ, μ, n; R_SI = a*1e3, ρ̄_SI = ρ_mean*1e3,
                aw = aw, Ω_SI = Ω_EARTH)
end
