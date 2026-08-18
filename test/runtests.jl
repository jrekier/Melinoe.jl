using Melinoe
using Melinoe.Models          # PREM, Mars, Ganymede, Kelvin (qualified submodule)
using ApproxFun: Fun, Chebyshev
using LinearAlgebra: eigen, norm, Diagonal
using Test

# Regression suite, in order of authority: analytic, then self-consistency checks
# that need no external reference, then published comparisons at their own
# tolerances. Baselines are stated to the digits the solver reproduces.

prem() = PREM(use_aw_oc = true, anelastic = true)   # AW outer core, anelastic μ

@testset "Melinoe" begin

# ══ 1. Analytic benchmarks ═══════════════════════════════════════════════════

    @testset "Kelvin analytic Love numbers" begin
        ln = love_numbers(Kelvin())        # ℓ=2, incompressible fluid
        @test -ln.h ≈ 2.5 atol = 1e-9      # h₂ = 5/2
        @test  ln.k ≈ 1.5 atol = 1e-9      # k₂ = 3/2
    end

    @testset "Kelvin l₂: the two non-commuting limits, and NaN between them" begin
        # h and k are pinned however you approach the fluid sphere; l is not.
        # (a) static + fluid: nothing pins V(R), so `l` must be NaN, not noise.
        for n in (40, 60, 80, 100)
            ln = love_numbers(Kelvin(n = n))
            @test -ln.h ≈ 2.5 atol = 1e-9
            @test ln.k  ≈ 1.5 atol = 1e-9
            @test isnan(ln.l)
        end
        @test isnan(love_numbers(dahlenize(Kelvin())).l)

        # (b) solid, μ̃ → 0: shear traction pins it ⇒ (3/4)/(1+μ̃).
        # l is the worst conditioned of the three and degrades another order of
        # magnitude as μ̃ → 0, where the problem is nearly degenerate — hence the
        # looser bound there. All three also move by ~10× that error between BLAS
        # implementations, so these tolerances leave room for the platform spread.
        for μ̃ in (1.0, 1e-2, 1e-4)
            ln = love_numbers(Kelvin(μ̃ = μ̃, κ̃ = 1e8))
            @test -ln.h * (1 + μ̃) ≈ 2.50 atol = 1e-4
            @test -ln.l * (1 + μ̃) ≈ 0.75 atol = (μ̃ ≥ 1e-2 ? 1e-4 : 5e-3)
            @test  ln.k * (1 + μ̃) ≈ 1.50 atol = 1e-4
        end

        # (c) fluid, ω² → 0⁺: inertia pins it ⇒ 5/4. A different limit from (b).
        for ω² in (1e-5, 1e-7)
            ln = love_numbers(Kelvin().layers, 2; ω²)
            @test -ln.h ≈ 2.50 atol = 1e-3
            @test -ln.l ≈ 1.25 atol = 1e-3
            @test  ln.k ≈ 1.50 atol = 1e-3
        end
    end

    @testset "Kelvin f-mode" begin
        # analytic ℓ=2 f-mode: ω² = 2ℓ(ℓ-1)/(2ℓ+1) = 4/5
        @test minimum(abs.(free_modes(Kelvin()).ωs.^2 .- 0.8)) < 5e-3
    end

    @testset "Exact polynomial integrals" begin
        c = [1.5, -0.3, 0.2]
        @test poly_r2_integral(c, 0.2, 0.8) ≈
              sum(Fun(r -> evalpoly(r, c) * r^2, Chebyshev(0.2..0.8))) atol = 1e-15
        @test poly_r4_integral(c, 0.2, 0.8) ≈
              sum(Fun(r -> evalpoly(r, c) * r^4, Chebyshev(0.2..0.8))) atol = 1e-15
        # mean-normalised density (3∫ρr²dr = 1) ⇒ g(1) = 1, by construction
        @test self_gravity([[1.0]], [0.0..1.0])[1](1.0) ≈ 1.0 atol = 1e-14
        @test self_gravity([[1.0]], [0.0..1.0])[1](0.0) == 0.0     # no 0/0 at the centre
    end

    @testset "background_potential is the antiderivative of g₀" begin
        # φ₀ is not used by the elastic solve, so nothing else would catch a sign
        # or constant error in it: differentiate it back and compare to g₀.
        φ = Melinoe.background_potential([[1.0]], [0.0..1.0])[1]
        g = Kelvin().layers[1].g₀
        @test maximum(abs((φ(r+1e-5) - φ(r-1e-5))/2e-5 - g(r)) for r in 0.1:0.1:0.9) < 1e-9
    end

# ══ 2. Operator verification (manufactured solution) ═════════════════════════

    @testset "Manufactured-profile operator verification (MMS)" begin
        # Non-trivial polynomial ρ, κ, μ: solve, then substitute U, V, δφ back into
        # HAND-WRITTEN copies of the ×r operators. Residual vs the tidal RHS must be
        # machine zero, verifying operators + assembly + forcing against no reference.
        # These transcriptions are the authoritative statement of the equation forms.
        ℓ = 2
        ρ_fn(r)  = 1.5 - 0.3r + 0.2r^2;  Dρ_fn(r) = -0.3 + 0.4r
        μ_fn(r)  = 0.5 + 0.1r^2;         Dμ_fn(r) = 0.2r
        κ_fn(r)  = 2.0 - 0.4r;           Dκ_fn(r) = -0.4
        g_fn(r)  = r == 0.0 ? 0.0 : 3*(1.5r^3/3 - 0.3r^4/4 + 0.2r^5/5)/r^2
        lay = Layer(domain = 0.0..1.0, n = 60, κ = κ_fn, μ = μ_fn, ρ₀ = ρ_fn, g₀ = g_fn)
        Amat, _, ops, ns, jrhs = assemble_planet([lay], ℓ)
        x = Amat \ potential_forcing(ops, ns; ℓ, jrhs)
        n = ns[1]; SL = ops[1].S
        U = Fun(SL, x[1:n]); V = Fun(SL, x[n+1:2n]); δφ = Fun(SL, x[2n+1:3n])
        DU, DDU = U', U''; DV, DDV = V', V''; Dφ, DDφ = δφ', δφ''
        L_U(r) = begin
            ρ, μ, κ = ρ_fn(r), μ_fn(r), κ_fn(r); Dμ, Dκ = Dμ_fn(r), Dκ_fn(r); rg = r^2*g_fn(r)
            -(3κ+4μ)*r^3/3*DDU(r) + (-r^2*(6κ+8μ+3r*Dκ+4r*Dμ)/3)*DU(r) +
            (r*(2κ+(8/3+ℓ+ℓ^2)*μ-2r*Dκ+(4/3)*r*Dμ) - (4ρ*rg - 3ρ^2*r^3))*U(r) +
            (r^2*ℓ*(1+ℓ)*(3κ+μ)/3)*DV(r) +
            (-ℓ*(1+ℓ)/3*(r*(3κ+7μ)+r^2*(-3Dκ+2Dμ)) + ℓ*(1+ℓ)*ρ*rg)*V(r) + r^3*ρ*Dφ(r)
        end
        L_V(r) = begin
            ρ, μ, κ = ρ_fn(r), μ_fn(r), κ_fn(r); Dμ = Dμ_fn(r); rg = r^2*g_fn(r)
            (-r^2*(3κ+μ)/3)*DU(r) + (r*(-2κ-(8/3)*μ-r*Dμ) + ρ*rg)*U(r) +
            -r^3*μ*DDV(r) + (-r^2*(2μ+r*Dμ))*DV(r) +
            r*(ℓ*(1+ℓ)*(κ+(4/3)*μ)+r*Dμ)*V(r) + r^2*ρ*δφ(r)
        end
        L_φ(r) = 3r^2*ρ_fn(r)*DU(r) + 3r*(2ρ_fn(r)+r*Dρ_fn(r))*U(r) -
                 3r*ℓ*(1+ℓ)*ρ_fn(r)*V(r) + r^2*DDφ(r) + 2r*Dφ(r) - ℓ*(1+ℓ)*δφ(r)
        for r in (0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95)
            @test abs(L_U(r) + ℓ*ρ_fn(r)*r^(ℓ+2)) < 1e-12   # = −f_U_tide
            @test abs(L_V(r) + ρ_fn(r)*r^(ℓ+2))   < 1e-12   # = −f_V_tide
            @test abs(L_φ(r))                      < 1e-12
        end
    end

# ══ 3. The composable `forced` API ═══════════════════════════════════════════

    @testset "forced/readers reproduce love_numbers field by field" begin
        m = prem(); d = forced(m, Tide()); ln = love_numbers(m)
        @test radial(d, 1.0)     ≈ ln.h atol = 1e-14
        @test tangential(d, 1.0) ≈ ln.l atol = 1e-14
        @test potential(d, 1.0)  ≈ ln.k atol = 1e-14
        @test surface_potential(d) ≈ ln.k atol = 1e-14
        @test read_love(d).h ≈ ln.h atol = 1e-14
        # the callable and sampled forms are the same function
        @test radial(d)(0.7) == radial(d, 0.7)
        # out of range must throw, not extrapolate a layer polynomial silently
        @test_throws DomainError radial(d, 1.5)
        @test_throws DomainError radial(d, -0.1)
        @test_throws DomainError radial_profile(m, :ρ₀)(1.5)
    end

    @testset "Forcings compose linearly and by construction" begin
        m = prem()
        Ω̂² = (rotation_rate(m) / m.ω_unit)^2
        tide = forced(m, Tide());  cent = forced(m, Centrifugal())
        both = forced(m, Tide() + Centrifugal())
        # one solve of the sum == sum of the solves (the operator is linear).
        # Relative, not absolute: the two sides are separate solves, so they differ
        # by conditioning × eps — ~1e-9 relative, and that varies across platforms.
        @test surface_potential(both) ≈
              surface_potential(tide) + surface_potential(cent) rtol = 1e-8
        # Centrifugal is exactly Potential(r²) at amplitude Ω̂²/3
        @test surface_potential(cent) ==
              surface_potential(forced(m, Potential(r -> r^2; amplitude = Ω̂²/3)))
        # a layer mask really restricts the forcing
        @test surface_potential(forced(m, Centrifugal(layers = :oc))) !=
              surface_potential(cent)
        @test_throws ErrorException forced(m, Centrifugal(layers = :nonsense))
    end

    @testset "moment: telescoping integral matches the compliance definition" begin
        m = prem(); Ω̂² = (rotation_rate(m) / m.ω_unit)^2
        fs  = forced(m, Potential(r -> r^2; amplitude = Ω̂²/3))
        Â_f = sum(Fun(r -> m.layers[2].ρ₀(r) * r^4, Chebyshev(m.layers[2].domain)))
        c   = compliances(m.layers, 2; Ω̂², core_layer = 2)
        @test moment(fs, 2) / ((8π/3) * Â_f) ≈ c.γ atol = 1e-15
        @test moment(fs, :whole) ≈ (4π/3) * surface_potential(fs) atol = 1e-15
    end

# ══ 4. Formulation invariances (self-consistency) ════════════════════════════

    @testset "compliances: layers form == PlanetModel form" begin
        # Two independent implementations: direct F(r)-telescoping, and forced/moment.
        m = prem(); Ω̂² = (rotation_rate(m) / m.ω_unit)^2
        a = compliances(m.layers, 2; Ω̂², core_layer = 2)
        b = compliances(m; core_layer = 2)
        for k in (:κ, :ξ, :γ, :β, :k₂)
            @test getfield(a, k) ≈ getfield(b, k) rtol = 1e-12
        end
    end

    @testset "Non-AW fluid operator (self-consistency on PREM outer core)" begin
        # PREM's κ_OC = −ρ²g/Dρ is exactly Adams-Williamson, so forcing the outer
        # core to the non-AW (raw ρ₀′) operator must reproduce the AW result.
        m = prem(); Ω̂² = (7.2921e-5 / m.ω_unit)^2
        oc = m.layers[2]
        ls = copy(m.layers)
        ls[2] = Layer(domain=oc.domain, n=oc.n, κ=oc.κ, μ=oc.μ, ρ₀=oc.ρ₀, g₀=oc.g₀, aw=false)
        c0 = compliances(m.layers, 2; Ω̂², core_layer=2)
        c1 = compliances(ls,        2; Ω̂², core_layer=2)
        @test c0.γ ≈ c1.γ atol = 1e-9
        @test c0.β ≈ c1.β atol = 1e-9
        @test love_numbers(m.layers, 2).k ≈ love_numbers(ls, 2).k atol = 1e-8
    end

    @testset "Dahlen-reduced fluid == standard, and restores Betti" begin
        # On a neutral (AW, N²=0) core the two formulations are the same physics,
        # so they must agree to solver precision — two different assemblies, so
        # ~1e-9 relative, not machine epsilon.
        m = prem(); md = dahlenize(m)
        ln, lnd = love_numbers(m), love_numbers(md)
        @test lnd.h ≈ ln.h rtol = 1e-8
        @test lnd.k ≈ ln.k rtol = 1e-8
        # ...and the Dahlen form is well posed at a solid wall, so its reciprocity
        # residual sits at machine precision rather than the Longman floor.
        cd = compliances(md; core_layer = 2)
        @test cd.γ ≈ 1.984454e-3 rtol = 1e-4
        @test abs(cd.betti) < 1e-12
        # Mars: stratified (non-AW) core, where the Longman floor was ~2.7e-5.
        cmd = compliances(dahlenize(Mars()); core_layer = 1, Ω_SI = 7.088e-5)
        @test cmd.γ ≈ 1.523817e-3 rtol = 0.01
        @test abs(cmd.betti) < 1e-10
        # the Dahlen closure drops fluid inertia, so an eigensolve is meaningless
        @test_throws ArgumentError free_modes(md)
    end

    @testset "PREM(ocean=true): free ocean surface sits on the geoid" begin
        mo = PREM(ocean = true)
        @test is_fluid(mo.layers[end])                # the only fluid-surface path
        ln = love_numbers(mo)
        @test isnan(ln.l)                             # undetermined, must not be faked
        # an equilibrium ocean surface is an equipotential of (tide + δφ), so the
        # displacement is (Φ_T(R) + δφ(R))/g(R) = 1 + k in these units
        @test -ln.h ≈ 1 + ln.k rtol = 1e-3
    end

# ══ 5. Published references ══════════════════════════════════════════════════

    @testset "PREM static Love numbers" begin
        ln = love_numbers(prem())
        @test -ln.h ≈ 0.6107814 atol = 1e-6   # vs PREM published 0.6032 (+1.3%)
        @test -ln.l ≈ 0.0859434 atol = 1e-6
        @test  ln.k ≈ 0.3022730 atol = 1e-6
    end

    @testset "PREM Sasao compliances + reciprocity" begin
        m = prem(); c = compliances(m; core_layer = 2)
        @test c.κ ≈ 1.050074e-3 atol = 5e-7   # reference (own research): 1.039e-3
        @test c.ξ ≈ 2.241380e-4 atol = 5e-7   #                            0.222e-3
        @test c.γ ≈ 1.984454e-3 atol = 5e-6   #                            1.965e-3
        @test c.β ≈ 6.226327e-4 atol = 5e-7   #                            0.616e-3
        # Â_tot·ξ = Â_f·γ (Saito eq. 58). The standard formulation's residual is
        # conditioning-limited, not exact — contrast the dahlenized 1e-12 above.
        @test abs(c.betti) < 1e-7
        @test c.k₂ ≈ 3 * c.δφ_s1 / (rotation_rate(m)/m.ω_unit)^2 atol = 1e-14
    end

    @testset "PREM ℓ=2 spheroidal modes vs MINEOS" begin
        fm = free_modes(prem())
        mineos = [53.89, 24.52, 17.77, 15.07, 9.68, 7.97, 6.91, 6.62]
        for (n, Tref) in enumerate(mineos)
            @test abs(fm.T_min[n] - Tref) / Tref < 0.02   # within 2%
        end
        @test issorted(fm.ωs)                             # ascending, as documented
        @test length(fm.T_min) == length(fm.ωs)
    end

    @testset "Mars (InSight-constrained) vs the observations it was not fitted to" begin
        m = Mars()
        @test m.layers[1].aw == false                 # fluid core to the centre
        @test m.R_SI ≈ 3389.5e3                       # IAU mean radius
        @test m.ρ̄_SI ≈ 3934.1 atol = 0.5              # observed mean density
        # geometry, straight from the papers
        @test rightendpoint(m.layers[1].domain)*m.R_SI/1e3 ≈ 1830 atol = 40   # Stähler+ 2021
        @test (1 - rightendpoint(m.layers[2].domain))*m.R_SI/1e3 ≈ 45 atol = 1  # 24–72 km
        # core density is an OUTPUT of the M + I fit, and must land in the
        # seismically inferred range 5700–6300 kg/m³ (Stähler+ 2021)
        ρ_core = 3*sum(Fun(r -> m.layers[1].ρ₀(r)*r^2, Chebyshev(m.layers[1].domain))) /
                 rightendpoint(m.layers[1].domain)^3 * m.ρ̄_SI
        @test 5700 ≤ ρ_core ≤ 6300
        # k₂ is left free by the fit, so it is an independent test of the moduli.
        # Published determinations disagree by more than their errors: 0.169 ± 0.006
        # (Konopliv+ 2016), 0.1697 ± 0.0009 (Genova+ 2016), 0.174 ± 0.008
        # (Konopliv+ 2020). Compared against the most recent.
        @test love_numbers(m).k ≈ 0.17835 atol = 1e-4              # locked in
        @test abs(love_numbers(m).k - 0.174) < 0.008               # within 1σ
        c = compliances(m; core_layer = 1, Ω_SI = 7.088e-5)
        @test isfinite(c.γ) && isfinite(c.β)          # F(c=0) finite (no NaN)
        @test c.γ ≈ 1.524e-3 rtol = 0.01              # Le Maistre S8 @1830 ~1.5e-3
        @test c.β ≈ 3.739e-4 rtol = 0.01              #                     ~0.36e-3
        @test abs(c.betti) < 1e-3                     # reciprocity
    end

    @testset "PREM pressure Love numbers vs Dumberry & Bloxham 2004 Table 2" begin
        # Unit pressure at the CMB (top of the outer core), elastic PREM. Their
        # published k_n, h_n for n = 2, 4, 6, 8 — four degrees over 2.5 decades,
        # matched with no free normalisation.
        DB = Dict(2 => (1.116e-1, 2.302e-1), 4 => (1.156e-2, 5.135e-2),
                  6 => (1.957e-3, 1.366e-2), 8 => (4.171e-4, 4.013e-3))
        m = PREM()                                    # elastic; anelastic drifts n ≥ 4 by 1.2%
        for n in (2, 4, 6, 8)
            p = pressure_love(m; ℓ = n, forcing_layer = 2)
            k_ref, h_ref = DB[n]
            @test p.kP ≈ k_ref rtol = 0.02
            @test p.hP ≈ h_ref rtol = 0.01
        end
        # n ≥ 4 is much tighter than the degree-2 whole-body response
        for n in (4, 6, 8)
            p = pressure_love(m; ℓ = n, forcing_layer = 2)
            @test p.kP ≈ DB[n][1] rtol = 0.0025
            @test p.hP ≈ DB[n][2] rtol = 0.0010
        end
    end

    @testset "Ganymede pressure Love k_P (Dumberry & Bloxham 2004 formulation)" begin
        g  = Ganymede(h_ice_km = 100.0)
        kp = pressure_love(g; forcing_layer = 4)                            # the ocean
        @test kp.kP ≈ 0.7672273 atol = 1e-6       # regression lock, not a published value
        @test kp.hP ≈ 2.3124562 atol = 1e-6       #   ""
        @test kp.kP == -kp.δφR                    # k_P is defined as −δφ(R)

        # the forcing layer is frozen, so its κ never enters the system
        ls = copy(g.layers)
        ls[4] = Layer(domain = ls[4].domain, n = ls[4].n, κ = r -> 2.0e9 / g.p_unit,
                      μ = ls[4].μ, ρ₀ = ls[4].ρ₀, g₀ = ls[4].g₀, aw = ls[4].aw,
                      dahlen = ls[4].dahlen, ν_SI = ls[4].ν_SI)
        @test pressure_love(ls; forcing_layer = 4).kP === kp.kP

        # C₅ couples the frozen layer's density into both interface rows, so k_P must
        # respond to it; if it does not, the ρ_f terms have been dropped (D&B eq. 41)
        ls2 = copy(g.layers)
        ls2[4] = Layer(domain = ls2[4].domain, n = ls2[4].n, κ = ls2[4].κ, μ = ls2[4].μ,
                       ρ₀ = r -> 1.05 * g.layers[4].ρ₀(r), g₀ = ls2[4].g₀,
                       aw = ls2[4].aw, dahlen = ls2[4].dahlen, ν_SI = ls2[4].ν_SI)
        @test !isapprox(pressure_love(ls2; forcing_layer = 4).kP, kp.kP; rtol = 1e-3)
        # the forcing layer must be a fluid sandwiched between solids
        @test_throws AssertionError pressure_love(Ganymede(); forcing_layer = 2)
        # tidal Love numbers land inside the spread Macrì & Casotto 2025 (A&A 699,
        # A261) Fig. A.1 find for Ganymede models with a subsurface ocean
        ln = love_numbers(Ganymede())
        @test 0.25 < ln.k  < 0.60
        @test 1.0  < -ln.h < 1.60
        # rigidities from Hussmann et al. 2016, CMDA 126, 131
        @test Ganymede().layers[2].μ(0.5) * Ganymede().p_unit ≈ 50.0e9 rtol = 1e-9
        @test Ganymede().layers[3].μ(0.8) * Ganymede().p_unit ≈  6.6e9 rtol = 1e-9
        @test Ganymede().layers[5].μ(0.99)* Ganymede().p_unit ≈  3.3e9 rtol = 1e-9
    end

    @testset "Bulk properties: mass and moment of inertia vs observation" begin
        # The two integral constraints every interior model must satisfy. For a
        # spherically symmetric body I = (8π/3)∫ρr⁴dr, so with the mean-normalised
        # density the MOI factor is just 2∫ρ̂x⁴dx.
        moi(m)  = 2 * sum(sum(Fun(r -> l.ρ₀(r)*r^4, Chebyshev(l.domain))) for l in m.layers)
        mass(m) = (4π/3) * m.R_SI^3 * m.ρ̄_SI
        @test moi(build_model("u", [0.0..1.0], [[1.0]], [r->Inf], [r->0.0], [20];
                              R_SI = 1.0, ρ̄_SI = 1.0)) ≈ 0.4 atol = 1e-12   # uniform sphere

        # PREM — M and I are what the model was built to fit
        @test mass(PREM()) ≈ 5.9722e24 rtol = 1e-3        # IERS
        @test moi(PREM())  ≈ 0.3307    rtol = 2e-3        # Earth MOI factor

        # Ganymede — M and C_nd from the Galileo gravity data (Anderson et al. 1996,
        # Nature 384, 541); radius and mean density from Baland & Van Hoolst 2010.
        # Both matched by construction.
        @test mass(Ganymede()) ≈ 1.482e23 rtol = 1e-6
        @test moi(Ganymede())  ≈ 0.3105   rtol = 1e-6
        @test Ganymede().R_SI ≈ 2631.2e3
        @test Ganymede().ρ̄_SI ≈ 1942 atol = 1
        # the dense liquid core is what makes it centrally condensed enough:
        # homogenising the interior would put C_nd at 0.354, 15σ high
        @test is_fluid(Ganymede().layers[1])
        @test Ganymede().layers[1].ρ₀(0.0) * Ganymede().ρ̄_SI > 5000

        # Mars — matched by construction: the core and mantle densities are scaled
        # to hit both. Konopliv et al. 2016 C/MR² = 0.363918 (polar, R = 3396 km)
        # → mean moment at the mean radius, 0.36401.
        @test mass(Mars()) ≈ 6.4171e23 rtol = 1e-6
        @test moi(Mars())  ≈ 0.36401   rtol = 1e-5
    end

# ══ 6. Model construction and I/O ════════════════════════════════════════════

    @testset "build_model: normalisation and derived metadata" begin
        # uniform sphere built by hand == Kelvin()
        m = build_model("unit", [0.0..1.0], [[1.0]], [r -> Inf], [r -> 0.0], [60];
                        R_SI = 6371.0e3, ρ̄_SI = 5515.0)
        @test m.layers[1].g₀(1.0) ≈ 1.0 atol = 1e-14      # mean-normalised ⇒ g(1) = 1
        @test m.ω_unit ≈ sqrt(4π * 6.674e-11 * 5515.0 / 3)
        @test m.p_unit ≈ (4π * 6.674e-11 / 3) * 5515.0^2 * (6371.0e3)^2
        @test m.Ω_SI == 0.0
        @test love_numbers(m).k ≈ love_numbers(Kelvin()).k atol = 1e-12
        @test length(layers_of(m)) == 1
        @test T_minutes(m, 1.0) ≈ 2π / m.ω_unit / 60
        @test occursin("unit", repr(m))
    end

    @testset "Model rotation metadata (Ω_SI defaults)" begin
        @test prem().Ω_SI ≈ 7.2921e-5
        @test Mars().Ω_SI ≈ 7.088e-5
        @test Kelvin().Ω_SI == 0.0
        @test rotation_rate(Kelvin()) ≈ 7.2921e-5        # Earth fallback for Ω_SI=0
        @test rotation_rate(prem(), 1e-4) == 1e-4        # explicit override wins
        # compliances default (model Ω) must equal the explicit-Earth call
        c0 = compliances(prem(); core_layer = 2)
        c1 = compliances(prem(); core_layer = 2, Ω_SI = 7.2921e-5)
        @test c0.γ == c1.γ && c0.β == c1.β
    end

    @testset "Layer/model predicates" begin
        m = prem()
        @test [is_fluid(l) for l in m.layers][1:3] == [false, true, false]
        @test is_incompressible(Kelvin().layers[1])
        @test !is_incompressible(m.layers[1])
        @test Melinoe.layer_at(m, 0.0) == 1
        @test Melinoe.layer_at(m, 1.0) == length(m.layers)
        # an interface belongs to the layer below it
        @test Melinoe.layer_at(m, rightendpoint(m.layers[1].domain)) == 1
        @test_throws DomainError Melinoe.layer_at(m, 1.5)
        @test_throws DomainError Melinoe.layer_at(m, NaN)
    end

    @testset "planet_coeffs (PREM: exact polynomial recovery, sane coefficients)" begin
        fits = planet_coeffs(prem())
        g(i, q) = fits[findfirst(f -> f.layer == i && f.q == q, fits)]
        ρ1 = g(1, "rho"); μ1 = g(1, "mu"); K2 = g(2, "K")
        # IC density: exact even polynomial, published ratio c₂/c₀ (unit-convention-free)
        @test length(ρ1.c) == 3 && ρ1.c[2] == 0.0
        @test ρ1.c[3]/ρ1.c[1] ≈ -8.8381/13.0885 rtol = 1e-9
        @test ρ1.err < 1e-9
        # IC μ = ρVs² products: degree 6, pure even (odd roundoff snapped), exact;
        # c₂/c₀ from the published polynomials (anelastic factor cancels in the ratio)
        @test length(μ1.c) == 7 && all(μ1.c[k] == 0.0 for k in (2, 4, 6))
        μ20 = (13.0885*(-2*3.6678*4.4475) - 8.8381*3.6678^2) / (13.0885*3.6678^2)
        @test μ1.c[3]/μ1.c[1] ≈ μ20 rtol = 1e-9
        @test μ1.err < 1e-9
        # AW outer-core K is rational: flagged as a fit, with BOUNDED coefficients
        @test 1e-9 < K2.err < 1e-3
        @test maximum(abs, K2.c) < 2e13          # no 1e15 monomial explosions
    end

    @testset "load_planet_csv on the published PREM table" begin
        # examples/PREM_1s.csv is real PREM as distributed: radius AND depth, km,
        # g/cm³, transversely isotropic vpv/vph/vsv/vsh + eta, Q columns we ignore,
        # rows running outward→inward, and no layer column — regions are marked by
        # repeated radii.
        f = joinpath(@__DIR__, "..", "examples", "PREM_1s.csv")
        m = load_planet_csv(f)
        @test length(m.layers) == 13
        @test m.R_SI ≈ 6371e3
        @test m.ρ̄_SI ≈ 5514.2 atol = 0.5              # PREM's own 5515, self-consistent
        # the discontinuities PREM actually has, recovered from the repeated radii
        edges = [rightendpoint(l.domain) * m.R_SI / 1e3 for l in m.layers]
        @test edges ≈ [1221.5, 3480.0, 3630.0, 5600.0, 5701.0, 5771.0, 5971.0,
                       6151.0, 6291.0, 6346.6, 6356.0, 6368.0, 6371.0] rtol = 1e-9
        # inner core solid, outer core fluid, ocean fluid, everything else solid
        @test findall(is_fluid, m.layers) == [2, 13]
        @test all(l -> !l.aw, m.layers[[2, 13]])       # fluids get the non-AW closure
        # ...and it reproduces the hand-coded model built from the same publication
        b = PREM(ocean = true)
        @test love_numbers(m).k ≈ love_numbers(b).k rtol = 1e-3
        @test love_numbers(m).h ≈ love_numbers(b).h rtol = 1e-3
        @test isnan(love_numbers(m).l)                 # fluid (ocean) surface
    end

    @testset "load_planet_csv: formats, units, and what it refuses" begin
        hdr = "layer,radius[unit=\"m\"],density[unit=\"kg/m^3\"],K[unit=\"Pa\"],mu[unit=\"Pa\"]"
        rows(lbl, r1, r2, ρ, K, μ) =
            [string(lbl, ",", r, ",", ρ, ",", K, ",", μ) for r in range(r1, r2, length = 25)]
        two = vcat(rows("core", 0, 3e6, 7000.0, 2e11, 0.0),
                   rows("mantle", 3e6, 6e6, 4000.0, 2e11, 1e11))
        write_tmp(ls) = (p = tempname()*".csv"; write(p, join(ls, "\n")); p)

        # moduli given directly, as here, are used as-is
        p = write_tmp([hdr; two]); m = load_planet_csv(p); rm(p)
        @test length(m.layers) == 2 && is_fluid(m.layers[1])
        @test m.layers[2].κ(0.9) * m.p_unit ≈ 2e11 rtol = 1e-6

        # the same body in km / g·cm⁻³ / GPa must give the same model
        hdr2 = "layer,radius[unit=\"km\"],density[unit=\"g/cm^3\"],K[unit=\"GPa\"],mu[unit=\"GPa\"]"
        two2 = vcat(rows("core", 0, 3e3, 7.0, 200.0, 0.0),
                    rows("mantle", 3e3, 6e3, 4.0, 200.0, 100.0))
        p = write_tmp([hdr2; two2]); m2 = load_planet_csv(p); rm(p)
        @test m2.R_SI ≈ m.R_SI && m2.ρ̄_SI ≈ m.ρ̄_SI
        @test love_numbers(m2).k ≈ love_numbers(m).k rtol = 1e-9

        # velocities instead of moduli: κ = ρ(vp²−4vs²/3), μ = ρvs²
        hdrv = "layer,radius[unit=\"m\"],density[unit=\"kg/m^3\"],vp[unit=\"m/s\"],vs[unit=\"m/s\"]"
        vp, vs = sqrt((2e11 + 4e11/3)/4000), sqrt(1e11/4000)
        vrows = vcat(rows("core", 0, 3e6, 7000.0, sqrt(2e11/7000), 0.0),
                     rows("mantle", 3e6, 6e6, 4000.0, vp, vs))
        p = write_tmp([hdrv; vrows]); mv = load_planet_csv(p); rm(p)
        @test mv.layers[2].κ(0.9) * mv.p_unit ≈ 2e11 rtol = 1e-6
        @test mv.layers[2].μ(0.9) * mv.p_unit ≈ 1e11 rtol = 1e-6

        # unrecognised columns are ignored, and depth substitutes for radius
        hdrq = "depth[unit=\"m\"],density[unit=\"kg/m^3\"],K[unit=\"Pa\"],mu[unit=\"Pa\"],Q-mu,Q-kappa"
        qrows = [string(6e6 - r, ",", 4000.0, ",", 2e11, ",", 1e11, ",", 312, ",", 57823)
                 for r in range(0, 6e6, length = 40)]
        p = write_tmp([hdrq; qrows]); mq = load_planet_csv(p); rm(p)
        @test mq.R_SI ≈ 6e6 && length(mq.layers) == 1

        # rotation survives write → load
        p = tempname()*".csv"; write_planet_csv(Mars(), p; n = 30)
        @test load_planet_csv(p).Ω_SI ≈ Mars().Ω_SI; rm(p)

        # ...and the malformed cases now fail loudly instead of loading something wrong
        p = write_tmp(two);                       @test_throws ArgumentError load_planet_csv(p); rm(p)   # no header
        p = write_tmp([hdr; two[1:25]; rows("mantle", 3.5e6, 6e6, 4000.0, 2e11, 1e11)...])
        @test_throws ArgumentError load_planet_csv(p); rm(p)                                            # gap
        p = write_tmp([hdr; rows("core", 1e6, 3e6, 7000.0, 2e11, 0.0)...;
                            rows("mantle", 3e6, 6e6, 4000.0, 2e11, 1e11)...])
        @test_throws ArgumentError load_planet_csv(p); rm(p)                                            # hollow centre
        p = write_tmp([hdr; "core,0.0,7000.0,2e11"]);  @test_throws ArgumentError load_planet_csv(p); rm(p)
        p = write_tmp([hdr; "core,0.0,nope,2e11,0.0"]); @test_throws ArgumentError load_planet_csv(p); rm(p)
        p = write_tmp([hdr]);                      @test_throws ArgumentError load_planet_csv(p); rm(p)
        # km values in a column declared as metres are caught by the radius sanity check
        p = write_tmp([hdr; rows("core", 0, 3e3, 7000.0, 2e11, 0.0)...])
        @test_throws ArgumentError load_planet_csv(p); rm(p)
    end

    @testset "CSV round-trip through both output layouts" begin
        # write_planet_csv emits either layout load_planet_csv reads, and the two
        # must describe the same model: :moduli carries K and mu in SI with an
        # explicit layer column, :velocity carries vp/vs in km/s with layers marked
        # by repeated radii, as published tables do.
        for mk in (Mars, PREM, () -> PREM(ocean = true), Ganymede, Kelvin)
            m0 = mk()
            got = map((:moduli, :velocity)) do fmt
                p = tempname() * ".csv"
                write_planet_csv(m0, p; n = 45, format = fmt)
                m1 = load_planet_csv(p); rm(p)
                @test length(m1.layers) == length(m0.layers)
                @test [is_fluid(l) for l in m1.layers] == [is_fluid(l) for l in m0.layers]
                # an incompressible layer is written as a literal Inf and read back
                # as incompressible — Kelvin is all of one, Ganymede is mixed
                @test [is_incompressible(l) for l in m1.layers] ==
                      [is_incompressible(l) for l in m0.layers]
                @test m1.ρ̄_SI ≈ m0.ρ̄_SI rtol = 1e-3    # sample → refit, not identical
                @test m1.Ω_SI ≈ m0.Ω_SI                 # rotation survives the trip
                love_numbers(dahlenize(m1))
            end
            ref = love_numbers(dahlenize(m0))
            for ln in got
                @test ln.k ≈ ref.k rtol = 2e-3          # tidal response preserved
                @test ln.h ≈ ref.h rtol = 2e-3
            end
            # the two layouts agree to their text precision: :velocity is written at
            # fixed decimals, like a published table, so it carries ~1e-9 relative
            @test got[1].k ≈ got[2].k rtol = 1e-7
            @test got[1].h ≈ got[2].h rtol = 1e-7
        end
        @test [is_fluid(l) for l in load_planet_csv(
                   write_planet_csv(Mars(), tempname()*".csv"; n = 45)).layers] ==
              [true, false, false]
        @test_throws ArgumentError write_planet_csv(Mars(), tempname()*".csv"; format = :nope)
        # ...but a layer may not be half incompressible
        p = tempname()*".csv"
        write(p, "layer,radius[unit=\"m\"],density[unit=\"kg/m^3\"],K[unit=\"Pa\"],mu[unit=\"Pa\"]\n" *
                 join(["a,$(r),3000.0,$(r < 3e6 ? "Inf" : "2e11"),0.0"
                       for r in range(0, 6e6, length = 20)], "\n"))
        @test_throws ArgumentError load_planet_csv(p); rm(p)
    end

    @testset "write_planet_coeffs renders a readable table" begin
        tmp = tempname() * ".txt"
        write_planet_coeffs(prem(), tmp)
        txt = read(tmp, String)
        @test occursin("PREM", txt)
        @test occursin("Layer 1", txt)
        @test occursin("rho =", txt) && occursin("mu  =", txt)
        @test occursin("fluid", txt)                  # the outer core is labelled
        @test occursin("(fitted", txt)                # the rational AW κ is flagged
        rm(tmp)
    end

    # optional held-back tests (nutation + rotating) — present only locally
    let p = joinpath(@__DIR__, "extras_local.jl")
        isfile(p) && include(p)
    end

end
