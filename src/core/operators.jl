# ── Per-layer spectral operator construction ─────────────────────────────────
#
# Solid layers carry state (U, V, δφ); fluid layers (U, P, δφ) with P = -κ∇·u
# the Lagrangian pressure, V eliminated via the kinematic relation
#   V = r/(ℓ(ℓ+1))·(U' + 2U/r + P/κ).
#
# Momentum equations are pre-multiplied by r so every gravity coefficient is a
# polynomial: g₀(r) ~ 1/r² where there is interior mass, but r²·g₀ = 3·(enclosed
# mass) is polynomial. The Poisson identity r³·D2φ₀ = 3ρ₀r³ − 2·(r²g₀) removes
# the second derivative of the background potential.

function eval_rows(S, D, r_pt, n)
    Dmat = Matrix(D[1:n, 1:n])
    ev   = Matrix(Evaluation(S, r_pt)[1:1, 1:n])
    Dev  = Matrix(Evaluation(rangespace(D), r_pt)[1:1, 1:n]) * Dmat
    return ev, Dev
end

"""
    gravitoelastic_blocks(layer, ℓ, is_inner) -> NamedTuple

Discretised gravito-elastic equations for one `layer` at degree `ℓ`, as a 3×3 grid
of spectral operator blocks. `is_inner` selects the Jacobi basis carrying `r^ℓ`
regularity at the centre.

Blocks are named `A_<eq><field>`: rows are the radial-momentum, tangential-momentum
and Poisson equations (`U`, `V`, `φ`); columns are the fields `U` (radial
displacement), `V` (tangential displacement in solids, Lagrangian pressure `P` in
fluids) and `φ` (`δφ`). So `A_Vφ` is the δφ column of the tangential row.

`B_UU`, `B_VU`, `B_VV` are the mass blocks (no `B_φφ` — self-gravity is
instantaneous). `A_φΦ` couples the Poisson row to an applied potential, nonzero only
under the Dahlen closure. Also returns `fluid`, the space `S`, the derivative `D`.
"""
function gravitoelastic_blocks(layer::Layer, ℓ::Int, is_inner::Bool)
    dom = layer.domain
    n   = layer.n
    S = is_inner ? Jacobi(0, ℓ, dom) : Chebyshev(dom)
    r = Fun(identity, S)
    D = Derivative(S)

    μ   = Fun(layer.μ,  S)
    ρ₀  = Fun(layer.ρ₀, S)
    Dμ  = μ'
    Dρ₀ = ρ₀'
    # r²·g₀ = 3·(enclosed mass) — the polynomial combination (see file header).
    r²g₀ = Fun(r -> r^2 * layer.g₀(r), S)

    if is_fluid(layer)
        if layer.dahlen
            # ── Dahlen-reduced static fluid (Dahlen 1974) ─────────────────────
            # Static fluid ⇒ ρ₁ = ρ₀′(δφ+Φ)/g₀, closing Poisson into one equation:
            #   (r²δφ″ + 2rδφ′ − ℓ(ℓ+1)δφ) + src·δφ = −src·Φ,   src = −3r²ρ₀′/g₀
            # (ρ₀′ from AW −ρ₀²g/κ or the raw profile; the −src·Φ forcing is added
            # by potential_forcing via A_φΦ). U and P are then slave fields:
            #   U:  r²g₀·U + r²(δφ+Φ) = 0        (hydrostatic radial balance)
            #   P:  P = −ρ₀(g₀U + δφ + Φ)        (tangential balance)
            # pinned by the junctions: [U]=0 at a solid wall, [P]=0 fluid-fluid,
            # traction passes the Dahlen load −ρ_f(g₀U + δφ + Φ). Static only (B≡0).
            # Derived in examples/2_dahlen_fluid_tutorial.ipynb.
            RS_P = rangespace(Derivative(S))
            cuP  = op -> Conversion(rangespace(op), RS_P) * op
            A_VU = Matrix(cuP(Multiplication(ρ₀ * r²g₀, S))[1:n, 1:n])
            A_VV = Matrix(cuP(Multiplication(r^2, S))[1:n, 1:n])
            A_Vφ = Matrix(cuP(Multiplication(r^2*ρ₀, S))[1:n, 1:n])

            # U-block: the algebraic U-definition on all but the last interior
            # row, then the top tangential-balance row placed in that last row
            # (index n if innermost, n−1 otherwise) so P carries no free mode.
            Mw   = Multiplication(r²g₀, S)
            RS_r = rangespace(Mw)
            cuR  = op -> Conversion(rangespace(op), RS_r) * op
            Mwm  = Matrix(Mw[1:n, 1:n])
            Mr2m = Matrix(cuR(Multiplication(r^2, S))[1:n, 1:n])
            n_rid = is_inner ? n-1 : n-2
            A_UU = zeros(n, n); A_UV = zeros(n, n); A_Uφ = zeros(n, n)
            A_UU[1:n_rid, :] = Mwm[1:n_rid, :]
            A_Uφ[1:n_rid, :] = Mr2m[1:n_rid, :]
            A_UU[n_rid+1, :] = A_VU[n, :]
            A_UV[n_rid+1, :] = A_VV[n, :]
            A_Uφ[n_rid+1, :] = A_Vφ[n, :]
            B_UU = zeros(n, n)

            # src = −3r²ρ₀′/g₀, one full coefficient sample per closure.
            # aw: ρ₀′ = −ρ₀²g/κ ⇒ src = 3r²ρ₀²/κ (κ = Inf ⇒ src ≡ 0).
            src = layer.aw ?
                Fun(rr -> 3 * layer.ρ₀(rr)^2 * rr^2 / layer.κ(rr), S) :
                Fun(rr -> -3 * rr^4 * Dρ₀(rr) / (rr^2 * layer.g₀(rr)), S)
            A33   = r^2 * D^2 + 2r * D - ℓ*(1+ℓ)
            RS_δφ = rangespace(A33)
            cuδ   = op -> Conversion(rangespace(op), RS_δφ) * op
            srcm  = Matrix(cuδ(Multiplication(src, S))[1:n, 1:n])
            A_φU  = zeros(n, n)
            A_φV  = zeros(n, n)
            A_φφ  = Matrix(A33[1:n, 1:n]) + srcm

            return (A_UU=A_UU, A_UV=A_UV, A_Uφ=A_Uφ,
                    A_VU=A_VU, A_VV=A_VV, A_Vφ=A_Vφ,
                    A_φU=A_φU, A_φV=A_φV, A_φφ=A_φφ,
                    B_UU=B_UU, B_VU=zeros(n, n), B_VV=zeros(n, n),
                    A_φΦ=srcm, fluid=true, S=S, D=D)
        end

        # ── Fluid operators: (U, P, δφ) ──────────────────────────────────────
        RS_U = rangespace(Derivative(S))
        cuU  = op -> Conversion(rangespace(op), RS_U) * op

        A_UU = (Matrix((ρ₀ * r * r²g₀ * D)[1:n, 1:n])
              + Matrix(cuU(Multiplication(3*ρ₀^2*r^3 - 2*ρ₀*r²g₀, S))[1:n, 1:n]))
        A_Uφ = Matrix((r^3*ρ₀ * D)[1:n, 1:n])
        B_UU = Matrix(cuU(Multiplication(r^3*ρ₀, S))[1:n, 1:n])

        RS_P = rangespace(Derivative(S))
        cuP  = op -> Conversion(rangespace(op), RS_P) * op

        A_VU = Matrix(cuP(Multiplication(ρ₀ * r²g₀, S))[1:n, 1:n])
        A_VV = Matrix(cuP(Multiplication(r^2, S))[1:n, 1:n])
        A_Vφ = Matrix(cuP(Multiplication(r^2*ρ₀, S))[1:n, 1:n])
        B_VU = (Matrix(((r^4*ρ₀/(ℓ*(1+ℓ))) * D)[1:n, 1:n])
              + Matrix(cuP(Multiplication(2*r^3*ρ₀/(ℓ*(1+ℓ)), S))[1:n, 1:n]))

        if is_incompressible(layer)
            # κ → ∞: drop all 1/κ terms; Poisson reverts to ∇²δφ = −3ρ₀'U
            A_UV = Matrix((r^3 * D)[1:n, 1:n])
            B_VV = zeros(n, n)

            A33   = r^2 * D^2 + 2r * D - ℓ*(1+ℓ)
            RS_δφ = rangespace(A33)
            cuδ   = op -> Conversion(rangespace(op), RS_δφ) * op

            A_φU = Matrix(cuδ(Multiplication(-3*r^2*Dρ₀, S))[1:n, 1:n])
            A_φV = zeros(n, n)
            A_φφ = Matrix(A33[1:n, 1:n])
        else
            # Compressible. κ may diverge as 1/r² (PREM outer core AW closure
            # κ = −ρ²g₀/Dρ). Never build Fun(κ,S) alone; only sample full
            # coefficient functions where any 1/r² is already cancelled.
            r3_ρ_g_over_κ = Fun(r -> r^3 * layer.ρ₀(r) * layer.g₀(r) / layer.κ(r), S)
            r4_ρ_over_κ   = Fun(r -> r^4 * layer.ρ₀(r) / (ℓ*(1+ℓ) * layer.κ(r)), S)

            A_UV = (Matrix((r^3 * D)[1:n, 1:n])
                  + Matrix(cuU(Multiplication(r3_ρ_g_over_κ, S))[1:n, 1:n]))
            B_VV = Matrix(cuP(Multiplication(r4_ρ_over_κ, S))[1:n, 1:n])

            if layer.aw
                # Adams-Williamson Poisson (ρ₀′ = −ρ₀²g/κ). × κr²:
                #   r²·κ·∇²δφ → (r⁴·κ)·D² + 2·(r³·κ)·D − ℓ(ℓ+1)·(r²·κ)
                r2_κ = Fun(r -> r^2 * layer.κ(r), S)
                r3_κ = Fun(r -> r^3 * layer.κ(r), S)
                r4_κ = Fun(r -> r^4 * layer.κ(r), S)
                A33   = r4_κ * D^2 + 2*r3_κ * D - ℓ*(1+ℓ) * r2_κ
                RS_δφ = rangespace(A33)
                cuδ   = op -> Conversion(rangespace(op), RS_δφ) * op
                A_φU = Matrix(cuδ(Multiplication(-3*ρ₀^2 * r^2 * r²g₀, S))[1:n, 1:n])
                A_φV = Matrix(cuδ(Multiplication(-3*r^4 * ρ₀, S))[1:n, 1:n])
                A_φφ = Matrix(A33[1:n, 1:n])
            else
                # Non-AW (raw) Poisson: keep the actual ρ₀′. From ∇·(ρ₀u) = ρ₀′U − ρ₀P/κ
                # after V-elimination, r²∇²δφ + 3r²ρ₀′U − 3r²(ρ₀/κ)P = 0. The +3r²ρ₀′
                # sign matches the solid branch (A31) and the AW form under ρ₀′=−ρ₀²g/κ.
                ρ_over_κ = Fun(r -> layer.ρ₀(r) / layer.κ(r), S)
                A33   = r^2 * D^2 + 2r * D - ℓ*(1+ℓ)
                RS_δφ = rangespace(A33)
                cuδ   = op -> Conversion(rangespace(op), RS_δφ) * op
                A_φU = Matrix(cuδ(Multiplication( 3*r^2 * Dρ₀,      S))[1:n, 1:n])
                A_φV = Matrix(cuδ(Multiplication(-3*r^2 * ρ_over_κ, S))[1:n, 1:n])
                A_φφ = Matrix(A33[1:n, 1:n])
            end
        end

        return (A_UU=A_UU, A_UV=A_UV, A_Uφ=A_Uφ,
                A_VU=A_VU, A_VV=A_VV, A_Vφ=A_Vφ,
                A_φU=A_φU, A_φV=A_φV, A_φφ=A_φφ,
                B_UU=B_UU, B_VU=B_VU, B_VV=B_VV,
                A_φΦ=zeros(n, n), fluid=true, S=S, D=D)
    else
        # ── Solid operators: (U, V, δφ) ──────────────────────────────────────
        κ  = Fun(layer.κ, S)
        Dκ = κ'

        A11 = -(3κ + 4μ)*r^3/3 * D^2 +
              (-r^2*(6κ + 8μ + 3r*Dκ + 4r*Dμ)/3) * D +
              Multiplication(r*(2κ + (8/3 + ℓ + ℓ^2)*μ - 2r*Dκ + (4/3)*r*Dμ)
                             - (4*ρ₀*r²g₀ - 3*ρ₀^2*r^3), S)
        A12 = (r^2*ℓ*(1+ℓ)*(3κ + μ)/3) * D +
              Multiplication(-ℓ*(1+ℓ)/3*(r*(3κ + 7μ) + r^2*(-3Dκ + 2Dμ))
                             + ℓ*(1+ℓ)*ρ₀*r²g₀, S)
        A13 = r^3*ρ₀ * D
        B11 = Multiplication(r^3*ρ₀, S)

        A22 = -r^3*μ * D^2 + (-r^2*(2μ + r*Dμ)) * D +
              Multiplication(r*(ℓ*(1+ℓ)*(κ + (4/3)*μ) + r*Dμ), S)
        A21 = (-r^2*(3κ + μ)/3) * D +
              Multiplication(r*(-2κ - (8/3)*μ - r*Dμ) + ρ₀*r²g₀, S)
        A23 = Multiplication(r^2*ρ₀, S)
        B22 = Multiplication(r^3*ρ₀, S)

        A33 = r^2 * D^2 + 2r * D - ℓ*(1+ℓ)
        A31 = 3r^2*ρ₀ * D + Multiplication(3r*(2ρ₀ + r*Dρ₀), S)
        A32 = Multiplication(-3r*ℓ*(1+ℓ)*ρ₀, S)

        RS   = rangespace(A11)
        conv = op -> Conversion(rangespace(op), RS) * op

        A_UU = Matrix(A11[1:n, 1:n])
        A_UV = Matrix(conv(A12)[1:n, 1:n])
        A_Uφ = Matrix(conv(A13)[1:n, 1:n])
        A_VU = Matrix(conv(A21)[1:n, 1:n])
        A_VV = Matrix(A22[1:n, 1:n])
        A_Vφ = Matrix(conv(A23)[1:n, 1:n])
        A_φU = Matrix(conv(A31)[1:n, 1:n])
        A_φV = Matrix(conv(A32)[1:n, 1:n])
        A_φφ = Matrix(conv(A33)[1:n, 1:n])
        B_UU = Matrix(conv(B11)[1:n, 1:n])
        B_VV = Matrix(conv(B22)[1:n, 1:n])

        return (A_UU=A_UU, A_UV=A_UV, A_Uφ=A_Uφ,
                A_VU=A_VU, A_VV=A_VV, A_Vφ=A_Vφ,
                A_φU=A_φU, A_φV=A_φV, A_φφ=A_φφ,
                B_UU=B_UU, B_VU=zeros(n, n), B_VV=B_VV,
                A_φΦ=zeros(n, n), fluid=false, S=S, D=D)
    end
end

# Surface BCs for a solid outer layer:
#   σ_rr = 0                      (radial-stress free)
#   σ_rθ = 0                      (tangential-stress free)
#   δφ' + (ℓ+1)δφ + 3ρ U = 0      (gravity matches outer vacuum solution)
function surface_bc_rows(S, D, n, ℓ, κ_fn, μ_fn, ρ_fn, r_surf)
    κ₁ = κ_fn(r_surf); μ₁ = μ_fn(r_surf); ρ₁ = ρ_fn(r_surf)
    ev, Dev = eval_rows(S, D, r_surf, n)
    σrr_U  = (2κ₁ - 4μ₁/3) .* ev .+ (κ₁ + 4μ₁/3) .* Dev
    σrr_V  = (ℓ*(1+ℓ)/3 * (-3κ₁ + 2μ₁)) .* ev
    σrθ_U  = μ₁ .* ev
    σrθ_V  = -μ₁ .* ev .+ μ₁ .* Dev
    grav_U  = 3ρ₁ .* ev
    grav_δφ = (1+ℓ) .* ev .+ Dev
    return (; σrr_U, σrr_V, σrθ_U, σrθ_V, grav_U, grav_δφ)
end
