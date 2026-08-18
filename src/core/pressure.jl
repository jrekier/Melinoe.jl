# ── Pressure Love numbers ─────────────────────────────────────────────────────
# Freezing the forcing layer drops its U and P columns, so the variable layout
# differs from `assemble_planet` and this keeps its own assembler — reusing `Layer`,
# `gravitoelastic_blocks` and the same σ_rr/σ_rθ stress forms.

function _pressure_love(layers::Vector{Layer}, ℓ::Int; forcing_layer::Int, ω² = 0.0)
    L  = length(layers)
    fi = forcing_layer
    @assert is_fluid(layers[fi])               "forcing layer must be fluid"
    @assert fi ≥ 2 && !is_fluid(layers[fi-1])  "layer below must be solid"
    @assert fi < L && !is_fluid(layers[fi+1])  "layer above must be solid"

    ops = [gravitoelastic_blocks(layers[i], ℓ, i == 1) for i in 1:L]
    ns  = [l.n for l in layers]
    is_frozen = [(i == fi) for i in 1:L]

    # Variable-grouped layout: frozen layer contributes only δφ.
    U_sizes  = [is_frozen[i] ? 0 : ns[i] for i in 1:L]
    U_cum    = cumsum([0; U_sizes])
    δφ_cum   = cumsum([0; ns])
    N_U      = U_cum[end];   N_V = N_U;   N_δφ = δφ_cum[end]
    N        = N_U + N_V + N_δφ

    col_U(i)  = (U_cum[i] + 1):U_cum[i+1]
    col_V(i)  = (N_U + U_cum[i] + 1):(N_U + U_cum[i+1])
    col_δφ(i) = (N_U + N_V + δφ_cum[i] + 1):(N_U + N_V + δφ_cum[i+1])
    row_U  = col_U;  row_V = col_V;  row_δφ = col_δφ

    Amat = zeros(N, N)
    Bmat = zeros(N, N)
    f    = zeros(N)

    evals(S, D, r_pt, n) = begin
        ev  = vec(Matrix(Evaluation(S, r_pt)[1:1, 1:n]))
        Dev = vec(Matrix(Evaluation(rangespace(D), r_pt)[1:1, 1:n]) * Matrix(D[1:n, 1:n]))
        return ev, Dev
    end

    # ── Interior equations ─────────────────────────────────────────────────
    for i in 1:L
        o  = ops[i]
        ni = ns[i]
        if is_frozen[i]
            Amat[row_δφ(i)[2:ni-1], col_δφ(i)] = o.A_φφ[1:ni-2, :]   # Laplace interior
            continue
        end

        rU = row_U(i);  rV = row_V(i);  rδφ = row_δφ(i)
        cU = col_U(i);  cV = col_V(i);  cδφ = col_δφ(i)

        if is_fluid(layers[i])
            if i == 1
                U_int  = rU[1:ni];      U_op  = 1:ni
                δφ_int = rδφ[1:ni-1];   δφ_op = 1:ni-1
            else
                U_int  = rU[2:ni];      U_op  = 1:ni-1
                δφ_int = rδφ[2:ni-1];   δφ_op = 1:ni-2
            end
            V_int = rV[1:ni-1];  V_op = 1:ni-1
        else
            if i == 1
                U_int  = rU[1:ni-1];    U_op  = 1:ni-1
                δφ_int = rδφ[1:ni-1];   δφ_op = 1:ni-1
            else
                U_int  = rU[2:ni-1];    U_op  = 1:ni-2
                δφ_int = rδφ[2:ni-1];   δφ_op = 1:ni-2
            end
            V_int = i == 1 ? rV[1:ni-1] : rV[2:ni-1]
            V_op  = i == 1 ? (1:ni-1)    : (1:ni-2)
        end

        Amat[U_int,  cU]  = o.A_UU[U_op, :]
        Amat[U_int,  cV]  = o.A_UV[U_op, :]
        Amat[U_int,  cδφ] = o.A_Uφ[U_op, :]
        Bmat[U_int,  cU]  = o.B_UU[U_op, :]

        Amat[V_int,  cU]  = o.A_VU[V_op, :]
        Amat[V_int,  cV]  = o.A_VV[V_op, :]
        Amat[V_int,  cδφ] = o.A_Vφ[V_op, :]
        Bmat[V_int,  cU]  = o.B_VU[V_op, :]
        Bmat[V_int,  cV]  = o.B_VV[V_op, :]

        Amat[δφ_int, cU]  = o.A_φU[δφ_op, :]
        Amat[δφ_int, cV]  = o.A_φV[δφ_op, :]
        Amat[δφ_int, cδφ] = o.A_φφ[δφ_op, :]
    end

    # ── Junction equations ─────────────────────────────────────────────────
    for i in 1:L-1
        li, lj = layers[i], layers[i+1]
        r0  = rightendpoint(li.domain)
        κL  = li.κ(r0);  μL = li.μ(r0);  ρL = li.ρ₀(r0)
        κR  = lj.κ(r0);  μR = lj.μ(r0);  ρR = lj.ρ₀(r0)
        evL, DevL = evals(ops[i].S,   ops[i].D,   r0, ns[i])
        evR, DevR = evals(ops[i+1].S, ops[i+1].D, r0, ns[i+1])

        flu_L = is_fluid(li);  fro_L = is_frozen[i]
        flu_R = is_fluid(lj);  fro_R = is_frozen[i+1]

        σrr_U(κ, μ, ev, Dev) = (2κ - 4μ/3) .* ev .+ (κ + 4μ/3) .* r0 .* Dev
        σrr_V(κ, μ, ev)      = (ℓ*(1+ℓ)/3*(-3κ + 2μ)) .* ev
        σrθ_U(μ, ev)     = μ .* ev
        σrθ_V(μ, ev, Dev) = -μ .* ev .+ μ .* r0 .* Dev

        # ── Frozen-layer interfaces (D&B 2004 eq. 41) ──────────────────────────
        # The frozen fluid has no U of its own, but it is not at rest: it takes the
        # hydrostatic displacement U_f = −δφ/g₀ that holds its surface on an
        # equipotential. The solid sits C₅ = U_s + δφ/g₀ off that surface, so U is
        # apparently discontinuous here — D&B's "apparent jump in the radial
        # displacement". Substituting C₅ leaves two rows, in L=below / R=above:
        #
        #   σ_rr^s − ρ_f g₀ U_s − ρ_f δφ = −Ψ         traction, Ψ at the top of `fi`
        #   3(ρ_L − ρ_R) U_s + Dδφ_L − Dδφ_R = 0      mass-anomaly jump in δφ′
        #
        # The second carries the density *difference* against the solid's U alone,
        # where an ordinary junction below carries 3ρU on each side separately: the
        # two δφ/g₀ pieces cancel. Setting U_f = 0 instead gives C₅ = U_s and quietly
        # drops both the buoyancy and the mass-anomaly correction.
        if fro_R
            g0  = li.g₀(r0)
            rrow = row_U(i)[end]
            Amat[rrow, col_U(i)]  = σrr_U(κL, μL, evL, DevL) .- (r0*ρR*g0) .* evL
            Amat[rrow, col_V(i)]  = σrr_V(κL, μL, evL)
            Amat[rrow, col_δφ(i)] = -(r0*ρR) .* evL

            srow = row_V(i)[end]
            Amat[srow, col_U(i)] = σrθ_U(μL, evL)
            Amat[srow, col_V(i)] = σrθ_V(μL, evL, DevL)

            drow = row_δφ(i+1)[1]
            Amat[drow, col_δφ(i)]   =  evL
            Amat[drow, col_δφ(i+1)] = -evR

            jrow = row_δφ(i)[end]
            Amat[jrow, col_U(i)]    =  (3*(ρL - ρR)) .* evL
            Amat[jrow, col_δφ(i)]   =  DevL
            Amat[jrow, col_δφ(i+1)] = -DevR

        elseif fro_L
            g0 = li.g₀(r0)
            Ψ  = (i == fi) ? 1.0 : 0.0

            rrow = row_U(i+1)[1]
            Amat[rrow, col_U(i+1)]  = σrr_U(κR, μR, evR, DevR) .- (r0*ρL*g0) .* evR
            Amat[rrow, col_V(i+1)]  = σrr_V(κR, μR, evR)
            Amat[rrow, col_δφ(i+1)] = -(r0*ρL) .* evR
            f[rrow] = -r0 * Ψ

            srow = row_V(i+1)[1]
            Amat[srow, col_U(i+1)] = σrθ_U(μR, evR)
            Amat[srow, col_V(i+1)] = σrθ_V(μR, evR, DevR)

            drow = row_δφ(i+1)[1]
            Amat[drow, col_δφ(i)]   =  evL
            Amat[drow, col_δφ(i+1)] = -evR

            jrow = row_δφ(i)[end]
            Amat[jrow, col_U(i+1)]  =  (3*(ρL - ρR)) .* evR
            Amat[jrow, col_δφ(i)]   =  DevL
            Amat[jrow, col_δφ(i+1)] = -DevR

        elseif !flu_L && !flu_R
            Amat[row_U(i)[end], col_U(i)]   =  σrr_U(κL, μL, evL, DevL)
            Amat[row_U(i)[end], col_V(i)]   =  σrr_V(κL, μL, evL)
            Amat[row_U(i)[end], col_U(i+1)] = -σrr_U(κR, μR, evR, DevR)
            Amat[row_U(i)[end], col_V(i+1)] = -σrr_V(κR, μR, evR)

            Amat[row_V(i)[end], col_U(i)]   =  σrθ_U(μL, evL)
            Amat[row_V(i)[end], col_V(i)]   =  σrθ_V(μL, evL, DevL)
            Amat[row_V(i)[end], col_U(i+1)] = -σrθ_U(μR, evR)
            Amat[row_V(i)[end], col_V(i+1)] = -σrθ_V(μR, evR, DevR)

            Amat[row_U(i+1)[1], col_U(i)]   =  evL
            Amat[row_U(i+1)[1], col_U(i+1)] = -evR
            Amat[row_V(i+1)[1], col_V(i)]   =  evL
            Amat[row_V(i+1)[1], col_V(i+1)] = -evR
            Amat[row_δφ(i+1)[1], col_δφ(i)]   =  evL
            Amat[row_δφ(i+1)[1], col_δφ(i+1)] = -evR

            Amat[row_δφ(i)[end], col_U(i)]    =  3ρL .* evL
            Amat[row_δφ(i)[end], col_U(i+1)]  = -3ρR .* evR
            Amat[row_δφ(i)[end], col_δφ(i)]   =  DevL
            Amat[row_δφ(i)[end], col_δφ(i+1)] = -DevR

        elseif !flu_L && flu_R
            Amat[row_U(i)[end], col_U(i)]   = σrr_U(κL, μL, evL, DevL)
            Amat[row_U(i)[end], col_V(i)]   = σrr_V(κL, μL, evL)
            Amat[row_U(i)[end], col_V(i+1)] = r0 .* evR

            Amat[row_V(i)[end], col_U(i)] = σrθ_U(μL, evL)
            Amat[row_V(i)[end], col_V(i)] = σrθ_V(μL, evL, DevL)

            Amat[row_U(i+1)[1], col_U(i)]   =  evL
            Amat[row_U(i+1)[1], col_U(i+1)] = -evR
            Amat[row_δφ(i+1)[1], col_δφ(i)]   =  evL
            Amat[row_δφ(i+1)[1], col_δφ(i+1)] = -evR

            Amat[row_δφ(i)[end], col_U(i)]    =  3ρL .* evL
            Amat[row_δφ(i)[end], col_U(i+1)]  = -3ρR .* evR
            Amat[row_δφ(i)[end], col_δφ(i)]   =  DevL
            Amat[row_δφ(i)[end], col_δφ(i+1)] = -DevR

        elseif flu_L && !flu_R
            Amat[row_V(i)[end], col_V(i)]   = r0 .* evL
            Amat[row_V(i)[end], col_U(i+1)] = σrr_U(κR, μR, evR, DevR)
            Amat[row_V(i)[end], col_V(i+1)] = σrr_V(κR, μR, evR)

            Amat[row_V(i+1)[1], col_U(i+1)] = σrθ_U(μR, evR)
            Amat[row_V(i+1)[1], col_V(i+1)] = σrθ_V(μR, evR, DevR)

            Amat[row_U(i+1)[1], col_U(i)]   =  evL
            Amat[row_U(i+1)[1], col_U(i+1)] = -evR
            Amat[row_δφ(i+1)[1], col_δφ(i)]   =  evL
            Amat[row_δφ(i+1)[1], col_δφ(i+1)] = -evR

            Amat[row_δφ(i)[end], col_U(i)]    =  3ρL .* evL
            Amat[row_δφ(i)[end], col_U(i+1)]  = -3ρR .* evR
            Amat[row_δφ(i)[end], col_δφ(i)]   =  DevL
            Amat[row_δφ(i)[end], col_δφ(i+1)] = -DevR
        else
            error("fluid-fluid junctions not supported in pressure_love")
        end
    end

    # ── Surface BC (outermost layer must be solid) ─────────────────────────
    @assert !is_fluid(layers[L]) "outermost layer must be solid"
    let i = L
        ni = ns[i]
        r_s = rightendpoint(layers[i].domain)
        κ1 = layers[i].κ(r_s);  μ1 = layers[i].μ(r_s);  ρ1 = layers[i].ρ₀(r_s)
        ev, Dev = evals(ops[i].S, ops[i].D, r_s, ni)

        Amat[row_U(i)[end], col_U(i)]   = (2κ1 - 4μ1/3) .* ev .+ (κ1 + 4μ1/3) .* Dev
        Amat[row_U(i)[end], col_V(i)]   = (ℓ*(1+ℓ)/3*(-3κ1 + 2μ1)) .* ev
        Amat[row_V(i)[end], col_U(i)]   = μ1 .* ev
        Amat[row_V(i)[end], col_V(i)]   = -μ1 .* ev .+ μ1 .* Dev
        Amat[row_δφ(i)[end], col_U(i)]  = 3ρ1 .* ev
        Amat[row_δφ(i)[end], col_δφ(i)] = (1+ℓ) .* ev .+ Dev
    end

    x = (Amat - ω² .* Bmat) \ f

    SL  = ops[L].S
    r_s = rightendpoint(layers[L].domain)
    δφR = Fun(SL, x[col_δφ(L)])(r_s)
    uR  = Fun(SL, x[col_U(L)])(r_s)
    return (; kP = -δφR, hP = uR, δφR, uR)
end

"""
    pressure_love(m::PlanetModel; ℓ=2, forcing_layer, ω²=0.0)
    pressure_love(layers, ℓ=2; forcing_layer, ω²=0.0)

Response to a unit pressure applied at the top of an internal fluid layer. Returns
`(; kP, hP, δφR, uR)`, where `kP = −δφ(R)` and `hP = u_r(R)`.

Where the tidal Love numbers `h, k, l` answer *how does the body deform under an
external gravitational potential*, these answer *how does it deform when its own
fluid pushes on the solid above*. A core or a subsurface ocean moving relative to
the shell — in a libration, wobble or nutation — presses on its ceiling; the solid
yields, and the resulting change in the external gravity field feeds back into the
rotational equations.

This is not a standard Love number. The definition used here is the one introduced
by Dumberry & Bloxham (2004), *Variations in the Earth's gravity field caused by
torsional oscillations in the core*, *GJI* **159**(2), 417–434, eq. (41). Pressure at
PREM's CMB reproduces their Table 2 at `ℓ = 2, 4, 6, 8` — within 0.25% for `ℓ ≥ 4`.

`forcing_layer` must be fluid with a solid layer on either side. It is *frozen*:
only the pressure it transmits to its boundaries matters, so its `U` and `P` leave
the system and it contributes just Laplace's equation for `δφ`. Its `κ` therefore
never enters, and an incompressible ocean gives a bit-identical `k_P`.

At a frozen interface the radial displacement is *apparently discontinuous*: the
fluid is taken to sit at the hydrostatic `U_f = −δφ/g₀` that keeps its surface on an
equipotential, while the solid is offset from it by `C₅ = U_s + δφ/g₀`. This is D&B's
"apparent jump in the radial displacement", and it is what makes the interface rows
differ from an ordinary fluid–solid junction — see the comment at the assembly.

```julia
pressure_love(Melinoe.Models.Ganymede(); forcing_layer = 4)   # the ocean
```
"""
pressure_love(layers::Vector{Layer}, ℓ::Int = 2; forcing_layer, ω² = 0.0) =
    _pressure_love(layers, ℓ; forcing_layer, ω²)

pressure_love(m::PlanetModel; ℓ::Int = 2, forcing_layer, ω² = 0.0) =
    _pressure_love(m.layers, ℓ; forcing_layer, ω²)
