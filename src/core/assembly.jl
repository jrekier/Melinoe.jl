# ── Global assembly ───────────────────────────────────────────────────────────

# Interior operator-row ranges for layer `i`, matching the layout `assemble_planet`
# reserves. `U_int/V_int/δφ_int` are the global rows carrying the U/V/Poisson operator
# rows; `*_op` the matching operator sub-block rows. The dropped first/last rows are the
# gaps left for BC and junction equations. Shared by assembly, forcing, and Love code.
function interior_ranges(i, ni, isfluid, Ntot, cumN)
    rU  = (cumN[i]+1):cumN[i+1]
    rV  = (Ntot+cumN[i]+1):(Ntot+cumN[i+1])
    rδφ = (2Ntot+cumN[i]+1):(2Ntot+cumN[i+1])
    if isfluid
        if i == 1
            U_int = rU[1:ni];    U_op = 1:ni
            δφ_int = rδφ[1:ni-1]; δφ_op = 1:ni-1
        else
            U_int = rU[2:ni];    U_op = 1:ni-1
            δφ_int = rδφ[2:ni-1]; δφ_op = 1:ni-2
        end
        V_int = rV[1:ni-1]; V_op = 1:ni-1
    else
        if i == 1
            U_int = rU[1:ni-1];  U_op = 1:ni-1
            δφ_int = rδφ[1:ni-1]; δφ_op = 1:ni-1
        else
            U_int = rU[2:ni-1];  U_op = 1:ni-2
            δφ_int = rδφ[2:ni-1]; δφ_op = 1:ni-2
        end
        V_int = i == 1 ? rV[1:ni-1] : rV[2:ni-1]
        V_op  = i == 1 ? (1:ni-1)   : (1:ni-2)
    end
    return (; U_int, U_op, V_int, V_op, δφ_int, δφ_op)
end

"""
    assemble_planet(layers, ℓ) -> (Amat, Bmat, ops, ns, jrhs)

Assemble the global generalised-eigenvalue pencil `A x = λ B x` for the
multi-layer gravito-elastic system at harmonic degree `ℓ`. Variables are grouped
`[U₁…U_L; V₁…V_L; δφ₁…δφ_L]` (V is the Lagrangian pressure P in fluid layers).
Returns the two matrices, the per-layer operator blocks `ops`, the mode counts
`ns`, and `jrhs` — the list of junction rows that carry an applied-potential RHS
(nonempty only for Dahlen-reduced fluids; consumed by `potential_forcing`).

BC/junction rows are rescaled to unit ∞-norm at the end (Bmat and all standard
RHS are zero on those rows, so this is invisible downstream).
"""
function assemble_planet(layers::Vector{Layer}, ℓ::Int)
    L   = length(layers)
    ns  = [l.n for l in layers]
    flu = [is_fluid(l) for l in layers]
    N   = 3 * sum(ns)

    Amat = zeros(N, N)
    Bmat = zeros(N, N)
    bc_rows = Int[]   # rows holding BC/junction equations — rescaled at the end
    # Junction rows that carry an applied-potential RHS (Dahlen-reduced fluids
    # only): potential_forcing adds coef·amplitude·potential(r0) to row when the
    # named layer is forced. Empty for standard formulations.
    jrhs = @NamedTuple{row::Int, layer::Int, coef::Float64, r0::Float64}[]

    ops  = [gravitoelastic_blocks(layers[i], ℓ, i == 1) for i in 1:L]
    Svec = [ops[i].S for i in 1:L]
    Dvec = [ops[i].D for i in 1:L]

    Ntot = sum(ns)
    cumN = cumsum([0; ns])
    col_U(i)  = cumN[i]+1 : cumN[i+1]
    col_V(i)  = Ntot  + cumN[i]+1 : Ntot  + cumN[i+1]
    col_δφ(i) = 2Ntot + cumN[i]+1 : 2Ntot + cumN[i+1]
    row_U(i)  = col_U(i)
    row_V(i)  = col_V(i)
    row_δφ(i) = col_δφ(i)

    # Interior operator rows
    for i in 1:L
        o = ops[i]; ni = ns[i]
        cU = col_U(i); cV = col_V(i); cδφ = col_δφ(i)
        rg = interior_ranges(i, ni, flu[i], Ntot, cumN)

        Amat[rg.U_int,  cU]  = o.A_UU[rg.U_op, :]
        Amat[rg.U_int,  cV]  = o.A_UV[rg.U_op, :]
        Amat[rg.U_int,  cδφ] = o.A_Uφ[rg.U_op, :]
        Bmat[rg.U_int,  cU]  = o.B_UU[rg.U_op, :]

        Amat[rg.V_int,  cU]  = o.A_VU[rg.V_op, :]
        Amat[rg.V_int,  cV]  = o.A_VV[rg.V_op, :]
        Amat[rg.V_int,  cδφ] = o.A_Vφ[rg.V_op, :]
        Bmat[rg.V_int,  cU]  = o.B_VU[rg.V_op, :]
        Bmat[rg.V_int,  cV]  = o.B_VV[rg.V_op, :]

        Amat[rg.δφ_int, cU]  = o.A_φU[rg.δφ_op, :]
        Amat[rg.δφ_int, cV]  = o.A_φV[rg.δφ_op, :]
        Amat[rg.δφ_int, cδφ] = o.A_φφ[rg.δφ_op, :]
    end

    # Junction conditions
    for i in 1:L-1
        li = layers[i]; li1 = layers[i+1]
        r0 = rightendpoint(li.domain)
        ni = ns[i]; ni1 = ns[i+1]

        evL, DevL = eval_rows(Svec[i],   Dvec[i],   r0, ni)
        evR, DevR = eval_rows(Svec[i+1], Dvec[i+1], r0, ni1)
        κL = li.κ(r0);  μL = li.μ(r0);  ρL = li.ρ₀(r0)
        κR = li1.κ(r0); μR = li1.μ(r0); ρR = li1.ρ₀(r0)
        dL = flu[i]   && li.dahlen      # Dahlen-reduced fluid below/above:
        dR = flu[i+1] && li1.dahlen     # traction/[P] rows are written directly
        g0 = li.g₀(r0)                  # on (U_solid, δφ, forcing)

        last_U  = row_U(i)[end]
        last_V  = row_V(i)[end]
        last_δφ = row_δφ(i)[end]
        push!(bc_rows, last_U, last_V, last_δφ)

        if !flu[i] && !flu[i+1]
            Amat[last_U, col_U(i)]   = (2κL-4μL/3).*evL .+ (κL+4μL/3).*r0.*DevL
            Amat[last_U, col_V(i)]   = (ℓ*(1+ℓ)/3*(-3κL+2μL)).*evL
            Amat[last_U, col_U(i+1)] = -((2κR-4μR/3).*evR .+ (κR+4μR/3).*r0.*DevR)
            Amat[last_U, col_V(i+1)] = -(ℓ*(1+ℓ)/3*(-3κR+2μR)).*evR
        elseif !flu[i] && flu[i+1]
            # solid below | fluid above: σ_rr(solid) = −P(fluid). Dahlen fluids write
            # the load −P = ρ_R(g₀U_solid + δφ + Φ_R) directly (the slave P field is
            # only defined up to truncation): σrr-form − r0ρ_R(g₀U_L + δφ_R) = r0ρ_RΦ_R.
            if dR
                Amat[last_U, col_U(i)]    = (2κL-4μL/3).*evL .+ (κL+4μL/3).*r0.*DevL .- (r0*ρR*g0).*evL
                Amat[last_U, col_V(i)]    = (ℓ*(1+ℓ)/3*(-3κL+2μL)).*evL
                Amat[last_U, col_δφ(i+1)] = -(r0*ρR) .* evR
                push!(jrhs, (row=last_U, layer=i+1, coef=r0*ρR, r0=r0))
            else
                Amat[last_U, col_U(i)]   = (2κL-4μL/3).*evL .+ (κL+4μL/3).*r0.*DevL
                Amat[last_U, col_V(i)]   = (ℓ*(1+ℓ)/3*(-3κL+2μL)).*evL
                Amat[last_U, col_V(i+1)] = r0 .* evR
            end
        elseif flu[i] && !flu[i+1]
            # fluid below | solid above (mirror of the case above)
            if dL
                Amat[last_V, col_U(i+1)] = (2κR-4μR/3).*evR .+ (κR+4μR/3).*r0.*DevR .- (r0*ρL*g0).*evR
                Amat[last_V, col_V(i+1)] = (ℓ*(1+ℓ)/3*(-3κR+2μR)).*evR
                Amat[last_V, col_δφ(i)]  = -(r0*ρL) .* evL
                push!(jrhs, (row=last_V, layer=i, coef=r0*ρL, r0=r0))
            else
                Amat[last_V, col_V(i)]   = r0 .* evL
                Amat[last_V, col_U(i+1)] = (2κR-4μR/3).*evR .+ (κR+4μR/3).*r0.*DevR
                Amat[last_V, col_V(i+1)] = (ℓ*(1+ℓ)/3*(-3κR+2μR)).*evR
            end
        else
            # fluid-fluid: pressure continuity. Dahlen writes the direct compatibility
            # row Δρ(g₀U+δφ) = ρ_RΦ_R − ρ_LΦ_L instead, which with [U]=0 pins the
            # shared material deflection U(r0) exactly.
            @assert dL == dR "mixed standard/Dahlen fluid-fluid junction unsupported"
            Δρ = ρL - ρR
            if dL && abs(Δρ) > 1e-12
                Amat[last_V, col_U(i)]  = (Δρ*g0) .* evL
                Amat[last_V, col_δφ(i)] = Δρ .* evL
                push!(jrhs, (row=last_V, layer=i+1, coef=ρR,  r0=r0))
                push!(jrhs, (row=last_V, layer=i,   coef=-ρL, r0=r0))
            else
                Amat[last_V, col_V(i)]   =  evL
                Amat[last_V, col_V(i+1)] = -evR
            end
        end

        Amat[last_δφ, col_U(i)]    =  3ρL .* evL
        Amat[last_δφ, col_δφ(i)]   =  DevL
        Amat[last_δφ, col_U(i+1)]  = -(3ρR .* evR)
        Amat[last_δφ, col_δφ(i+1)] = -DevR

        first_U = row_U(i+1)[1]
        push!(bc_rows, first_U)
        Amat[first_U, col_U(i)]   =  evL
        Amat[first_U, col_U(i+1)] = -evR

        first_δφ = row_δφ(i+1)[1]
        push!(bc_rows, first_δφ)
        Amat[first_δφ, col_δφ(i)]   =  evL
        Amat[first_δφ, col_δφ(i+1)] = -evR

        if !flu[i] && !flu[i+1]
            Amat[last_V, col_U(i)]   =  μL .* evL
            Amat[last_V, col_V(i)]   = -μL .* evL .+ μL .* r0 .* DevL
            Amat[last_V, col_U(i+1)] = -(μR .* evR)
            Amat[last_V, col_V(i+1)] =  μR .* evR .- μR .* r0 .* DevR
            first_V = row_V(i+1)[1]
            push!(bc_rows, first_V)
            Amat[first_V, col_V(i)]   =  evL
            Amat[first_V, col_V(i+1)] = -evR
        elseif !flu[i] && flu[i+1]
            Amat[last_V, col_U(i)] =  μL .* evL
            Amat[last_V, col_V(i)] = -μL .* evL .+ μL .* r0 .* DevL
        elseif flu[i] && !flu[i+1]
            first_V = row_V(i+1)[1]
            push!(bc_rows, first_V)
            Amat[first_V, col_U(i+1)] =  μR .* evR
            Amat[first_V, col_V(i+1)] = -μR .* evR .+ μR .* r0 .* DevR
        end
    end

    # Surface BCs
    i      = L
    r_surf = rightendpoint(layers[i].domain)
    ni     = ns[i]

    if flu[i]
        # Fluid surface: free-surface normal stress reduces to P = 0 (μ=0 ⇒
        # σ_rr = −P). The U-momentum row at r=R stays an interior equation.
        ev, Dev = eval_rows(Svec[i], Dvec[i], r_surf, ni)
        ρ₁ = layers[i].ρ₀(r_surf)
        Amat[row_V(i)[end],  col_V(i)]   = ev
        Amat[row_δφ(i)[end], col_U(i)]   = 3ρ₁ .* ev
        Amat[row_δφ(i)[end], col_δφ(i)]  = (1+ℓ) .* ev .+ Dev
        push!(bc_rows, row_V(i)[end], row_δφ(i)[end])
    else
        bc = surface_bc_rows(Svec[i], Dvec[i], ni, ℓ,
                             layers[i].κ, layers[i].μ, layers[i].ρ₀, r_surf)

        Amat[row_U(i)[end],  col_U(i)]  = bc.σrr_U
        Amat[row_U(i)[end],  col_V(i)]  = bc.σrr_V
        Amat[row_V(i)[end],  col_U(i)]  = bc.σrθ_U
        Amat[row_V(i)[end],  col_V(i)]  = bc.σrθ_V
        Amat[row_δφ(i)[end], col_U(i)]  = bc.grav_U
        Amat[row_δφ(i)[end], col_δφ(i)] = bc.grav_δφ
        push!(bc_rows, row_U(i)[end], row_V(i)[end], row_δφ(i)[end])
    end

    # Rescale BC/junction rows to unit ∞-norm — invisible downstream (Bmat and the
    # standard RHS vanish there) but tightens conditioning. jrhs coefficients are
    # rescaled identically so potential_forcing stays consistent.
    rowscale = Dict{Int, Float64}()
    for k in unique(bc_rows)
        s = maximum(abs, view(Amat, k, :))
        s > 0 && (Amat[k, :] ./= s; rowscale[k] = s)
    end
    jrhs = [(row=e.row, layer=e.layer, coef=e.coef/get(rowscale, e.row, 1.0), r0=e.r0)
            for e in jrhs]

    return Amat, Bmat, ops, ns, jrhs
end
