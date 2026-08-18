# ── Applied-potential body forcing ────────────────────────────────────────────
#
# An external potential Φ enters the momentum equations through ρ₀∇Φ — exactly how
# δφ enters. So the RHS is f = −A_δφcol · vΦ, the δφ-column operators applied to Φ's
# spectral coefficients and negated, on the interior U- and V-momentum rows only.
# Poisson and BC rows take no forcing when ∇²Φ = 0, true for Φ ∝ r^ℓ.

"""
    potential_forcing(ops, ns; ℓ, potential=(r->r^ℓ), amplitude=1.0,
                      layer_mask=trues(length(ops))) -> f::Vector

Body-force RHS for the applied potential `amplitude·potential(r)`, restricted to
the layers flagged in `layer_mask`. Row indexing matches `assemble_planet`.
"""
function potential_forcing(ops, ns; ℓ::Int, potential = (r -> r^ℓ),
                           amplitude::Real = 1.0,
                           layer_mask = trues(length(ops)),
                           jrhs = nothing)
    L    = length(ops)
    Ntot = sum(ns)
    cumN = cumsum([0; ns])

    f = zeros(3Ntot)
    for i in 1:L
        layer_mask[i] || continue
        ni = ns[i]
        c  = Fun(r -> amplitude * potential(r), ops[i].S).coefficients
        vΦ = length(c) >= ni ? c[1:ni] : [c; zeros(ni - length(c))]

        rg = interior_ranges(i, ni, ops[i].fluid, Ntot, cumN)
        f[rg.U_int] = -ops[i].A_Uφ[rg.U_op, :] * vΦ
        f[rg.V_int] = -ops[i].A_Vφ[rg.V_op, :] * vΦ
        # A_φΦ is nonzero only for Dahlen fluids, whose closed δφ equation carries
        # a source; a harmonic Φ has none.
        f[rg.δφ_int] = -ops[i].A_φΦ[rg.δφ_op, :] * vΦ
    end
    # Junction rows carrying a forcing RHS (Dahlen fluid boundaries). Already
    # rescaled with their rows by assemble_planet.
    if jrhs !== nothing
        for e in jrhs
            layer_mask[e.layer] || continue
            f[e.row] += e.coef * amplitude * potential(e.r0)
        end
    end
    return f
end
