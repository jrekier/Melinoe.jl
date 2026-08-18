# ── Free-oscillation eigenvalue solve ─────────────────────────────────────────
# Poisson carries no time derivative, so B's whole δφ row and column are zero and
# the pencil is singular. QZ handles that; the spurious values it returns for the
# null directions are filtered out here.

"""
    solve_modes(Amat, Bmat; tol_imag=1e-5, tol_abs=1e-8, tol_λmin=1e-4)
        -> (ωs, sorted_idx, vs, λs)

Generalised eigensolve `A x = λ B x` (`λ = ω²`), ascending. `vs[:, sorted_idx]`
are the kept eigenvectors; `λs` is everything, unfiltered.

Discarded: non-finite `λ`, from the zero rows of `B`; complex `λ`, since the
problem is self-adjoint; and `λ ≤ tol_λmin`, junk clustered near zero rather than
a physical band.
"""
function solve_modes(Amat, Bmat; tol_imag=1e-5, tol_abs=1e-8, tol_λmin=1e-4)
    λs, vs = eigen(Amat, Bmat)
    good = isfinite.(λs) .&
           (abs.(imag.(λs)) .< tol_imag * abs.(real.(λs)) .+ tol_abs) .&
           (real.(λs) .> tol_λmin)
    good_idx = findall(good)
    perm = sortperm(real.(λs[good_idx]))
    sorted_idx = good_idx[perm]
    ωs = sqrt.(real.(λs[sorted_idx]))
    return ωs, sorted_idx, vs, λs
end
