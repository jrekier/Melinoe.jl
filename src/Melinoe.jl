"""
    Melinoe

Melinoë — internal geodynamics of planetary bodies, on an ApproxFun spectral
discretisation (P-formulation in fluid layers).

Build one `PlanetModel` — from the `Models` submodule (`Melinoe.Models.PREM()`,
`.Mars()`, `.Ganymede()`, `.Kelvin()`, …) or your own via `build_model` /
`load_planet_csv` — then ask it for `love_numbers`, `free_modes`, `pressure_love`
or `compliances`. `forced(model, forcing)` leaves the solved fields open, read with
`radial` / `tangential` / `potential` / `moment`.

The `examples/` notebooks are the guided tour.
"""
module Melinoe

using ApproxFun
using LinearAlgebra
using Printf

# ── gravito-elastic core solver ───────────────────────────────────────────────
include("core/layer.jl")
include("core/gravity.jl")
include("core/operators.jl")
include("core/assembly.jl")
include("core/forcing.jl")
include("core/love.jl")
include("core/modes.jl")
# ── model layer + high-level solvers ─────────────────────────────────────────
include("core/model.jl")
include("core/pressure.jl")
include("core/compliance.jl")
include("core/forced.jl")                 # composable Forcing → forced → readers
include("core/load.jl")
# ── planet models — qualified only, e.g. `Melinoe.Models.PREM()` ──────────────
module Models
    using ApproxFun
    using ..Melinoe
    import ..Melinoe: Ω_EARTH, G_SI   # constants the models need, not exported
    include("models/kelvin.jl")
    include("models/ganymede.jl")
    include("models/prem.jl")
    include("models/mars.jl")
    export PREM, Mars, Ganymede, Kelvin
end

export
    # domain helpers, re-exported so `using Melinoe` is enough to build a model
    (..), leftendpoint, rightendpoint,
    # model
    Layer, is_fluid, is_incompressible,
    PlanetModel, build_model,
    load_planet_csv, write_planet_csv, write_planet_coeffs, planet_coeffs,
    layers_of, radial_profile, ω_unit, T_minutes, dahlenize, rotation_rate,
    # assembly / operators
    gravitoelastic_blocks, surface_bc_rows, assemble_planet, interior_ranges,
    # forcing / solves
    potential_forcing, forced_solve, read_love,
    # composable forced API
    Forcing, Tide, Centrifugal, Tilt, Potential, forced, Forced,
    radial, tangential, potential, surface_potential, moment,
    love_numbers, free_modes, pressure_love, compliances,
    # polynomial utilities
    poly_r2_integral, poly_r4_integral, self_gravity

# ── optional local extensions ─────────────────────────────────────────────────
# `extras_local.jl` is gitignored: absent in a clone, so this stays a pure
# gravito-elastic core; a development checkout may add it to load more modules.
let extra = joinpath(@__DIR__, "extras_local.jl")
    isfile(extra) && include(extra)
end

end # module
