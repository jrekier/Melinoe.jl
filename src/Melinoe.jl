"""
    Melinoe

Melinoë — internal geodynamics of planetary bodies, on an ApproxFun spectral
discretisation (P-formulation in fluid layers).

Build one `PlanetModel` — from the `Models` submodule (`Melinoe.Models.PREM()`,
`.Mars()`, `.Ganymede()`, `.Kelvin()`, …) or your own via `build_model` /
`load_planet_csv` — then ask it for `love_numbers`, `free_modes`, `pressure_love`
or `compliances`. `forced(model, forcing)` leaves the solved fields open, read with
`radial` / `tangential` / `potential` / `moment`.

Rotation and nutation: `flattening` (Clairaut) for the hydrostatic figure, then
either `sic_compliances` / `sic_nutation_modes` for a solid inner core, or
`bmo_compliances` / `nutation_modes` / `nutation_residues` / `nutation_transfer`
for a fluid core under a basal magma layer.

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
# ── rotation: hydrostatic figure, compliances, nutation ───────────────────────
include("nutation/flattening.jl")
include("nutation/compliance_sic.jl")     # solid inner core (Mathews–Herring–Buffett)
include("nutation/nutation_sic.jl")
include("nutation/compliance_bmo.jl")     # fluid core under a basal magma layer
include("nutation/nutation_bmo.jl")
# ── planet models — qualified only, e.g. `Melinoe.Models.PREM()` ──────────────
module Models
    using ApproxFun
    using ..Melinoe
    import ..Melinoe: Ω_EARTH, G_SI   # constants the models need, not exported
    include("models/kelvin.jl")
    include("models/ganymede.jl")
    include("models/prem.jl")
    include("models/mars.jl")
    include("models/mercury.jl")
    include("models/europa.jl")
    include("models/callisto.jl")
    export PREM, Mars, Mercury, Ganymede, Europa, Callisto, Kelvin
end

export
    # domain helpers, re-exported so `using Melinoe` is enough to build a model
    (..), leftendpoint, rightendpoint,
    # model
    Layer, is_fluid, is_incompressible,
    PlanetModel, build_model,
    load_planet_csv, load_planet_profiles, planet_from_table, write_planet_csv, write_planet_coeffs, planet_coeffs,
    layers_of, radial_profile, ω_unit, T_minutes, dahlenize, rotation_rate,
    # assembly / operators
    gravitoelastic_blocks, surface_bc_rows, assemble_planet, interior_ranges,
    # forcing / solves
    potential_forcing, forced_solve, read_love,
    # composable forced API
    Forcing, Tide, Centrifugal, Tilt, Potential, forced, Forced,
    radial, tangential, potential, surface_potential, moment,
    love_numbers, free_modes, pressure_love, compliances,
    # rotation: hydrostatic figure
    flattening,
    # nutation with a solid inner core (MHB 4×4)
    sic_compliances, sic_alphas, sic_nutation_modes, sic_nutation_transfer,
    # nutation with a basal magma layer (3×3)
    bmo_compliances, nutation_modes, nutation_transfer, nutation_residues,
    ekman_K_bc, ekman_K_bm,
    # polynomial utilities
    poly_r2_integral, poly_r4_integral, self_gravity

# ── optional local extensions ─────────────────────────────────────────────────
# `extras_local.jl` is gitignored: absent in a clone, so this stays a pure
# gravito-elastic core; a development checkout may add it to load more modules.
let extra = joinpath(@__DIR__, "extras_local.jl")
    isfile(extra) && include(extra)
end

end # module
