# ocean convection with plankton example using oceananigans
# https://clima.github.io/OceananigansDocumentation/latest/generated/ocean_convection_with_plankton/

using Random, Printf, Plots
using Oceananigans, Oceananigans.Utils, Oceananigans.Grids

Nz = 128
Lz = 64.0
N2 = 1e-5
Qb = 1e-8
end_time = day / 2

grid = RegularCartesianGrid(size=(Nz, 1, Nz), extent=(Lz, Lz, Lz))

buoyancy_bcs = TracerBoundaryConditions(grid,    top = BoundaryCondition(Flux, Qb),
                                              bottom = BoundaryCondition(Gradient, N2))


growth_and_decay(x, y, z, t) = exp(z/16) - 1

# Instantiate the model
model = IncompressibleModel(
                   grid = grid,
                closure = IsotropicDiffusivity(ν=1e-4, κ=1e-4),	
               coriolis = FPlane(f=1e-4),
                tracers = (:b, :plankton),
               buoyancy = BuoyancyTracer(),
                forcing = (plankton=growth_and_decay,),
    boundary_conditions = (b=buoyancy_bcs,)
)

# Set initial condition. Initial velocity and salinity fluctuations needed for AMD.
E(z) = randn() * z / Lz * (1 + z / Lz) # noise
b0(x, y, z) = N2 * z + N2 * Lz * 1e-6 * E(z)
set!(model, b=b0)

# A wizard for managing the simulation time-step.
wizard = TimeStepWizard(cfl=0.1, Δt=1.0, max_change=1.1, max_Δt=90.0)

simulation = Simulation(model, Δt=wizard, stop_iteration=0, iteration_interval=100)

anim = @animate for i = 1:100
    simulation.stop_iteration += 100
    walltime = @elapsed run!(simulation)

    # Print a progress message
    @printf("progress: %.1f %%, i: %04d, t: %s, dt: %s, wall time: %s\n",
            model.clock.time / end_time * 100, model.clock.iteration,
            prettytime(model.clock.time), prettytime(wizard.Δt), prettytime(walltime))

    # Coordinate arrays for plotting
    xC, zF, zC = xnodes(Cell, grid)[:], znodes(Face, grid)[:], znodes(Cell, grid)[:]

    # Fields to plot (converted to 2D arrays).
    w = Array(interior(model.velocities.w))[:, 1, :]
    p = Array(interior(model.tracers.plankton))[:, 1, :]

    # Plot the fields.
    w_plot = heatmap(xC, zF, w', xlabel="x (m)", ylabel="z (m)", color=:balance, clims=(-1e-2, 1e-2))
    p_plot = heatmap(xC, zC, p', xlabel="x (m)", ylabel="z (m)", color=:matter) #, legend=false)

    # Arrange the plots side-by-side.
    plot(w_plot, p_plot, layout=(1, 2), size=(1000, 400),
         title=["vertical velocity (m/s)" "Plankton concentration"])
end
