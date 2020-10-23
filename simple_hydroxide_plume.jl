# # A simple hydroxide plume

using Pkg
using Statistics
using Printf
using JLD2
using Plots

using Oceananigans
using Oceananigans.Advection
using Oceananigans.Buoyancy
using Oceananigans.BoundaryConditions
using Oceananigans.Grids
using Oceananigans.Forcings

using Oceananigans.Fields: AveragedField
using Oceananigans.Utils: minute, hour, GiB, prettytime
using Oceananigans.OutputWriters: JLD2OutputWriter, FieldSlicer

using LESbrary.Utils: SimulationProgressMessenger

plot_only = false

# To start, we ensure that all packages in the LESbrary environment are installed:

Pkg.instantiate()

# Domain
#
# We use a three-dimensional domain that's twice as wide as it is deep.
# We choose this aspect ratio so that the horizontal scale is 4x larger
# than the boundary layer depth when the boundary layer penetrates half
# the domain depth.

Nh = 64
Nz = 64
Lz = 64
Lx = Ly = 128

surface_buoyancy_flux = 1e-7      
surface_momentum_flux = 0 # -1e-4  # "kinematic momentum flux", eg τ / ρ
N² = 1e-5
surface_temperature = 20

grid = RegularCartesianGrid(size=(Nh, Nh, Nz), x=(-Lx/2, Lx/2), y=(-Ly/2, Ly/2), z=(-Lz, 0))

# Buoyancy and boundary conditions

coriolis = FPlane(f=1e-4)

buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(α=2e-4), constant_salinity=35.0)

α = buoyancy.equation_of_state.α
g = buoyancy.gravitational_acceleration

Qᶿ = surface_buoyancy_flux / (α * g)
dTdz = N² / (α * g)

T_bcs = TracerBoundaryConditions(grid, top = BoundaryCondition(Flux, Qᶿ),
                                       bottom = BoundaryCondition(Gradient, dTdz))

u_bcs = UVelocityBoundaryConditions(grid, top = BoundaryCondition(Flux, surface_momentum_flux))

# Tracer forcing

# # Initial condition and sponge layer

OH⁻_source_func(x, y, z, t, p) = 1/p.τ * exp(-z^2 / (2 * p.depth^2) - (x^2 + y^2) / (2 * p.radius^2))
OH⁻_source = Forcing(OH⁻_source_func, parameters=(τ=hour/4, depth=2, radius=5))

model = IncompressibleModel(architecture = CPU(),
                             timestepper = :RungeKutta3,
                               advection = UpwindBiasedFifthOrder(),
                                    grid = grid,
                                 tracers = (:T, :OH⁻),
                                buoyancy = buoyancy,
                                coriolis = coriolis,
                                 closure = AnisotropicMinimumDissipation(),
                     boundary_conditions = (T=T_bcs, u=u_bcs),
                                 forcing = (OH⁻=OH⁻_source,))

# # Set Initial condition

## Noise with 8 m decay scale
Ξ(z) = rand() * exp(z / 8)
                   
linear_temperature_profile(x, y, z) = surface_temperature + dTdz * z

# Modify temperature profile to have a surface mixed layer...
mixed_layer_depth = 20
mixed_layer_temperature = linear_temperature(0, 0, -mixed_layer_depth)

mixed_layer_temperature_profile(x, y, z) =
    z < -mixed_layer_depth ? linear_temperature_profile(x, y, z) : mixed_layer_temperature

set!(model, T=mixed_layer_temperature_profile)
    
# # Prepare the simulation

# Adaptive time-stepping
wizard = TimeStepWizard(cfl=1.0, Δt=1.0, max_change=1.1, max_Δt=30.0)

simulation = Simulation(model, Δt=wizard, stop_time=4hour, iteration_interval=10,
                        progress=SimulationProgressMessenger(model, wizard))

# Prepare Output

prefix = @sprintf("hydroxide_plume_Qu%.1e_Qb%.1e_Nh%d_Nz%d", abs(Qᵘ), Qᵇ, grid.Nx, grid.Nz)

# Fields and slices

if !plot_only
    simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers); 
                                                          time_interval = 1hour, # every quarter period
                                                                 prefix = prefix * "_fields",
                                                                  force = true)
    
    simulation.output_writers[:slices] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                          time_interval = 1minute,
                                                                 prefix = prefix * "_slices",
                                                           field_slicer = FieldSlicer(j=floor(Int, grid.Ny/2)),
                                                                  force = true)
    
    # Averages
    u, v, w = model.velocities
    θ, oh⁻ = model.tracers
    
    U = AveragedField(u, dims=(1, 2))
    V = AveragedField(v, dims=(1, 2))
    T = AveragedField(θ, dims=(1, 2))
    OH⁻ = AveragedField(oh⁻, dims=(1, 2))
    
    # Averaged fluxes
    wT = AveragedField(w * θ, dims=(1, 2))
    wOH⁻ = AveragedField(w * oh⁻, dims=(1, 2))
    
    averaged_fields_and_fluxes = (u=U, v=V, T=T, OH⁻=OH⁻, wT=wT, wOH⁻=wOH⁻)
    
    simulation.output_writers[:averages] = JLD2OutputWriter(model, averaged_fields_and_fluxes,
                                                            time_interval = 1minute,
                                                                  prefix = prefix * "_averages",
                                                                   force = true)
    
    # # Run
    
    run!(simulation)
end

# # Load and plot turbulence statistics

""" Returns colorbar levels equispaced from `(-clim, clim)` and encompassing the extrema of `c`. """
function divergent_levels(c, clim, nlevels=30)
    levels = range(-clim, stop=clim, length=nlevels)
    cmax = maximum(abs, c)
    return ((-clim, clim), clim > cmax ? levels : levels = vcat([-cmax], levels, [cmax]))
end

""" Returns colorbar levels equispaced between `clims` and encompassing the extrema of `c`."""
function sequential_levels(c, clims, nlevels=30)
    levels = range(clims[1], stop=clims[2], length=nlevels)
    cmin, cmax = minimum(c), maximum(c)
    cmin < clims[1] && (levels = vcat([cmin], levels))
    cmax > clims[2] && (levels = vcat(levels, [cmax]))
    return clims, levels
end

xw, yw, zw = nodes(model.velocities.w)
xu, yu, zu = nodes(model.velocities.u)
xc, yc, zc = nodes(model.tracers.OH⁻)

file = jldopen(prefix * "_slices.jld2")

iterations = parse.(Int, keys(file["timeseries/t"]))

# This utility is handy for calculating nice contour intervals:

@info "Making an animation from the saved data..."

anim = @animate for (i, iter) in enumerate(iterations)

    @info "Drawing frame $i from iteration $iter \n"

    t = file["timeseries/t/$iter"]

    ## Load 3D fields from file
    w = file["timeseries/w/$iter"][:, 1, :]
    u = file["timeseries/u/$iter"][:, 1, :]
    OH⁻ = file["timeseries/OH⁻/$iter"][:, 1, :]

    wlims, wlevels = divergent_levels(w, 0.8*maximum(abs, w) + 1e-9)
    ulims, ulevels = divergent_levels(u, 0.8*maximum(abs, u) + 1e-9)
    OH⁻lims, OH⁻levels = sequential_levels(OH⁻, (0, 0.8*maximum(abs, OH⁻) + 1e-9))

     kwargs = (linewidth=0, xlabel="x (m)", ylabel="z (m)", aspectratio=1,
               xlims=(-grid.Lx/2, grid.Lx/2), ylims=(-grid.Lz, 0))

    w_plot = contourf(xw, zw, w'; color=:balance, clims=wlims, levels=wlevels, kwargs...)
    u_plot = contourf(xu, zu, u'; color=:balance, clims=ulims, levels=ulevels, kwargs...)
    OH⁻_plot = contourf(xc, zc, OH⁻'; color=:thermal,  clims=OH⁻lims, levels=OH⁻levels, kwargs...)

    w_title = @sprintf("w(y=0, t=%s) (m s⁻¹), ", prettytime(t))
    u_title = @sprintf("u(y=0, t=%s) (m s⁻¹), ", prettytime(t))
    OH⁻_title = @sprintf("OH⁻(y=0, t=%s) (m s⁻¹), ", prettytime(t))

    ## Arrange the plots side-by-side.
    plot(w_plot, u_plot, OH⁻_plot, layout=(1, 3), size=(2000, 400),
         title=[w_title u_title OH⁻_title])

    iter == iterations[end] && close(file)
end

gif(anim, prefix * ".gif", fps = 8)
