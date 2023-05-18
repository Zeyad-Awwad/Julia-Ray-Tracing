
using CUDA, Random, Plots#, Images

function shortwave_bounce!(intersections::CuDeviceVector{Float32, 1}, pos::CuDeviceMatrix{Float32, 1}, direction::CuDeviceMatrix{Float32, 1}, energy::CuDeviceVector{Float32, 1}, 
    objects::CuDeviceMatrix{Int32, 1}, offsets::CuDeviceMatrix{Float32, 1}, R_layers::CuDeviceVector{Float32, 1}, C_layers::CuDeviceMatrix{Float32, 1}, 
    Q_grid::CuDeviceArray{Float32, 3, 1}, albedo_grid::CuDeviceArray{Float32, 3, 1}, sw_flux::Float32, dlat::Float32, dlon::Float32 )
    
    index, stride = threadIdx().x, blockDim().x
    for i = index:stride:length(intersections)

        # Initialize all rays on a uniform random disk representing the incoming stellar light 
        intersections[i], objects[i,1], objects[i,2] = -1, 0, 0
        pos[i,1] = R_layers[1] * (2*rand() - 1) 
        pos[i,2] = -1.1 * R_layers[end]
        pos[i,3] = R_layers[1] * (2*rand() - 1)
        direction[i,1] = 0.
        direction[i,2] = 1.
        direction[i,3] = 0.
        energy[i] = sw_flux

        # Check only the spheres that interact with shortwave light (for now, only the planetary surface at j=1)
        j = 1
        a, b, c = 0, 0, 0
        for axis in 1:3
            @inbounds offsets[i,axis] = pos[i,axis] - C_layers[j,axis]
            a = a + direction[i,axis] * direction[i,axis]
            b = b + offsets[i,axis] * direction[i,axis]
            c = c + offsets[i,axis] * offsets[i,axis] 
        end

        c -= R_layers[j]*R_layers[j]
        discriminant = b*b - a*c
        if discriminant >= 0
            t = (-b - sqrt(discriminant))/a 
            if (t >= 1e-4) && ( (intersections[i] < 0) || (t < intersections[i]) )
                @inbounds intersections[i] =  t
            end
            t = (-b + sqrt(discriminant))/a 
            if (t >= 1e-4) && ( (intersections[i] < 0) || (t < intersections[i]) )
                @inbounds intersections[i] =  t
            end
        end
        
        # Compute the grid cell where the ray strikes the planetary surface 
        if intersections[i] >= 0
            
            a = pos[i,1] + intersections[i] * direction[i,1] - C_layers[j,1] 
            b = pos[i,2] + intersections[i] * direction[i,2] - C_layers[j,2]
            c = pos[i,3] + intersections[i] * direction[i,3] - C_layers[j,3]
            lat = atan(b, a)
            lon = atan(sqrt(a*a + b*b), c)
            a1 = 1 + Int( floor( (lat+pi) / dlat ) )
            a2 = 1 + Int( floor( lon / dlon ) )
            # Probability of absorbing or reflecting into space from albedo (0 always absorbs, 1 always reflects)
            if rand() > albedo_grid[j,a1,a2]   
                Q_grid[j,a1,a2] += energy[i]
            end
        end
    end
    return  
end

function longwave_bounce!(intersections::CuDeviceVector{Float32, 1}, pos::CuDeviceMatrix{Float32, 1}, direction::CuDeviceMatrix{Float32, 1}, energy::CuDeviceVector{Float32, 1}, 
    objects::CuDeviceMatrix{Int32, 1}, offsets::CuDeviceMatrix{Float32, 1}, R_layers::CuDeviceVector{Float32, 1}, C_layers::CuDeviceMatrix{Float32, 1}, 
    T_grid::CuDeviceArray{Float32, 3, 1}, Q_grid::CuDeviceArray{Float32, 3, 1}, normalization::CuDeviceMatrix{Float32, 1}, opacity_grid::CuDeviceMatrix{Float32, 1}, dlat::Float32, dlon::Float32)
    """
    DISCLAIMER: There is a rare out-of-bounds error that didn't show up in any of my tests, or the results that I ran over two days, but 
                appeared after restarting the kernel in the final 12 hours. I didn't have time to track it down, or figure out what change broke things,
                but it tends to occur after 100 or so timesteps. It didn't affect the results presented in my report, since they used an earlier version 
                (which ran to 1000 timesteps without issue) but should be noted for anyone who plans to use this version"""
                
    N_layers = length(R_layers)
    N_rays = length(intersections)
    N_lat = size(T_grid,2)
    N_lon = size(T_grid,3)
    index, stride = threadIdx().x, blockDim().x

    for i = index:stride:N_rays

        # Initialize rays uniformly across all grid cells (with an energy weighted by cell area)
        intersections[i], objects[i,1], objects[i,2] = -1, 0, 0
        j = 1 + Int32( floor(N_layers*rand()) )
        lon = 2*pi*rand()
        lat = pi*rand()
        a1 = 1 + Int( floor( lat / dlat ) )
        a2 = 1 + Int( floor( lon / dlon ) )

        # Check for out-of-bounds errors (I started getting one last-minute after restarting my kernel, and I haven't tracked it down yet)
        if (j > N_layers) j = N_layers
        elseif (j < 1) j = 1 end
        if (j > N_layers) j = N_layers
        elseif (j < 1) j = 1 end
        if (a1 > N_lat) a1 = N_lat 
        elseif (a1 < 1) a1 = 1 end
        if (a2 > N_lon) a2 = N_lon 
        elseif (a2 < 1) a2 = 1 end

        pos[i,1] = R_layers[j] * cos(lon) * sin(lat) 
        pos[i,2] = R_layers[j] * sin(lon) * sin(lat)
        pos[i,3] = R_layers[j] * cos(lat)

        t = T_grid[j,a1,a2]
        energy[i] = normalization[a1,a2] * (t*t*t*t) 
        Q_grid[j,a1,a2] -= energy[i]
        
        # Emit in a random (spherically symmetric) direction using rejection sampling
        len = 2
        while len > 1
            xr, yr, zr = 2*rand()-1, 2*rand()-1, 2*rand()-1
            len = xr*xr + yr*yr + zr*zr
        end
        len = sqrt(len)
        xr /= len
        yr /= len
        zr /= len
        direction[i,1] = xr  
        direction[i,2] = yr   
        direction[i,3] = zr 
        
        # Compute direction factor D (estimates increased path length for non-radial motion)  
        len = sqrt(pos[i,1]*pos[i,1] + pos[i,2]*pos[i,2] + pos[i,3]*pos[i,3])
        xr = pos[i,1] / len
        yr = pos[i,2] / len
        zr = pos[i,3] / len
        D = xr*direction[i,1] + yr*direction[i,2] + zr*direction[i,3]
        
        if D < 0
            D = -1/D
            column = 2
            lower, upper = N_layers - j, N_layers
        else
            D = 1/D
            column = 1
            lower, upper = j, N_layers
        end

        
        # Find which layer (if any) absorbs based on the opacity
        counter = 0
        p = rand()
        if column == 1 && p < exp(-D*opacity_grid[j,1])
            # Longwave light passes through the atmosphere into space
            continue
        elseif column == 2 && p < exp(-D* (opacity_grid[1,1] - opacity_grid[j,1]) )
            # Longwave light is absorbed by the surface
            j = 1
        else
            # Longwave light is absorbed somewhere in the atmosphere
            current = Int32(floor( 0.5*(lower + upper) ) )
            prob = exp(-D* (opacity_grid[current,column] - opacity_grid[j,1]) ) 
            while upper - lower > 1 && counter < 10
                current = Int32(floor( 0.5*(lower + upper) ) )
                #prob = exp(-D*opacity_grid[current,column]) - exp(-D*opacity_grid[j,1])
                prob = exp(-D* (opacity_grid[current,column] - opacity_grid[j,1]) ) 
                counter += 1
                if p > prob 
                    lower = current
                elseif p < prob
                    upper = current 
                elseif upper - lower == 1
                    break
                end
            end
            j = lower
        end

        a, b, c = 0, 0, 0
        for axis in 1:3
            @inbounds offsets[i,axis] = pos[i,axis] - C_layers[j,axis]
            a = a + direction[i,axis] * direction[i,axis]
            b = b + offsets[i,axis] * direction[i,axis]
            c = c + offsets[i,axis] * offsets[i,axis] 
        end

        c -= R_layers[j]*R_layers[j]
        discriminant = b*b - a*c
        if discriminant >= 0
            t = (-b - sqrt(discriminant))/a 
            if (t >= 1e-4) && ( (intersections[i] < 0) || (t < intersections[i]) )
                @inbounds intersections[i] =  t
            end
            t = (-b + sqrt(discriminant))/a 
            if (t >= 1e-4) && ( (intersections[i] < 0) || (t < intersections[i]) )
                @inbounds intersections[i] =  t
            end
        end
        
        
        # Compute the grid cell where the ray strikes the absorbing layer  
        if intersections[i] >= 0
            a = pos[i,1] + intersections[i] * direction[i,1] - C_layers[j,1] 
            b = pos[i,2] + intersections[i] * direction[i,2] - C_layers[j,2]
            c = pos[i,3] + intersections[i] * direction[i,3] - C_layers[j,3]
            lat = atan(b, a)
            lon = atan(sqrt(a*a + b*b), c)
            a1 = 1 + Int( floor( 2*(lat+pi) / dlat ) )
            a2 = 1 + Int( floor( lon / dlon ) )
            Q_grid[j,a1,a2] += energy[i]
        end
    end
end

# Number of emitted rays at each shortwave/longwave step
N_rays = Int(2.5e7)

# Stefan-Boltzmann constant (SI: W m^−2 K^−4) after a m^-2 => km^-2 conversion (multiplying by 1e6)
stefan_boltzmann = Float32(5.67e-2)


# Scale height and planet/orbit (in km) and resulting geometries of all planetary layers 
scale_height = 8.5
R_planet = 6370.0
R_star = 695700. 
R_orbit = 150.0e6
N_layers = 101
# The depth of the temperature-varying layer of the surface (in km) 
surface_depth = 4.0

# Thermodynamic properties of the star and planet (temperatures in Kelvin, pressures in Pa)
T_star    = 5700.
T_planet = 200.
P_s      = 1001.
P_toa    = 54.51
gamma    = 0.6655
lw_opacity0 = 5.
freezing_point = 273.15
frozen_albedo = 0.9
melted_albedo = 0.1


# Angular resolution of the spherical grid (defines lat/long step sizes in radians)
N_lat, N_lon = 16, 24
lat, lon   = LinRange(-pi/2, pi/2, N_lat+1), LinRange(-pi, pi, N_lon+1)
dlat, dlon = lat[2]-lat[1], lon[2]-lon[1] 
lat_grid = lat' .* ones(N_lon+1)
lon_grid = ones(N_lat+1)' .* lon 

# Compute the area of each lat-lon cell (in km^2) and scale with the SB contant
areas_km = ( sin.(lat_grid[2:end,2:end]) .- sin.(lat_grid[1:end-1,1:end-1]) ) * R_planet^2 
areas_km .*= (lon_grid[2:end,2:end] - lon_grid[1:end-1,1:end-1]) .% (2*pi)
areas_km = areas_km'
#normalization = stefan_boltzmann * (1 ./ areas_km) * N_layers * N_lat * N_lon / N_rays
normalization = stefan_boltzmann * (areas_km / sum(areas_km) ) * N_layers * N_lat * N_lon / N_rays
normalization = stefan_boltzmann * areas_km * N_layers * N_lat * N_lon / N_rays


# 1D grids for pressure and shortwave/longwave opacity (which which vary in elevation only) as well as the atmospheric elevation in km 
P_grid   = LinRange(P_s, P_toa, N_layers)
opacity_grid = lw_opacity0 * (P_grid / P_s).^4
opacity_grid = hcat( opacity_grid, opacity_grid[1] .- opacity_grid[end:-1:1] )
R_layers = R_planet .- scale_height * log.(P_grid / P_s)
C_layers = zeros(N_layers, 3)


# 3D grids for radiative heat exchange, temperature, and albedo (which all vary across each layer)
Q_grid  = zeros(Float32, N_layers, N_lat, N_lon)
T_mean  = T_planet * (P_grid / P_s ).^gamma
T_grid  = cat([ T * ones(N_lat, N_lon) for T in T_mean ]..., dims=3)
T_grid  = permutedims(T_grid, [3,1,2])
T_grid  = Float32.(T_grid)

# Assigns a nonzero albedo to the surface (lower than Earth's ~0.3 because we are neglecting clouds)
albedo_grid  = zeros(N_layers, N_lat, N_lon)

if T_planet < freezing_point
    albedo_grid[1,:,:] .= frozen_albedo
else
    albedo_grid[1,:,:] .= melted_albedo
end


# Ray initialization (shortwave only)
sw_flux = stefan_boltzmann * (T_star^4) * (4 * pi * R_star^2) / (4 * pi * R_orbit^2)
sw_flux *= pi * (R_layers[end]^2) / N_rays
x = R_layers[end] * (2*rand(N_rays) .- 1) 
z = R_layers[end] * (2*rand(N_rays) .- 1)
y = -(1.1*R_layers[end]) * ones(N_rays) 
pos = Float32.( [ x y z ] ) 
direction = Float32.( [ zeros(N_rays) ones(N_rays) zeros(N_rays) ] )


# Convert everything to a GPU-friendly format
sw_flux, dlat, dlon = Float32(sw_flux), Float32(dlat), Float32(dlon)
pos_gpu = cu(pos)
direction_gpu = cu(direction)
R_gpu   = cu(R_layers)
C_gpu   = cu(C_layers)
T_gpu   = cu(T_grid)
albedo_gpu   = cu(albedo_grid)
Q_gpu   = cu(Q_grid)
opacity_gpu = cu( Array(opacity_grid) )
norm_gpu = cu(normalization)
energy = cu( zeros(Float32, N_rays) )
objects = cu( zeros(Int32, N_rays, 2) )
offsets = cu( zeros(Float32, N_rays, 3) )
intersections = cu(-ones(Float32, N_rays) ) 

# Defines how temperature depends with heat exchange, considering a planet covered by surface water
heat_capacity = 4180.           # J / kg*K
mean_area = sum(areas_km) / length(areas_km)
scaling = mean_area * surface_depth #* 1e12 
for i in 2:N_layers
    global scaling
    scaling = cat(scaling, mean_area * (R_layers[i]-R_layers[i-1]), dims=3)
end
scaling = permutedims(scaling, [3,1,2])
scaling *= heat_capacity



N_timesteps = 750
dt = 1e-4


for iter in 1:3

    if iter == 1
        jacobi_weight = 1.0
    elseif iter == 2
        jacobi_weight = 0.0
    elseif iter == 3
        jacobi_weight = 0.5
    end
        
    T_bounds, Q_bounds = zeros(N_timesteps,2), zeros(N_timesteps,2)
    filename = string(Int(T_planet)) * "K-albedo-" * string(Int(floor(100*jacobi_weight))) 
    filename *= "jacobi-" * string(rand(1:100000000)) * "-hysteresis"
    println("Starting RT loop")


    for timestep = 1:N_timesteps
        
        global T_grid, Q_grid, albedo_grid, T_gpu, Q_gpu, albedo_gpu, scaling, dt, sw_flux


        Q_gpu = cu(zeros(size(Q_gpu)))


        @cuda threads=256 shortwave_bounce!(intersections, pos_gpu, direction_gpu, energy, objects, offsets, R_gpu, C_gpu, Q_gpu, albedo_gpu, sw_flux2, dlat, dlon)
        synchronize() 

        sw_energy = sum(Array(energy))
        println("\nStep ", timestep)
        println("\tTotal shortwave energy: ", sw_energy, " W" )

        @cuda threads=256 longwave_bounce!(intersections, pos_gpu, direction_gpu, energy, objects, offsets, R_gpu, C_gpu, T_gpu, Q_gpu, norm_gpu, opacity_gpu, dlat, dlon)
        synchronize()
        lw_energy = sum(Array(energy))
        println("\tTotal longwave energy: ", lw_energy, " W" )

        Q_grid = Array(Q_gpu)
        dT = dt * Q_grid ./ scaling 
        Q_bounds[timestep,:] .= minimum(dT), maximum(dT)
        println( "\tTemperature changes range: ", Q_bounds[timestep,:] )

        T_grid = Array(T_gpu)
        T_grid .+= dT 
        T_grid2 = copy(T_grid)
        for i in 1:N_lat
            for k in 1:N_lon
                for j in 1:N_layers
                    i1, i2, k1, k2 = i+1, i-1, k+1, k-1
                    if (i1 == N_lat+1) i1 = 1 end
                    if (i2 == 0) i2 = N_lat end
                    if (k1 == N_lon+1) k1 = 1 end
                    if (k2 == 0) k2 = N_lon end
                    T_grid2[j,i,k] = (1-jacobi_weight)*T_grid[j,i,k] + 0.25*jacobi_weight*(T_grid[j,i1,k] + T_grid[j,i2,k] + T_grid[j,i,k1] + T_grid[j,i,k2] )
                end
                if T_grid2[1,i,k] < freezing_point
                    albedo_grid[1,i,k] = frozen_albedo
                else
                    albedo_grid[1,i,k] = melted_albedo 
                end
            end
        end
        T_bounds[timestep,:] .= minimum(T_grid2[1,:,:]), maximum(T_grid2[1,:,:])
        
        T_gpu = cu(T_grid2)
        albedo_gpu = cu(albedo_grid)
    end
end