using CUDA, Random, Plots, Images

function find_intersections!(intersections::CuDeviceVector{Float32, 1}, pos::CuDeviceMatrix{Float32, 1}, direction::CuDeviceMatrix{Float32, 1}, 
                                R_cir::CuDeviceVector{Float32, 1}, C_cir::CuDeviceMatrix{Float32, 1}, V_tri::CuDeviceArray{Float32, 3, 1}, objects::CuDeviceMatrix{Int32, 1},
                                offsets::CuDeviceMatrix{Float32, 1}, E1::CuDeviceMatrix{Float32, 1}, E2::CuDeviceMatrix{Float32, 1}, T::CuDeviceMatrix{Float32, 1}, 
                                P::CuDeviceMatrix{Float32, 1}, Q::CuDeviceMatrix{Float32, 1} )
    index, stride = threadIdx().x, blockDim().x
    for i = index:stride:length(intersections)

        intersections[i], objects[i,1], objects[i,2] = -1, 0, 0
        # Check all spheres for intersections
        for j = 1:length(R_cir)
            a, b, c = 0, 0, 0
            for axis in 1:3
                @inbounds offsets[i,axis] = pos[i,axis] - C_cir[j,axis]
                a = a + direction[i,axis] * direction[i,axis]
                b = b + offsets[i,axis] * direction[i,axis]
                c = c + offsets[i,axis] * offsets[i,axis] 
            end
            c -= R_cir[j]*R_cir[j]
            discriminant = b*b - a*c
            if discriminant >= 0
                t = (-b - sqrt(discriminant))/a 
                if (t >= 1e-4) && ( (intersections[i] < 0) || (t < intersections[i]) )
                    @inbounds intersections[i] =  t
                    @inbounds objects[i,1] = 1
                    @inbounds objects[i,2] = j
                end
                t = (-b + sqrt(discriminant))/a 
                if (t >= 1e-4) && ( (intersections[i] < 0) || (t < intersections[i]) )
                    @inbounds intersections[i] =  t
                    @inbounds objects[i,1] = 1
                    @inbounds objects[i,2] = j
                end
            end 
        end

        # Check all triangles for intersections
        for j in 1:size(V_tri,3)
            for axis in 1:3
                @inbounds E1[i,axis] = V_tri[2,axis,j] - V_tri[1,axis,j]
                @inbounds E2[i,axis] = V_tri[3,axis,j] - V_tri[1,axis,j]
                @inbounds  T[i,axis] = pos[i,axis] - V_tri[1,axis,j]
            end
            
            # Compute cross products
            @inbounds P[i,1] = direction[i,2]*E2[i,3]-direction[i,3]*E2[i,2] 
            @inbounds P[i,2] = direction[i,3]*E2[i,1]-direction[i,1]*E2[i,3]
            @inbounds P[i,3] = direction[i,1]*E2[i,2]-direction[i,2]*E2[i,1]
            @inbounds Q[i,1] = T[i,2]*E1[i,3] - T[i,3]*E1[i,2]
            @inbounds Q[i,2] = T[i,3]*E1[i,1] - T[i,1]*E1[i,3]
            @inbounds Q[i,3] = T[i,1]*E1[i,2] - T[i,2]*E1[i,1] 
            
            t, u, v, denom = 0, 0, 0, 0
            for d in 1:3
                t += Q[i,d] * E2[i,d]
                u += P[i,d] * T[i,d]
                v += Q[i,d] * direction[i,d]
                denom += P[i,d] * E1[i,d]
            end
            t /= denom
            u /= denom
            v /= denom
            
            if (0 <= u+v <= 1) && (0<=u<=1) && (0<=v<=1)
                if (t >= 1e-6) && ( (intersections[i] < 0) || (t < intersections[i]) ) 
                    @inbounds intersections[i] =  t
                    @inbounds objects[i,1] = 2
                    @inbounds objects[i,2] = j
                end
            end
        end

    end
    return  
end



function bounce!(intersections::CuDeviceVector{Float32, 1}, pos::CuDeviceMatrix{Float32, 1}, direction::CuDeviceMatrix{Float32, 1}, 
                 R_cir::CuDeviceVector{Float32, 1}, C_cir::CuDeviceMatrix{Float32, 1}, V_tri::CuDeviceArray{Float32, 3, 1}, objects::CuDeviceMatrix{Int32, 1},
                 ray_color::CuDeviceMatrix{Float32, 1}, cir_color::CuDeviceMatrix{Float32, 1}, tri_color::CuDeviceMatrix{Float32, 1}, cir_material::CuDeviceVector{Int64, 1},
                 image::CuDeviceArray{Float32, 3, 1}, I::CuDeviceVector{Int32, 1}, J::CuDeviceVector{Int32, 1}, dx::Float32, dz::Float32 )
    index, stride = threadIdx().x, blockDim().x
    for i = index:stride:length(intersections)

        if intersections[i] <= 0 || (ray_color[i,1] + ray_color[i,2] + ray_color[i,3]) < 0.1 
            # Replace ray
            image[1,I[i],J[i]] += ray_color[i,1] * 0.65
            image[2,I[i],J[i]] += ray_color[i,2] * 0.65
            image[3,I[i],J[i]] += ray_color[i,3] 
            image[4,I[i],J[i]] += 1

            ray_color[i,1], ray_color[i,2], ray_color[i,3] = 1, 1, 1
            pos[i,1], pos[i,2], pos[i,3] = 10*rand(), -5, 5*rand()
            direction[i,1], direction[i,2], direction[i,3] = 0, 1, -0.1
            I[i], J[i] = 1+floor(Int32,pos[i,1]/dx), 1+floor(Int32, pos[i,3]/dz)
        else

            pos[i,1] += intersections[i]*direction[i,1]
            pos[i,2] += intersections[i]*direction[i,2]
            pos[i,3] += intersections[i]*direction[i,3]
            
            j = objects[i,2]
            if objects[i,1] == 1
                ray_color[i,1] *= cir_color[j,1]
                ray_color[i,2] *= cir_color[j,2]
                ray_color[i,3] *= cir_color[j,3]
                
                nx = pos[i,1] - C_cir[j,1]
                ny = pos[i,2] - C_cir[j,2]
                nz = pos[i,3] - C_cir[j,3]
                total = sqrt(nx*nx + ny*ny + nz*nz)
                nx /= total
                ny /= total
                nz /= total
                if cir_material[j] == 1
                    # reflect perfectly (TODO: add some diffusivity to specular reflections)
                    b = 2 * (nx*direction[i,1] + ny*direction[i,2] + nz*direction[i,3]) 
                    direction[i,1] -= b * nx
                    direction[i,2] -= b * ny
                    direction[i,3] -= b * nz
                elseif cir_material[j] == 2
                    # Diffusively reflect in a random (Lambertian) direction using rejection sampling
                    total2 = 2
                    while total2 > 1
                        xr, yr, zr = 2*rand()-1, 2*rand()-1, 2*rand()-1
                        total2 = xr*xr + yr*yr + zr*zr
                    end
                    total2 = sqrt(total2)
                    xr /= total2
                    yr /= total2
                    zr /= total2
                    direction[i,1] = nx + xr  
                    direction[i,2] = ny + yr   
                    direction[i,3] = nz + zr  

                elseif cir_material[j] == 3
                    # transmit/refract
                    
                    dot = sqrt(direction[i,1]*direction[i,1] + direction[i,2]*direction[i,2] + direction[i,3]*direction[i,3])
                    direction[i,1] /= dot
                    direction[i,2] /= dot
                    direction[i,3] /= dot
                    
                    dot = -(nx*direction[i,1] + ny*direction[i,2] + nz*direction[i,3])
                    refraction = 1.5
                    if dot > 0
                        refraction = 1. / refraction
                    end
                    
                    cos_theta = -dot
                    if cos_theta > 1
                        cos_theta = 1.
                    end
                    sin_theta = sqrt(1 - cos_theta*cos_theta)
                    if refraction * sin_theta > 1
                        # reflect perfectly (update to add slight diffusivity)
                        b = 2 * dot 
                        direction[i,1] -= b * nx
                        direction[i,2] -= b * ny
                        direction[i,3] -= b * nz
                    else
                        # refract with Snell's law
                        p1 = refraction * (direction[i,1] + nx*cos_theta)
                        p2 = refraction * (direction[i,2] + ny*cos_theta)
                        p3 = refraction * (direction[i,3] + nz*cos_theta)
                        p_tot = p1*p1 + p2*p2 + p3*p3

                        direction[i,1] = p1 - nx*sqrt( abs(1 - p_tot) ) 
                        direction[i,2] = p2 - ny*sqrt( abs(1 - p_tot) ) 
                        direction[i,3] = p3 - nz*sqrt( abs(1 - p_tot) ) 
                    end

                end

            elseif objects[i,1] == 2
                ray_color[i,1] *= tri_color[j,1]  
                ray_color[i,2] *= tri_color[j,2] 
                ray_color[i,3] *= tri_color[j,3] 
                # Diffusively reflect off ground (generalize! this hard-codes a [0,0,1] normal)
                total2 = 2
                while total2 > 1
                    xr, yr, zr = 2*rand()-1, 2*rand()-1, 2*rand()-1
                    total2 = xr*xr + yr*yr + zr*zr
                end
                total2 = sqrt(total2)
                xr /= total2
                yr /= total2
                zr /= total2
                direction[i,1] = xr  
                direction[i,2] = yr   
                direction[i,3] = abs(zr)  

            end
        end
    end
    return
end


N_rays = Int(2.5e7)
N_spheres = 25
N_triangles = 2

# Size of each camera pixel in spatial coordinates
dx, dz = Float32(0.01), Float32(0.01)


R_cir = 0.25*Float32.( 1 .+ rand(N_spheres) )


C_cir = zeros(N_spheres,2)
for i in 1:N_spheres
    overlap = true
    sample = [0, R_cir[i]] .+ [-5, -5] .+ rand(2) .* [15, 25]
    while overlap == true
        sample = [0, R_cir[i]] .+ [-5, -5] .+ rand(2) .* [15, 25]
        overlap = false 
        for j in 1:(i-1)
            if sum((sample .- C_cir[j,:]).^2 ) < (R_cir[i]+R_cir[j])^2
                overlap = true
            end
        end
    end
    C_cir[i,:] .= sample
end
C_cir = Float32.( [ C_cir R_cir ]  )
V_tri = Float32.( cat( [-100 -100 0 ; -100 20 0; 100 20 0], [-100 -100 0; 100 -100 0; 100 20 0], dims=3) )


# Generate random rays to cast from the camera
x = 10*rand(N_rays) 
z = 5*rand(N_rays) 
y = -5*ones(N_rays) 
pos = Float32.( [ x y z ] ) 
direction = Float32.( [ zeros(N_rays) ones(N_rays) -0.1*ones(N_rays) ] )

I, J = cu( Int32(1) .+ floor.(Int32, x/dx)), Int32(1) .+ cu(floor.(Int32,z/dz))
xdim = 1 .+ ceil.(Int32, 10/dx) 
zdim = 1 .+ ceil.(Int32, 5/dz)
image_gpu = cu( zeros(Float32, 4, xdim, zdim) )


# Create all GPU data structures
pos_gpu = cu(pos)
direction_gpu = cu(direction)
R_cir_gpu = cu(R_cir)
C_cir_gpu = cu(C_cir)
V_tri_gpu = cu(V_tri)


# Some data structures used to handle intermediate computations (to lessen the burden on registers)
objects = cu( zeros(Int32, N_rays, 2) )
offsets = cu( zeros(Float32, N_rays, 3) )
E1 = cu( zeros(Float32, N_rays,3) )
E2 = cu( zeros(Float32, N_rays,3) )
T = cu( zeros(Float32, N_rays,3) )
Q = cu( zeros(Float32, N_rays,3) )
P = cu( zeros(Float32, N_rays,3) )


# Assign colors and materials to all objects
ray_color = cu( ones(Float32, N_rays,3) )
tri_color = cu( ones(Float32, N_triangles,3) .* [0.5 1 0.5] )

cir_color = 0.25 .+ 0.75 * rand(Float32, N_spheres,3) 
too_green = (cir_color[:,2] .> cir_color[:,1]) .* (cir_color[:,2] .> cir_color[:,3])
temp = cir_color[too_green,2]
cir_color[too_green,2] .= cir_color[too_green,1]
cir_color[too_green,1] .= temp

materials = zeros(N_spheres)
for i in 1:N_spheres
    p = rand()
    if p < 0.35
        materials[i] = 1
    elseif p < 0.85  
        materials[i] = 2
    else
        materials[i] = 3
        cir_color[i,:] .=  0.8 .+ 0.2*rand(3)
    end
end
cir_material = cu( Int64.(materials) )
cir_color = cu(cir_color)

intersections = cu(-ones(Float32, N_rays) ) 


println("Starting RT loop")
for i_bounce in 1:10
    @cuda threads=256 find_intersections!(intersections, pos_gpu, direction_gpu, R_cir_gpu, C_cir_gpu, V_tri_gpu, objects, offsets, E1, E2, T, P, Q)
    synchronize()

    @cuda threads=256 bounce!(intersections, pos_gpu, direction_gpu, R_cir_gpu, C_cir_gpu, V_tri_gpu, objects, ray_color, cir_color, tri_color, cir_material, image_gpu, I, J, dx, dz)
    synchronize()

    println("  Completed bounce ", i_bounce)
    
end

image = Array(image_gpu) 
image ./= (1 .+ image[4:4,:,:])
img = colorview(RGB, permutedims(image[1:3,:,end:-1:1], (1,3,2)) )
display(img)