"""
Optimized temporal schemes that reuse pre-allocated buffers
All schemes follow the pattern: scheme(U, dt, t, F, [buffers])
"""

# Estructura para contener los buffers temporales de RK4
struct RK4Buffers{T,N,ArrayType}
    k1::ArrayType
    k2::ArrayType
    k3::ArrayType
    k4::ArrayType
    U_temp::ArrayType
    result::ArrayType
end

# Constructor para crear buffers del mismo tamaño que U
function RK4Buffers(U::AbstractArray{T,N}) where {T,N}
    ArrayType = typeof(U)
    return RK4Buffers{T,N,ArrayType}(
        similar(U), similar(U), similar(U), similar(U),
        similar(U), similar(U)
    )
end

# Versión optimizada de Euler (sin cambios necesarios)
function Euler_optimized(U, dt, t, F)
    return U .+ dt .* F(U, t)
end

# Versión optimizada de RK4 usando buffers pre-allocados
function RK4_optimized!(U, dt, t, F, buffers::RK4Buffers)
    k1, k2, k3, k4, U_temp, result = buffers.k1, buffers.k2, buffers.k3, buffers.k4, buffers.U_temp, buffers.result
    
    # Etapa 1: k1 = F(U, t)
    k1 .= F(U, t)
    
    # Etapa 2: k2 = F(U + dt*k1/2, t + dt/2)
    @. U_temp = U + dt * k1 / 2
    k2 .= F(U_temp, t + dt/2)

    # Etapa 3: k3 = F(U + dt*k2/2, t + dt/2)
    @. U_temp = U + dt * k2 / 2
    k3 .= F(U_temp, t + dt/2)
    
    # Etapa 4: k4 = F(U + dt*k3, t + dt)
    @. U_temp = U + dt * k3
    k4 .= F(U_temp, t + dt)
    
    # Resultado final: U + dt*(k1 + 2*k2 + 2*k3 + k4)/6
    @. result = U + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

    return result
end

# Wrapper que mantiene la compatibilidad con la interfaz anterior
function RK4_optimized(U, dt, t, F, buffers::RK4Buffers)
    return RK4_optimized!(U, dt, t, F, buffers)
end

# Mantener las funciones originales para compatibilidad hacia atrás
function Euler(U, dt, t, F)
    return U .+ dt .* F(U, t)
end

function RK4(U, dt, t, F)
    k1 = F(U, t)
    k2 = F(U .+ dt .* k1/2, t .+ dt/2)
    k3 = F(U .+ dt .* k2/2, t .+ dt/2)
    k4 = F(U .+ dt .* k3, t .+ dt)
    return U .+ dt .* (k1 .+ 2*k2 .+ 2*k3 .+ k4)/6
end
