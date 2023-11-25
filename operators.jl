shift_right(a::Float32) = reinterpret(Float32, reinterpret(Int32, a) >> 1)
shift_left(a::Float32) = reinterpret(Float32, reinterpret(Int32, a) << 1)

function scaled_sigmoid(x::Float32, k::Float32)
    return 1 / (1 + exp(-k * x))
end

map_float32_to_uint32(f::Float32) = floor(UInt32, ((scaled_sigmoid(f, 0.1f0)) * (2^32 - 2^7 - 1)))
magic_add(val::Float32, magic::Float32) = magic == 0f0 ? val : reinterpret(Float32, reinterpret(UInt32, val) + map_float32_to_uint32(magic)) 

magic_inverse(val::Float32) = reinterpret(Float32, -reinterpret(Int32, val))

function fisr(x::Float32, magicnum::Float32)
    x2 = x * Float32(0.5)
    i1 = magic_inverse(x)
    i2 = shift_right(i1)
    y  = magic_add(i2, magicnum)
    y  = y * (Float32(1.5) - (x2 * y * y))
end

# see the mapping function, maybe this should just be a linear transformation? 
plot(range(-100, 100, 100), map_float32_to_uint32.(Float32.(range(-1000, 1000, 100))))