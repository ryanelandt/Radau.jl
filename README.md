# Radau.jl

[![Build Status](https://travis-ci.com/ryanelandt/Radau.jl.svg?branch=master)](https://travis-ci.com/ryanelandt/Radau.jl)
[![codecov.io](https://codecov.io/github/ryanelandt/Radau.jl/coverage.svg?branch=master)](https://codecov.io/github/ryanelandt/Radau.jl?branch=master)

A Julia implementation of Radau as described by [Hairer](http://www.unige.ch/~hairer/preprints/coimbra.pdf).
This implementation is designed to be lightweight and takes a single step at a time.
It is desirable for users who want maintain control of their workflow.
Radau methods are implicit collocation methods that used to integrate stiff differential equations.
Implementations of implicit integrators require a Jacobian to solve the stage values.
This package was primary written to integrate multi-rigid-body dynamics with soft-contact (i.e. state-dependent contact forces).

### Current features
- Adaptive order
- Adaptive time-step

### Future features
- Dense output

### Using 'Radau.jl'
This implementation of Radau requires a structure with field named `de` than contains a differential equation that takes three inputs explained in this paragraph.
The differential equation takes a state vector `x` and writes the derivative of this state vector to an output vector `xx`.
In addition, the differential equation takes the structure with the field `de` as a third argument.
The function you used needs to be callable with both `Float64` and with [`Dual`](https://github.com/JuliaDiff/ForwardDiff.jl).
Relevant code is below.

```Julia
struct SimpleStiffSystem
    λ::Float64
    de::Function
    function SimpleStiffSystem(de::Function)
        return new(1.0, de)
    end
end

function stiff_de!(xx::Vector{T}, x::Vector{T}, s::SimpleStiffSystem) where {T}
    xx .= -s.λ * x
    return nothing
end
```

See `test/basic_test.jl` to see how this is used in a minimum working example.
