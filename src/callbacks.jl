abstract type AbstractCallback end

struct Register <: AbstractCallback
    tape::BitVector
end

function Register(; sizehint::Int=-1)
    tape = BitVector()
    sizehint > 0 && sizehint!(tape, sizehint)
    return Register(tape)
end

function (register::Register)(outcomes::AbstractVector{Bool}, state::SparseState)
    append!(register.tape, outcomes)
    return nothing
end

Base.parent(register::Register) = register.tape
Base.BitVector(register::Register) = parent(register)
Base.Vector(register::Register) = Vector(parent(register))
Base.collect(register::Register) = Vector(register)
Base.empty!(register::Register) = empty!(parent(register))

# Iterator interface
Base.length(register::Register) = length(parent(register))
Base.iterate(register::Register, args...) = iterate(parent(register), args...)
Base.firstindex(register::Register) = firstindex(parent(register))
Base.lastindex(register::Register) = lastindex(parent(register))
Base.getindex(register::Register, i) = getindex(parent(register), i)

function Base.show(io::IO, register::Register)
    return print(io, "Register($(register.tape))")
end

struct Feedback{F<:Function,T<:Tuple} <: AbstractCallback
    trigger::F
    ops::T
end

Feedback(trigger::Function, op::AbstractOperator) = Feedback(trigger, (op,))
Feedback(ops::Tuple) = Feedback(identity, ops)
Feedback(ops::AbstractOperator...) = Feedback(ops)

function (feedback::Feedback)(outcomes::AbstractVector{Bool}, state::SparseState)
    (; trigger, ops) = feedback
    values = trigger(outcomes)
    for (value, op) in zip(values, ops)
        value && apply!(op, state)
    end
    return nothing
end

function Base.show(io::IO, feedback::Feedback)
    print(io, "Feedback", "(")
    print(io, (feedback.trigger == identity) ? "" : "$(feedback.trigger), ")
    print(io, (length(feedback.ops) == 1) ? only(feedback.ops) : feedback.ops)
    print(io, ")")
    return nothing
end
