# SparseStates.jl


This package efficiently simulates quantum circuits that don't generate a lot of superposition in the computational basis.
In other words, they are few basis states to keep track of.
Many common quantum gates, such as *X*, *Z*, *S*, *CX*, *CZ* along with several non-Clifford gates such as _T_ or _CCZ_ map each computational state to a single computational state, i.e. they are not **branching**.
If there are few branching gates, such as the Hadamard gate *H*, the final state can be computed efficiently, since the state is effectively very **sparse**.
Computationally, the complexity grows exponentially with the number of *H* gates, both in time and space.

## Installation

<p>
    SparseStates is a <a href="https://julialang.org"><img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">&nbsp;Julia Language</a> package.
    To install this package, please clone this repository.
    Open <a href="https://docs.julialang.org/en/v1/manual/getting-started/">Julia's interactive session</a> (known as REPL) and press the <kbd>]</kbd> key in the REPL to use the package mode.
    Then type the following command
</p>

```
pkg> add <local_path>
```

where `<local_path>` is the path to the downloaded repository.

## Getting started

To generate a quantum register with two qubits initialized in the *|00⟩* state, we just have do the following

```julia
using SparseStates

state_initial = SparseState(2)
```

`SparseState`s are concrete types of `AbstractDict`, but should behave like `AbstractArrays` for basic algebraic manipulations.

We can then generate a quantum circuit

```julia
circuit = H(1) * CX(1, 2)
```

Gates can be defined to act on multiple qubits simultaneously, such as `CX([(1, 2), (3, 4)]` or equivalently `CX([1, 3], [2, 4])`. 

Let's now apply it to the state

```julia
state_final = circuit(state_initial)
```

We can also use Julia's built-in pipe operator `|>` to write the more suggestive
```julia
state_final = state_initial |> circuit
```

A variety of common gates are supported as well as quantum channels such as `Measure` and `Reset`.
Notice that in these cases a measurement outcome is chosen stochastically, so the result may differ between simulations.
In such cases one has to take over multiple simulations, for example

```julia

using LinearAlgebra

shots = 100
outcomes = [abs2(dot(state_initial, Reset(2), state_final)) for _ in 1:shots]

sum(outcomes) / shots
```




## Compatibility

This package is compatible with Julia 1.9 and above.
