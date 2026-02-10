using TensorKit

""" 
    getCorrelation(Tpure, Timp1, Timp2, dist, niter, trunc)

Computes the correlation function by running the HOTRG algorithm with pure and impurity tensors.

Arguments:
- `Tpure`: Pure tensor.
- `Timp1`, `Timp2`: Impurity tensors.
- `dist`: Distance between impurities.
- `niter`: Number of HOTRG iterations.
- `trunc`: Truncation scheme (`TensorKit.TruncationScheme`).

Returns: The ratio of impurity to pure tensor norms after HOTRG.
"""
function getCorrelation(Tpure, Timp1, Timp2, dist, niter, trunc::TensorKit.TruncationScheme)
    Tpure, Timp = CorrelationHOTRG(Tpure, Timp1, Timp2, dist, niter, trunc)
    npure = norm(@tensor Tpure[1 2; 2 1])
    nimp = norm(@tensor Timp[1 2; 2 1])
    return nimp / npure
end

"""
    CorrelationHOTRG(Tpure, Timp1, Timp2, dist, niter, trunc)

Runs the HOTRG algorithm for a given number of iterations, separating the process into three phases.

Arguments:
- `Tpure`: Pure tensor.
- `Timp1`, `Timp2`: Impurity tensors.
- `dist`: Distance between impurities.
- `niter`: Number of HOTRG iterations.
- `trunc`: Truncation scheme.

Returns: Final pure and impurity tensors after HOTRG.
"""
function CorrelationHOTRG(Tpure, Timp1, Timp2, dist, niter, trunc::TensorKit.TruncationScheme)
    # Assert realistic calculation
    @assert niter > dist "niter must be larger than dist"

    # PHASE 1 (dist times)
    Tpure1 = Tpure
    Tpure2 = Tpure

    for _ in 1:dist
        # Tensor 1
        Tpure1, Timp1 = phase1(Tpure1, Timp1, trunc)
        # Tensor 2
        Tpure2, Timp2 = phase1(Tpure2, Timp2, trunc)   
        # normalize
        Tpure1, Tpure2, Timp1, Timp2 = finalize_phase1(Tpure1, Timp1, Timp2)
    end
    
    # PHASE 2 (1 time)
    Tpure, Timp = phase2(Tpure1, Timp1, Timp2, trunc)
    # normalize
    Tpure, Timp = finalize_phase23(Tpure, Timp)

    # PHASE 3 (niter - dist - 1 times)
    n = niter - dist - 1
    
    # PHASE 1 (dist times)
    for _ in 1:n
        Tpure, Timp = phase3(Tpure, Timp, trunc)
        # normalize
        Tpure, Timp = finalize_phase23(Tpure, Timp)
    end

    return Tpure, Timp
end

"""
    phase1(Tpure, Timp, trunc)

Performs one HOTRG step for phase 1, updating pure and impurity tensors.

Arguments:
- `Tpure`: Pure tensor.
- `Timp`: Impurity tensor.
- `trunc`: Truncation scheme.

Returns: Updated pure and impurity tensors.
"""
function phase1(Tpure, Timp, trunc::TensorKit.TruncationScheme)
    Uy, _ = _get_hotrg_yproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_x(Tpure, Tpure, Uy)
    Timp = 0.5 * (_step_hotrg_x(Timp, Tpure, Uy) + _step_hotrg_x(Tpure, Timp, Uy))
    Tpure = Tpure_temp

    Ux, _ = _get_hotrg_xproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_y(Tpure, Tpure, Ux)
    Timp = 0.5 * (_step_hotrg_y(Timp, Tpure, Ux) + _step_hotrg_y(Tpure, Timp, Ux))
    Tpure = Tpure_temp

    return Tpure, Timp
end

"""
    phase2(Tpure, Timp1, Timp2, trunc)

Performs one HOTRG step for phase 2, combining two impurity tensors.

Arguments:
- `Tpure`: Pure tensor.
- `Timp1`, `Timp2`: Impurity tensors.
- `trunc`: Truncation scheme.

Returns: Updated pure and impurity tensors.
"""
function phase2(Tpure, Timp1, Timp2, trunc::TensorKit.TruncationScheme)
    Uy, _ = _get_hotrg_yproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_x(Tpure, Tpure, Uy)
    Timp = 0.5 * (_step_hotrg_x(Timp1, Timp2, Uy) + _step_hotrg_x(Timp2, Timp1, Uy))
    Tpure = Tpure_temp

    Ux, _ = _get_hotrg_xproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_y(Tpure, Tpure, Ux)
    Timp = 0.5 * (_step_hotrg_y(Timp, Tpure, Ux) + _step_hotrg_y(Tpure, Timp, Ux))
    Tpure = Tpure_temp

    return Tpure, Timp
end

"""
    phase3(Tpure, Timp, trunc)

Performs one HOTRG step for phase 3, updating pure and impurity tensors.

Arguments:
- `Tpure`: Pure tensor.
- `Timp`: Impurity tensor.
- `trunc`: Truncation scheme.

Returns: Updated pure and impurity tensors.
"""
function phase3(Tpure, Timp, trunc::TensorKit.TruncationScheme)
    Uy, _ = _get_hotrg_yproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_x(Tpure, Tpure, Uy)
    Timp = 0.5 * (_step_hotrg_x(Timp, Tpure, Uy) + _step_hotrg_x(Tpure, Timp, Uy))
    Tpure = Tpure_temp

    Ux, _ = _get_hotrg_xproj(Tpure, Tpure, trunc)

    Tpure_temp = _step_hotrg_y(Tpure, Tpure, Ux)
    Timp = 0.5 * (_step_hotrg_y(Timp, Tpure, Ux) + _step_hotrg_y(Tpure, Timp, Ux))
    Tpure = Tpure_temp

    return Tpure, Timp
end


"""
    finalize_phase1(Tpure, Timp1, Timp2)

Normalizes pure and impurity tensors after phase 1.

Arguments:
- `Tpure`: Pure tensor.
- `Timp1`, `Timp2`: Impurity tensors.

Returns: Normalized pure tensor (twice) and impurity tensors.
"""
function finalize_phase1(Tpure, Timp1, Timp2)
    npure = norm(@tensor Tpure[1 2; 2 1])
    
    Tpure /= npure
    Timp1 /= npure
    Timp2 /= npure

    return Tpure, Tpure, Timp1, Timp2
end

"""
    finalize_phase23(Tpure, Timp)

Normalizes pure and impurity tensors after phases 2 and 3.

Arguments:
- `Tpure`: Pure tensor.
- `Timp`: Impurity tensor.

Returns: Normalized pure and impurity tensors.
"""
function finalize_phase23(Tpure, Timp)
    npure = norm(@tensor Tpure[1 2; 2 1])
    
    Tpure /= npure
    Timp /= npure

    return Tpure, Timp
end


#=
    HELPERFUNCTIONS
=#


function _step_hotrg_x(
        A1::TensorMap{E, S, 2, 2}, A2::TensorMap{E, S, 2, 2},
        U::TensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the x-direction
                -3
                |
            ┌3--U--4┐
            |       |
        -1--A1--5---A2-- -4
            |       |
            └1--U†-2┘
                |
                -2
    =#

    @tensor T[-1 -2; -3 -4] :=
        A1[-1 1; 3 5] * A2[5 2; 4 -4] * conj(U[1 2; -2]) * U[3 4; -3]
    return T
end

function _step_hotrg_y(
        A1::TensorMap{E, S, 2, 2}, A2::TensorMap{E, S, 2, 2},
        U::TensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the y-direction
                    -3
                    |
            ┌---1---A2---3--┐
            |       |       |
        -1--U†      5       U-- -4
            |       |       |
            └---2---A1---4--┘
                    |
                    -2
    =#

    @tensor T[-1 -2; -3 -4] :=
        conj(U[1 2; -1]) * U[3 4; -4] * A2[1 5; -3 3] * A1[2 -2; 5 4]
    return T
end

function _get_hotrg_xproj(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    #= join in y-direction, keep x-indices open (A1 below A2)
    M M†                        M† M
            ┌---1---┐                   ┌---1---┐
            ↓       ↑                   ↑       ↓
    -1 -←--A2-←-2--A2†-←- -3    -1 -←--A2†--2-←-A2-←- -3
            ↓       ↑                   ↑       ↓
            5       6                   5       6
            ↓       ↑                   ↑       ↓
    -2 -←--A1-←-4--A1†-←- -4    -2 -←--A1†--4-←-A1-←- -4
            ↓       ↑                   ↑       ↓
            └---3---┘                   └---3---┘
    =#
    # get left unitary
    @plansor MM[-1 -2; -3 -4] :=
        A2[-1 5; 1 2] * A1[-2 3; 5 4] *
        conj(A2[-3 6; 1 2]) * conj(A1[-4 3; 6 4])
    U, s, _, ε = tsvd!(MM; trunc)
    # get right unitary
    @plansor MM[-1 -2; -3 -4] :=
        conj(A2[2 5; 1 -1]) * conj(A1[4 3; 5 -2]) *
        A2[2 6; 1 -3] * A1[4 3; 6 -4]
    _, s′, U′, ε′ = tsvd!(MM; trunc)
    if ε > ε′
        U, s, ε = adjoint(U′), s′, ε′
    end
    return U, s, ε
end

function _get_hotrg_yproj(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        trunc::TensorKit.TruncationScheme
    ) where {E, S}
    #= join in x-direction, keep y-indices open (A1 on the left of A2)
    M M†                        M† M
            -3      -4              -3      -4
            ↓       ↓               ↓       ↓
        ┌-→-A1†--6-→A2†-→┐      ┌-←-A1-←-6--A2-←-┐
        ↑   ↓       ↓    ↓      ↑   ↓       ↓    ↓
        1   2       4    3      1   2       4    3
        ↑   ↓       ↓    ↓      ↑   ↓       ↓    ↓
        └-←-A1-←-5--A2-←-┘      └-→-A1†--5-→A2†-→┘
            ↓       ↓               ↓       ↓
            -1      -2              -1      -2
    =#
    # get bottom unitary
    @plansor MM[-1 -2; -3 -4] :=
        A1[1 -1; 2 5] * A2[5 -2; 4 3] *
        conj(A1[1 -3; 2 6]) * conj(A2[6 -4; 4 3])
    U, s, _, ε = tsvd!(MM; trunc)
    # get top unitary
    @plansor MM[-1 -2; -3 -4] :=
        conj(A1[1 2; -1 5]) * conj(A2[5 4; -2 3]) *
        A1[1 2; -3 6] * A2[6 4; -4 3]
    _, s′, U′, ε′ = tsvd!(MM; trunc)
    if ε > ε′
        U, s, ε = adjoint(U′), s′, ε′
    end
    return U, s, ε
end

function _step_hotrg_y(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        Ux::AbstractTensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the y-direction
                    -3
                    |
            ┌---1---A2---3--┐
            |       |       |
        -1--Ux†      5      Ux-- -4
            |       |       |
            └---2---A1---4--┘
                    |
                    -2
    =#
    @tensor T[-1 -2; -3 -4] :=
        conj(Ux[1 2; -1]) * Ux[3 4; -4] * A2[1 5; -3 3] * A1[2 -2; 5 4]
    return T
end

function _step_hotrg_x(
        A1::AbstractTensorMap{E, S, 2, 2}, A2::AbstractTensorMap{E, S, 2, 2},
        Uy::AbstractTensorMap{E, S, 2, 1}
    ) where {E, S}
    #= compression along the x-direction
                -3
                |
            ┌3--Uy-4┐
            |       |
        -1--A1--5---A2-- -4
            |       |
            └1-Uy†-2┘
                |
                -2
    =#
    @tensor T[-1 -2; -3 -4] :=
        A1[-1 1; 3 5] * A2[5 2; 4 -4] * conj(Uy[1 2; -2]) * Uy[3 4; -3]
    return T
end