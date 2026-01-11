using TNRKit
using TensorKit

"""
N   
    Distance bewteen the two impurities: 2^N
    Watch out:
        - N >= 1
        - Each step in algorithm aggregates √2

L 
    Number of steps to aggregate
    Watch out:
        - Each step in algorithm aggregates √2, so 2 steps aggregate factor 2
        - L > N
Tpure
    Pure tensor

Timp
    Impurity tensor

Convention
```
    p   p
    |   |  
p---1---2---p
    |   |
p---4---3---p
    |   |
    p   p
```
```
    p   p   p
    |   |   |  
p---1---2---3---p
    |   |   |
p---6---5---4---p
    |   |   |
    p   p   p
```
"""
function getCorrelation(N::Integer, L::Integer, Tpure, Timp, trunc::TensorKit.TruncationScheme)
    Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6 = CorrelationFunction(N, L, Tpure, Timp, trunc)

    nimp =  norm(@tensoropt Timp1[3 7;8 1] * Timp2[1 9;10 2] * Timp3[2 11;12 3] * Timp4[6 8;7 4] * Timp5[4 10;9 5] * Timp6[5 12;11 6])
    npure =  norm(@tensoropt Tpure[3 7;8 1] * Tpure[1 9;10 2] * Tpure[2 11;12 3] * Tpure[6 8;7 4] * Tpure[4 10;9 5] * Tpure[5 12;11 6])
    return nimp / npure
end

"""
N   
    Distance bewteen the two impurities: 2^N
    Watch out:
        - N >= 1
        - Each step in algorithm aggregates √2

L 
    Number of steps to aggregate
    Watch out:
        - Each step in algorithm aggregates √2, so 2 steps aggregate factor 2
        - L > N
Tpure
    Pure tensor

Timp
    Impurity tensor

Convention
```
    p   p
    |   |  
p---1---2---p
    |   |
p---4---3---p
    |   |
    p   p
```
```
    p   p   p
    |   |   |  
p---1---2---3---p
    |   |   |
p---6---5---4---p
    |   |   |
    p   p   p
```
"""
function CorrelationFunction(N::Integer, L::Integer, Tpure, Timp, trunc::TensorKit.TruncationScheme)
    # Assert realistic calculation
    @assert L >= N "L must be larger than N"

    # Preparation
    Timp1 = Timp
    Timp2 = Tpure
    Timp3 = Tpure
    Timp4 = Tpure

    # Fase 1: Seperate
    println("PART 1\n--------")
    for i in 1:N-1
        println("Step $i")
        # Do each step twice
        Tpure, Timp1, Timp2, Timp3, Timp4 = step_4tensors(Tpure, Timp1, Timp2, Timp3, Timp4, trunc)
        #Tpure, Timp1, Timp2, Timp3, Timp4 = step_4tensors(Tpure, Timp1, Timp2, Timp3, Timp4, trunc)
        Tpure, Timp1, Timp2, Timp3, Timp4 = finalize_4tensor(Tpure, Timp1, Timp2, Timp3, Timp4)
        println("‖Tpure‖ = ", norm(Tpure))
        println("‖Timp1‖ = ", norm(Timp1))
        println("‖Timp2‖ = ", norm(Timp2))

    end

    # Fase 2: Merge
    println("PART 2\n--------")
    Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6 = step_8tensors(Tpure, Timp1, Timp2, Timp3, Timp4, trunc)
    Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6 = permuteIndicesAll(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6)
    Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6 = finalize_6tensor(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6)
    println("‖Tpure‖ = ", norm(Tpure))
    println("‖Timp1‖ = ", norm(Timp1))
    println("‖Timp2‖ = ", norm(Timp2))

    # Fase 3: Combined
    println("PART 3\n--------")
    nstep = L-N

    for i in 1:nstep
        println("Step $i")
        Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6 = step_6tensors(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6, trunc)
        Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6 = permuteIndicesAll(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6)
        Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6 = finalize_6tensor(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6)
        println("‖Tpure‖ = ", norm(Tpure))
        println("‖Timp1‖ = ", norm(Timp1))
        println("‖Timp2‖ = ", norm(Timp2))
    end

    println("DONE")

    return Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6
end

function step_4tensors(Tpure, Timp1, Timp2, Timp3, Timp4, trunc::TensorKit.TruncationScheme)
    # Tensor1
    A1, B1 = SVD12(Timp1, trunc)

    # Tensor2
    tensor2p = transpose(Timp2, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(Timp3, trunc)

    # Tensor4
    tensor4p = transpose(Timp4, ((2, 4), (1, 3)))
    C4, D4 = SVD12(tensor4p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(Tpure, trunc)
    tensorpurep = transpose(Tpure, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Contract
    @planar Tpure_new[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp1_new[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar Timp2_new[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp3_new[-1, -2; -3, -4] := D4[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * A3[3 2; -4]
    @planar Timp4_new[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C4[4 3; -3] * A1[3 2; -4]

    return Tpure_new, Timp1_new, Timp2_new, Timp3_new, Timp4_new
end

"""
```
    p   p   p   p
    |   |   |   |  
p---1---2---1---2---p
    |   |   |   |
p---4---3---4---3---p
    |   |   |   |
    p   p   p   p
```
transforms to
```
    2       2
  /   \\   /   \\
1       3       4
  \\  /   \\    /
    5       5
```
and finally to
```
    p   p   p
    |   |   | 
p---1---2---3---p
    |   |   |
p---6---5---4---p
    |   |   |
    p   p   p
```
"""
function step_8tensors(Tpure, Timp1, Timp2, Timp3, Timp4, trunc::TensorKit.TruncationScheme)
    #####################################
    #           PART 1: 8->7            #
    #####################################
    # Tensor1
    A1, B1 = SVD12(Timp1, trunc)

    # Tensor2
    tensor2p = transpose(Timp2, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(Timp3, trunc)

    # Tensor4
    tensor4p = transpose(Timp4, ((2, 4), (1, 3)))
    C4, D4 = SVD12(tensor4p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(Tpure, trunc)
    tensorpurep = transpose(Tpure, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Contract
    @planar Tpure_temp[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4] 
    @planar Timp1_temp[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C4[4 3; -3] * A1[3 2; -4] 
    @planar Timp2_temp[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar Timp3_temp[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * C4[4 3; -3] * A1[3 2; -4]
    @planar Timp4_temp[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp5_temp[-1, -2; -3, -4] := D4[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * A3[3 2; -4]

    #####################################
    #           PART 2: 7->6            #
    #####################################
    # Tensor1
    A1, B1 = SVD12(Timp1_temp, trunc)

    # Tensor2
    tensor2p = transpose(Timp2_temp, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(Timp3_temp, trunc)

    # Tensor4
    A4, B4 = SVD12(Timp4_temp, trunc)

    # Tensor5
    tensor5p = transpose(Timp5_temp, ((2, 4), (1, 3)))
    C5, D5 = SVD12(tensor5p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(Tpure_temp, trunc)
    tensorpurep = transpose(Tpure_temp, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Contract
    @planar Tpure_new[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp1_new[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar Timp2_new[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar Timp3_new[-1, -2; -3, -4] := D2[-2; 1 2] * B4[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp4_new[-1, -2; -3, -4] := D5[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * A4[3 2; -4]
    @planar Timp5_new[-1, -2; -3, -4] := D5[-2; 1 2] * Bp[-1; 4 1] * C5[4 3; -3] * A3[3 2; -4]
    @planar Timp6_new[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C5[4 3; -3] * A1[3 2; -4]

    return Tpure_new, Timp1_new, Timp2_new, Timp3_new, Timp4_new, Timp5_new, Timp6_new
end

"""
From
```
    p   p   p
    |   |   | 
p---1---2---3---p
    |   |   |
p---6---5---4---p
    |   |   |
    p   p   p
```
to
```
    2       4
  /   \\   /
1       3
  \\  /   \\
    6       5
```
and finally to
```
    p   p   p
    |   |   | 
p---1---2---3---p
    |   |   |
p---6---5---4---p
    |   |   |
    p   p   p
```
"""
function step_6tensors(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6, trunc::TensorKit.TruncationScheme)
    #####################################
    #           PART 1                  #
    #####################################
    # Tensor1
    A1, B1 = SVD12(Timp1, trunc)

    # Tensor2
    tensor2p = transpose(Timp2, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(Timp3, trunc)

    # Tensor4
    tensor4p = transpose(Timp4, ((2, 4), (1, 3)))
    C4, D4 = SVD12(tensor4p, trunc)

    # Tensor5
    A5, B5 = SVD12(Timp5, trunc)

    # Tensor6
    tensor6p = transpose(Timp6, ((2, 4), (1, 3)))
    C6, D6 = SVD12(tensor6p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(Tpure, trunc)
    tensorpurep = transpose(Tpure, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Contract
    @planar Tpure_temp[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4] 
    @planar Timp1_temp[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C6[4 3; -3] * A1[3 2; -4] 
    @planar Timp2_temp[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar Timp3_temp[-1, -2; -3, -4] := D2[-2; 1 2] * B5[-1; 4 1] * C4[4 3; -3] * A3[3 2; -4]
    @planar Timp4_temp[-1, -2; -3, -4] := Dp[-2; 1 2] * B3[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp5_temp[-1, -2; -3, -4] := D4[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp6_temp[-1, -2; -3, -4] := D6[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * A5[3 2; -4]


    #####################################
    #           PART 2:                 #
    #####################################
    # Tensor1
    A1, B1 = SVD12(Timp1_temp, trunc)

    # Tensor2
    tensor2p = transpose(Timp2_temp, ((2, 4), (1, 3)))
    C2, D2 = SVD12(tensor2p, trunc)

    # Tensor3
    A3, B3 = SVD12(Timp3_temp, trunc)

    # Tensor4
    tensor4p = transpose(Timp4_temp, ((2, 4), (1, 3)))
    C4, D4 = SVD12(tensor4p, trunc)

    # Tensor5
    tensor5p = transpose(Timp5_temp, ((2, 4), (1, 3)))
    C5, D5 = SVD12(tensor5p, trunc)

    # Tensor6
    tensor6p = transpose(Timp6_temp, ((2, 4), (1, 3)))
    C6, D6 = SVD12(tensor6p, trunc)

    # Tensor pure
    Ap, Bp = SVD12(Tpure_temp, trunc)
    tensorpurep = transpose(Tpure_temp, ((2, 4), (1, 3)))
    Cp, Dp = SVD12(tensorpurep, trunc)

    # Contract
    @planar Tpure_new[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp1_new[-1, -2; -3, -4] := Dp[-2; 1 2] * B1[-1; 4 1] * C2[4 3; -3] * Ap[3 2; -4]
    @planar Timp2_new[-1, -2; -3, -4] := D2[-2; 1 2] * B3[-1; 4 1] * C4[4 3; -3] * Ap[3 2; -4]
    @planar Timp3_new[-1, -2; -3, -4] := D2[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp4_new[-1, -2; -3, -4] := D5[-2; 1 2] * Bp[-1; 4 1] * Cp[4 3; -3] * Ap[3 2; -4]
    @planar Timp5_new[-1, -2; -3, -4] := D6[-2; 1 2] * Bp[-1; 4 1] * C5[4 3; -3] * A3[3 2; -4]
    @planar Timp6_new[-1, -2; -3, -4] := Dp[-2; 1 2] * Bp[-1; 4 1] * C6[4 3; -3] * A1[3 2; -4]

    return Tpure_new, Timp1_new, Timp2_new, Timp3_new, Timp4_new, Timp5_new, Timp6_new
end

function finalize_4tensor(Tpure, Timp1, Timp2, Timp3, Timp4)
    # First normalize everything by the pure tensor
    npure = norm(@tensor Tpure[1 2; 2 1])
    Timp1 /= npure
    Timp2 /= npure
    Timp3 /= npure
    Timp4 /= npure
    Tpure /= npure

    return Tpure, Timp1, Timp2, Timp3, Timp4
end

function finalize_6tensor(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6)
    # First normalize everything by the pure tensor
    npure = norm(@tensor Tpure[1 2;2 1])
    Timp1 /= npure
    Timp2 /= npure
    Timp3 /= npure
    Timp4 /= npure
    Timp5 /= npure
    Timp6 /= npure
    Tpure /= npure
return Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6
end


# Helper SVD functions

function SVD12(T::AbstractTensorMap{E, S, 2, 2}, trunc::TensorKit.TruncationScheme) where {E, S}
    U, s, V, e = tsvd(T; trunc = trunc)
    return U * sqrt(s), sqrt(s) * V
end


function permuteIndices(T)
    T_arr = Array(T[:,:,:,:])
    T__arr_perm = permutedims(T_arr, (2, 4, 3, 1))
    return TensorMap(T__arr_perm, space(T))
end

function permuteIndicesAll(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6)
    Tpure = permuteIndices(Tpure)
    Timp1 = permuteIndices(Timp1)
    Timp2 = permuteIndices(Timp2)
    Timp3 = permuteIndices(Timp3)
    Timp4 = permuteIndices(Timp4)
    Timp5 = permuteIndices(Timp5)
    Timp6 = permuteIndices(Timp6)
    return Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6
end


"TEST"

T = classical_ising()
T_imp = classical_ising_impurity()

getCorrelation(1, 10, T, T_imp, truncdim(12))
getCorrelation(2, 10, T, T_imp, truncdim(12))
getCorrelation(3, 10, T, T_imp, truncdim(12))
getCorrelation(4, 10, T, T_imp, truncdim(12))
getCorrelation(5, 10, T, T_imp, truncdim(12))
getCorrelation(6, 10, T, T_imp, truncdim(12))
getCorrelation(7, 10, T, T_imp, truncdim(12))
getCorrelation(8, 10, T, T_imp, truncdim(12))
getCorrelation(9, 10, T, T_imp, truncdim(12))



# OLD
"""
function finalize!(Tpure, Timp1, Timp2, Timp3, Timp4; get_nimp=false)
    # First normalize everything by the pure tensor
    npure = norm(@tensor Tpure[1 2; 2 1])
    Timp1 /= npure
    Timp2 /= npure
    Timp3 /= npure
    Timp4 /= npure
    Tpure /= npure

    # Then calculate the contracted/traced 4 impurity tensors
    nimp = norm(@tensoropt Timp1[5 4;6 1] * Timp2[1 2;7 5] * Timp3[3 7;2 8] * Timp4[8 6;4 3])

    return npure, nimp
end

function finalize!(Tpure, Timp1, Timp2, Timp3, Timp4, Timp5, Timp6)
    # First normalize everything by the pure tensor
    npure = norm(@tensor Tpure[1 2; 2 1])
    Timp1 /= npure
    Timp2 /= npure
    Timp3 /= npure
    Timp4 /= npure
    Timp5 /= npure
    Timp6 /= npure
    Tpure /= npure

    # Then calculate the contracted/traced 4 impurity tensors
    nimp = norm(@tensoropt Timp1[1 2;3 4] * Timp2[4 5;6 7] * Timp3[7 8;9 1] * Timp4[10 9;8 11] * Timp5[12 6;5 10] * Timp6[11 3;2 12])

    return npure, nimp
end
"""