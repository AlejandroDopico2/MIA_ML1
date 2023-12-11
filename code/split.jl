function holdOut(N::Int, P::Float64)
    # Verify that P is a valid probability
    @assert ((P >= 0.0) && (P <= 1.0))

    # Generate random permutation of integers from 1 to N
    indexes = randperm(N)

    # Calculate the size of the test set
    test_size = Int(ceil(N * P))

    # Select the indexes of the test and training sets
    test_indexes = indexes[1:test_size]
    train_indexes = indexes[test_size+1:end]

    # Verify that the length of the training and test sets is correct
    @assert length(train_indexes) + length(test_indexes) == N

    # Return the indexes of the training and test sets
    return (train_indexes, test_indexes)
end

function holdOut(N::Int, Pval::Float64, Ptest::Float64)
    # Verify that Pval and Ptest are valid probabilities
    @assert ((Pval >= 0.0) && (Pval <= 1.0))
    @assert ((Ptest >= 0.0) && (Ptest <= 1.0))
    @assert ((Pval + Ptest) <= 1.0)

    train_val_indexes, test_indexes = holdOut(N, Ptest)

    train_val_size = length(train_val_indexes)

    train_indexes, val_indexes = holdOut(train_val_size, Pval * N / train_val_size)

    train_indexes = train_val_indexes[train_indexes]
    val_indexes = train_val_indexes[val_indexes]

    @assert length(train_indexes) + length(val_indexes) + length(test_indexes) == N

    return (train_indexes, val_indexes, test_indexes)
end