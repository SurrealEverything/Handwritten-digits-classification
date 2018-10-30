function confidenceIntervals()
    error = 0.108
    n = 5340
    const = [1.64, 1.96, 2.33, 2.58]
    a = error - const * sqrt( (error * (1 - error)) / n)
    b = error + const * sqrt( (error * (1 - error)) / n)
end