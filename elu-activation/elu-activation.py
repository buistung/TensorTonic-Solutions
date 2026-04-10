def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    def eluu(x,alpha):
        if x > 0:
            return x
        else:
            return alpha * (math.exp(x) - 1)
    y = []
    for i in x:
        y.append(eluu(i,alpha))

    return y