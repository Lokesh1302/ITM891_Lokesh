def fib(n: int):
    """ Generates fibonacci number """
    if n == 0 or n == 1:
        return n
    f0 = 0
    f1 = 1
    for i in range(1, n - 1):
        f2 = f0 + f1
        f0 = f1
        f1 = f2
    return f2