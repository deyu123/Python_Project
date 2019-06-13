from functools import reduce


def f(x):
    return x * x


def fn(x, y):
    return x * 10 + y


def is_odd(n):
    return n % 2 == 1


if __name__ == '__main__':
    # map 输入
    r = map(f, [1, 2, 3, 4, 5, 6])
    print(list(r))

    # reduce 输入
    r1 = reduce(fn, [1, 2, 3, 4, 5, 6])
    print(r1)

    # filter
    r = filter(is_odd, [1, 2, 3, 4, 5, 6])
    print(list(r))
