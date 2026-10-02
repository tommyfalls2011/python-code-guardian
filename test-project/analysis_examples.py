import os
import os


def hello(name):
    return f"Hello {name}"


def hello(name):
    return f"Hi {name}"


if False:
    print("This will never run")


while True:
    break


def broken_reference():
    return definitely_not_defined + 1


def unused_function():
    unused_value = 123
    return 42
