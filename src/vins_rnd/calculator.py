def add(a: int, b: int) -> int:
    """Add two integers and return the result."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two integers and return the result."""
    return a - b


def multiply(a: int, b: int) -> int:
    """Multiply two integers and return the result."""
    return a * b


def divide(a: int, b: int) -> float:
    """Divide two integers and return the result."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
