def grid_factor(region_number: int) -> tuple[int, int]:
    """
    Resolve grid amount by mulriplying odd values.

    Args:
        region_number: int - number of regions

    Returns:
        tuple[int, int] - tuple of the number of regions in the left and right direction

    Examples:
        >>> grid_factor(8)
        (2, 4)
        >>> grid_factor(4)
        (2, 2)
        >>> grid_factor(2)
        (1, 2)
        >>> grid_factor(1)
        (1, 1)

    Mostly used for grid the frame into 4,8 regions in feature tracker.

    """
    if region_number <= 1:
        return 1, 1
    if region_number % 2 != 0:
        return region_number, 1

    left = region_number
    right = 1
    while left % 2 == 0 and left > right:
        left //= 2
        right *= 2

    return left, right
