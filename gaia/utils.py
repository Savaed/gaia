def check_kepid(kepid: int) -> None:
    if not 0 < kepid < 1_000_000_000:
        raise ValueError(f"'kepid' must be in range 1 to 999 999 999 inclusive, but got {kepid=}")
