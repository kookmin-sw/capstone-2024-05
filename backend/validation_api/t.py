from typing import List


def repeat(message: str, times: int = 2) -> List[str]:
    return [message] * times


repeat("Hi", "3")