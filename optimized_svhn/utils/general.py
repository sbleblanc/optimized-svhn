import numpy as np
from typing import List, Any, Iterable


def convert_to_list(obj: Any) -> List[Any]:
    if isinstance(obj, np.ndarray) and not obj.shape:
        return [obj.item()]
    if issubclass(obj.__class__, Iterable):
        return [
            e.item() if isinstance(e, np.ndarray) else e
            for e in obj
        ]
    else:
        return [obj]
