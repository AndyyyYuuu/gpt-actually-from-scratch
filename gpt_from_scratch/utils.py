from typing import Any, Union

def _enforce_type(received: Any, expected_type: type, content_type:type=None) -> None: 

    """
    Enforces type checking on the received value, raising TypeError when necessary.

    Parameters:
    ----------
    received : Any
        The value to be checked against the expected type.
        
    expected_type : type
        The type that the received value is expected to match. 
        Can be a single type (e.g., int, float) or a collection type 
        (e.g., list, tuple).
        
    content_type : type, optional
        If `expected_type` is a collection type, this parameter specifies
        the type that each element within the collection should match. 
        Defaults to None.

    Raises:
    ------
    TypeError
        If the received value does not match the expected type or if a 
        collection contains elements of unexpected types.

    Notes:
    -----
    This function allows integers to be accepted in place of floats 
    and lists to be accepted in place of tuples during type checking.
    """

    if not _is_instance(received, expected_type): 
        raise TypeError(f"Expected {expected_type.__name__}, got {type(received).__name__}")
    elif expected_type in (list, tuple) and (not content_type is None) and len(received) > 0:
        bad_types = []
        for i in received: 
            if not _is_instance(i, content_type) and type(i) not in bad_types: 
                bad_types.append(type(i))
        if len(bad_types) > 0: 
            raise TypeError(f"Expected {expected_type.__name__} of {content_type.__name__}, got {type(received).__name__} containing {', '.join(bad_type.__name__ for bad_type in bad_types)}")

def _is_instance(received: Any, expected: type) -> bool: 
    return isinstance(received, expected) or (expected is float and isinstance(received, int)) or (expected is tuple and isinstance(received, list))