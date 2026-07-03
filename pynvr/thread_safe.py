"""Thread-safe container implementations.

Provides lightweight thread-safe wrappers around dict, list, and set
behaviors used elsewhere in the project. These wrappers use reentrant locks
to ensure safe concurrent access and support context-manager locking for
batch operations.
"""
from threading import RLock

class ThreadSafeSet:
    """A thread-safe set implementation using a threading.RLock."""
    def __init__(self, initial_data=None):
        self._set = set(initial_data if initial_data is not None else [])
        self._lock = RLock()

    # -------------------------
    # Basic operations
    # -------------------------
    def add(self, item):
        with self._lock:
            self._set.add(item)

    def remove(self, item):
        with self._lock:
            self._set.remove(item)

    def update(self, *others):
        with self._lock:
            self._set.update(*others)

    def clear(self):
        with self._lock:
            self._set.clear()

    def __contains__(self, item):
        with self._lock:
            return item in self._set

    def __len__(self):
        with self._lock:
            return len(self._set)

    def unsafe_len(self):
        return len(self._set)

    def __iter__(self):
        with self._lock:
            return iter(set(self._set))

    def __repr__(self):
        with self._lock:
            return f"ThreadSafeSet({self._set!r})"

    # -------------------------
    # Set operator overloads
    # -------------------------

    # Union: a | b
    def __or__(self, other):
        with self._lock:
            if isinstance(other, ThreadSafeSet):
                with other._lock:
                    return ThreadSafeSet(self._set | other._set)
            return ThreadSafeSet(self._set | set(other))

    # Intersection: a & b
    def __and__(self, other):
        with self._lock:
            if isinstance(other, ThreadSafeSet):
                with other._lock:
                    return ThreadSafeSet(self._set & other._set)
            return ThreadSafeSet(self._set & set(other))

    # Difference: a - b
    def __sub__(self, other):
        with self._lock:
            if isinstance(other, ThreadSafeSet):
                with other._lock:
                    return ThreadSafeSet(self._set - other._set)
            return ThreadSafeSet(self._set - set(other))

    # In-place union: a |= b
    def __ior__(self, other):
        with self._lock:
            if isinstance(other, ThreadSafeSet):
                with other._lock:
                    self._set |= other._set
            else:
                self._set |= set(other)
        return self

    # In-place intersection: a &= b
    def __iand__(self, other):
        with self._lock:
            if isinstance(other, ThreadSafeSet):
                with other._lock:
                    self._set &= other._set
            else:
                self._set &= set(other)
        return self

    # In-place difference: a -= b
    def __isub__(self, other):
        with self._lock:
            if isinstance(other, ThreadSafeSet):
                with other._lock:
                    self._set -= other._set
            else:
                self._set -= set(other)
        return self

from threading import RLock

class ThreadSafeList:
    """A thread-safe list implementation using a threading.RLock."""

    def __init__(self, initial_data=None):
        self._list = list(initial_data) if initial_data is not None else []
        self._lock = RLock()

    # -------------------------
    # Basic list operations
    # -------------------------
    def append(self, item):
        with self._lock:
            self._list.append(item)

    def extend(self, items):
        with self._lock:
            self._list.extend(items)

    def insert(self, index, item):
        with self._lock:
            self._list.insert(index, item)

    def remove(self, item):
        with self._lock:
            self._list.remove(item)

    def pop(self, index=-1):
        with self._lock:
            return self._list.pop(index)

    def clear(self):
        with self._lock:
            self._list.clear()

    # -------------------------
    # Accessors
    # -------------------------
    def __getitem__(self, index):
        with self._lock:
            return self._list[index]

    def __setitem__(self, index, value):
        with self._lock:
            self._list[index] = value

    def __delitem__(self, index):
        with self._lock:
            del self._list[index]

    def __len__(self):
        with self._lock:
            return len(self._list)

    def unsafe_len(self):
        return len(self._list)

    def __contains__(self, item):
        with self._lock:
            return item in self._list

    # -------------------------
    # Iteration
    # -------------------------
    def __iter__(self):
        with self._lock:
            # Return a snapshot iterator so iteration is safe
            return iter(list(self._list))

    # -------------------------
    # Representation
    # -------------------------
    def __repr__(self):
        with self._lock:
            return f"ThreadSafeList({self._list!r})"
