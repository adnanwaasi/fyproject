from collections import OrderedDict

class LRUCache:
    """Implements a Least Recently Used (LRU) Cache."""
    def __init__(self, capacity: int):
        # Assumption: Capacity is a positive integer. Handle non-positive capacity.
        if capacity <= 0:
            self.capacity = 0
            self.cache = OrderedDict()
            return

        self.capacity = capacity
        # OrderedDict maintains insertion order and allows moving items to the end (MRU)
        self.cache = OrderedDict()

    def get(self, key: int) -> any:
        """Retrieves the value associated with the key. Updates usage status."""
        if key not in self.cache:
            return None  # Key does not exist

        # Move the accessed item to the end to mark it as Most Recently Used (MRU)
        value = self.cache[key]
        self.cache.move_to_end(key)
        return value

    def put(self, key: int, value: any) -> None:
        """Inserts or updates a key-value pair. Handles eviction if capacity is reached."""
        if self.capacity == 0:
            return

        if key in self.cache:
            # Key exists: Update value and mark as MRU
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # Key is new
            if len(self.cache) >= self.capacity:
                # Cache is at capacity, evict the LRU item (the first item)
                # popitem(last=False) removes and returns the (key, value) pair that was inserted first
                self.cache.popitem(last=False)
            
            # Insert new item (it automatically becomes the MRU)
            self.cache[key] = value
            # Explicitly move to end just in case, though standard assignment usually handles this for OrderedDict
            self.cache.move_to_end(key)

    def __str__(self):
        return f"LRUCache(Capacity={self.capacity}, Size={len(self.cache)}): {dict(self.cache)}"

# Note on Thread Safety: The problem specification assumes thread safety might be required.
# For true thread safety in a production environment, the entire class should be wrapped
# with a threading.Lock, e.g., self.lock = threading.Lock(), and all methods should
# start with 'with self.lock:'
