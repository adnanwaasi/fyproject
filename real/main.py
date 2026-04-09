from collections import OrderedDict

class LRUCache:
    """Implements a Least Recently Used (LRU) Cache.

    Uses OrderedDict to maintain insertion/access order, ensuring O(1) complexity
    for get and put operations.
    """
    def __init__(self, capacity: int):
        # OrderedDict maintains insertion order. The beginning is the LRU end,
        # and the end is the Most Recently Used (MRU) end.
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int: 
        """Retrieves the value associated with the key. Updates usage order.

        :param key: The key to retrieve.
        :return: The value if the key exists, otherwise -1 (or appropriate indicator).
        """
        if key not in self.cache:
            return -1  # Key not found
        
        # Move the accessed item to the end (MRU position)
        value = self.cache[key]
        self.cache.move_to_end(key)
        return value

    def put(self, key: int, value: int) -> int: 
        """Inserts or updates a key-value pair. Handles eviction if capacity is reached.

        :param key: The key to insert/update.
        :param value: The value to associate with the key.
        :return: The previously stored value for the key, or -1 if the key was new.
        """
        # 1. Update existing key (if present)
        if key in self.cache:
            # Get the old value before updating
            old_value = self.cache[key]
            # Update value and move to MRU end
            self.cache[key] = value
            self.cache.move_to_end(key)
            return old_value
        
        # 2. New key insertion
        
        # Check for eviction necessity
        if len(self.cache) >= self.capacity:
            # Pop the first item (LRU item)
            # last=False ensures we pop the first item (FIFO/LRU end)
            lru_key, _ = self.cache.popitem(last=False)
            # In a real system, we might return the evicted key, but here we just ensure eviction.
            pass
        
        # Insert new item (it automatically goes to the MRU end)
        self.cache[key] = value
        return -1 # Indicates a new insertion (no previous value)

# Example Usage (Not required for final output, but useful for verification):
# cache = LRUCache(2)
# print(f"Put (1, 1): {cache.put(1, 1)}") # -1
# print(f"Put (2, 2): {cache.put(2, 2)}") # -1
# print(f"Get (1): {cache.get(1)}") # 1 (1 becomes MRU)
# print(f"Put (3, 3): {cache.put(3, 3)}") # Evicts key 2 (LRU). Returns -1
# print(f"Get (2): {cache.get(2)}") # -1 (Evicted)
# print(f"Get (3): {cache.get(3)}") # 3 (3 becomes MRU)
# print(f"Put (1, 10): {cache.put(1, 10)}") # Updates 1. Returns 1
