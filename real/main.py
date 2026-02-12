class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.cache = {}
        self.head = None
        self.tail = None

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        if node == self.head:
            return node.value
        self.move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self.move_to_head(node)
            return
        node = Node(key, value)
        self.cache[key] = node
        if self.size == self.capacity:
            self.remove_tail()
            self.size -= 1
        self.size += 1
        self.add_to_head(node)

    def move_to_head(self, node):
        if node == self.head:
            return
        # Remove node
        node_prev = node.prev
        node_next = node.next
        if node_prev:
            node_prev.next = node_next
        else:
            # node is head, but we checked that
            pass
        if node_next:
            node_next.prev = node_prev
        else:
            # node is tail
            self.tail = node_prev
        # Add to head
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        self.head = node

    def add_to_head(self, node):
        node.next = self.head
        node.prev = None
        if self.head:
            self.head.prev = node
        else:
            # list was empty
            self.tail = node
        self.head = node

    def remove_tail(self):
        if not self.tail:
            return
        removed_node = self.tail
        if self.head == self.tail:
            self.head = None
            self.tail = None
        else:
            self.tail = self.tail.prev
            self.tail.next = None
        del self.cache[removed_node.key]
