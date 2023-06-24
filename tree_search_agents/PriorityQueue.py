class PriorityQueue:
    """
        This class is an implementation of PriorityQueue data structor.
    """
    queue: list  # List of objects

    def __init__(self):
        """
            Constructor
        """
        self.queue = []

    def enqueue(self, obj: object, val: float):
        """
            Enqueue operation appends the object into the queue based on the given value.

            **Note**: Queue is sorted in **DESCENDING** order.
        :param obj: Object that will be added
        :param val: Priority of the object
        :return: Nothing
        """
        self.queue.append([obj, val])

        for i in range(len(self.queue) - 2, -1, -1):
            if self.queue[i][1] < self.queue[i + 1][1]:
                self.queue[i], self.queue[i + 1] = self.queue[i + 1], self.queue[i]

    def dequeue(self):
        """
            Dequeue operation pops and returns the top of queue
        :return: Next object
        """
        return self.queue.pop(0)[0]

    def is_empty(self) -> bool:
        """
            This method checks if this priority queue is empty, or not.
        :return: Whether the priority queue is empty, or not.
        """

        return len(self) == 0

    def __len__(self):
        """
            This method provides the length of queue.
        :return: Length of queue
        """
        return len(self.queue)
