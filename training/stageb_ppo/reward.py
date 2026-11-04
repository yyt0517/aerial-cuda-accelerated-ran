"""Online FIFO accounting: full payload counts only if every segment succeeds."""
from collections import deque
import numpy as np


class PacketReward:
    def __init__(self, users=36, packet_size=3000, ttl=400):
        self.queues = [deque() for _ in range(users)]
        self.size, self.ttl = packet_size, ttl
        self.expired = 0

    def before(self, tti, pre_queue):
        self.expired = 0
        for u, queue in enumerate(self.queues):
            while queue and tti - queue[0][0] + 1 > self.ttl:
                queue.popleft()
                self.expired += self.size
            arrived = int(pre_queue[u]) - sum(p[1] for p in queue)
            assert arrived >= 0 and arrived % self.size == 0, (tti, u, arrived)
            queue.extend([tti, self.size, False] for _ in range(arrived // self.size))

    def after(self, served, good):
        complete = failed_complete = 0
        for u, queue in enumerate(self.queues):
            left = int(served[u])
            assert good[u] == 0 or good[u] == left
            failed = good[u] < left
            while left:
                take = min(left, queue[0][1])
                queue[0][1] -= take
                queue[0][2] |= failed
                left -= take
                if queue[0][1] == 0:
                    _, _, damaged = queue.popleft()
                    complete += 0 if damaged else self.size
                    failed_complete += self.size if damaged else 0
        remaining = sum(p[1] for q in self.queues for p in q)
        reward = (complete - self.expired - .01 * remaining) / 86400.
        return dict(reward=reward, complete_payload_bytes=complete,
                    failed_complete_bytes=failed_complete, expired_payload_bytes=self.expired,
                    post_queue_bytes=remaining, goodput_bytes=float(np.sum(good)))
