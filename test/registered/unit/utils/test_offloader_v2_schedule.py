"""CPU-only tests for the OffloaderV2 static-buffer schedule."""

import unittest

from sglang.srt.utils.offloader import _build_slot_prefetch_schedule
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestOffloaderV2Schedule(CustomTestCase):
    def test_empty_schedule(self):
        self.assertEqual(_build_slot_prefetch_schedule(0, 3), [])

    def test_non_divisible_module_count_keeps_slot_ownership(self):
        self.assertEqual(
            _build_slot_prefetch_schedule(7, 3),
            [3, 4, 5, 6, 1, 2, 0],
        )

    def test_more_prefetch_slots_than_modules(self):
        self.assertEqual(_build_slot_prefetch_schedule(3, 8), [0, 1, 2])

    def test_each_slot_forms_one_cycle(self):
        num_offloaders = 42
        prefetch_step = 3
        schedule = _build_slot_prefetch_schedule(num_offloaders, prefetch_step)

        for slot in range(prefetch_step):
            expected = set(range(slot, num_offloaders, prefetch_step))
            visited = set()
            current = slot
            while current not in visited:
                visited.add(current)
                current = schedule[current]
            self.assertEqual(current, slot)
            self.assertEqual(visited, expected)


if __name__ == "__main__":
    unittest.main()
