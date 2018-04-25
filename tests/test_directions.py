import sys
sys.path.append('../src')
import unittest

from state import *
from world import *


class DirectionTest(unittest.TestCase):

    def test_is_in_direction(self):
        self.assertTrue(Direction.is_in_direction(
            Ennemy((10,0), Direction.W),
            GameEntity(6, 0)
        ))
        self.assertFalse(Direction.is_in_direction(
            Ennemy((10,0), Direction.E),
            GameEntity(6,0)
        ))

        self.assertTrue(Direction.is_in_direction(
            Ennemy((10,10), Direction.NE),
            GameEntity(12, 13)
        ))

        self.assertTrue(Direction.is_in_direction(
            Ennemy((10,0), Direction.W),
            GameEntity(6, 20)
        ))

    def test_is_in_collision_course(self):
        self.assertTrue(Direction.is_in_collision_course(
            Ennemy((10,0), Direction.W),
            GameEntity(6, 0),
            10
        ))

        self.assertTrue(Direction.is_in_collision_course(
            Ennemy((10,0), Direction.W),
            GameEntity(6, 1),
            10
        ))

        self.assertFalse(Direction.is_in_collision_course(
            Ennemy((10,0), Direction.E),
            GameEntity(6, 1),
            10
        ))

        self.assertFalse(Direction.is_in_collision_course(
            Ennemy((10,0), Direction.W),
            GameEntity(6, 20),
            10
        ))
