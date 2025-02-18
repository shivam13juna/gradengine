## import libraries
import sys
import pyrootutils
root = pyrootutils.setup_root(sys.path[0], pythonpath=True, cwd=True)


from src.entity import Entity
import pytest


def test_entity_addition():
	test1 = Entity(5, 'test1')
	test2 = Entity(5, 'test2')
	test3 = test1 + test2
	assert test3.data == 10

def test_scalar_addition():
	test1 = Entity(5, 'test1')
	test2 = 5 + test1
	assert test2.data == 10

def test_entity_division():
	test4 = Entity(10, 'test4')
	test3 = Entity(5, 'test3')
	test5 = test4 / test3
	assert test5.data == 2

def test_entity_multiplication():
	test5 = Entity(2, 'test5')
	test4 = Entity(10, 'test4')
	test6 = test5 * test4
	assert test6.data == 20

def test_entity_exponentiation():
	test6 = Entity(20, 'test6')
	test7 = test6 ** 5
	assert test7.data == 3200000

def test_entity_addition_chain():
	test1 = Entity(5, 'test1')
	test2 = Entity(5, 'test2')
	test3 = test1 + test2
	test4 = 5 + test3
	test5 = test4 / test3
	test6 = test5 * test4
	test7 = test6 ** 5
	test8 = test1 + test2 + test3 + test4 + test5 + test6 + test7
	assert test8.data == 5766562.90625




#test8 = test1 + test2 + test3 + test4 + test5 + test6 + test7; test8.label = 'test8'