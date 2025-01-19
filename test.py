import unittest
from aads import Node, ReferenceBasedBinaryTree, ArrayBasedBinaryTree

class SimpleBinaryTreeCases(unittest.TestCase):
    def test_BinaryTree(self):
        bt=ReferenceBasedBinaryTree()
        bt.add(10)
        bt.print_tree()
        bt.add(4)
        bt.print_tree()
        bt.add(7)
        bt.print_tree()
        bt.add(1)
        bt.print_tree()
        bt.add(15)
        bt.print_tree()
        bt.add(17)
        bt.print_tree()
        bt.add(14)
        bt.print_tree()
        bt.add(12)
        bt.print_tree()
        bt.add(11)
        bt.print_tree()
        bt.add(13)
        bt.print_tree()
        bt.add(16)
        bt.print_tree()
        bt.add(18)
        bt.print_tree()

        bt.delete(10)
        
        bt.print_tree()

    def test_SimpleBinaryTree(self):
        bt=ReferenceBasedBinaryTree()
        bt.add(10)
        bt.add(6)
        bt.add(4)
        bt.add(9)
        bt.add(16)
        bt.add(19)
        bt.add(11)
        self.assertIs(bt.find(11) is not None, bt.find(12) is None)

    def test_RemoveSimpleBinaryTree(self):
        bt=ReferenceBasedBinaryTree()
        bt.rootNode=Node(14)
        bt.rootNode.setLeftChild(Node(6))
        bt.rootNode.setRightChild(Node(21))
        bt.rootNode.leftChild.setLeftChild(Node(4))
        bt.rootNode.leftChild.setRightChild(Node(11))
        bt.rootNode.leftChild.rightChild.setLeftChild(Node(9))
        bt.rootNode.leftChild.rightChild.setRightChild(Node(12))
        bt.rootNode.rightChild.setLeftChild(Node(15))
        bt.rootNode.rightChild.setRightChild(Node(25))
        bt.rootNode.rightChild.leftChild.setRightChild(Node(17))
        bt.rootNode.rightChild.leftChild.rightChild.setLeftChild(Node(16))
        bt.delete(21)
        self.assertTrue(bt.find(21) is None)

    def test_arrayBased(self) :
        abt = ArrayBasedBinaryTree(capacity=15)
        # Add nodes
        abt.add(10)
        abt.pretty_print()
        abt.add(5)
        abt.pretty_print()
        abt.add(15)
        abt.pretty_print()
        abt.add(16)
        abt.pretty_print()
        abt.add(7)
        abt.pretty_print()
        abt.add(12)
        abt.pretty_print()
        abt.add(18)

        # Print the tree structure
        abt.pretty_print()

        # Find a node
        index = abt.find(12)
        print(f"Index of 12: {index}")

        # Delete a node
        abt.delete(10)
        abt.pretty_print()

        index = abt.find(12)
        print(f"Index of 12: {index}")

if __name__ == '__main__':
    unittest.main()