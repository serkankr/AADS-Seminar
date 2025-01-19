import numpy as np

class Node :
    value = None
    parent = None
    leftChild = None
    rightChild = None
    h = 1

    def __init__(self, value) :
        self.value = value

    def setLeftChild(self, child) :
        self.leftChild = child
        if child != None : child.parent=self

    def setRightChild(self, child) :
        self.rightChild = child
        if child != None : child.parent=self

    def addNode(self, newValue) :
        if newValue < self.value :
            if self.leftChild is None :
                self.setLeftChild(Node(newValue))
            else :
                self.leftChild.addNode(newValue)
        elif newValue >= self.value :
            if self.rightChild is None :
                self.setRightChild(Node(newValue))
            else :
                self.rightChild.addNode(newValue)

    def findNode(self, target) :
        if self.value == target : return self
        elif (self.value < target) and (self.rightChild != None) :
            return self.rightChild.findNode(target)
        elif (self.value > target) and (self.leftChild != None) :
            return self.leftChild.findNode(target)
        else : return None

    def rightMost(self):
        if self.rightChild is not None :
            return self.rightChild.rightMost()
        else : return self

    def leftMost(self):
        if self.leftChild is not None:
            return self.leftChild.leftMost()
        else : return self

    def get_depth(self):
      left_depth = self.leftChild.get_depth() if self.leftChild else 0
      right_depth = self.rightChild.get_depth() if self.rightChild else 0
      return max(left_depth, right_depth) + 1

    def get_balance(self):
        left_depth = self.leftChild.get_depth() if self.leftChild else 0
        right_depth = self.rightChild.get_depth() if self.rightChild else 0
        return left_depth - right_depth

    def is_balanced(self):
        if self is None:
            return True

        # Check balance factor of current node
        balance = self.get_balance()
        if abs(balance) > 1:
            return False

        # Recursively check left and right subtrees
        left_balanced = True if self.leftChild is None else self.leftChild.is_balanced()
        right_balanced = True if self.rightChild is None else self.rightChild.is_balanced()

        return left_balanced and right_balanced


class ReferenceBasedBinaryTree :
    rootNode = None

    def add(self, newValue) :
        if self.rootNode == None :
            self.rootNode = Node(newValue)
        else : self.rootNode.addNode(newValue)

    def find(self, target) :
        if self.rootNode == None :
            return None
        else :
            return self.rootNode.findNode(target)

    def delete(self, target) :
        if self.rootNode == None :
            return False
        else :
            targetNode = self.find(target)
            if targetNode == None :
                return False
            else :
                if (targetNode.leftChild == None) and (targetNode.rightChild == None) :
                    if targetNode != self.rootNode :
                        # leaf node
                        parentNode = targetNode.parent
                        if parentNode.leftChild == targetNode :
                            parentNode.setLeftChild(None)
                        elif parentNode.rightChild == targetNode :
                            parentNode.setRightChild(None)
                        targetNode = None
                        return True
                    else :
                        # root node and has no child
                        targetNode = None
                        return True

                elif (targetNode.leftChild != None) and (targetNode.rightChild == None) :
                    # target node has 1 child and it is on the left
                    if targetNode != self.rootNode :
                        # target node is not root
                        parentNode = targetNode.parent
                        if parentNode.leftChild == targetNode :
                            # if target node is left child
                            parentNode.setLeftChild(targetNode.leftChild)
                        elif parentNode.rightChild == targetNode :
                            # if target node is right child
                            parentNode.setRightChild(targetNode.leftChild)
                        targetNode = None
                        return True
                    else :
                        # target node is root
                        targetNode.leftChild.parent = None
                        self.rootNode = targetNode.leftChild
                        targetNode = None
                        return True

                elif (targetNode.leftChild == None) and (targetNode.rightChild != None) :
                    # target node has 1 child and it is on the right
                    if targetNode != self.rootNode :
                        # target node is not root
                        parentNode = targetNode.parent
                        if parentNode.leftChild == targetNode :
                            # if target node is left child
                            parentNode.setLeftChild(targetNode.rightChild)
                        elif parentNode.rightChild == targetNode :
                            # if target node is right child
                            parentNode.setRightChild(targetNode.rightChild)
                        targetNode = None
                        return True
                    else :
                        # target node is root
                        targetNode.rightChild.parent = None
                        self.rootNode = targetNode.rightChild
                        targetNode = None
                        return True

                elif (targetNode.leftChild != None) and (targetNode.rightChild != None) :
                    # target node has 2 childs
                    if targetNode != self.rootNode :
                        # target node is not root
                        parentNode = targetNode.parent
                        if parentNode.leftChild == targetNode :
                            # if target node is left subtree
                            parentNode.setLeftChild(targetNode.rightChild)
                            # merge subtree is target -> right
                            leftMost = parentNode.leftChild.leftMost()
                            leftMost.setLeftChild(targetNode.leftChild)
                            # target -> left goes to the leftmost of the merge subtree
                            targetNode = None
                            return True

                        elif parentNode.rightChild == targetNode :
                            # if target node is right subtree
                            parentNode.setRightChild(targetNode.leftChild)
                            # merge subtree is target -> left
                            rightMost = parentNode.rightChild.rightMost()
                            rightMost.setRightChild(targetNode.rightChild)
                            # target -> right goes to the rightmost of the merge subtree
                            targetNode = None
                            return True
                    else :
                        # target node is root
                        successor = targetNode.rightChild.leftMost()
                        # leftmost node in the right subtree – successor
                        copiedValue = successor.value
                        # we copy its value
                        deleteSuccessor = self.delete(successor.value)
                        # remove the successor
                        if deleteSuccessor == True :
                            self.rootNode.value = copiedValue
                            # change root value with copied value
                            return True
                        else : return False
    def get_depth(self):
      if self.rootNode is None:
          return 0
      return self.rootNode.get_depth()

    def is_balanced(self):
        if self.rootNode is None:
            return True
        return self.rootNode.is_balanced()

    def print_tree(self):
        def print_tree2(node, level=0, prefix="Root: "):
            if node is not None:
                print(" " * (level * 4) + prefix + str(node.value))
                if node.leftChild or node.rightChild:  # If the node has children
                    print_tree2(node.leftChild, level + 1, "L--- ")
                    print_tree2(node.rightChild, level + 1, "R--- ")
            else:
                print(" " * (level * 4) + prefix + "None")  # Print None for empty nodes

        print_tree2(self.rootNode)
        print()  




class ArrayBasedBinaryTree:
    def __init__(self, capacity=10000):
        self.tree = np.full(capacity, None, dtype=object)  # Initialize tree with None values
        self.size = 0  # Current number of nodes in the tree
        self.capacity = capacity

    def add(self, value):
        if self.size >= self.capacity:
            raise OverflowError("Tree is full. Cannot add more elements.")
        
        # If tree is empty, add at root
        if self.size == 0:
            self.tree[0] = value
            self.size += 1
            return

        # Use level-order traversal to find the first valid position
        queue = [0]  # Start with root index
        while queue:
            current_index = queue.pop(0)
            
            # Decide whether to go left or right based on value
            if value < self.tree[current_index]:
                left_index = 2 * current_index + 1
                if left_index >= self.capacity:
                    raise OverflowError("Tree capacity exceeded during insertion.")
                
                # If left child position is empty, insert here
                if left_index >= self.size or self.tree[left_index] is None:
                    self.tree[left_index] = value
                    self.size = max(self.size, left_index + 1)
                    return
                queue.append(left_index)
            else:
                right_index = 2 * current_index + 2
                if right_index >= self.capacity:
                    raise OverflowError("Tree capacity exceeded during insertion.")
                
                # If right child position is empty, insert here
                if right_index >= self.size or self.tree[right_index] is None:
                    self.tree[right_index] = value
                    self.size = max(self.size, right_index + 1)
                    return
                queue.append(right_index)

    def find(self, value):
        return self._find_recursive(value, 0)
    
    def _find_recursive(self, value, current_index):
        if current_index >= self.size or self.tree[current_index] is None:
            return None
        
        if self.tree[current_index] == value:
            return current_index
        
        if value < self.tree[current_index]:
            return self._find_recursive(value, 2 * current_index + 1)
        return self._find_recursive(value, 2 * current_index + 2)

    def delete(self, value):
        index = self.find(value)
        if index is None:
            return False

        # Helper function to find minimum value index in a subtree
        def find_min_index(start_index):
            current = start_index
            while current < self.size:
                left = 2 * current + 1
                if left >= self.size or self.tree[left] is None:
                    return current
                current = left
            return current

        # Get child indices
        left_index = 2 * index + 1
        right_index = 2 * index + 2
        
        # Case 1: Leaf node
        if (left_index >= self.size or self.tree[left_index] is None) and \
        (right_index >= self.size or self.tree[right_index] is None):
            self.tree[index] = None
        
        # Case 2: Node has only left child
        elif right_index >= self.size or self.tree[right_index] is None:
            # Move the entire left subtree up
            subtree = capture_subtree(self, left_index)
            clear_subtree(self, index)
            restore_subtree(self, subtree, index)
        
        # Case 3: Node has only right child
        elif left_index >= self.size or self.tree[left_index] is None:
            # Move the entire right subtree up
            subtree = capture_subtree(self, right_index)
            clear_subtree(self, index)
            restore_subtree(self, subtree, index)
        
        # Case 4: Node has both children
        else:
            # Find the successor (minimum value in right subtree)
            successor_index = find_min_index(right_index)
            successor_value = self.tree[successor_index]
            
            if successor_index == right_index:
                # If the successor is the immediate right child
                self.tree[index] = successor_value
                right_subtree = capture_subtree(self, 2 * successor_index + 2)
                clear_subtree(self, successor_index)
                if right_subtree:
                    restore_subtree(self, right_subtree, right_index)
            else:
                # If the successor is deeper in the right subtree
                self.tree[index] = successor_value
                
                # Handle successor's right child if it exists
                successor_right = 2 * successor_index + 2
                if successor_right < self.size and self.tree[successor_right] is not None:
                    successor_parent = (successor_index - 1) // 2
                    subtree = capture_subtree(self, successor_right)
                    clear_subtree(self, successor_index)
                    restore_subtree(self, subtree, successor_index)
                else:
                    self.tree[successor_index] = None

        # Update size
        self._update_size()
        return True

    def _update_size(self):
        while self.size > 0 and self.tree[self.size - 1] is None:
            self.size -= 1

    def left_child(self, index):
        child_index = 2 * index + 1
        if child_index < self.size:
            return child_index
        return None

    def right_child(self, index):
        child_index = 2 * index + 2
        if child_index < self.size:
            return child_index
        return None

    def parent(self, index):
        if index == 0 or index >= self.size:
            return None
        return (index - 1) // 2

    def get_depth_at_index(self, index):
      if index >= self.size or self.tree[index] is None:
          return 0

      left_depth = self.get_depth_at_index(2 * index + 1)
      right_depth = self.get_depth_at_index(2 * index + 2)

      return max(left_depth, right_depth) + 1

    def get_depth(self):
        if self.size == 0:
            return 0
        return self.get_depth_at_index(0)

    def get_balance_at_index(self, index):
        if index >= self.size or self.tree[index] is None:
            return 0

        left_depth = self.get_depth_at_index(2 * index + 1)
        right_depth = self.get_depth_at_index(2 * index + 2)

        return left_depth - right_depth

    def is_balanced_at_index(self, index):
        if index >= self.size or self.tree[index] is None:
            return True

        # Check balance factor
        balance = self.get_balance_at_index(index)
        if abs(balance) > 1:
            return False

        # Check children
        left_index = 2 * index + 1
        right_index = 2 * index + 2

        return (self.is_balanced_at_index(left_index) and
                self.is_balanced_at_index(right_index))

    def is_balanced(self):
        if self.size == 0:
            return True
        return self.is_balanced_at_index(0)

    def pretty_print(self):
        def print_tree(index, level=0, prefix="Root: "):
            # Only print if the index is within the valid size of the tree
            if index < self.size and self.tree[index] is not None:
                print(" " * (level * 4) + prefix + str(self.tree[index]))

                # Get the left and right children indices
                left_index = self.left_child(index)
                right_index = self.right_child(index)

                # Recursively print left and right children if they exist
                if left_index is not None and left_index < self.size and self.tree[left_index] is not None:
                    print_tree(left_index, level + 1, "L--- ")
                if right_index is not None and right_index < self.size and self.tree[right_index] is not None:
                    print_tree(right_index, level + 1, "R--- ")
            # If no valid node exists at the given index, don't print anything

        # Start printing from the root (index 0)
        print_tree(0)
        print()  # Newline for better readability



from math import log2, log, floor
def left_rotate_reference(tree, node):
    if node is None or node.rightChild is None:
        return
    
    right_child = node.rightChild
    parent = node.parent
    
    # Update parent pointers
    node.setRightChild(right_child.leftChild)
    right_child.setLeftChild(node)
    right_child.parent = parent
    
    # Update tree root if necessary
    if parent is None:
        tree.rootNode = right_child
    else:
        if parent.leftChild == node:
            parent.setLeftChild(right_child)
        else:
            parent.setRightChild(right_child)

def right_rotate_reference(tree, node):
    if node is None or node.leftChild is None:
        return
    
    left_child = node.leftChild
    parent = node.parent
    
    # Update parent pointers
    node.setLeftChild(left_child.rightChild)
    left_child.setRightChild(node)
    left_child.parent = parent
    
    # Update tree root if necessary
    if parent is None:
        tree.rootNode = left_child
    else:
        if parent.leftChild == node:
            parent.setLeftChild(left_child)
        else:
            parent.setRightChild(left_child)

def create_backbone_reference(tree):
    root = tree.rootNode
    while root is not None:
        while root.leftChild is not None:
            right_rotate_reference(tree, root)
            root = root.parent
        root = root.rightChild

def dsw_reference(tree):
    if tree.rootNode is None:
        return
    
    # Create the backbone
    create_backbone_reference(tree)
    
    # Count nodes
    n = 0
    current = tree.rootNode
    while current is not None:
        n += 1
        current = current.rightChild
    
    # Calculate number of leaves in bottom level
    m = 2 ** (int(log(n + 1, 2))) - 1
    
    # First round of rotations
    DSW_rotates_reference(tree, n - m)
    
    # Subsequent rounds
    while m > 1:
        m = m // 2
        DSW_rotates_reference(tree, m)

def DSW_rotates_reference(tree, count):
    current = tree.rootNode
    for _ in range(count):
        next_root = current.rightChild
        left_rotate_reference(tree, current)
        current = next_root.rightChild



# Array-based implementation
def create_backbone_array(tree):
    if tree.size == 0:
        return
    
    # First, collect all values in sorted order
    values = []
    def inorder_collect(index):
        if index >= len(tree.tree) or tree.tree[index] is None:
            return
        inorder_collect(2 * index + 1)  # Left
        values.append(tree.tree[index])  # Current
        inorder_collect(2 * index + 2)   # Right
    
    # Collect all values
    inorder_collect(0)
   
    # Clear current tree
    tree.tree = [None] * tree.capacity
    
    # Create backbone - place values in a right-leaning chain
    if values:
        # Place first value at root
        tree.tree[0] = values[0]
        
        # Place each subsequent value as right child
        current_index = 0
        for i in range(1, len(values)):
            right_child_index = 2 * current_index + 2
            
            # If we need more space, expand the array
            if right_child_index >= len(tree.tree):
                new_capacity = max(tree.capacity * 2, right_child_index + 1)
                new_tree = [None] * new_capacity
                for j in range(len(tree.tree)):
                    new_tree[j] = tree.tree[j]
                tree.tree = new_tree
                tree.capacity = new_capacity
            
            tree.tree[right_child_index] = values[i]
            current_index = right_child_index
        
        # Update tree size to include the entire chain
        tree.size = current_index + 1

def right_rotate_array(tree, index):
    if index >= tree.size:
        return False
    
    left_index = 2 * index + 1
    if left_index >= tree.size or tree.tree[left_index] is None:
        return False

    # Store values
    root_value = tree.tree[index]
    left_value = tree.tree[left_index]
    left_right_index = 2 * left_index + 2
    
    # Get left's right child if it exists
    left_right_value = None
    if left_right_index < tree.size:
        left_right_value = tree.tree[left_right_index]
    
    # Perform rotation
    tree.tree[index] = left_value
    tree.tree[left_index] = root_value
    
    # Handle the left's right child
    right_index = 2 * index + 2
    if left_right_value is not None and right_index < tree.size:
        tree.tree[right_index] = left_right_value
        tree.tree[left_right_index] = None
    
    return True


def left_rotate_array(tree, index):
    right_index = index
    if right_index >= tree.size or tree.tree[right_index] is None:
        return False
    
    index = (index - 1) // 2
    if index >= tree.size or tree.tree[index] is None:
        return False

    # Store values
    root_val = tree.tree[index]
    right_val = tree.tree[right_index]

    # Move right child value to root
    tree.tree[index] = right_val
    left_index = 2 * index + 1
    # Move left child of root to one level down, in order to keep same children after move
    if tree.tree[left_index] is not None:
        move_subtree(tree, left_index , 2*left_index+1)

    # Move root value to left child
    tree.tree[left_index] = root_val

    # Move right's left child (if exists) to root's right child
    right_left_index = 2 * right_index + 1
    if right_left_index < tree.size and tree.tree[right_left_index] is not None:
        move_subtree(tree, right_left_index, 2 * left_index + 2)

    # Move right child's right child
    move_subtree(tree, 2 * right_index + 2 , right_index)
    return True


def capture_subtree(tree, index):
    if index >= tree.size or tree.tree[index] is None:
        return None
    
    return {
        'value': tree.tree[index],
        'left': capture_subtree(tree, 2 * index + 1),
        'right': capture_subtree(tree, 2 * index + 2)
    }


def restore_subtree(tree, subtree, index):
    if subtree is None or index >= tree.size:
        return
    tree.tree[index] = subtree['value']
    restore_subtree(tree, subtree['left'], 2 * index + 1)
    restore_subtree(tree, subtree['right'], 2 * index + 2)


def clear_subtree(tree, index):
    if index >= tree.size or tree.tree[index] is None:
        return
    tree.tree[index] = None
    clear_subtree(tree, 2 * index + 1)
    clear_subtree(tree, 2 * index + 2)


def move_subtree(tree, from_index, to_index):
    if from_index >= tree.size or tree.tree[from_index] is None:
        return
    
    # Capture the subtree we are moving
    subtree_to_move = capture_subtree(tree, from_index)
    
    # Clear the original position and destination
    clear_subtree(tree, from_index)
    clear_subtree(tree, to_index)
    
    # Restore the subtree at the new position
    restore_subtree(tree, subtree_to_move, to_index)


def perform_rotations(tree, count):
    index = 0
    index = 2 * index + 2
    for _ in range(count):
        left_rotate_array(tree, index)
        index = 2 * index + 2
           

def create_balanced_tree_from_backbone(tree):
    # Count the number of nodes in the backbone
    n = sum(1 for val in tree.tree if val is not None)
    if n <= 1:
        return

    # Calculate the largest perfect subtree size
    m = 2 ** (floor(log2(n + 1))) - 1

    # First set of rotations
    perform_rotations(tree, n - m)

    # Continue performing rotations
    while m > 1:
        m = m//2
        perform_rotations(tree, m)


def dsw_array(tree):
    if tree.size == 0:
        return

    # Step 1: Create the backbone
    create_backbone_array(tree)

    # Step 2: Create a balanced binary tree from the backbone
    create_balanced_tree_from_backbone(tree)


#   Example usage

# print("\nReference-Based Binary Tree Example:")
# ref_tree = ReferenceBasedBinaryTree()
# values = [60,53,80,45,49,48,33,25,94,36]
# for value in values:
#     ref_tree.add(value)

# print("Initial Tree (Reference-Based):")
# depth = ref_tree.get_depth()
# print(f"Tree depth is: {depth}")
# ref_tree.print_tree()
# print("Is balanced:", ref_tree.is_balanced())

# # Create backbone and print intermediate state
# create_backbone_reference(ref_tree)
# print("\nAfter creating backbone (Reference-Based):")
# depth = ref_tree.get_depth()
# print(f"Tree depth is: {depth}")
# ref_tree.print_tree()

# # Complete the DSW algorithm
# dsw_reference(ref_tree)
# print("\nAfter DSW (Reference-Based):")
# depth = ref_tree.get_depth()
# print(f"Tree depth is: {depth}")
# ref_tree.print_tree()
# print("Is balanced:", ref_tree.is_balanced())

# print("\n" + "="*50 + "\n")

# print("Array-Based Binary Tree Example:")
# array_tree = ArrayBasedBinaryTree()
# for value in values:  # Same values as reference-based example
#     array_tree.add(value)

# print("Initial Tree (Array-Based):")
# depth = array_tree.get_depth()
# print(f"Tree depth is: {depth}")
# array_tree.pretty_print()
# print("Is balanced:", array_tree.is_balanced())

# array_tree.delete(33)
# print("array tree after deletion : ")
# array_tree.pretty_print()

# array_tree.delete(48)
# print("array tree after deletion : ")
# array_tree.pretty_print()

# array_tree.delete(60)
# print("array tree after deletion : ")
# array_tree.pretty_print()

# # Create backbone and print intermediate state
# create_backbone_array(array_tree)
# print("\nAfter creating backbone (Array-Based):")
# depth = array_tree.get_depth()
# print(f"Tree depth is: {depth}")
# array_tree.pretty_print()


# # Complete the DSW algorithm
# dsw_array(array_tree)
# print("\nAfter DSW (Array-Based):")
# depth = array_tree.get_depth()
# print(f"Tree depth is: {depth}")
# array_tree.pretty_print()
# print("Is balanced:", array_tree.is_balanced())