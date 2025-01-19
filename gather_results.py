import time
import memory_profiler
import os
from graphviz import Digraph
import numpy as np
from datetime import datetime

from aads import ReferenceBasedBinaryTree, ArrayBasedBinaryTree, dsw_reference, dsw_array, create_backbone_reference, create_backbone_array

class TreeVisualizer:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def visualize_reference_tree(self, tree, filename):
        dot = Digraph(comment='Binary Tree')
        dot.attr(rankdir='TB')
        
        def add_nodes_edges(node):
            if node:
                dot.node(str(id(node)), str(node.value))
                if node.leftChild:
                    dot.edge(str(id(node)), str(id(node.leftChild)), 'L')
                    add_nodes_edges(node.leftChild)
                if node.rightChild:
                    dot.edge(str(id(node)), str(id(node.rightChild)), 'R')
                    add_nodes_edges(node.rightChild)
        
        add_nodes_edges(tree.rootNode)
        dot.render(f"{self.output_dir}/{filename}", view=False)
    
    def visualize_array_tree(self, tree, filename):
        dot = Digraph(comment='Binary Tree')
        dot.attr(rankdir='TB')
        
        def add_nodes_edges(index):
            if index < tree.size and tree.tree[index] is not None:
                dot.node(str(index), str(tree.tree[index]))
                left_idx = 2 * index + 1
                right_idx = 2 * index + 2
                
                if left_idx < tree.size and tree.tree[left_idx] is not None:
                    dot.edge(str(index), str(left_idx), 'L')
                    add_nodes_edges(left_idx)
                if right_idx < tree.size and tree.tree[right_idx] is not None:
                    dot.edge(str(index), str(right_idx), 'R')
                    add_nodes_edges(right_idx)
        
        add_nodes_edges(0)
        dot.render(f"{self.output_dir}/{filename}", view=False)

class StepAnalyzer:
    def __init__(self, data_file, base_output_dir='results'):
        data_name = os.path.splitext(os.path.basename(data_file))[0]
        self.output_dir = f"{base_output_dir}_{data_name}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.results_file = f"{self.output_dir}/step_analysis_results.txt"
        self.visualizer = TreeVisualizer(self.output_dir)
        
    def measure_operation(self, operation_name, func, *args):
        # Memory and time measurement
        start_time = time.time()
        memory_usage = memory_profiler.memory_usage((func, args), max_iterations=1)
        end_time = time.time()
        
        execution_time = end_time - start_time
        memory_used = max(memory_usage) - memory_usage[0]
        
        with open(self.results_file, 'a') as f:
            f.write(f"\n{operation_name}:\n")
            f.write(f"Execution Time: {execution_time:.6f} seconds\n")
            f.write(f"Memory Usage: {memory_used:.6f} MB\n")
            f.write("-" * 50 + "\n")
        
        return execution_time, memory_used

    def analyze_tree_operations(self, data_file):
        # Read data
        with open(data_file, 'r') as f:
            values = [int(x) for x in f.read().strip().split()]
        
        # Initialize results file
        with open(self.results_file, 'w') as f:
            f.write(f"Analysis Results for {data_file}\n")
            f.write(f"Started at: {datetime.now()}\n")
            f.write("=" * 50 + "\n")
        
        # Initialize trees
        ref_tree = ReferenceBasedBinaryTree()
        array_tree = ArrayBasedBinaryTree()
        
        # Step 1: Add elements one by one and measure
        print("Step 1: Adding elements...")
        with open(self.results_file, 'a') as f:
            f.write("\nStep 1: Adding Elements\n")
            f.write("=" * 30 + "\n")
        
        for i, value in enumerate(values):
            self.measure_operation(
                f"Adding {value} to reference-based tree (Element {i+1}/{len(values)})",
                ref_tree.add,
                value
            )

            self.measure_operation(
                f"Adding {value} to array-based tree (Element {i+1}/{len(values)})",
                array_tree.add,
                value
            )
        
        self.visualizer.visualize_reference_tree(ref_tree, "ref_tree_after_insertion")
        self.visualizer.visualize_array_tree(array_tree, "array_tree_after_insertion")
        
        # Step 2: Search for element
        print("Step 2: Searching for element...")
        with open(self.results_file, 'a') as f:
            f.write("\nStep 2: Searching Element\n")
            f.write("=" * 30 + "\n")
        
        self.measure_operation("Searching element in reference-based tree", ref_tree.find, 55)
        self.measure_operation("Searching element in array-based tree", array_tree.find, 55)
        
        # Step 3: Delete elements
        print("Step 3: Deleting elements...")
        with open(self.results_file, 'a') as f:
            f.write("\nStep 3: Deleting Elements\n")
            f.write("=" * 30 + "\n")
        
        for value in [33, 48, 60]:
            self.measure_operation(f"Deleting {value} from reference-based tree", ref_tree.delete, value)
            self.measure_operation(f"Deleting {value} from array-based tree", array_tree.delete, value)

        self.visualizer.visualize_reference_tree(ref_tree, "ref_tree_after_deletion")
        self.visualizer.visualize_array_tree(array_tree, "array_tree_after_deletion")
        
        # Step 4: Add elements back
        print("Step 4: Adding elements back...")
        with open(self.results_file, 'a') as f:
            f.write("\nStep 4: Adding Elements Back\n")
            f.write("=" * 30 + "\n")
        
        for value in [33, 48, 60]:
            self.measure_operation(f"Re-adding {value} to reference-based tree", ref_tree.add, value)
            self.measure_operation(f"Re-adding {value} to array-based tree", array_tree.add, value)

        # Step 5: Create Backbone
        print("Step 5: Creating backbones for array-based and reference-based trees...")
        with open(self.results_file, 'a') as f:
            f.write("\nStep 5: Creating Backbones\n")
            f.write("=" * 30 + "\n")

        self.measure_operation("Create backbone (reference-based tree)", create_backbone_reference, ref_tree)
        self.measure_operation("Create backbone (array-based tree)", create_backbone_array, array_tree)
        self.visualizer.visualize_reference_tree(ref_tree, "ref_tree_backbone")
        self.visualizer.visualize_array_tree(array_tree, "array_tree_backbone")
        
        # Step 6: Run DSW
        print("Step 6: Running DSW algorithm...")
        with open(self.results_file, 'a') as f:
            f.write("\nStep 6: DSW Algorithm\n")
            f.write("=" * 30 + "\n")
        
        self.measure_operation("DSW on reference-based tree", dsw_reference, ref_tree)
        self.measure_operation("DSW on array-based tree", dsw_array, array_tree)
        self.visualizer.visualize_reference_tree(ref_tree, "ref_tree_final")
        self.visualizer.visualize_array_tree(array_tree, "array_tree_final")

def main():
    # Change data name
    data_file = 'data_20.txt'
    analyzer = StepAnalyzer(data_file)
    try:
        analyzer.analyze_tree_operations(data_file)
        print("Analysis complete. Check the 'results' directory for detailed output.")
    except FileNotFoundError:
        print("Error: text file not found!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()