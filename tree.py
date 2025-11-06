from treelib import Node, Tree

tree_dict = {}

# Create a new tree
tree_dict["company_structure"] = Tree()
tree = tree_dict["company_structure"]
tree.create_node("haha", "root")
tree_dict["company_structure"].show()

# Add root node
# tree.create_node("Company", "company")

# # Add departments
# tree.create_node("Engineering", "eng", parent="company")
# tree.create_node("Sales", "sales", parent="company")
# tree.create_node("HR", "hr", parent="company")

# # Add team members
# tree.create_node("Alice (CTO)", "alice", parent="eng")
# tree.create_node("Bob (Developer)", "bob", parent="eng")
# tree.create_node("Carol (Sales Manager)", "carol", parent="sales")
# tree.create_node("Dave (HR Manager)", "dave", parent="hr")

# # Display the tree
# tree.show()