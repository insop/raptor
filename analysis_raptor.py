# # Notebook for analysis RAPTOR

import argparse
from raptor import RetrievalAugmentation
from dotenv import load_dotenv

load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run RAPTOR analysis with a query string.")
    parser.add_argument("--query", type=str, default="", 
                       help="The query string to be answered by RAPTOR.")
    parser.add_argument("--save_path", type=str, default="demo/cinderella_3_small", 
                       help="The path to the saved RAPTOR tree.")
    parser.add_argument("--collapse_tree", type=bool, default=False, 
                       help="Whether to collapse the tree.")
    return parser.parse_args()

def load_index(save_path:str):
    return RetrievalAugmentation(tree=save_path)

def main():
    args = parse_arguments()
    
    # ### Test load using the saved RA
    RA = load_index(save_path=args.save_path)

    if args.query == "":
        # Read Cinderella story from sample.txt
        with open('demo/sample.txt', 'r') as file:
            text = file.read()
        # Construct the tree
        RA.add_documents(text, use_saved_leaf_nodes=True, leaf_node_file_path="demo/sample_leaf_nodes.pkl")
        print("Tree constructed successfully!")
    else:
        # Test question answering
        # question = "what happened to sisters by a bird?"
        print("Query:", args.query)
        print("Collapsing tree:", args.collapse_tree)

        answer, layer_information = RA.answer_question(question=args.query, collapse_tree=args.collapse_tree, return_layer_information=True)
        print("\nQuestion:", args.query)
        print("Answer:", answer)
        print("Layer Information:", layer_information)

if __name__ == "__main__":
    main()
