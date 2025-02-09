import argparse
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig, TreeBuilderConfig, TreeRetrieverConfig
from dotenv import load_dotenv
import os

load_dotenv()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Ingest text into RAPTOR and build the tree.")
    parser.add_argument("--input_file", type=str, required=True,
                       help="Path to the input text file to be ingested.")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Path where the RAPTOR tree will be saved.")
    return parser.parse_args()

def load_text(file_path: str) -> str:
    """Load text from the input file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        raise

def main():
    args = parse_arguments()
    
    # Create save directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    RAC = RetrievalAugmentationConfig(
        tb_max_tokens=200,
    )
    
    # Initialize RAPTOR
    RA = RetrievalAugmentation(RAC)
    
    # Load and process the text
    print(f"Loading text from {args.input_file}...")
    text = load_text(args.input_file)
    
    
    # Build the tree
    print("Building RAPTOR tree...")
    RA.add_documents(
        text,
        leaf_node_file_path="{}_leaf_nodes.pkl".format(args.input_file)
    )

    print(f"Tree constructed successfully and saved to {args.save_path}!")

if __name__ == "__main__":
    main() 