import os
import json
from pathlib import Path


def get_folder_structure(path, indent=0, max_depth=None, current_depth=0):
    """
    Recursively get folder structure

    Args:
        path: Directory path to explore
        indent: Current indentation level for printing
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current depth in recursion
    """
    if max_depth is not None and current_depth > max_depth:
        return

    try:
        # Get all items in the directory
        items = os.listdir(path)

        # Separate directories and files
        dirs = [item for item in items if os.path.isdir(os.path.join(path, item))]
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]

        # Print current directory name
        if indent == 0:
            print(f"\n📁 {os.path.basename(path)}/")
        else:
            print(f"{' ' * indent}📁 {os.path.basename(path)}/")

        # Print subdirectories
        for dir_name in sorted(dirs):
            sub_path = os.path.join(path, dir_name)
            get_folder_structure(sub_path, indent + 4, max_depth, current_depth + 1)

        # Print files (with count if there are many)
        if files and current_depth < 2:  # Only show files in top levels
            for file_name in sorted(files)[:10]:  # Limit to first 10 files
                print(f"{' ' * (indent + 4)}📄 {file_name}")
            if len(files) > 10:
                print(f"{' ' * (indent + 4)}... and {len(files) - 10} more files")
        elif files:
            print(f"{' ' * (indent + 4)}📄 {len(files)} files")

    except PermissionError:
        print(f"{' ' * indent}⚠️ Permission denied: {path}")
    except Exception as e:
        print(f"{' ' * indent}❌ Error accessing {path}: {e}")


def get_detailed_structure(path):
    """Get detailed information about the folder structure"""
    structure = {}

    try:
        for root, dirs, files in os.walk(path):
            # Get relative path
            rel_path = os.path.relpath(root, path)
            if rel_path == '.':
                rel_path = os.path.basename(path)

            structure[rel_path] = {
                'directories': sorted(dirs),
                'files': sorted(files),
                'file_count': len(files),
                'dir_count': len(dirs)
            }
    except Exception as e:
        print(f"Error walking directory: {e}")

    return structure


def analyze_datasets(target_dir: str = "./data"):
    """Main function to analyze the datasets folder"""

    # Check if directory exists
    if not os.path.exists(target_dir):
        print(f"❌ Directory not found: {target_dir}")
        print("Please check if the path is correct.")
        return

    print("=" * 80)
    print("📊 DATASETS FOLDER STRUCTURE ANALYSIS")
    print("=" * 80)
    print(f"📍 Path: {target_dir}")

    # Get basic info
    try:
        total_size = sum(os.path.getsize(os.path.join(root, file))
                         for root, _, files in os.walk(target_dir)
                         for file in files)

        total_files = sum(len(files) for _, _, files in os.walk(target_dir))
        total_dirs = sum(len(dirs) for _, dirs, _ in os.walk(target_dir))

        print(f"\n📈 Summary:")
        print(f"   - Total directories: {total_dirs}")
        print(f"   - Total files: {total_files}")
        print(f"   - Total size: {total_size / (1024 ** 3):.2f} GB")

    except Exception as e:
        print(f"⚠️ Could not calculate size: {e}")

    # Show folder structure
    print("\n" + "=" * 80)
    print("📁 FOLDER STRUCTURE")
    print("=" * 80)
    get_folder_structure(target_dir, max_depth=3)  # Limit depth to 3 for readability

    # Get detailed JSON structure
    print("\n" + "=" * 80)
    print("📋 DETAILED STRUCTURE (JSON format)")
    print("=" * 80)
    detailed = get_detailed_structure(target_dir)

    # Print first few items of detailed structure
    for i, (folder, info) in enumerate(detailed.items()):
        if i < 5:  # Show first 5 folders
            print(f"\n📂 {folder}")
            print(f"   Subdirectories: {info['directories'][:5]}")
            if len(info['directories']) > 5:
                print(f"   ... and {len(info['directories']) - 5} more")
            print(f"   Files: {info['file_count']}")
            print(f"   Sample files: {info['files'][:3]}")
        elif i == 5:
            print("\n... and more folders")
            break

    # Save full structure to a JSON file
    output_file = os.path.join(target_dir, "folder_structure_report.json")
    try:
        with open(output_file, 'w') as f:
            json.dump(detailed, f, indent=2)
        print(f"\n✅ Full folder structure saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Could not save JSON report: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Analyze datasets folder structure")
    parser.add_argument("--target_dir", type=str, default="./data",
                        help="Path to the datasets directory (default: ./data)")
    args = parser.parse_args()
    analyze_datasets(args.target_dir)
