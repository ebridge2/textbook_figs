import os
import shutil
import argparse

def copy_svg_files(base_dir, output_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.svg'):
                # Get the full path of the source file
                src_path = os.path.join(root, file)
                
                # Calculate the relative path
                rel_path = os.path.relpath(root, base_dir)
                
                # Remove 'Figures/svg' from the path
                path_parts = rel_path.split(os.sep)
                if 'Figures' in path_parts and 'svg' in path_parts:
                    figures_index = path_parts.index('Figures')
                    svg_index = path_parts.index('svg')
                    if svg_index == figures_index + 1:
                        path_parts = path_parts[:figures_index] + path_parts[svg_index+1:]
                
                # Reconstruct the path without 'Figures/svg'
                new_rel_path = os.path.join(*path_parts) if path_parts else ''
                
                # Construct the destination path
                dest_path = os.path.join(output_dir, new_rel_path, file)
                
                # Create the destination directory if it doesn't exist
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                
                # Copy the file
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} -> {dest_path}")

def copy_ai_files(base_dir, output_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.ai'):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, base_dir)
                
                # Remove 'Figures/ai' from the path
                path_parts = rel_path.split(os.sep)
                if 'Figures' in path_parts and 'ai' in path_parts:
                    figures_index = path_parts.index('Figures')
                    ai_index = path_parts.index('ai')
                    if ai_index == figures_index + 1:
                        path_parts = path_parts[:figures_index] + path_parts[ai_index+1:]

                if 'Figures' in path_parts and 'svg' in path_parts:
                    figures_index = path_parts.index('Figures')
                    svg_index = path_parts.index('svg')
                    if svg_index == figures_index + 1:
                        path_parts = path_parts[:figures_index] + path_parts[svg_index+1:]
                
                new_rel_path = os.path.join(*path_parts) if path_parts else ''
                dest_path = os.path.join(output_dir, new_rel_path, file)
                
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"Copied: {src_path} -> {dest_path}")

def main():
    parser = argparse.ArgumentParser(description="Copy SVG and AI files while preserving a streamlined directory structure.")
    parser.add_argument("base_dir", help="Base directory to search for SVG and AI files")
    parser.add_argument("output_dir", help="Output directory to copy SVG files to")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.isdir(base_dir):
        print(f"Error: The specified base directory does not exist: {base_dir}")
        return

    copy_svg_files(base_dir, output_dir)
    print("SVG file copying completed.")
    copy_ai_files(base_dir, output_dir)
    print("AI file copying completed.")

if __name__ == "__main__":
    main()
