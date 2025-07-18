#!/usr/bin/env python3
"""
Combine all project files into a single text file for easy upload to Claude.
This creates one file that Claude can read to understand your entire project.
"""

import os
from pathlib import Path
import datetime

def combine_project_files():
    """Combine all project files into one text file."""
    
    project_root = Path.cwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"project_combined_{timestamp}.txt"
    
    # File extensions to include (text files only)
    include_extensions = {
        '.py', '.json', '.yaml', '.yml', '.txt', '.md', '.cfg', '.ini', 
        '.sql', '.sh', '.bat', '.js', '.html', '.css', '.xml'
    }
    
    # Directories to include
    include_dirs = {'src', 'config', 'scripts', 'tests', 'docs'}
    
    # Files to include from root
    root_files = ['main.py', 'setup.py', 'requirements.txt', 'README.md']
    
    # Files/folders to exclude
    exclude_patterns = {
        '__pycache__', '.git', '.vscode', '.idea', 'node_modules',
        '.env', '.pytest_cache', 'htmlcov', 'venv', 'env', '.venv'
    }
    
    def should_include_file(file_path):
        """Check if file should be included."""
        # Check extension
        if file_path.suffix not in include_extensions:
            return False
        
        # Check size (skip files larger than 100KB)
        try:
            if file_path.stat().st_size > 100 * 1024:
                return False
        except:
            return False
        
        # Check if in excluded pattern
        for part in file_path.parts:
            if part in exclude_patterns:
                return False
        
        return True
    
    print(f"📝 Combining project files into: {output_file}")
    print(f"📁 Project root: {project_root}")
    
    files_combined = 0
    total_lines = 0
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Write header
        outfile.write("=" * 80 + "\n")
        outfile.write("QUOTEX TRADING BOT - COMPLETE PROJECT FILES\n")
        outfile.write("=" * 80 + "\n")
        outfile.write(f"Generated: {datetime.datetime.now()}\n")
        outfile.write(f"Project: {project_root.name}\n")
        outfile.write("=" * 80 + "\n\n")
        
        # Add table of contents placeholder
        toc_position = outfile.tell()
        outfile.write("\n" + "=" * 80 + "\n")
        outfile.write("TABLE OF CONTENTS\n")
        outfile.write("=" * 80 + "\n\n")
        
        # We'll come back to write TOC
        toc_placeholder = outfile.tell()
        outfile.write("\n" * 50)  # Reserve space for TOC
        
        file_list = []
        
        # Process root files first
        for filename in root_files:
            file_path = project_root / filename
            if file_path.exists() and should_include_file(file_path):
                file_list.append(('ROOT', file_path))
        
        # Process directories
        for include_dir in include_dirs:
            dir_path = project_root / include_dir
            if dir_path.exists():
                for file_path in sorted(dir_path.rglob('*')):
                    if file_path.is_file() and should_include_file(file_path):
                        relative_dir = file_path.parent.relative_to(project_root)
                        file_list.append((str(relative_dir), file_path))
        
        # Sort files by directory then name
        file_list.sort(key=lambda x: (x[0], x[1].name))
        
        # Write files
        toc_entries = []
        
        for relative_dir, file_path in file_list:
            relative_path = file_path.relative_to(project_root)
            
            print(f"  📄 Adding: {relative_path}")
            
            # Write file separator
            outfile.write("\n" + "=" * 80 + "\n")
            outfile.write(f"FILE: {relative_path}\n")
            outfile.write(f"DIRECTORY: {relative_dir}\n")
            outfile.write(f"SIZE: {file_path.stat().st_size} bytes\n")
            outfile.write("=" * 80 + "\n\n")
            
            # Add to TOC
            toc_entries.append(f"{len(toc_entries)+1:2d}. {relative_path} ({relative_dir})")
            
            # Write file content
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    
                    # Count lines
                    file_lines = len(content.splitlines())
                    total_lines += file_lines
                    
                    if not content.endswith('\n'):
                        outfile.write('\n')
                
                files_combined += 1
                
            except Exception as e:
                outfile.write(f"[ERROR: Could not read file - {e}]\n")
                print(f"    ❌ Error reading {relative_path}: {e}")
        
        # Write final summary
        outfile.write("\n" + "=" * 80 + "\n")
        outfile.write("PROJECT SUMMARY\n")
        outfile.write("=" * 80 + "\n")
        outfile.write(f"Total files combined: {files_combined}\n")
        outfile.write(f"Total lines of code: {total_lines}\n")
        outfile.write(f"Generated: {datetime.datetime.now()}\n")
        outfile.write("=" * 80 + "\n")
        
        # Go back and write TOC
        current_position = outfile.tell()
        outfile.seek(toc_placeholder)
        
        for entry in toc_entries:
            outfile.write(entry + "\n")
        
        # Fill remaining TOC space
        outfile.write("\n" + "=" * 80 + "\n\n")
        
        # Return to end
        outfile.seek(current_position)
    
    # Get file size
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"\n🎉 Project files combined successfully!")
    print(f"📄 Output file: {output_file}")
    print(f"📊 Files combined: {files_combined}")
    print(f"📏 Total lines: {total_lines}")
    print(f"💾 File size: {file_size_mb:.2f} MB")
    print(f"\n📤 You can now upload {output_file} to Claude!")
    
    return output_file

def create_file_listing():
    """Create a detailed file listing without content."""
    
    project_root = Path.cwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"project_listing_{timestamp}.txt"
    
    print(f"📋 Creating file listing: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("QUOTEX TRADING BOT - PROJECT FILE LISTING\n")
        outfile.write("=" * 60 + "\n\n")
        
        total_files = 0
        total_size = 0
        
        for root, dirs, files in os.walk(project_root):
            # Skip hidden and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
            level = root.replace(str(project_root), '').count(os.sep)
            indent = '  ' * level
            outfile.write(f"{indent}{os.path.basename(root)}/\n")
            
            subindent = '  ' * (level + 1)
            for file in sorted(files):
                if not file.startswith('.') and not file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    total_files += 1
                    
                    # Get file extension for type
                    ext = Path(file).suffix
                    
                    if file_size > 1024:
                        size_str = f"{file_size // 1024}KB"
                    else:
                        size_str = f"{file_size}B"
                    
                    outfile.write(f"{subindent}{file} ({size_str}) {ext}\n")
        
        outfile.write(f"\nSUMMARY:\n")
        outfile.write(f"Total files: {total_files}\n")
        outfile.write(f"Total size: {total_size // 1024}KB\n")
    
    print(f"✅ File listing created: {output_file}")
    return output_file

def create_source_summary():
    """Create a summary of just the important source files."""
    
    project_root = Path.cwd()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"source_summary_{timestamp}.txt"
    
    # Key files to include in summary
    key_files = [
        'main.py',
        'setup.py', 
        'requirements.txt',
        'src/utils/__init__.py',
        'src/utils/logger.py',
        'src/utils/helpers.py',
        'src/utils/data_processor.py'
    ]
    
    print(f"📋 Creating source summary: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        outfile.write("QUOTEX TRADING BOT - SOURCE CODE SUMMARY\n")
        outfile.write("=" * 60 + "\n\n")
        
        for file_path in key_files:
            full_path = project_root / file_path
            if full_path.exists():
                outfile.write(f"\n{'='*60}\n")
                outfile.write(f"FILE: {file_path}\n")
                outfile.write(f"{'='*60}\n\n")
                
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Limit content to first 100 lines for summary
                        lines = content.splitlines()
                        if len(lines) > 100:
                            outfile.write('\n'.join(lines[:100]))
                            outfile.write(f"\n\n[... truncated - {len(lines)} total lines ...]")
                        else:
                            outfile.write(content)
                except Exception as e:
                    outfile.write(f"[Error reading file: {e}]")
            else:
                outfile.write(f"\n[FILE NOT FOUND: {file_path}]\n")
    
    print(f"✅ Source summary created: {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine project files for Claude")
    parser.add_argument("--mode", choices=['combine', 'listing', 'summary'], 
                       default='combine', help="Output mode")
    
    args = parser.parse_args()
    
    if args.mode == 'combine':
        combine_project_files()
    elif args.mode == 'listing':
        create_file_listing()
    elif args.mode == 'summary':
        create_source_summary()