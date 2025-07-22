#!/bin/bash

# Git Cleanup Script for Huskybot Workspace
# Solves git pull conflicts by properly handling build artifacts

set -e

echo "🧹 ========================================================"
echo "   HUSKYBOT GIT CLEANUP & PULL SCRIPT"
echo "🧹 ========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

print_status "Working in workspace: $WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"

# Check git status
print_status "Checking git status..."
git status --porcelain

# Check if there are untracked files that need to be removed
print_status "Cleaning build artifacts..."

# Remove build artifacts that should not be in git
if [ -d "build/" ]; then
    print_warning "Removing build/ directory..."
    rm -rf build/
fi

if [ -d "install/" ]; then
    print_warning "Removing install/ directory..."
    rm -rf install/
fi

if [ -d "log/" ]; then
    print_warning "Removing log/ directory..."
    rm -rf log/
fi

# Add .gitignore entries if not present
print_status "Ensuring proper .gitignore..."
GITIGNORE_CONTENT="# ROS2 Build Artifacts
build/
install/
log/

# Models
models/*.engine
models/*.onnx
models/*.pt

# Python cache
__pycache__/
*.pyc
*.pyo

# IDE
.vscode/settings.json
.vscode/launch.json

# Temporary files
*.tmp
*.swp
*~

# OS specific
.DS_Store
Thumbs.db"

if [ ! -f ".gitignore" ]; then
    echo "$GITIGNORE_CONTENT" > .gitignore
    print_success "Created .gitignore file"
elif ! grep -q "build/" .gitignore; then
    echo -e "\n$GITIGNORE_CONTENT" >> .gitignore
    print_success "Updated .gitignore file"
fi

# Try git pull again
print_status "Attempting git pull..."
if git pull; then
    print_success "✅ Git pull successful!"
else
    print_error "❌ Git pull failed. Manual intervention may be required."
    print_status "You may need to run:"
    echo "  git reset --hard HEAD"
    echo "  git clean -fd"
    echo "  git pull"
    exit 1
fi

print_success "🎯 Git cleanup and pull completed successfully!"
print_status "You can now run: ./src/huskybot_deepstream_ultra/scripts/build_and_test.sh"
