# create-github-repo.ps1

param(
    [ValidateSet('private', 'public')]
    [string]$Visibility = 'private'  # Default to private if not specified
)

Write-Host "🔒 Repository visibility set to: $Visibility"

# Get the absolute path of the script's directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Function to find Git executable
function Find-Git {
    $gitPaths = @(
        "C:\Program Files\Git\cmd\git.exe",
        "C:\Program Files (x86)\Git\cmd\git.exe",
        "${env:ProgramFiles}\Git\cmd\git.exe",
        "${env:ProgramFiles(x86)}\Git\cmd\git.exe",
        "${env:LOCALAPPDATA}\Programs\Git\cmd\git.exe"
    )
    
    foreach ($path in $gitPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    return $null
}

# Function to find GitHub CLI executable
function Find-GitHubCLI {
    $ghPaths = @(
        "C:\Program Files\GitHub CLI\gh.exe",
        "C:\Program Files (x86)\GitHub CLI\gh.exe",
        "${env:ProgramFiles}\GitHub CLI\gh.exe",
        "${env:ProgramFiles(x86)}\GitHub CLI\gh.exe",
        "${env:LOCALAPPDATA}\Programs\GitHub CLI\gh.exe"
    )
    
    foreach ($path in $ghPaths) {
        if (Test-Path $path) {
            return $path
        }
    }
    return $null
}

# Find Git and add to PATH if needed
$gitPath = Find-Git
if ($null -eq $gitPath) {
    Write-Host "❌ Error: Git not found in common locations. Please install Git or add it to PATH."
    exit 1
}

# Add Git to PATH for this session
$gitDir = Split-Path -Parent (Split-Path -Parent $gitPath)
$env:Path = "$gitDir\cmd;$gitDir\bin;$env:Path"

# Find GitHub CLI and add to PATH if needed
$ghPath = Find-GitHubCLI
if ($null -eq $ghPath) {
    Write-Host "❌ Error: GitHub CLI not found in common locations. Please install GitHub CLI using:"
    Write-Host "winget install GitHub.cli"
    exit 1
}

# Add GitHub CLI to PATH for this session
$ghDir = Split-Path -Parent $ghPath
$env:Path = "$ghDir;$env:Path"

# Verify Git is now accessible
try {
    $null = & git --version
    Write-Host "✅ Git found and accessible"
} catch {
    Write-Host "❌ Error: Git is still not accessible. Please ensure Git is properly installed."
    exit 1
}

# Configure Git user if not already configured
$userEmail = & git config --global user.email
$userName = & git config --global user.name

if (-not $userEmail) {
    Write-Host "🔧 Setting Git user email..."
    & git config --global user.email "graham@grahama.co"
}

if (-not $userName) {
    Write-Host "🔧 Setting Git user name..."
    & git config --global user.name "Graham Anderson"
}

Write-Host "✅ Git user configured as: $(git config --global user.name) <$(git config --global user.email)>"

# Verify GitHub CLI is accessible
try {
    $null = & gh --version
    Write-Host "✅ GitHub CLI found and accessible"
} catch {
    Write-Host "❌ Error: GitHub CLI is still not accessible. Please ensure GitHub CLI is properly installed."
    exit 1
}

# Change to project root directory
Set-Location $ProjectRoot

# Get GitHub username early for cleanup
try {
    $username = gh api user --jq '.login'
    if (-not $username) {
        throw "Empty username returned"
    }
    Write-Host "👤 Authenticated as: $username"
} catch {
    Write-Host "❌ Error: Could not get GitHub username. Please run 'gh auth login' first."
    exit 1
}

# Clean up existing repository if it exists
Write-Host "🧹 Checking for existing repository..."
if (Test-Path ".git") {
    Write-Host "Found local git repository. Removing..."
    Remove-Item -Recurse -Force .git
    Write-Host "✅ Local repository removed"
}

# Check if remote repository exists and delete it
$ProjectName = "cursor-clicker"
try {
    $repoExists = gh repo view "$username/$ProjectName" 2>$null
    if ($?) {
        Write-Host "Found existing GitHub repository. Deleting..."
        gh repo delete "$username/$ProjectName" --yes
        Write-Host "✅ Remote repository deleted"
    }
} catch {
    Write-Host "No existing remote repository found"
}

Write-Host "🔄 Starting fresh repository creation..."

# Load environment variables from project root
if (Test-Path "$ProjectRoot\.env") {
    Write-Host "🌍 Loading environment variables from project root .env..."
    Get-Content "$ProjectRoot\.env" | ForEach-Object {
        if (-not $_.StartsWith('#') -and $_.Length -gt 0) {
            $key, $value = $_ -split '=', 2
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
} else {
    Write-Host "⚠️ No .env file found in the project root. Continuing without it..."
}

# Use PROJECT_ROOT from .env or fallback to calculated root
if (-not $env:PROJECT_ROOT) {
    $env:PROJECT_ROOT = $ProjectRoot
}
Write-Host "📂 PROJECT_ROOT: $($env:PROJECT_ROOT)"

# Get the project name from the root directory
$ProjectName = "cursor-clicker"
Write-Host "🏷️ Project Name: $ProjectName"

# Check if already authenticated with GitHub
try {
    $null = & gh auth status
    Write-Host "✅ Already authenticated with GitHub"
} catch {
    Write-Host "🔑 Please authenticate with GitHub..."
    & gh auth login
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ GitHub authentication failed"
        exit 1
    }
}

# Create GitHub repository
Write-Host "🐙 Creating GitHub repository ($Visibility)..."
gh repo create $ProjectName --$Visibility --confirm

# Create .gitignore file
Write-Host "📝 Creating .gitignore file..."
@"
# Environment variables
.env
.env.*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
"@ | Out-File -FilePath "$ProjectRoot\.gitignore" -Encoding UTF8

# Initialize git repository if not already initialized
if (-not (Test-Path ".git")) {
    Write-Host "🔄 Initializing Git repository..."
    git init
} else {
    Write-Host "📂 Git repository already initialized."
}

# Ensure branch is set to main
git branch -M main

# Add all files and commit if changes exist
$status = git status --porcelain
if ($status) {
    Write-Host "📦 Adding and committing files..."
    git add .  # Add all files
    git commit -m "🎉 Initial commit"
} else {
    Write-Host "⚠️ No changes to commit."
}

# Handle existing remote
$remotes = git remote
if ($remotes -match "origin") {
    Write-Host "🔄 Remote 'origin' already exists. Updating it..."
    git remote remove origin
}

git remote add origin "https://github.com/$username/$ProjectName.git"

# Push to the remote repository
Write-Host "🚀 Pushing code to remote repository..."
try {
    git push -u origin main
    Write-Host "✅ Repository created and code pushed successfully!"
} catch {
    Write-Host "❌ Failed to push to remote. Error: $_"
    exit 1
}