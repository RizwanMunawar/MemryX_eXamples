"""
============
Information:
============
File Name: example_launcher.py

============
Description:
============
Flask version of the MemryX Examples Launcher - Standalone version
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
import json
import os
import re
import socket
import subprocess
import psutil
import logging
import webbrowser

from flask import Flask, render_template, jsonify, request, send_from_directory, Response, stream_with_context
from bs4 import BeautifulSoup
from markdown import markdown as md_to_html
import threading
import queue
import time

###############################################################################
# Global constants and paths ##################################################
###############################################################################

def _looks_like_repo_root(path: Path) -> bool:
    return (path / "README.md").is_file() and (path / "gui").is_dir()


def _find_repo_root_from(start: Path) -> Optional[Path]:
    for candidate in (start, *start.parents):
        if _looks_like_repo_root(candidate):
            return candidate
    return None


def _resolve_repo_root() -> Path:
    env_root = os.environ.get("MEMRYX_EXAMPLES_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _looks_like_repo_root(candidate):
            return candidate

    module_root = Path(__file__).resolve().parent.parent
    if _looks_like_repo_root(module_root):
        return module_root

    cwd_root = _find_repo_root_from(Path.cwd().resolve())
    if cwd_root:
        return cwd_root

    return module_root


REPO_ROOT = _resolve_repo_root()
README_PATH = REPO_ROOT / "README.md"
GITHUB_BASE = "https://github.com/memryx/memryx_examples_internal/tree/release/"

EXAMPLE_BLACKLIST = [
    "Mario RL",       
    "Gesture Scrolling",  
    "Aimbot",
    "Facial Cartoonizer",
    "Chrome Dino Game",
    "Intrusion Detection",
    "Point Cloud from Depth",
    "Optimized YOLOv8 OD",
    "PCB Defect Detection",
    "CenterNet",
    "Luggage counting",
    "FootballCV",
    "PPE Detection",
    "Depth to 3D",
    "Object Tracking",
    "Detection with H/W Decoding",
    "Fitness Mirror",
    "Cartoonizer + Pose (Side-by-Side)"
]

CATEGORY_BLACKLIST = [
    "Image Inference",
    "Audio Inference",
    "Accuracy Calculation",
]


def _should_open_browser() -> bool:
    value = os.environ.get("MEMRYX_OPEN_BROWSER", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _wait_for_server_and_open_browser(url: str, port: int, timeout_seconds: float = 15.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                webbrowser.open(url, new=2)
                print(f"[STARTUP] Opened browser at {url}")
                return
        except OSError:
            time.sleep(0.2)
    print(f"[STARTUP] Server did not become reachable within {int(timeout_seconds)}s; open {url} manually.")

###############################################################################
# Data Model ##################################################################
###############################################################################

@dataclass
class ExampleItem:
    title: str
    description: str
    models: List[str] = field(default_factory=list)
    code_langs: List[str] = field(default_factory=list)
    os_list: List[str] = field(default_factory=list)
    preview_src: Optional[str] = None
    readme_rel: Optional[str] = None
    tutorial_url: Optional[str] = None
    run_sh_path: Optional[Path] = None
    has_run_sh: bool = False

    @property
    def readme_url(self) -> Optional[str]:
        if not self.readme_rel:
            return None
        return f"{GITHUB_BASE}{self.readme_rel}"

    @property
    def preview_url(self) -> Optional[str]:
        if not self.preview_src:
            return None
        if self.preview_src.startswith("http://") or self.preview_src.startswith("https://"):
            return self.preview_src
        rel = self.preview_src.lstrip("/")
        return f"/repo/{rel}"


@dataclass
class Category:
    name: str
    items: List[ExampleItem] = field(default_factory=list)

###############################################################################
# README Command Extractor ####################################################
###############################################################################

class ReadmeCommandExtractor:
    """
    Extracts setup and run commands from example README files.
    Robust handling for compound commands (&&), directory context, and cleanup.
    """
    
    def __init__(self, readme_path: Path, mode: str = "python"):
        self.readme_path = readme_path
        self.readme_text = readme_path.read_text(encoding="utf-8", errors="ignore")
        self.mode = mode
        
    def extract_commands(self) -> Tuple[List[str], Optional[str]]:
        setup_commands = self._extract_setup_commands()
        run_command = self._extract_run_command()
        return setup_commands, run_command
    
    def _extract_setup_commands(self) -> List[str]:
        """Extract only essential setup commands (wget, unzip, mkdir) from trusted sources."""
        commands = []
        seen_urls = set()
        
        # Trusted URL Paths (Safe list)
        ALLOWED_PATHS = [
            "https://developer.memryx.com/example_files/",
            "https://developer.memryx.com/model_explorer/",
        ]
        
        # Valid actions for setup
        SETUP_ACTIONS = ["wget", "curl", "mkdir", "unzip", "tar", "mv", "cd"]
        
        # Block-list for things that belong in run phase or system setup
        # We use regex word boundaries so 'pip' doesn't ban 'mediapipe'
        IGNORE_REGEX = r'\b(pip|install|sudo|apt-get|brew|dnf|yum|mx_nc|python|python3|g\+\+|cmake|make|mv objectDet_poseEst_yolov8/* ./|mv YOLOX_medium_640_640_3_onnx.dfp yolox_m.dfp|mv YOLOX_medium_640_640_3_onnx_post.onnx yolox_m_post.onnx)\b'

        def is_valid_url(cmd: str) -> bool:
            """Ensure downloads come from trusted domains."""
            if "wget" not in cmd and "curl" not in cmd:
                return True # Not a download command, so url check doesn't apply
            
            url_match = re.search(r'(https?://[^\s"\']+)', cmd)
            if not url_match: return False
            return any(allowed in url_match.group(1) for allowed in ALLOWED_PATHS)

        def process_raw_line(line: str):
            """Process a raw line, handling && splits."""
            # Split compound commands like 'mkdir models && cd models'
            parts = line.split('&&')
            for part in parts:
                process_single_command(part.strip())

        def process_single_command(cmd: str):
            # Cleanup
            cmd = re.sub(r'^sudo\s+', '', cmd).strip()
            if not cmd or cmd.startswith('#'): return
            if cmd == "mv objectDet_poseEst_yolov8/* ./":
                return
            
            # Skip build-related commands in python mode
            if self.mode == "python":
                if cmd in ["mkdir build", "cd build"] or cmd.startswith("mkdir build") or cmd.startswith("cd build"):
                    return
            
            # 1. Filter by allowed Actions (must start with wget, mkdir, etc.)
            if not any(cmd.startswith(action) for action in SETUP_ACTIONS):
                return

            # 2. Filter by Ignore Keywords (using regex boundaries)
            if re.search(IGNORE_REGEX, cmd):
                return
            
            # 3. Specific check for "./" execution which is usually a run step
            if cmd.startswith("./"):
                return
            
            # 4. URL Validation for downloads
            if not is_valid_url(cmd):
                return

            # 5. Dedup downloads
            if "wget" in cmd or "curl" in cmd:
                url_match = re.search(r'(https?://[^\s"\']+)', cmd)
                if url_match:
                    url = url_match.group(1)
                    if url in seen_urls: return
                    seen_urls.add(url)
            
            if cmd not in commands:
                commands.append(cmd)

        # --- Extraction Logic ---

        # 1. Broad Search for "Step 1" or "Download" blocks
        priority_pattern = r'(?i)###\s*(?:Step\s*1|Download).*?```bash\s*(.*?)```'
        matches = re.findall(priority_pattern, self.readme_text, re.DOTALL)
        
        if matches:
            for match in matches:
                for line in match.split('\n'):
                    process_raw_line(line)
        else:
            # 2. Fallback: Scan ALL bash blocks if no specific setup section found
            blocks = re.findall(r'```bash\s*(.*?)```', self.readme_text, re.DOTALL | re.IGNORECASE)
            for block in blocks:
                for line in block.split('\n'):
                    process_raw_line(line)
        
        return commands
    
    def _extract_run_command(self) -> Optional[str]:
        """Extract the cleanest single run command."""
        candidates = []
        blocks = re.findall(r'```bash\s*(.*?)```', self.readme_text, re.DOTALL | re.IGNORECASE)
        
        for block in blocks:
            lines = block.strip().split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or line.startswith('#'): continue

                # Skip commands with user placeholders like <file>
                if re.search(r'<[^>]+>|\[[^\]]+\]', line): continue

                cmd = None
                if self.mode == "python":
                    if line.startswith('python') or line.startswith('python3'):
                        cmd = line
                        # Look back for preceding 'cd' command (skip comments and blanks)
                        for j in range(i-1, -1, -1):
                            prev_line = lines[j].strip()
                            if not prev_line or prev_line.startswith('#'):
                                continue
                            if prev_line.startswith('cd '):
                                cmd = f"{prev_line}\n{line}"
                            break  # Stop at first non-comment/blank line
                
                elif self.mode == "cpp":
                    if line.startswith('./') and not line.endswith('.sh'):
                        cmd = line
                        # Look back for preceding 'cd' command (skip comments and blanks)
                        for j in range(i-1, -1, -1):
                            prev_line = lines[j].strip()
                            if not prev_line or prev_line.startswith('#'):
                                continue
                            if prev_line.startswith('cd '):
                                cmd = f"{prev_line}\n{line}"
                            break  # Stop at first non-comment/blank line

                if cmd: candidates.append(cmd)

        if not candidates: return None
        return min(candidates, key=len) # shortest usually best (e.g. "python run.py")
    
    def generate_run_script(self) -> str:
        setup_commands, run_command = self.extract_commands()
        
        script_lines = [
            "#!/bin/bash",
            "",
            f"# Auto-generated run script ({self.mode} mode)",
            "# Generated by MemryX Examples Launcher",
            "",
            "set -e",
            ""
        ]
        
        # --- Smart Setup Block ---
        if setup_commands:
            # Guess the check directory (e.g., 'models')
            check_dir = "models" 
            # for cmd in setup_commands:
            #     if "mkdir" in cmd:
            #         parts = cmd.split()
            #         # Find the arg after mkdir (ignoring flags like -p)
            #         for part in parts[1:]:
            #             if not part.startswith("-"):
            #                 check_dir = part
            #                 break
            
            script_lines.append(f"# Check if setup is needed (looks for '{check_dir}' folder)")
            script_lines.append(f'if [ ! -d "{check_dir}" ]; then')
            script_lines.append(f"    echo 'Directory {check_dir} not found. Setting up...'")
            
            # Save current directory state
            script_lines.append("    pushd . > /dev/null")
            
            for cmd in setup_commands:
                script_lines.append(f"    {cmd}")
            
            # Restore directory state
            script_lines.append("    popd > /dev/null")
                
            script_lines.append("else")
            script_lines.append(f"    echo 'Directory {check_dir} exists. Skipping setup.'")
            script_lines.append("fi")
            script_lines.append("")
        
        # --- Run Block ---
        if run_command:
            script_lines.append("echo 'Starting application...'")
            script_lines.append(run_command)
        else:
            script_lines.append("echo 'Error: No valid run command found in README'")
            script_lines.append("exit 1")
        
        return "\n".join(script_lines)

###############################################################################
# README Parser ###############################################################
###############################################################################

class ReadmeParser:
    """Parses the MemryX_eXamples README.md to extract categories and items."""

    BADGE_MAP = {
        "python-badge": "Python",
        "cpp-badge": "C++",
    }
    OS_NAMES = {"Linux", "Windows"}

    def __init__(self, readme_text: str):
        self.readme_text = readme_text

    @classmethod
    def from_file(cls, path: Path) -> "ReadmeParser":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return cls(text)

    def parse(self) -> List[Category]:
        html = md_to_html(self.readme_text, extensions=["tables", "fenced_code"])
        soup = BeautifulSoup(html, "html.parser")
        categories: List[Category] = []
        for h in soup.find_all(["h3"]):
            cat_name = self._clean_text(h.get_text())
            
            # Skip blacklisted categories
            if any(blacklisted.lower() in cat_name.lower() for blacklisted in CATEGORY_BLACKLIST):
                continue
            
            table = self._find_next_table(h)
            if not table:
                continue
            items = self._parse_examples_table(table)
            if items:
                categories.append(Category(name=cat_name, items=items))
        return categories

    def _find_next_table(self, heading_tag):
        for sib in heading_tag.find_all_next():
            if sib.name in ["h1", "h2", "h3"]:
                if sib is not heading_tag:
                    return None
            if sib.name == "table":
                return sib
        return None

    def _parse_examples_table(self, table_tag) -> List[ExampleItem]:
        items: List[ExampleItem] = []
        rows = table_tag.find_all("tr")
        if not rows:
            return items

        first_row_cells = rows[0].find_all(["th", "td"])
        first_row_text = ''.join([self._clean_text(c.get_text()) for c in first_row_cells])
        
        # Check if first row contains <th> tags (true headers) or just <td> tags (data)
        has_header_tags = len(rows[0].find_all("th")) > 0
        is_image_grid = not has_header_tags or len(first_row_text.strip()) == 0 or all(not self._clean_text(c.get_text()) for c in first_row_cells)
        
        def is_blacklisted(title_str: str, rel_path: Optional[str]) -> bool:
            if any(b.lower() == title_str.lower() for b in EXAMPLE_BLACKLIST):
                return True
            if rel_path:
                parts = rel_path.replace('\\', '/').split('/')
                for part in parts:
                    if any(b.lower() == part.lower() for b in EXAMPLE_BLACKLIST):
                        return True
            return False

        def process_item(title, readme_rel, tutorial_url, description, models, code_langs, os_list, preview_src):
            has_run_sh = False
            run_sh_path = None
            
            if readme_rel:
                example_dir = REPO_ROOT / Path(readme_rel).parent
                run_sh_file = example_dir / "run.sh"
                
                if run_sh_file.exists():
                    has_run_sh = True
                    run_sh_path = run_sh_file
                else:
                    example_readme = example_dir / "README.md"
                    if example_readme.exists():
                        try:
                            mode = "python"
                            src_dir = example_dir / "src"
                            has_python = False
                            if (src_dir / "python").exists():
                                has_python = True
                            elif src_dir.exists() and list(src_dir.glob("*.py")):
                                has_python = True
                            
                            if not has_python:
                                has_cpp = False
                                if (src_dir / "c++").exists():
                                    has_cpp = True
                                elif src_dir.exists() and list(src_dir.glob("*.cpp")):
                                    has_cpp = True
                                elif (example_dir / "CMakeLists.txt").exists():
                                    has_cpp = True
                                if has_cpp:
                                    mode = "cpp"
                            
                            extractor = ReadmeCommandExtractor(example_readme, mode=mode)
                            script_content = extractor.generate_run_script()
                            run_sh_file.write_text(script_content, encoding="utf-8")
                            run_sh_file.chmod(0o755)
                            has_run_sh = True
                            run_sh_path = run_sh_file
                            print(f"[SUCCESS] Generated run.sh ({mode}) for: {title}")
                        except Exception as e:
                            print(f"[ERROR] Failed to generate run.sh for {title}: {e}")

            return ExampleItem(
                title=title, description=description, models=models,
                code_langs=code_langs, os_list=os_list, preview_src=preview_src,
                readme_rel=readme_rel, tutorial_url=tutorial_url,
                has_run_sh=has_run_sh, run_sh_path=run_sh_path,
            )

        if is_image_grid:
            # For image grids, skip first row only if it appears to be headers
            # If table has only 1 row, don't skip it
            start_row = 1 if len(rows) > 1 and (not first_row_text.strip() or all(not self._clean_text(c.get_text()) for c in first_row_cells)) else 0
            for row in rows[start_row:]:
                tds = row.find_all("td")
                for td in tds:
                    if not td.get_text().strip(): continue
                    title, readme_rel, tutorial_url = self._parse_application_cell(td)
                    if not title or is_blacklisted(title, readme_rel): continue
                    item = process_item(
                        title, readme_rel, tutorial_url,
                        self._extract_description(td),
                        self._extract_models(td),
                        self._parse_code_langs(td),
                        self._parse_os(td),
                        self._parse_preview(td)
                    )
                    items.append(item)
        else:
            headers = [self._clean_text(th.get_text()) for th in first_row_cells]
            col_idx = {name.lower(): i for i, name in enumerate(headers)}
            
            def cell(row, key: str):
                i = col_idx.get(key.lower())
                if i is None or i >= len(row.find_all(["td", "th"])): 
                    return None
                return row.find_all(["td", "th"])[i]
            
            for row in rows[1:]:
                tds = row.find_all("td")
                if not tds: continue
                app_cell = cell(row, "Application") or cell(row, "Task")
                title, readme_rel, tutorial_url = self._parse_application_cell(app_cell)
                if is_blacklisted(title, readme_rel): continue
                item = process_item(
                    title, readme_rel, tutorial_url,
                    self._clean_text(cell(row, "Description").get_text()) if cell(row, "Description") else "",
                    self._split_models(cell(row, "Models") or cell(row, "Model")),
                    self._parse_code_langs(cell(row, "Code")),
                    self._parse_os(cell(row, "OS")),
                    self._parse_preview(cell(row, "Preview"))
                )
                items.append(item)
        return items

    def _parse_application_cell(self, cell) -> Tuple[str, Optional[str], Optional[str]]:
        if not cell: return ("", None, None)
        title = ""
        readme_rel = None
        tutorial_url = None
        cell_text = self._clean_text(cell.get_text())
        links = cell.find_all("a")
        for link in links:
            href = link.get("href", "")
            text = self._clean_text(link.get_text())
            if "tutorial" in text.lower():
                if href.startswith("http"): 
                    tutorial_url = href
                continue
            if href.endswith("README.md") and not href.startswith("http"):
                readme_rel = href.replace("\\", "/")
                if not title:
                    title = text if text else (link.find_parent("strong") or link.find_parent("b") or link).get_text().strip()
            elif ("tutorial" in href or "docs" in href or "developer.memryx.com" in href) and href.startswith("http"):
                tutorial_url = href
        if not title:
            strong = cell.find("strong") or cell.find("b")
            title = self._clean_text(strong.get_text()) if strong else cell_text.split('\n')[0].strip()
        title = re.sub(r'\s*\[.*?\]\s*', '', title)
        return (title.strip(), readme_rel, tutorial_url)

    def _split_models(self, cell) -> List[str]:
        if not cell: return []
        text = self._clean_text(cell.get_text())
        return [m.strip() for m in text.split(",") if m.strip()]
    
    def _extract_description(self, cell) -> str:
        subs = cell.find_all('sub')
        if subs and len(subs) > 0:
            desc = self._clean_text(subs[0].get_text())
            if not desc.startswith('Model') and not desc.startswith('!['):
                return desc
        return ""
    
    def _extract_models(self, cell) -> List[str]:
        subs = cell.find_all('sub')
        # Models are always in the second <sub> tag (index 1)
        if len(subs) >= 2:
            text = self._clean_text(subs[1].get_text())
            # Remove "Model:" or "Models:" prefix if present
            if text.startswith('Model:') or text.startswith('Models:'):
                models_text = text.split(':', 1)[1].strip()
            else:
                models_text = text
            
            # Parse model names (handle separators: &, and, comma)
            if models_text and not models_text.startswith('!['):
                return [m.strip() for m in re.split(r'\s+&\s+|\s+and\s+|,\s*', models_text) if m.strip()]
        return []

    def _parse_code_langs(self, cell) -> List[str]:
        if not cell: return []
        langs = []
        text = cell.get_text()
        if 'python-badge' in text: langs.append('Python')
        if 'cpp-badge' in text or 'c++-badge' in text: langs.append('C++')
        for img in cell.find_all("img"):
            src = img.get("src", "")
            for badge_kw, lang in self.BADGE_MAP.items():
                if badge_kw in src.lower() and lang not in langs:
                    langs.append(lang)
        if not langs:
            if re.search(r'\bpython\b', text, re.IGNORECASE): langs.append('Python')
            if re.search(r'\bc\+\+\b', text, re.IGNORECASE): langs.append('C++')
            if re.search(r'\bjava\b', text, re.IGNORECASE): langs.append('Java')
        return sorted(set(langs))

    def _parse_os(self, cell) -> List[str]:
        if not cell: return []
        os_set = set()
        for img in cell.find_all("img"):
            alt = img.get("alt", "")
            for os_name in self.OS_NAMES:
                if os_name.lower() in alt.lower():
                    os_set.add(os_name)
        text = self._clean_text(cell.get_text())
        for os_name in self.OS_NAMES:
            if os_name.lower() in text.lower():
                os_set.add(os_name)
        return sorted(os_set)

    def _parse_preview(self, cell) -> Optional[str]:
        if not cell: return None
        img = cell.find("img")
        if img and img.get("src", "").strip():
            return img.get("src", "").strip()
        link = cell.find("a")
        if link and any(link.get("href", "").lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"]):
            return link.get("href", "").strip()
        return None

    @staticmethod
    def _clean_text(s: str) -> str:
        # Remove emojis using Unicode ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+", 
            flags=re.UNICODE
        )
        s = emoji_pattern.sub('', s)
        return " ".join(s.split()).strip()

###############################################################################
# Process Management ##########################################################
###############################################################################

class ProcessManager:
    """Manages running example processes with graceful shutdown."""
    
    def __init__(self):
        self.running_proc = None
        self.current_example_name = None
        self.status_listeners = []  # SSE clients listening for status changes
        self.process_output = []  # Capture process output for error detection
        self.health_monitor_thread = None
        self.stop_monitor = threading.Event()
        self.last_status = None
        self.debug_mode = False  # Debug mode flag
        self.debug_listeners = []  # SSE clients listening for debug output
        self.max_debug_lines = 1000  # Maximum number of debug lines to keep
        
        # Initialize lock file path
        self.lock_file = Path("/tmp/memryx_example_running.lock")
        
        # Clean up stale lock file on startup
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
                print("[STARTUP] Removed stale lock file")
            except Exception as e:
                print(f"[STARTUP] Warning: Could not remove stale lock file: {e}")

    @staticmethod
    def _enqueue_latest(listener_queue: queue.Queue, payload: dict) -> bool:
        """Queue the latest event for an SSE client, dropping stale data if needed."""
        try:
            listener_queue.put_nowait(payload)
            return True
        except queue.Full:
            try:
                listener_queue.get_nowait()
            except queue.Empty:
                return False
            try:
                listener_queue.put_nowait(payload)
                return True
            except queue.Full:
                return False
    
    def is_running(self) -> bool:
        """Check if a process is running and clean up if it finished"""
        if self.running_proc is None:
            return False
        
        # Poll the process to update its status
        poll_result = self.running_proc.poll()
        
        # If process finished, clean up
        if poll_result is not None:
            exit_code = poll_result
            print(f"Process exited with code {exit_code}")
            
            # Analyze exit reason
            if exit_code != 0:
                self._analyze_process_failure(exit_code)
            
            self.cleanup()
            return False
        
        return True
    
    def _analyze_process_failure(self, exit_code: int):
        """Analyze process output to determine failure reason"""
        output_text = '\n'.join(self.process_output[-100:])  # Last 100 lines
        
        error_patterns = {
            'segfault': ['segmentation fault', 'sigsegv', 'core dumped'],
            'camera': ['no camera', 'camera not found', 'cannot open camera', 'video capture'],
            'qt_libs': ['qt.qpa', 'could not load the qt platform plugin', 'libqt', 'display'],
            'dfp': ['dfp not found', 'invalid dfp', 'cannot load dfp', 'model file'],
            'module_lock': ['module already in use', 'device busy', 'resource busy'],
            'display_thread': ['display thread', 'gui thread', 'window creation failed']
        }
        
        detected_errors = []
        for error_type, patterns in error_patterns.items():
            if any(pattern in output_text.lower() for pattern in patterns):
                detected_errors.append(error_type)
        
        if detected_errors:
            print(f"[FAILURE ANALYSIS] Detected errors: {', '.join(detected_errors)}")
            print(f"[FAILURE ANALYSIS] Exit code: {exit_code}")
            
            # Broadcast failure event to SSE listeners
            self._broadcast_status_change({
                'status': 'failed',
                'exit_code': exit_code,
                'errors': detected_errors,
                'message': f"Process failed: {', '.join(detected_errors)}"
            })
        else:
            print(f"[FAILURE ANALYSIS] Process exited with code {exit_code}, no specific error pattern detected")
    
    def start_example(self, script_path: Path, example_name: str) -> Optional[subprocess.Popen]:
        # Clean up any previous process first
        if self.is_running():
            print(f"Another example is still running: {self.current_example_name}")
            return None

        try:
            # Ensure script is executable
            stat = script_path.stat()
            if not (stat.st_mode & 0o111):
                script_path.chmod(stat.st_mode | 0o755)
                print(f"Made script executable: {script_path}")
            
            # Create lock file
            self.lock_file.touch()
            self.current_example_name = example_name
            
            print(f"Starting example: {example_name}")
            print(f"Working directory: {script_path.parent}")
            print(f"Script: {script_path}")
            
            # Start process with proper GUI support
            # Keep DISPLAY and terminal connection for OpenCV/GUI apps
            env = os.environ.copy()
            
            # Ensure DISPLAY is set for GUI applications
            if 'DISPLAY' not in env:
                env['DISPLAY'] = ':0'
            
            # Clear process output buffer
            self.process_output = []
            
            # Log startup details in debug mode
            if self.debug_mode:
                startup_msg = f"🚀 Starting {example_name}\n" + \
                             f"   Script: {script_path}\n" + \
                             f"   Working Dir: {script_path.parent}\n" + \
                             f"   Environment: {len(env)} variables\n" + \
                             f"   DISPLAY: {env.get('DISPLAY', 'not set')}"
                self._broadcast_debug_output(startup_msg)
            
            proc = subprocess.Popen(
                [str(script_path)],
                cwd=script_path.parent,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                start_new_session=False  # Keep terminal session for GUI
            )
            
            self.running_proc = proc
            print(f"[SUCCESS] Process started with PID: {proc.pid}")
            
            # Start health monitoring and output capture
            self.stop_monitor.clear()
            self.health_monitor_thread = threading.Thread(
                target=self._monitor_process_health,
                args=(proc,),
                daemon=True
            )
            self.health_monitor_thread.start()
            
            # Broadcast status change
            self._broadcast_status_change({
                'status': 'started',
                'example_name': example_name,
                'pid': proc.pid
            })
            
            return proc
            
        except FileNotFoundError:
            print(f"[ERROR] Script not found: {script_path}")
            self.cleanup()
            return None
        except PermissionError:
            print(f"[ERROR] Permission denied: {script_path}")
            self.cleanup()
            return None
        except Exception as e:
            print(f"[ERROR] Failed to start example: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            return None
    
    def stop_gracefully(self, timeout: int = 5) -> bool:
        if not self.running_proc:
            print("No process to stop")
            return False
        
        try:
            print(f"Stopping example: {self.current_example_name} (PID: {self.running_proc.pid})")
            
            try:
                proc = psutil.Process(self.running_proc.pid)
            except psutil.NoSuchProcess:
                print("Process already finished")
                self.cleanup()
                return True
            
            # Get all child processes before terminating
            try:
                children = proc.children(recursive=True)
                print(f"Found {len(children)} child process(es)")
            except psutil.NoSuchProcess:
                children = []
            
            # Terminate children first
            for child in children:
                try:
                    print(f"Terminating child PID {child.pid}")
                    child.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"Could not terminate child {child.pid}: {e}")
            
            # Terminate main process
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                print("Main process already gone")
                self.cleanup()
                return True
            
            # Wait for graceful shutdown
            try:
                proc.wait(timeout=timeout)
                print(f"[SUCCESS] Process terminated gracefully")
                self.cleanup()
                return True
            except psutil.TimeoutExpired:
                print(f"[WARNING] Timeout after {timeout}s, forcing kill...")
                
                # Force kill children
                for child in children:
                    try:
                        if child.is_running():
                            print(f"Killing child PID {child.pid}")
                            child.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Force kill main process
                try:
                    if proc.is_running():
                        proc.kill()
                        proc.wait(timeout=2)
                except psutil.NoSuchProcess:
                    pass
                
                print("[SUCCESS] Process killed")
                self.cleanup()
                return True
                
        except Exception as e:
            print(f"[ERROR] Error stopping process: {e}")
            import traceback
            traceback.print_exc()
            self.cleanup()
            return False
    
    def _monitor_process_health(self, proc: subprocess.Popen):
        """Monitor process health after spawn to detect immediate failures"""
        startup_window = 5  # Monitor first 5 seconds for immediate failures
        check_interval = 0.5  # Check every 500ms
        
        start_time = time.time()
        line_count = 0
        
        try:
            while not self.stop_monitor.is_set():
                # Check if process is still alive
                if proc.poll() is not None:
                    elapsed = time.time() - start_time
                    if elapsed < startup_window:
                        msg = f"[HEALTH] Process died within {elapsed:.1f}s of startup - likely failed"
                        print(msg)
                        if self.debug_mode:
                            self._broadcast_debug_output(msg)
                        self._broadcast_status_change({
                            'status': 'failed',
                            'message': f'Process crashed {elapsed:.1f}s after startup',
                            'exit_code': proc.returncode
                        })
                    break
                
                # Capture output lines
                try:
                    line = proc.stdout.readline()
                    if line:
                        line = line.rstrip()
                        timestamp = time.strftime('%H:%M:%S')
                        formatted_line = f"[{timestamp}] {line}"
                        
                        self.process_output.append(formatted_line)
                        line_count += 1
                        
                        # Broadcast to debug listeners if debug mode is enabled
                        if self.debug_mode:
                            self._broadcast_debug_output(formatted_line)
                        
                        # Keep only last lines to prevent memory issues
                        max_lines = self.max_debug_lines if self.debug_mode else 500
                        if len(self.process_output) > max_lines:
                            self.process_output = self.process_output[-max_lines:]
                        
                        # Check for immediate error patterns
                        line_lower = line.lower()
                        if any(err in line_lower for err in ['error', 'failed', 'exception', 'segfault', 'abort']):
                            error_msg = f"[HEALTH] Detected error in output: {line}"
                            print(error_msg)
                            if self.debug_mode:
                                self._broadcast_debug_output(f"🚨 {error_msg}")
                except Exception:
                    pass
                
                # After startup window, check less frequently
                elapsed = time.time() - start_time
                if elapsed > startup_window:
                    if line_count == 0 and elapsed > 3:
                        warning_msg = f"[HEALTH] Warning: No output after {elapsed:.1f}s, process may be hung"
                        print(warning_msg)
                        if self.debug_mode:
                            self._broadcast_debug_output(f"⚠️ {warning_msg}")
                    time.sleep(2)  # Check every 2 seconds after startup
                else:
                    time.sleep(check_interval)
        
        except Exception as e:
            error_msg = f"[HEALTH] Monitor thread error: {e}"
            print(error_msg)
            if self.debug_mode:
                self._broadcast_debug_output(f"💥 {error_msg}")
    
    def _broadcast_status_change(self, status_data: dict):
        """Broadcast status change to all SSE listeners"""
        self.last_status = dict(status_data)
        active_listeners = []

        for listener_queue in self.status_listeners:
            if self._enqueue_latest(listener_queue, dict(status_data)):
                active_listeners.append(listener_queue)

        self.status_listeners = active_listeners
    
    def _broadcast_debug_output(self, output_line: str):
        """Broadcast debug output to all debug SSE listeners"""
        event_payload = {
            'type': 'output',
            'line': output_line,
            'timestamp': time.time()
        }
        active_listeners = []

        for listener_queue in self.debug_listeners:
            if self._enqueue_latest(listener_queue, dict(event_payload)):
                active_listeners.append(listener_queue)

        self.debug_listeners = active_listeners
    
    def set_debug_mode(self, enabled: bool):
        """Enable or disable debug mode"""
        self.debug_mode = enabled
        if enabled:
            print("[DEBUG] Debug mode enabled - capturing detailed output")
            # Send current output buffer to new debug listeners
            if self.process_output:
                for line in self.process_output[-50:]:  # Send last 50 lines
                    self._broadcast_debug_output(line)
        else:
            print("[DEBUG] Debug mode disabled")
            # Clear debug listeners
            self.debug_listeners.clear()
    
    def get_debug_output(self) -> list:
        """Get current process output for debugging"""
        return self.process_output.copy() if self.process_output else []
    
    def cleanup(self):
        """Clean up process state and lock file"""
        # Stop health monitor
        if self.health_monitor_thread and self.health_monitor_thread.is_alive():
            self.stop_monitor.set()
            self.health_monitor_thread.join(timeout=2)
        
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
                print("Removed lock file")
            except Exception as e:
                print(f"Warning: Could not remove lock file: {e}")
        
        if self.running_proc:
            # Reap zombie process if needed
            try:
                self.running_proc.poll()
            except Exception:
                pass
        
        # Broadcast stopped status
        self._broadcast_status_change({
            'status': 'stopped',
            'example_name': self.current_example_name
        })
        
        self.running_proc = None
        self.current_example_name = None
        self.process_output = []
        print("Cleanup complete")

###############################################################################
# Utility Functions ###########################################################
###############################################################################

def no_chip_available():
    """Check if MemryX chip is available by running mx_bench --hello"""
    try:
        result = subprocess.run(
            ["mx_bench", "--hello"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        if result.returncode != 0:
            print("[CHIP CHECK] mx_bench returned non-zero exit code")
            return True
    except FileNotFoundError:
        print("[CHIP CHECK] mx_bench command not found")
        return True
    except subprocess.TimeoutExpired:
        print("[CHIP CHECK] mx_bench timed out")
        return True
    except Exception as e:
        print(f"[CHIP CHECK] Error: {e}")
        return True
    
    print("[CHIP CHECK] MemryX chip is available")
    return False


def resolve_run_script_path(path_value: str) -> Optional[Path]:
    """Resolve and validate a user-provided script path."""
    try:
        candidate = Path(path_value).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None

    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError:
        return None

    if resolved.name != "run.sh" or not resolved.is_file():
        return None

    return resolved


def sse_data(payload: dict) -> str:
    """Serialize payload as a Server-Sent Events data frame."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

###############################################################################
# Flask Application ###########################################################
###############################################################################

app = Flask(__name__)
app.config['SECRET_KEY'] = 'memryx-examples-launcher-2026'

# Grid layout configuration for responsive design
GRID_CONFIG = {
    'small': {'breakpoint': 1366, 'columns': 2, 'card_width': '320px', 'image_height': '140px'},
    'medium': {'breakpoint': 1920, 'columns': 3, 'card_width': '360px', 'image_height': '165px'},
    'large': {'breakpoint': 2560, 'columns': 4, 'card_width': '400px', 'image_height': '180px'},
    'xlarge': {'breakpoint': 3840, 'columns': 5, 'card_width': '440px', 'image_height': '200px'},
}

# Initialize global objects
process_manager = ProcessManager()
categories = []
chip_available = True  # Will be checked on startup

def load_examples():
    """Load and parse examples from README"""
    global categories
    if README_PATH.exists():
        parser = ReadmeParser.from_file(README_PATH)
        categories = parser.parse()
    else:
        categories = []
    return categories

@app.route('/')
def index():
    """Main page"""
    cats = load_examples()
    return render_template('index.html', categories=cats, github_base=GITHUB_BASE, grid_config=GRID_CONFIG)

@app.route('/api/examples')
def api_examples():
    """API endpoint to get all examples"""
    cats = load_examples()
    data = []
    for cat in cats:
        cat_data = {
            'name': cat.name,
            'items': []
        }
        for item in cat.items:
            cat_data['items'].append({
                'title': item.title,
                'description': item.description,
                'models': item.models,
                'code_langs': item.code_langs,
                'os_list': item.os_list,
                'preview_url': item.preview_url,
                'readme_url': item.readme_url,
                'tutorial_url': item.tutorial_url,
                'has_run_sh': item.has_run_sh,
                'run_sh_path': str(item.run_sh_path) if item.run_sh_path else None
            })
        data.append(cat_data)
    return jsonify(data)

@app.route('/api/search')
def api_search():
    """Search examples"""
    query = request.args.get('q', '').lower()
    category = request.args.get('category', None)
    
    cats = load_examples()
    results = []
    
    for cat in cats:
        if category and cat.name != category:
            continue
            
        cat_results = {
            'name': cat.name,
            'items': []
        }
        
        for item in cat.items:
            search_text = ' '.join([
                item.title, item.description,
                ' '.join(item.models),
                ' '.join(item.code_langs),
                ' '.join(item.os_list)
            ]).lower()
            
            if not query or query in search_text:
                cat_results['items'].append({
                    'title': item.title,
                    'description': item.description,
                    'models': item.models,
                    'code_langs': item.code_langs,
                    'os_list': item.os_list,
                    'preview_url': item.preview_url,
                    'readme_url': item.readme_url,
                    'tutorial_url': item.tutorial_url,
                    'has_run_sh': item.has_run_sh,
                    'run_sh_path': str(item.run_sh_path) if item.run_sh_path else None
                })
        
        if cat_results['items']:
            results.append(cat_results)
    
    return jsonify(results)

@app.route('/api/run', methods=['POST'])
def api_run():
    """Run an example"""
    data = request.get_json(silent=True) or {}
    script_path = data.get('script_path')
    example_name = data.get('name')

    script_path = script_path.strip() if isinstance(script_path, str) else ""
    example_name = example_name.strip() if isinstance(example_name, str) else ""
    
    if not script_path or not example_name:
        return jsonify({'success': False, 'error': 'Missing parameters'}), 400
    
    if process_manager.is_running():
        return jsonify({'success': False, 'error': 'Another example is already running'}), 409
    
    resolved_script = resolve_run_script_path(script_path)
    if not resolved_script:
        return jsonify({'success': False, 'error': 'Invalid script path'}), 400
    
    proc = process_manager.start_example(resolved_script, example_name)
    if proc:
        return jsonify({'success': True, 'message': f'Started {example_name}'})
    else:
        return jsonify({'success': False, 'error': 'Failed to start example'}), 500

@app.route('/api/stop', methods=['POST'])
def api_stop():
    """Stop running example"""
    if not process_manager.is_running():
        return jsonify({'success': False, 'error': 'No example is running'}), 400
    
    success = process_manager.stop_gracefully()
    return jsonify({'success': success})

@app.route('/api/kill', methods=['POST'])
def api_kill():
    """Force kill running example"""
    if not process_manager.is_running():
        return jsonify({'success': False, 'error': 'No example is running'}), 400
    
    try:
        proc = psutil.Process(process_manager.running_proc.pid)
        for child in proc.children(recursive=True):
            child.kill()
        proc.kill()
        process_manager.cleanup()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status')
def api_status():
    """Get process status (for compatibility, prefer SSE)"""
    is_running = process_manager.is_running()
    return jsonify({
        'is_running': is_running,
        'example_name': process_manager.current_example_name if is_running else None,
        'chip_available': chip_available
    })

@app.route('/api/status/stream')
def api_status_stream():
    """Server-Sent Events endpoint for real-time status updates"""
    def event_stream():
        # Create a queue for this client
        client_queue = queue.Queue(maxsize=10)
        process_manager.status_listeners.append(client_queue)
        
        try:
            # Send initial status
            is_running = process_manager.is_running()
            initial_status = {
                'status': 'running' if is_running else 'idle',
                'example_name': process_manager.current_example_name,
                'chip_available': chip_available,
                'debug_mode': process_manager.debug_mode
            }
            yield sse_data(initial_status)
            
            # Stream updates
            while True:
                try:
                    # Wait for status change (with timeout to send keepalive)
                    status_data = client_queue.get(timeout=30)
                    status_event = {
                        **status_data,
                        'chip_available': chip_available,
                        'debug_mode': process_manager.debug_mode
                    }
                    yield sse_data(status_event)
                except queue.Empty:
                    # Send keepalive comment
                    yield ": keepalive\n\n"
        finally:
            # Remove this client's queue when disconnected
            if client_queue in process_manager.status_listeners:
                process_manager.status_listeners.remove(client_queue)
    
    return Response(stream_with_context(event_stream()), mimetype='text/event-stream')

@app.route('/api/debug/toggle', methods=['POST'])
def api_debug_toggle():
    """Toggle debug mode"""
    data = request.get_json(silent=True) or {}
    enabled = data.get('enabled')
    if enabled is None:
        enabled = not process_manager.debug_mode
    elif isinstance(enabled, str):
        enabled = enabled.strip().lower() in {"1", "true", "yes", "on"}
    else:
        enabled = bool(enabled)
    process_manager.set_debug_mode(enabled)
    return jsonify({
        'success': True,
        'debug_mode': process_manager.debug_mode,
        'message': f'Debug mode {"enabled" if process_manager.debug_mode else "disabled"}'
    })

@app.route('/api/debug/output')
def api_debug_output():
    """Get current debug output"""
    output = process_manager.get_debug_output()
    return jsonify({
        'output': output,
        'debug_mode': process_manager.debug_mode,
        'line_count': len(output)
    })

@app.route('/api/debug/stream')
def api_debug_stream():
    """Server-Sent Events endpoint for real-time debug output"""
    def debug_event_stream():
        # Create a queue for this debug client
        client_queue = queue.Queue(maxsize=50)  # Larger buffer for debug output
        process_manager.debug_listeners.append(client_queue)
        
        try:
            # Send initial debug info
            initial_data = {
                'type': 'init',
                'debug_mode': process_manager.debug_mode,
                'message': 'Debug stream connected',
                'timestamp': time.time()
            }
            yield sse_data(initial_data)
            
            # If debug mode is enabled and there's existing output, send recent lines
            if process_manager.debug_mode and process_manager.process_output:
                for line in process_manager.process_output[-20:]:  # Last 20 lines
                    historical_data = {
                        'type': 'historical',
                        'line': line,
                        'timestamp': time.time()
                    }
                    yield sse_data(historical_data)
            
            # Stream real-time debug updates
            while True:
                try:
                    # Wait for debug output (with timeout for keepalive)
                    debug_data = client_queue.get(timeout=30)
                    yield sse_data(debug_data)
                except queue.Empty:
                    # Send keepalive
                    keepalive_data = {
                        'type': 'keepalive',
                        'timestamp': time.time()
                    }
                    yield sse_data(keepalive_data)
        finally:
            # Remove this client's queue when disconnected
            if client_queue in process_manager.debug_listeners:
                process_manager.debug_listeners.remove(client_queue)
    
    return Response(stream_with_context(debug_event_stream()), mimetype='text/event-stream')

@app.route('/api/chip/check')
def api_chip_check():
    """Check if MemryX chip is available"""
    global chip_available
    chip_available = not no_chip_available()
    return jsonify({
        'chip_available': chip_available,
        'message': 'Chip is available' if chip_available else 'Chip not available or mx_bench failed'
    })

@app.route('/repo/<path:filename>')
def serve_repo(filename):
    """Serve files from the repository"""
    return send_from_directory(REPO_ROOT, filename)


def main() -> None:
    global chip_available
    launcher_url = "http://localhost:8080"
    launcher_port = 8080

    # Check chip availability on startup
    print("\n" + "="*60)
    print("MemryX Examples Launcher - Starting Up")
    print("="*60)

    print(f"\n[STARTUP] Repository root: {REPO_ROOT}")
    if not README_PATH.exists():
        print(f"[STARTUP] WARNING: README not found at {README_PATH}")
        print("[STARTUP] Set MEMRYX_EXAMPLES_ROOT or run from inside a cloned repository.")
    
    print("\n[STARTUP] Checking MemryX chip availability...")
    chip_available = not no_chip_available()
    
    if chip_available:
        print("[STARTUP] ✓ MemryX chip is available and ready")
    else:
        print("[STARTUP] ✗ WARNING: MemryX chip check failed")
    
    print("\n[STARTUP] Cleaning up stale processes and locks...")
    # Process manager init already does cleanup
    
    print("\n[STARTUP] Loading examples from README...")
    load_examples()
    print(f"[STARTUP] Loaded {sum(len(cat.items) for cat in categories)} examples")
    
    print("\n" + "="*60)
    print("Server starting on http://0.0.0.0:8080")
    print("Use Server-Sent Events at /api/status/stream for real-time updates")
    print("="*60 + "\n")

    if _should_open_browser():
        threading.Thread(
            target=_wait_for_server_and_open_browser,
            args=(launcher_url, launcher_port),
            daemon=True,
        ).start()
    else:
        print("[STARTUP] Browser auto-open disabled (set MEMRYX_OPEN_BROWSER=1 to re-enable)")
    
    # Disable Werkzeug request logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(debug=False, host='0.0.0.0', port=launcher_port, threaded=True)


if __name__ == '__main__':
    main()
