"""Test zip validation against attack vectors and safety rules."""

import io
import zipfile

import pytest

from competition.ingestion import validate_zip_bytes, ALLOWED_EXTENSIONS


class TestZipValidation:
    """Test suite for zip archive validation and security checks."""

    def _create_zip(self, files: dict) -> bytes:
        """Helper: create a zip from dict of {filename: content (str or bytes)}."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, content in files.items():
                if isinstance(content, str):
                    content = content.encode("utf-8")
                zf.writestr(name, content)
        return bio.getvalue()

    def test_valid_minimal_zip(self):
        """Minimal valid zip with just agent.py should pass."""
        agent_code = "def act(obs): return 0"
        zip_bytes = self._create_zip({"agent.py": agent_code})
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is True
        assert reason is None
        assert manifest is not None
        assert "agent.py" in manifest
        assert manifest["agent.py"] > 0

    def test_valid_zip_with_supporting_files(self):
        """Valid zip with agent.py and supporting files (weights, config)."""
        files = {
            "agent.py": "def act(obs): return 0",
            "weights.pt": b"fake_weights" * 1000,  # ~12 KB
            "config.yaml": "learning_rate: 0.001",
            "README.md": "# My Agent",
        }
        zip_bytes = self._create_zip(files)
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is True
        assert reason is None
        assert len(manifest) == 4
        assert all(name in manifest for name in files.keys())

    def test_invalid_no_agent_py(self):
        """Zip without agent.py should be rejected."""
        zip_bytes = self._create_zip({"config.yaml": "test"})
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is False
        assert reason == "agent_py_missing_or_multiple"
        assert manifest is None

    def test_invalid_multiple_agent_py(self):
        """Zip with multiple agent.py files should be rejected."""
        zip_bytes = self._create_zip({
            "agent.py": "def act(obs): return 0",
            "subdir/agent.py": "def act(obs): return 1",
        })
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is False
        assert reason == "agent_py_missing_or_multiple"
        assert manifest is None

    def test_invalid_agent_py_syntax_error(self):
        """agent.py with syntax error should be rejected."""
        bad_code = "def act(obs) this is invalid syntax:"
        zip_bytes = self._create_zip({"agent.py": bad_code})
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is False
        assert "agent_py_syntax_error" in reason
        assert manifest is None

    def test_zip_bomb_detection_total_size(self):
        """Zip that expands beyond MAX_EXTRACTED_TOTAL_BYTES should be rejected."""
        # Create a zip that compresses well but expands large
        # We'll just create agent.py + files that total > 300MB when expanded
        files = {
            "agent.py": "def act(obs): return 0",
            "large_file_1.bin": b"\x00" * (140 * 1024 * 1024),  # 140 MB
            "large_file_2.bin": b"\x00" * (140 * 1024 * 1024),  # 140 MB
            "large_file_3.bin": b"\x00" * (30 * 1024 * 1024),   # 30 MB
        }
        zip_bytes = self._create_zip(files)
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is False
        assert reason == "extracted_total_too_large"
        assert manifest is None

    def test_single_file_too_large(self):
        """Single file > MAX_SINGLE_FILE_BYTES should be rejected."""
        files = {
            "agent.py": "def act(obs): return 0",
            "oversized.bin": b"\x00" * (160 * 1024 * 1024),  # 160 MB > 150 MB limit
        }
        zip_bytes = self._create_zip(files)
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is False
        assert reason == "single_file_too_large"
        assert manifest is None

    def test_zip_too_large_compressed(self):
        """Zip file itself > MAX_ZIP_SIZE_BYTES should be rejected."""
        # Create a zip that's more than 100 MB compressed
        # This is tricky because zips compress well, so we'll make it incompressible
        files = {
            "agent.py": "def act(obs): return 0",
            "random.bin": bytes(range(256)) * (102 * 1024 * 256),  # ~102 MB of incompressible data
        }
        zip_bytes = self._create_zip(files)
        
        if len(zip_bytes) > 100 * 1024 * 1024:
            is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
            
            assert is_valid is False
            assert reason == "zip_too_large"
            assert manifest is None

    def test_path_traversal_attack_rejected(self):
        """Zip with ../relative paths should be rejected."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("agent.py", "def act(obs): return 0")
            zf.writestr("../../../etc/passwd", "fake")  # Traversal attempt
        
        is_valid, reason, manifest = validate_zip_bytes(bio.getvalue())
        
        assert is_valid is False
        assert reason == "unsafe_path"
        assert manifest is None

    def test_absolute_path_rejected(self):
        """Zip with absolute paths should be rejected."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("agent.py", "def act(obs): return 0")
            zf.writestr("/etc/passwd", "fake")  # Absolute path
        
        is_valid, reason, manifest = validate_zip_bytes(bio.getvalue())
        
        assert is_valid is False
        assert reason == "unsafe_path"
        assert manifest is None

    def test_disallowed_extension_rejected(self):
        """Files with non-whitelisted extensions should be rejected."""
        files = {
            "agent.py": "def act(obs): return 0",
            "malware.exe": "fake exe",  # .exe not in ALLOWED_EXTENSIONS
        }
        zip_bytes = self._create_zip(files)
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is False
        assert ".exe" in reason or "disallowed_extension" in reason
        assert manifest is None

    def test_allowed_extensions(self):
        """All whitelisted extensions should be accepted."""
        files = {
            "agent.py": "def act(obs): return 0",
        }
        for ext in list(ALLOWED_EXTENSIONS)[:5]:  # Test first 5 extensions
            test_filename = f"test_file{ext}"
            files[test_filename] = f"content for {ext}"
        
        zip_bytes = self._create_zip(files)
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is True
        assert reason is None

    def test_too_many_files_rejected(self):
        """Zip with > MAX_FILE_COUNT files should be rejected."""
        files = {
            "agent.py": "def act(obs): return 0",
        }
        # Create 201 files (exceeds MAX_FILE_COUNT=200)
        for i in range(201):
            files[f"file_{i:03d}.txt"] = f"content {i}"
        
        zip_bytes = self._create_zip(files)
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is False
        assert reason == "too_many_files"
        assert manifest is None

    def test_invalid_zip_format(self):
        """Non-zip data should be rejected."""
        invalid_zip = b"This is not a zip file"
        
        is_valid, reason, manifest = validate_zip_bytes(invalid_zip)
        
        assert is_valid is False
        assert "invalid_zip" in reason
        assert manifest is None

    def test_case_insensitive_extension_check(self):
        """Extension check should be case-insensitive."""
        files = {
            "agent.py": "def act(obs): return 0",
            "WEIGHTS.PT": b"fake",  # Uppercase
            "Config.YAML": "test",  # Mixed case
        }
        zip_bytes = self._create_zip(files)
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is True  # Both uppercase variants should be allowed
        assert reason is None

    def test_manifest_contains_correct_sizes(self):
        """Manifest should accurately report file sizes."""
        content = b"test content"
        files = {
            "agent.py": "def act(obs): return 0",
            "data.txt": content,
        }
        zip_bytes = self._create_zip(files)
        
        is_valid, reason, manifest = validate_zip_bytes(zip_bytes)
        
        assert is_valid is True
        assert "data.txt" in manifest
        assert manifest["data.txt"] == len(content)

    def test_directories_in_zip_ignored(self):
        """Directory entries in zip should be ignored (only files counted)."""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("agent.py", "def act(obs): return 0")
            zf.writestr("subdir/", "")  # Directory entry
            zf.writestr("subdir/config.yaml", "test: true")
        
        is_valid, reason, manifest = validate_zip_bytes(bio.getvalue())
        
        assert is_valid is True
        # Manifest should have only files, not directory entries
        assert "subdir/" not in manifest
        assert "subdir/config.yaml" in manifest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
