"""
Ultra-fast test training - ~15 minute target on GTX 1060 3GB.
"""
import subprocess
import sys

if __name__ == "__main__":
    cmd = [
        sys.executable,
        "train.py",
        "--data_path", "ai-challenge-data_ai_challenge_data_anonymized_19744.jsonl",
        "--limit", "100",           # Minimal docs
        "--epochs", "2",            # 2 epochs
        "--batch_size", "32",       # Max batch for 3GB VRAM
        "--learning_rate", "2e-5",
        "--chunk_size", "128",      # Very small chunks
        "--overlap", "32",          # Minimal overlap
    ]

    print("=" * 60)
    print("ULTRA-FAST TEST TRAINING (~15 min)")
    print("=" * 60)
    print("Settings:")
    print("  - Documents: 100")
    print("  - Epochs: 2")
    print("  - Batch size: 32")
    print("  - Chunk size: 128")
    print("  - Overlap: 32")
    print("=" * 60)
    print()

    subprocess.run(cmd)
