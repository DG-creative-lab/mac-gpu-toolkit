import os
import time
import sys
import shutil
import tempfile
from pathlib import Path

# Toggle for speed on long PDFs:
DISABLE_IMAGES = True     # set to False if you want images extracted
PAGINATE_OUTPUT = True    # insert page break markers in Markdown

def check_marker_installation() -> bool:
    """Check if Marker CLI is available and whether MPS is usable."""
    print("=== Marker Installation Check ===")
    cli = shutil.which("marker_single")
    if cli:
        print(f"✅ Found marker_single at: {cli}")
    else:
        # We'll try python -m convert_single later, but warn the user.
        print("ℹ️  'marker_single' not found on PATH; will try 'python -m convert_single' fallback.")

    # Check Torch/MPS (optional)
    try:
        import torch  # noqa: F401
        has_mps = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        if has_mps:
            print("✅ MPS backend available for GPU acceleration")
        else:
            print("⚠️  MPS not available, will use CPU (slower)")
        return True
    except Exception:
        print("⚠️  Torch not installed or cannot be imported. CLI will still run on CPU.")
        return True

def create_sample_pdf() -> Path | None:
    """Create a sample PDF for testing (if ReportLab available)."""
    print("🔄 Creating sample PDF for testing...")
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
    except Exception:
        print("⚠️  ReportLab not available, skipping PDF creation")
        print("💡 Install with: uv pip install reportlab")
        return None

    temp_dir = Path(tempfile.gettempdir())
    pdf_path = temp_dir / "sample_document.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Simple two-page document
    c.drawString(100, 750, "Sample Document for GPU Processing Test")
    c.drawString(100, 720, "=" * 50)
    y = 680
    for line in [
        "This is a sample document created for testing GPU-accelerated",
        "document processing with Marker. The document contains multiple",
        "paragraphs with different types of content to simulate real-world",
        "document processing scenarios.",
        "",
        "Key Benefits of GPU Acceleration:",
        "• Faster processing of complex layouts",
        "• Improved text extraction accuracy",
        "• Better handling of images and tables",
        "• Significant speed improvements for batch processing",
    ]:
        c.drawString(100, y, line)
        y -= 20
    c.showPage()
    c.drawString(100, 750, "Page 2: Additional Content")
    c.drawString(100, 720, "=" * 30)
    c.drawString(100, 700, "Tables, lists, and mixed content often benefit from better layout parsing.")
    c.save()

    print(f"✅ Sample PDF created: {pdf_path}")
    return pdf_path

def _marker_cmd(pdf: Path, outdir: Path, *, paginate: bool, disable_images: bool) -> list[str]:
    """Build a marker CLI command (prefer marker_single, else python -m convert_single)."""
    if shutil.which("marker_single"):
        cmd = ["marker_single", str(pdf)]
    else:
        cmd = [sys.executable, "-m", "convert_single", str(pdf)]
    cmd += ["--output_dir", str(outdir), "--output_format", "markdown"]
    if paginate:
        cmd += ["--paginate_output"]
    if disable_images:
        cmd += ["--disable_image_extraction"]
    return cmd

def _run_marker(pdf_path: Path, device: str) -> tuple[str | None, float]:
    """
    Run Marker CLI on the PDF and return (markdown_text, elapsed_seconds).
    Reads the generated .md from the output directory.
    """
    out_dir = Path("marker_output")
    out_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    if device in {"cpu", "mps"}:
        env["TORCH_DEVICE"] = device

    cmd = _marker_cmd(pdf_path, out_dir, paginate=PAGINATE_OUTPUT, disable_images=DISABLE_IMAGES)
    print(">> Running:", " ".join(cmd), f"(TORCH_DEVICE={env.get('TORCH_DEVICE','unset')})")

    t0 = time.time()
    try:
        shutil.rmtree(out_dir / f"{pdf_path.stem}_images", ignore_errors=True)  # clean old assets
        for f in out_dir.glob(f"{pdf_path.stem}*.*"):
            f.unlink(missing_ok=True)
        import subprocess
        subprocess.run(cmd, check=True, env=env)
    except Exception as e:
        print(f"❌ Marker CLI failed: {e}")
        return None, 0.0

    elapsed = time.time() - t0

    # Pick the main markdown file
    md_candidates = sorted(out_dir.glob(f"{pdf_path.stem}*.md"))
    if not md_candidates:
        print("❌ No Markdown produced.")
        return None, elapsed

    md_path = md_candidates[0]
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"❌ Failed to read {md_path}: {e}")
        return None, elapsed

    return text, elapsed

def process_document_cpu(pdf_path: Path) -> tuple[str | None, float]:
    print("\n=== CPU Processing ===")
    text, secs = _run_marker(pdf_path, device="cpu")
    if text is not None:
        print(f"✅ CPU processing completed in {secs:.2f}s")
        print(f"📊 Generated {len(text)} characters of markdown")
    return text, secs

def process_document_gpu(pdf_path: Path) -> tuple[str | None, float]:
    print("\n=== GPU Processing (MPS) ===")
    # If torch/mps isn’t available, this will still run on CPU via CLI env
    try:
        import torch  # noqa: F401
        mps_ok = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    except Exception:
        mps_ok = False

    device = "mps" if mps_ok else "cpu"
    if not mps_ok:
        print("⚠️  MPS not available; falling back to CPU.")

    text, secs = _run_marker(pdf_path, device=device)
    if text is not None:
        print(f"✅ {device.upper()} processing completed in {secs:.2f}s")
        print(f"📊 Generated {len(text)} characters of markdown")
    return text, secs

def save_results_example(markdown_text: str | None):
    if not markdown_text:
        return
    out_dir = Path("marker_output")
    out_dir.mkdir(exist_ok=True)
    md_file = out_dir / "converted_document.md"
    md_file.write_text(markdown_text, encoding="utf-8")
    print(f"\n✅ Markdown saved to: {md_file}  ({md_file.stat().st_size} bytes)")

    lines = markdown_text.splitlines()
    print("\n📄 Document preview (first 10 lines):")
    print("=" * 40)
    for i, line in enumerate(lines[:10]):
        print(f"{i+1:2d}: {line}")
    if len(lines) > 10:
        print(f"... ({len(lines) - 10} more lines)")

def main():
    print("Marker Document Processing (CLI-based)")
    print("=" * 50)

    if not check_marker_installation():
        print("❌ Cannot proceed without Marker.")
        return

    # If a PDF path is provided as the first CLI arg, use it; else create a sample.
    pdf_path: Path | None = None
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".pdf"):
        pdf_path = Path(sys.argv[1]).expanduser().resolve()
        if not pdf_path.exists():
            print(f"❌ PDF not found: {pdf_path}")
            return
    else:
        pdf_path = create_sample_pdf()
        if not pdf_path:
            print("❌ No PDF to process.")
            return

    try:
        md_cpu, t_cpu = process_document_cpu(pdf_path)
        md_gpu, t_gpu = process_document_gpu(pdf_path)

        if t_cpu > 0 and t_gpu > 0:
            speedup = (t_cpu / t_gpu) if t_gpu else 0.0
            print(f"\n🚀 GPU Speedup (vs CPU): {speedup:.2f}x")

        save_results_example(md_gpu or md_cpu)

    finally:
        # cleanup only if we created the sample file
        if pdf_path and pdf_path.name == "sample_document.pdf":
            try:
                pdf_path.unlink()
            except Exception:
                pass

    print("\n✅ Document processing completed!")

if __name__ == "__main__":
    main()
