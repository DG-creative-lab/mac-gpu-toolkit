"""
Document processing example using Marker with GPU acceleration
This demonstrates one of the key use cases for Mac GPU acceleration
"""
import time
import os
from pathlib import Path
import tempfile
import sys

def check_marker_installation():
    """Check if Marker is installed and configured"""
    print("=== Marker Installation Check ===")
    
    try:
        import marker
        print("✅ Marker package found")
        
        # Check if we have GPU acceleration
        import torch
        if torch.backends.mps.is_available():
            print("✅ MPS backend available for GPU acceleration")
            return True
        else:
            print("⚠️  MPS not available, will use CPU (slower)")
            return False
            
    except ImportError:
        print("❌ Marker not installed")
        print("💡 Install with: pip install marker-pdf")
        return False

def create_sample_pdf():
    """Create a sample PDF for testing (if needed)"""
    print("🔄 Creating sample PDF for testing...")
    
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        # Create temporary PDF
        temp_dir = Path(tempfile.gettempdir())
        pdf_path = temp_dir / "sample_document.pdf"
        
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        
        # Add some content
        c.drawString(100, 750, "Sample Document for GPU Processing Test")
        c.drawString(100, 720, "=" * 50)
        
        # Add multiple paragraphs
        content = [
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
            "",
            "Performance Comparison:",
            "CPU processing typically takes 30-60 seconds per page for",
            "complex documents, while GPU processing can reduce this to",
            "5-10 seconds per page, representing a 5-6x speedup.",
        ]
        
        y_position = 680
        for line in content:
            c.drawString(100, y_position, line)
            y_position -= 20
            
            if y_position < 100:  # Start new page
                c.showPage()
                y_position = 750
        
        # Add a second page with more content
        c.showPage()
        c.drawString(100, 750, "Page 2: Additional Content")
        c.drawString(100, 720, "=" * 30)
        
        more_content = [
            "This second page contains additional text to make the",
            "document processing more realistic and provide better",
            "benchmarking results.",
            "",
            "Tables and structured data:",
            "Item 1    Value A    Description X",
            "Item 2    Value B    Description Y", 
            "Item 3    Value C    Description Z",
            "",
            "The GPU acceleration is particularly beneficial when",
            "processing documents with:",
            "- Complex formatting",
            "- Multiple columns", 
            "- Images and diagrams",
            "- Tables and structured data",
            "- Mathematical equations",
            "- Mixed fonts and styles",
        ]
        
        y_position = 680
        for line in more_content:
            c.drawString(100, y_position, line)
            y_position -= 20
        
        c.save()
        
        print(f"✅ Sample PDF created: {pdf_path}")
        return pdf_path
        
    except ImportError:
        print("⚠️  ReportLab not available, skipping PDF creation")
        print("💡 Install with: pip install reportlab")
        return None

def process_document_cpu(pdf_path):
    """Process document using CPU only"""
    print("\n=== CPU Processing ===")
    
    try:
        from marker import convert_single_pdf
        import torch
        
        # Force CPU processing
        device = 'cpu'
        
        print(f"🔄 Processing {pdf_path.name} on CPU...")
        start_time = time.time()
        
        # Convert PDF to markdown
        markdown_text, images, metadata = convert_single_pdf(
            str(pdf_path),
            device=device
        )
        
        cpu_time = time.time() - start_time
        
        print(f"✅ CPU processing completed in {cpu_time:.2f}s")
        print(f"📊 Generated {len(markdown_text)} characters of markdown")
        print(f"📊 Extracted {len(images)} images")
        
        return markdown_text, cpu_time
        
    except Exception as e:
        print(f"❌ CPU processing failed: {e}")
        return None, 0

def process_document_gpu(pdf_path):
    """Process document using GPU acceleration"""
    print("\n=== GPU Processing ===")
    
    try:
        from marker import convert_single_pdf
        import torch
        
        if not torch.backends.mps.is_available():
            print("⚠️  GPU not available, skipping GPU processing")
            return None, 0
        
        # Use MPS device for GPU acceleration
        device = 'mps'
        
        print(f"🔄 Processing {pdf_path.name} on GPU...")
        start_time = time.time()
        
        # Convert PDF to markdown with GPU acceleration
        markdown_text, images, metadata = convert_single_pdf(
            str(pdf_path),
            device=device
        )
        
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU processing completed in {gpu_time:.2f}s")
        print(f"📊 Generated {len(markdown_text)} characters of markdown")
        print(f"📊 Extracted {len(images)} images")
        
        return markdown_text, gpu_time
        
    except Exception as e:
        print(f"❌ GPU processing failed: {e}")
        return None, 0

def batch_processing_example():
    """Demonstrate batch processing with multiple documents"""
    print("\n=== Batch Processing Example ===")
    
    # Create multiple sample PDFs
    sample_pdfs = []
    
    for i in range(3):
        pdf_path = create_sample_pdf()
        if pdf_path:
            # Rename to include index
            new_path = pdf_path.parent / f"sample_document_{i+1}.pdf"
            pdf_path.rename(new_path)
            sample_pdfs.append(new_path)
    
    if not sample_pdfs:
        print("⚠️  No sample PDFs created, skipping batch processing")
        return
    
    print(f"📁 Created {len(sample_pdfs)} sample documents")
    
    # Process batch on CPU
    print("\n🔄 Batch processing on CPU...")
    cpu_start = time.time()
    cpu_results = []
    
    for pdf_path in sample_pdfs:
        try:
            from marker import convert_single_pdf
            
            markdown_text, images, metadata = convert_single_pdf(
                str(pdf_path),
                device='cpu'
            )
            cpu_results.append(len(markdown_text))
            
        except Exception as e:
            print(f"❌ Failed to process {pdf_path.name}: {e}")
    
    cpu_batch_time = time.time() - cpu_start
    print(f"CPU batch processing: {cpu_batch_time:.2f}s for {len(sample_pdfs)} documents")
    
    # Process batch on GPU (if available)
    import torch
    if torch.backends.mps.is_available():
        print("\n🔄 Batch processing on GPU...")
        gpu_start = time.time()
        gpu_results = []
        
        for pdf_path in sample_pdfs:
            try:
                from marker import convert_single_pdf
                
                markdown_text, images, metadata = convert_single_pdf(
                    str(pdf_path),
                    device='mps'
                )
                gpu_results.append(len(markdown_text))
                
            except Exception as e:
                print(f"❌ Failed to process {pdf_path.name}: {e}")
        
        gpu_batch_time = time.time() - gpu_start
        print(f"GPU batch processing: {gpu_batch_time:.2f}s for {len(sample_pdfs)} documents")
        
        if gpu_batch_time > 0:
            speedup = cpu_batch_time / gpu_batch_time
            print(f"🚀 Batch processing speedup: {speedup:.2f}x")
    
    # Cleanup
    print("\n🧹 Cleaning up temporary files...")
    for pdf_path in sample_pdfs:
        try:
            pdf_path.unlink()
        except:
            pass

def save_results_example(markdown_text):
    """Save processing results to files"""
    if not markdown_text:
        return
    
    print("\n=== Saving Results ===")
    
    # Save markdown to file
    output_dir = Path("marker_output")
    output_dir.mkdir(exist_ok=True)
    
    markdown_file = output_dir / "converted_document.md"
    
    with open(markdown_file, 'w', encoding='utf-8') as f:
        f.write(markdown_text)
    
    print(f"✅ Markdown saved to: {markdown_file}")
    print(f"📊 File size: {markdown_file.stat().st_size} bytes")
    
    # Show preview
    lines = markdown_text.split('\n')
    print("\n📄 Document preview (first 10 lines):")
    print("=" * 40)
    for i, line in enumerate(lines[:10]):
        print(f"{i+1:2d}: {line}")
    if len(lines) > 10:
        print(f"... ({len(lines) - 10} more lines)")

def main():
    """Run document processing examples"""
    print("Marker Document Processing with Mac GPU")
    print("=" * 50)
    
    # Check installation
    if not check_marker_installation():
        print("❌ Cannot run examples without Marker installation")
        return
    
    # Create or use sample PDF
    pdf_path = create_sample_pdf()
    if not pdf_path:
        print("❌ Cannot create sample PDF for testing")
        return
    
    try:
        # Process with CPU
        markdown_cpu, cpu_time = process_document_cpu(pdf_path)
        
        # Process with GPU
        markdown_gpu, gpu_time = process_document_gpu(pdf_path)
        
        # Compare results
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\n🚀 GPU Speedup: {speedup:.2f}x")
        
        # Save results
        if markdown_gpu or markdown_cpu:
            save_results_example(markdown_gpu or markdown_cpu)
        
        # Batch processing example
        batch_processing_example()
        
    finally:
        # Cleanup
        try:
            pdf_path.unlink()
        except:
            pass
    
    print("\n✅ Document processing examples completed!")

if __name__ == "__main__":
    main()
