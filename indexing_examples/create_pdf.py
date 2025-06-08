from fpdf import FPDF

def create_sample_pdf():
    # Create PDF object
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set font
    pdf.set_font("Arial", size=12)

    # Add content
    pdf.cell(200, 10, txt="Sample PDF Document", ln=1, align="C")
    pdf.cell(200, 10, txt="", ln=1, align="L")

    pdf.multi_cell(0, 10, txt="This is a sample PDF document created for testing RAG functionality.")
    pdf.multi_cell(0, 10, txt="The document contains information about various topics that can be used to test the indexing and retrieval capabilities of our RAG system.")

    pdf.cell(200, 10, txt="", ln=1, align="L")
    pdf.cell(200, 10, txt="Key Features:", ln=1, align="L")
    pdf.multi_cell(0, 10, txt="1. Structured content\n2. Multiple paragraphs\n3. Different formatting\n4. Sample text for testing")

    # Save the PDF
    pdf.output("data/sample.pdf")

if __name__ == "__main__":
    create_sample_pdf()