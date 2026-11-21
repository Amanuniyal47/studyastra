"""Synthetic PDF generator for Stage 11.1 reproducible benchmarking."""

import io
from typing import List, Tuple
import fitz  # PyMuPDF

SAMPLE_ACADEMIC_PARAGRAPHS: List[Tuple[str, str]] = [
    (
        "Photosynthesis and Energy Conversion in Chloroplasts",
        "Photosynthesis is a vital biochemical process through which green plants, algae, and specific cyanobacteria "
        "convert radiant solar energy into chemical energy stored in carbohydrates. Occurring within specialized "
        "double-membraned organelles known as chloroplasts, the reaction relies upon chlorophyll pigments embedded in "
        "thylakoid membranes to absorb electromagnetic radiation in the blue and red wavelengths while reflecting green. "
        "The overall chemical equation synthesizes glucose and releases molecular oxygen as a byproduct from carbon "
        "dioxide and water through coupled light-dependent reactions and the light-independent Calvin-Benson cycle."
    ),
    (
        "Cellular Respiration and Mitochondrial ATP Synthesis",
        "Cellular respiration encompasses the metabolic pathways through which organic macromolecules are oxidatively "
        "catabolized to generate adenosine triphosphate (ATP), the universal energetic currency of biological systems. "
        "The process proceeds through four distinct stages: anaerobic glycolysis in the cytosol, pyruvate decarboxylation, "
        "the tricarboxylic acid (Krebs) cycle in the mitochondrial matrix, and oxidative phosphorylation along the inner "
        "mitochondrial cristae. Chemiosmotic coupling across the proton gradient drives the rotational catalysis of ATP synthase."
    ),
    (
        "Deoxyribonucleic Acid Structure and Replication",
        "Deoxyribonucleic acid (DNA) is a double-stranded antiparallel helical polymer that encodes the hereditary genomic "
        "blueprints of all known cellular organisms. The structural framework consists of alternating phosphate groups and "
        "deoxyribose sugar rings connected via phosphodiester linkages. Nitrogenous bases project inwardly, adhering to "
        "Chargaff base-pairing rules where adenine pairs with thymine via two hydrogen bonds and guanine pairs with cytosine "
        "via three hydrogen bonds. Semiconservative replication is orchestrated by DNA polymerases, helicases, and primases."
    ),
    (
        "Operating System Kernels and Memory Management",
        "Modern operating systems provide essential hardware abstraction and process virtualization through an "
        "architectural separation of kernel space and user space privilege rings. The kernel oversees processor scheduling, "
        "inter-process communication, secondary storage file hierarchies, and memory virtualization via multi-level paging "
        "and segmentation. Virtual memory decouples physical address space from logical addressing, utilizing translation "
        "lookaside buffers (TLBs) and page tables to implement memory protection, copy-on-write optimizations, and demand paging."
    ),
    (
        "Relational Database Transactions and ACID Guarantees",
        "Relational database management systems ensure data consistency and durability across concurrent execution "
        "through adherence to the ACID properties: Atomicity, Consistency, Isolation, and Durability. Concurrency control "
        "mechanisms such as two-phase locking (2PL), multi-version concurrency control (MVCC), and write-ahead logging (WAL) "
        "prevent anomalies such as dirty reads, non-repeatable reads, and phantom records. Recovery algorithms using checkpointing "
        "enable full state reconstruction following unpredicted system crashes."
    )
]

BENCHMARK_QUERIES = [
    {
        "query": "Where do light-dependent reactions occur during photosynthesis?",
        "expected_topic": "Photosynthesis",
        "follow_up": "What pigments absorb light in those membranes?"
    },
    {
        "query": "What role does ATP synthase play in the mitochondria?",
        "expected_topic": "Cellular Respiration",
        "follow_up": "How does it use the proton gradient to produce energy?"
    },
    {
        "query": "Explain how virtual memory paging protects processes.",
        "expected_topic": "Operating Systems",
        "follow_up": "What hardware component caches those page table translations?"
    }
]


def generate_synthetic_pdf(page_count: int) -> bytes:
    """Generate an in-memory PDF document containing the requested page count."""
    doc = fitz.open()
    for page_num in range(1, page_count + 1):
        page = doc.new_page()
        # Header
        header_text = f"Chapter {page_num}: Advanced Scientific & Computational Systems (Page {page_num})\n\n"
        
        # Insert paragraphs
        content = header_text
        for idx, (heading, body) in enumerate(SAMPLE_ACADEMIC_PARAGRAPHS):
            content += f"Section {page_num}.{idx + 1}: {heading}\n"
            content += f"{body}\n\n"
            
        page.insert_text((50, 60), content, fontsize=9)

    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def generate_synthetic_pdf_stream(page_count: int) -> io.BytesIO:
    """Generate an in-memory BytesIO stream of the synthetic PDF."""
    return io.BytesIO(generate_synthetic_pdf(page_count))
