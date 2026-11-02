import json

from open_agentinstruct.utils.text_extraction import (
    extract_text_chunks_from_dataset,
    extract_text_chunks_from_pdf,
)


def test_current_datasets_reads_and_chunks_local_json_without_network(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    data = tmp_path / "rows.jsonl"
    data.write_text("\n".join(json.dumps({"text": value}) for value in ("hello", "world")))
    chunks = extract_text_chunks_from_dataset(
        "json", dataset_kwargs={"data_files": str(data)}, chunk_size=5
    )
    assert chunks == ["hello", " worl", "d"]


def test_current_pdfminer_extracts_text_from_a_local_pdf(tmp_path):
    # A tiny valid PDF fixture with an ordinary Helvetica text stream.
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 300] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    content = b"BT /F1 12 Tf 50 250 Td (Security test) Tj ET"
    objects.append(b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream")
    data = b"%PDF-1.4\n"
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data += str(index).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    startxref = len(data)
    data += b"xref\n0 6\n0000000000 65535 f \n"
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    data += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + str(startxref).encode() + b"\n%%EOF\n"
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(data)
    assert extract_text_chunks_from_pdf(pdf) == ["Security test"]
