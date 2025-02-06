import pytest
from processors.document_processor import DocumentProcessor

@pytest.fixture
def processor():
    return DocumentProcessor()

@pytest.mark.asyncio
async def test_process_text_file(processor):
    content = "Hello, World!".encode('utf-8')
    result = await processor.process_document(content, "test.txt")
    assert result == "Hello, World!"

@pytest.mark.asyncio
async def test_invalid_extension(processor):
    with pytest.raises(ValueError, match="Unsupported file type: .invalid"):
        await processor.process_document(b"content", "test.invalid")

@pytest.mark.asyncio
async def test_no_extension(processor):
    with pytest.raises(ValueError, match="Filename has no extension"):
        await processor.process_document(b"content", "testfile") 