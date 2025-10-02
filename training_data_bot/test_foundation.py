from core.models import Document, DocumentType, TrainingExample, TaskType
from core.exceptions import DocumentLoadError

# Test data model creation
doc = Document(
    title="Test Document",
    content="This is a test document content.",
    source="/path/to/test.txt",
    doc_type=DocumentType.TXT
)

print(f"Created document: {doc.title} with ID: {doc.id}")
print(f"Word count: {doc.word_count}")