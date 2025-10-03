"""
Streamlit UI for Training Data Bot.

A visual interface for document upload, processing, and dataset management.
"""

import streamlit as st
import sys
from pathlib import Path
import asyncio

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training_data_bot import TrainingDataBot
from training_data_bot.core import TaskType, ExportFormat
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Training Data Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'bot' not in st.session_state:
    st.session_state.bot = None
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False


def initialize_bot():
    """Initialize the Training Data Bot."""
    if st.session_state.bot is None:
        with st.spinner("Initializing bot..."):
            bot = TrainingDataBot()
            
            # Configure AI client - Streamlit Cloud uses st.secrets
            try:
                # Streamlit Cloud secrets
                openai_key = st.secrets["TDB_OPENAI_API_KEY"]
            except (KeyError, FileNotFoundError):
                # Fallback to environment variable
                openai_key = os.getenv("TDB_OPENAI_API_KEY")

            if openai_key:
                bot.set_ai_client(
                    provider="openai",
                    api_key=openai_key,
                    model="gpt-3.5-turbo"
                )
            else:
                st.error("⚠️ OpenAI API key not configured. Please add TDB_OPENAI_API_KEY to Streamlit secrets.")

            
            st.session_state.bot = bot
    
    return st.session_state.bot

def show_navigation_buttons(current_page):
    """Show navigation buttons at bottom of page."""
    pages = [
        "📤 Upload Documents",
        "⚙️ Process Data", 
        "📊 View Results",
        "ℹ️ About"
    ]
    
    current_index = pages.index(current_page)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_index > 0:
            if st.button(f"← {pages[current_index - 1]}", use_container_width=True):
                st.session_state.page = pages[current_index - 1]
                st.rerun()
    
    with col3:
        if current_index < len(pages) - 1:
            if st.button(f"{pages[current_index + 1]} →", use_container_width=True):
                st.session_state.page = pages[current_index + 1]
                st.rerun()


def main():
    """Main application."""
    
    # Header
    st.markdown('<div class="main-header">🤖 Training Data Bot</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Initialize page in session state
    if 'page' not in st.session_state:
        st.session_state.page = "📤 Upload Documents"

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        pages = ["📤 Upload Documents", "⚙️ Process Data", "📊 View Results", "ℹ️ About"]
        
        page = st.radio(
            "Go to",
            pages,
            index=pages.index(st.session_state.page),
            label_visibility="collapsed"
        )

        st.session_state.page = page
        
        st.markdown("---")
        
        # Status
        st.subheader("Status")
        bot = initialize_bot()
        
        if bot:
            stats = bot.get_statistics()
            st.metric("Documents Loaded", stats['documents']['total'])
            st.metric("Datasets Created", stats['datasets']['total'])
            st.metric("Total Examples", stats['datasets']['total_examples'])
        
        st.markdown("---")
        st.caption("Training Data Bot v0.1.0")
    
    # Main content
    if page == "📤 Upload Documents":
        show_upload_page()
    elif page == "⚙️ Process Data":
        show_process_page()
    elif page == "📊 View Results":
        show_results_page()
    elif page == "ℹ️ About":
        show_about_page()


def show_upload_page():
    """Document upload page."""
    st.header("📤 Upload Documents")
    st.write("Upload documents to process into training data.")
    
    bot = initialize_bot()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['txt', 'pdf', 'docx', 'md', 'html', 'csv', 'json'],
        accept_multiple_files=True,
        help="Upload one or more documents"
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected**")
        
        # Show file details
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024:.2f} KB)")
        
        if st.button("Load Documents", type="primary"):
            with st.spinner("Loading documents..."):
                try:
                    # Save uploaded files temporarily
                    temp_dir = Path("temp/uploads")
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    
                    file_paths = []
                    for file in uploaded_files:
                        file_path = temp_dir / file.name
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())
                        file_paths.append(str(file_path))
                    
                    # Load documents
                    documents = asyncio.run(bot.load_documents(file_paths))
                    st.session_state.documents = documents
                    
                    st.success(f"✅ Successfully loaded {len(documents)} document(s)!")
                    
                    # Show document details
                    for doc in documents:
                        with st.expander(f"📄 {doc.title}"):
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Type", doc.doc_type.value)
                            col2.metric("Words", doc.word_count)
                            col3.metric("Characters", doc.char_count)
                            
                            st.text_area(
                                "Content Preview",
                                doc.content[:500] + "..." if len(doc.content) > 500 else doc.content,
                                height=150
                            )
                
                except Exception as e:
                    st.error(f"❌ Error loading documents: {e}")
    
    # Show loaded documents
    if st.session_state.documents:
        st.markdown("---")
        st.subheader("Loaded Documents")
        st.write(f"**{len(st.session_state.documents)} document(s) ready for processing**")
        
        for i, doc in enumerate(st.session_state.documents, 1):
            st.write(f"{i}. {doc.title} ({doc.word_count} words)")

    st.markdown("---")
    show_navigation_buttons("📤 Upload Documents")


def show_process_page():
    """Processing configuration page."""
    st.header("⚙️ Process Data")
    
    if not st.session_state.documents:
        st.warning("⚠️ No documents loaded. Please upload documents first.")
        return
    
    bot = initialize_bot()
    
    st.write(f"**Processing {len(st.session_state.documents)} document(s)**")
    
    # Configuration
    st.subheader("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        task_types = st.multiselect(
            "Task Types",
            ["qa_generation", "summarization", "classification"],
            default=["qa_generation", "summarization"],
            help="Select which types of tasks to generate"
        )
    
    with col2:
        quality_filter = st.checkbox(
            "Enable Quality Filtering",
            value=True,
            help="Filter out low-quality examples"
        )
        
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Minimum quality score (0-1)"
        )
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=5000)
        chunk_overlap = st.number_input("Chunk Overlap", value=200, min_value=0, max_value=500)
    
    # Process button
    if st.button("🚀 Start Processing", type="primary"):
        if not task_types:
            st.error("Please select at least one task type")
            return
        
        # Convert task types
        selected_tasks = [TaskType(t) for t in task_types]
        
        # Processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process documents
            status_text.text("Processing documents...")
            progress_bar.progress(25)
            
            dataset = asyncio.run(
                bot.process_documents(
                    documents=st.session_state.documents,
                    task_types=selected_tasks,
                    quality_filter=quality_filter
                )
            )
            
            progress_bar.progress(50)
            
            # Step 2: Evaluate
            status_text.text("Evaluating quality...")
            report = asyncio.run(bot.evaluate_dataset(dataset))
            
            progress_bar.progress(75)
            
            # Step 3: Complete
            st.session_state.dataset = dataset
            st.session_state.processing_complete = True
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Show results
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.write("### Processing Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Examples Generated", len(dataset.examples))
            col2.metric("Quality Score", f"{report.overall_score:.2f}")
            col3.metric("Passed Quality", "✅" if report.passed else "❌")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.info("💡 Go to 'View Results' to see and download your dataset")
        
        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            import traceback
            st.code(traceback.format_exc())

    st.markdown("---")
    show_navigation_buttons("⚙️ Process Data")


def show_results_page():
    """Results viewer page."""
    st.header("📊 View Results")
    
    if not st.session_state.processing_complete or not st.session_state.dataset:
        st.warning("⚠️ No results available. Please process documents first.")
        return
    
    dataset = st.session_state.dataset
    bot = initialize_bot()
    
    # Dataset summary
    st.subheader("Dataset Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Examples", len(dataset.examples))
    col2.metric("Dataset ID", dataset.id.hex[:8] + "...")
    col3.metric("Created", dataset.created_at.strftime("%Y-%m-%d %H:%M"))
    col4.metric("Name", dataset.name)
    
    st.markdown("---")
    
    # Task distribution
    st.subheader("Task Distribution")
    task_dist = {}
    for example in dataset.examples:
        task_type = example.task_type.value
        task_dist[task_type] = task_dist.get(task_type, 0) + 1
    
    col1, col2, col3 = st.columns(3)
    for i, (task, count) in enumerate(task_dist.items()):
        if i == 0:
            col1.metric(task.replace("_", " ").title(), count)
        elif i == 1:
            col2.metric(task.replace("_", " ").title(), count)
        else:
            col3.metric(task.replace("_", " ").title(), count)
    
    st.markdown("---")
    
    # Example viewer
    st.subheader("Examples")

    # Check if there are examples
    if len(dataset.examples) == 0:
        st.warning("⚠️ No examples were generated. This might be because:")
        st.write("- The document was too short")
        st.write("- Quality filtering removed all examples")
        st.write("- The AI client encountered an error")
        return
    
    # Filters
    col1, col2 = st.columns([1, 3])
    with col1:
        filter_task = st.selectbox(
            "Filter by task",
            ["All"] + list(task_dist.keys())
        )

    with col2:
        # Only show slider if there are multiple examples
        if len(dataset.examples) > 1:
            max_examples = min(20, len(dataset.examples))
            default_examples = min(5, len(dataset.examples))
            
            num_examples = st.slider(
                "Number of examples to show",
                min_value=1,
                max_value=max_examples,
                value=default_examples
            )
        else:
            num_examples = 1
            st.info("Showing the only example available")


    
    # Filter examples
    filtered_examples = dataset.examples
    if filter_task != "All":
        filtered_examples = [
            ex for ex in dataset.examples
            if ex.task_type.value == filter_task
        ]
    
    # Display examples
    for i, example in enumerate(filtered_examples[:num_examples], 1):
        with st.expander(f"Example {i} - {example.task_type.value}"):
            st.write("**Input:**")
            st.text_area("", example.input_text, height=100, key=f"input_{i}", label_visibility="collapsed")
            
            st.write("**Output:**")
            st.text_area("", example.output_text, height=150, key=f"output_{i}", label_visibility="collapsed")
            
            if example.quality_scores:
                st.write("**Quality Scores:**")
                score_cols = st.columns(len(example.quality_scores))
                for j, (metric, score) in enumerate(example.quality_scores.items()):
                    score_cols[j].metric(metric, f"{score:.2f}")
    
    st.markdown("---")
    
    # Export section
    st.subheader("Export Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        export_format = st.selectbox(
            "Format",
            ["jsonl", "json", "csv"],
            help="Select export format"
        )
    
    with col2:
        split_data = st.checkbox(
            "Split into train/val/test",
            value=True,
            help="Split dataset into training, validation, and test sets"
        )
    
    if st.button("📥 Export Dataset", type="primary"):
        try:
            with st.spinner("Exporting..."):
                # Ensure output directory exists
                output_dir = Path("output")
                output_dir.mkdir(parents=True, exist_ok=True)

                # Create proper file path
                output_path = output_dir / f"{dataset.name}.{export_format}"
                
                result_path = asyncio.run(
                    bot.export_dataset(
                        dataset=dataset,
                        output_path=output_path,
                        format=ExportFormat(export_format),
                        split_data=split_data
                    )
                )
                                
                # Provide download button
                st.success(f"✅ Dataset exported successfully!")

                # Find and provide download buttons for the actual files
                if split_data:
                    # Look for train/val/test files
                    base_name = dataset.name
                    train_file = output_dir / f"{base_name}_train.{export_format}"
                    val_file = output_dir / f"{base_name}_val.{export_format}"
                    test_file = output_dir / f"{base_name}_test.{export_format}"
                    
                    col1, col2, col3 = st.columns(3)
                    
                    if train_file.exists():
                        with open(train_file, "rb") as f:
                            col1.download_button(
                                "📥 Training Set",
                                f.read(),
                                file_name=train_file.name,
                                mime="application/octet-stream"
                            )
                    
                    if val_file.exists():
                        with open(val_file, "rb") as f:
                            col2.download_button(
                                "📥 Validation Set",
                                f.read(),
                                file_name=val_file.name,
                                mime="application/octet-stream"
                            )
                    
                    if test_file.exists():
                        with open(test_file, "rb") as f:
                            col3.download_button(
                                "📥 Test Set",
                                f.read(),
                                file_name=test_file.name,
                                mime="application/octet-stream"
                            )
                else:
                    # Single file
                    if output_path.exists():
                        with open(output_path, "rb") as f:
                            st.download_button(
                                "📥 Download Dataset",
                                f.read(),
                                file_name=output_path.name,
                                mime="application/octet-stream"
                            )
        
        except Exception as e:
            st.error(f"❌ Export failed: {e}")


def show_about_page():
    """About page."""
    st.header("ℹ️ About Training Data Bot")
    
    st.write("""
    **Training Data Bot** is an enterprise-grade system for curating high-quality training data for LLM fine-tuning.
    
    ### Features
    - 📄 Multi-format document loading (PDF, DOCX, TXT, MD, HTML, CSV, JSON)
    - 🤖 AI-powered task generation (Q&A, Classification, Summarization)
    - ✅ Quality evaluation and filtering
    - 📊 Multiple export formats (JSONL, JSON, CSV)
    - 🔄 Batch processing capabilities
    
    ### Technology Stack
    - **Backend:** FastAPI, Python 3.13
    - **AI:** OpenAI GPT-3.5/4, Anthropic Claude
    - **UI:** Streamlit
    - **Deployment:** Docker, Docker Compose
    
    ### Version
    0.1.0
    
    ### Documentation
    For more information, visit the project repository or check the API documentation at `/docs`.
    """)

    st.markdown("---")
    show_navigation_buttons("ℹ️ About")


if __name__ == "__main__":
    main()