import os
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_with_citation

load_dotenv(override=True)

st.set_page_config(
    page_title="Tư vấn Tuyển sinh Đại học — RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Xin chào! Tôi là trợ lý AI tư vấn tuyển sinh đại học (HUST, NEU, VinUni & Quy chế Bộ GD&ĐT). "
                "Mọi câu trả lời của tôi đều có nguồn trích dẫn và bằng chứng rõ ràng. Bạn muốn tìm hiểu thông tin gì?"
            ),
            "sources": [],
            "retrieval_source": "none",
        }
    ]

# Sidebar cấu hình & thông tin nhóm
with st.sidebar:
    st.title("🎓 RAG Pipeline")
    st.caption("Nhóm L3A — Day 8 RAG Project")
    st.divider()

    st.subheader("👥 Thành viên nhóm")
    st.markdown("- **Chu Văn Nhân** *(Leader)*: Architecture, Generation, UI\n"
                "- **Nguyễn Khắc Quang**: Data, Markdown, Indexing\n"
                "- **Dương Dương**: BM25 Lexical, Reranking, PageIndex")
    st.divider()

    st.subheader("⚙️ Cấu hình Retrieval")
    top_k = st.slider("Số lượng tài liệu truy xuất (top-k)", min_value=3, max_value=10, value=5)
    
    provider = os.getenv("LLM_PROVIDER", "openai").upper()
    api_key_set = bool(
        os.getenv("OPENAI_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("ANTHROPIC_API_KEY")
    )
    if api_key_set:
        st.success(f"**LLM Provider:** {provider} (Online)\n\n**Embedding:** BAAI/bge-m3\n\n**Fusion:** RRF (k=60)")
    else:
        st.warning(f"**LLM Provider:** {provider} (Chưa có API key - Chế độ trích xuất offline)\n\n**Embedding:** BAAI/bge-m3\n\n**Fusion:** RRF (k=60)")

    st.divider()
    if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Tiêu đề chính
st.title("🎓 Hệ thống Hỏi đáp & Tra cứu Tuyển sinh Đại học")
st.caption(
    "Chatbot ứng dụng kiến trúc Hybrid Retrieval (Dense Semantic + BM25 Lexical + RRF Reranking + Cosine Fallback) "
    "đảm bảo câu trả lời trung thực, chính xác và có thể kiểm chứng nguồn gốc."
)

# Hiển thị lịch sử chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        sources = message.get("sources", [])
        retrieval_source = message.get("retrieval_source", "none")

        if sources:
            with st.expander(f"📚 Nguồn trích dẫn ({len(sources)} tài liệu | Retrieval: {retrieval_source.upper()})"):
                for idx, src in enumerate(sources, 1):
                    meta = src.get("metadata", {})
                    score = src.get("score", 0.0)
                    title = meta.get("title", "Tài liệu")
                    source_name = meta.get("source", "N/A")
                    url = meta.get("url")
                    chunk_idx = meta.get("chunk_index", 0)

                    st.markdown(f"**[Document {idx}] {title}**")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        if url:
                            st.caption(f"🔗 [Xem nguồn bài viết]({url}) | File: `{source_name}` (Chunk #{chunk_idx})")
                        else:
                            st.caption(f"📄 Nguồn: `{source_name}` (Chunk #{chunk_idx})")
                    with col2:
                        st.caption(f"Score: `{score:.4f}`")

                    st.markdown(f"> {src.get('content', '').strip()}")
                    st.divider()

# Input câu hỏi
query = st.chat_input("Nhập câu hỏi về tuyển sinh, chỉ tiêu, điểm chuẩn, quy chế...")

if query:
    # 1. Hiển thị câu hỏi người dùng
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Xử lý qua RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu và tổng hợp câu trả lời..."):
            result = generate_with_citation(query, top_k=top_k)
            answer = result["answer"]
            sources = result.get("sources", [])
            retrieval_source = result.get("retrieval_source", "none")

            st.markdown(answer)

            if sources:
                with st.expander(f"📚 Nguồn trích dẫn ({len(sources)} tài liệu | Retrieval: {retrieval_source.upper()})"):
                    for idx, src in enumerate(sources, 1):
                        meta = src.get("metadata", {})
                        score = src.get("score", 0.0)
                        title = meta.get("title", "Tài liệu")
                        source_name = meta.get("source", "N/A")
                        url = meta.get("url")
                        chunk_idx = meta.get("chunk_index", 0)

                        st.markdown(f"**[Document {idx}] {title}**")
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            if url:
                                st.caption(f"🔗 [Xem nguồn bài viết]({url}) | File: `{source_name}` (Chunk #{chunk_idx})")
                            else:
                                st.caption(f"📄 Nguồn: `{source_name}` (Chunk #{chunk_idx})")
                        with col2:
                            st.caption(f"Score: `{score:.4f}`")

                        st.markdown(f"> {src.get('content', '').strip()}")
                        st.divider()
            elif retrieval_source == "none":
                st.info("ℹ️ Không tìm thấy bằng chứng phù hợp trong cơ sở tri thức đã thu thập.")

    # 3. Lưu lại lịch sử
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })

