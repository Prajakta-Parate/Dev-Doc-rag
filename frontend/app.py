import os

import requests
import streamlit as st

API_BASE_URL = st.secrets.get(
    "API_BASE_URL",
    os.getenv("API_BASE_URL", "http://localhost:8000")
).rstrip("/")


st.set_page_config(
    page_title="DevDocRag",
    page_icon="💻",
    layout="wide",
)

st.title("💻 DevDocRag")
st.caption("Ask questions about your codebase and documentation.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge base")
    st.write("Upload a ZIP of your project or documentation.")

    uploaded_zip = st.file_uploader(
        "Repository ZIP",
        type=["zip"],
    )

    if st.button(
        "Index repository",
        type="primary",
        disabled=uploaded_zip is None,
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Reading files, creating embeddings, and indexing..."
            ):
                response = requests.post(
                    f"{API_URL}/ingest",
                    files={
                        "file": (
                            uploaded_zip.name,
                            uploaded_zip.getvalue(),
                            "application/zip",
                        )
                    },
                    timeout=600,
                )

            if response.ok:
                result = response.json()
                st.success(result["message"])
                st.write(
                    f"Indexed chunks: {result['chunks_indexed']}"
                )
                st.session_state.messages = []
            else:
                st.error(
                    response.json().get(
                        "detail",
                        "Indexing failed.",
                    )
                )

        except requests.RequestException as exc:
            st.error(f"Could not reach the backend: {exc}")

    st.divider()
    st.caption("Backend status")

    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=5,
            )
            response.raise_for_status()
            st.success("API is running.")
        except requests.RequestException:
            st.error("API is not reachable.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("View retrieved source files"):
                for source in message["sources"]:
                    st.markdown(
                        f"**[{source['id']}] "
                        f"{source['file_path']}** "
                        f"(lines {source['start_line']}-"
                        f"{source['end_line']})"
                    )
                    st.code(
                        source["snippet"],
                        language="text",
                    )


question = st.chat_input(
    "Ask about your codebase..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your codebase..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "question": question,
                        "top_k": 5,
                    },
                    timeout=180,
                )

                response.raise_for_status()
                result = response.json()

                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("View retrieved source files"):
                        for source in result["sources"]:
                            st.markdown(
                                f"**[{source['id']}] "
                                f"{source['file_path']}** "
                                f"(lines {source['start_line']}-"
                                f"{source['end_line']})"
                            )
                            st.code(
                                source["snippet"],
                                language="text",
                            )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                })

            except requests.RequestException as exc:
                error_message = f"Could not get an answer: {exc}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                })