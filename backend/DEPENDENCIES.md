# Dependencies Documentation

## Why These Dependencies? Where Are They Used?

---

## CORE WEB FRAMEWORK

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **fastapi** | Modern, fast web framework for building REST APIs | `app/main.py` - creates the API endpoints |
| **uvicorn** | ASGI server to run FastAPI | Terminal command: `uvicorn app.main:app` |
| **starlette** | FastAPI's foundation (comes with fastapi) | Underlying ASGI toolkit |
| **httptools** | Faster HTTP parsing for uvicorn | Performance optimization in uvicorn |
| **watchfiles** | Auto-reload on code changes | Development mode: `--reload` flag |

---

## DATA VALIDATION & CONFIGURATION

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **pydantic** | Validate request/response data, ensure type safety | `app/schemas/*.py` - all request/response models |
| **pydantic-core** | Core validation logic for Pydantic | Internal dependency of pydantic |
| **pydantic-settings** | Load environment variables from .env with type validation | `app/core/config.py` - Settings class |
| **python-dotenv** | Load .env file into environment | Fallback for pydantic-settings |
| **typing-extensions** | Backports of modern typing features | Used across all Python files |
| **typing-inspection** | Runtime typing inspection utilities | Pydantic internals |
| **annotated-types** | Support for `Annotated` type hints | Pydantic validation |

---

## LOGGING

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **loguru** | Structured logging with JSON output, log rotation | `app/core/logging.py` - all logging |
| **colorama** | Colored console output on Windows | Loguru console formatting |
| **win32-setctime** | Windows file creation time support | Loguru file rotation on Windows |

---

## MACHINE LEARNING & DATA PROCESSING

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **numpy** | Numerical operations, array manipulation | `app/services/ml_predictor.py` - feature extraction |
| **pandas** | Data manipulation, DataFrame operations | Notebooks: data cleaning, feature engineering |
| **scikit-learn** | ML models (Random Forest, Logistic Regression) | `notebooks/09_ml_model_training.ipynb` |
| **xgboost** | Gradient boosting classifier (best model) | `notebooks/09_ml_model_training.ipynb` |
| **joblib** | Save/load trained models | `app/services/ml_predictor.py` - load .pkl file |
| **threadpoolctl** | Control thread parallelism in scikit-learn | scikit-learn internals |
| **scipy** | Scientific computing (used by scikit-learn) | Internal dependency |

---

## TEXT PROCESSING & FEATURE ENGINEERING

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **nltk** | Text tokenization, stopwords, stemming | `app/utils/text_processing.py` - text cleaning |
| **textblob** | Sentiment analysis (-1 to 1 score) | `app/utils/text_processing.py` - sentiment scoring |
| **regex** | Advanced regular expression operations | Text cleaning and pattern matching |
| **tiktoken** | Count tokens for LLM cost calculation | `app/utils/token_counter.py` - cost tracking |

---

## EMBEDDINGS & VECTOR DATABASE (RAG)

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **sentence-transformers** | Generate embeddings from text | `app/services/rag_retriever.py` - embed queries |
| **transformers** | Transformer models (used by sentence-transformers) | Internal dependency |
| **tokenizers** | Tokenization for transformer models | Internal dependency |
| **torch** | PyTorch backend for sentence-transformers | Running embedding models |
| **chromadb** | Vector database for storing and retrieving embeddings | `app/services/rag_retriever.py` - similarity search |
| **onnxruntime** | Faster inference for embeddings (optional) | ChromaDB optimization |
| **safetensors** | Safe tensor storage format | HuggingFace model loading |
| **huggingface-hub** | Download models from HuggingFace | Downloading sentence-transformers |
| **hf-xet** | HuggingFace dataset acceleration | Internal dependency |
| **filelock** | Prevent concurrent file access | Model caching |
| **fsspec** | File system interface for remote storage | HuggingFace model downloads |
| **mmh3** | MurmurHash3 - fast hashing for ChromaDB | ChromaDB indexing |
| **pypika** | SQL query builder for ChromaDB | ChromaDB internal queries |
| **pybase64** | Fast base64 encoding | ChromaDB serialization |
| **overrides** | Method override decorators | ChromaDB internal |
| **build** | Python package building | ChromaDB build process |
| **pyproject-hooks** | PEP 517 build hooks | ChromaDB build process |
| **grpcio** | gRPC for distributed ChromaDB | ChromaDB client-server communication |
| **protobuf** | Protocol buffers for gRPC | ChromaDB serialization |
| **opentelemetry-*** | Telemetry and observability | ChromaDB monitoring |
| **kubernetes** | K8s client (ChromaDB might use for discovery) | ChromaDB clustering |
| **oauthlib** | OAuth authentication | Kubernetes client |
| **requests-oauthlib** | OAuth for requests | Kubernetes client |
| **websocket-client** | WebSocket client | ChromaDB real-time communication |
| **durationpy** | Duration parsing | Kubernetes client |
| **googleapis-common-protos** | Google API protos | gRPC internals |
| **bcrypt** | Password hashing | ChromaDB auth (if enabled) |
| **typer** | CLI building | ChromaDB CLI |
| **rich** | Beautiful console output | ChromaDB CLI |
| **shellingham** | Detect shell type | Typer/ChromaDB |
| **pygments** | Syntax highlighting | Rich console |
| **markdown-it-py** | Markdown parsing | Rich console |
| **mdurl** | URL parsing for markdown | markdown-it-py |
| **click** | CLI framework | Various internal tools |

---

## OPENAI / LLM

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **openai** | Call OpenAI API (GPT-3.5, GPT-4) | `app/services/llm_client.py` - LLM calls |
| **httpx** | HTTP client for OpenAI | `app/services/llm_client.py` - API requests |
| **httpcore** | Core HTTP logic for httpx | Internal dependency |
| **anyio** | Async networking support | httpx internal |
| **h11** | HTTP/1.1 protocol implementation | httpcore internal |
| **certifi** | SSL certificate verification | HTTPS requests |
| **idna** | Internationalized domain names | URL handling |
| **sniffio** | Detect async library in use | anyio internal |
| **distro** | OS distribution detection | OpenAI client |
| **jiter** | Fast JSON parsing | OpenAI response parsing |
| **tqdm** | Progress bars | HuggingFace downloads |
| **tenacity** | Retries with exponential backoff | `app/services/llm_client.py` - retry failed API calls |
| **requests** | HTTP client (alternative) | Some internal dependencies |

---

## ASYNCIO & CONCURRENCY

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **anyio** | Async networking | httpx, OpenAI client |
| **sniffio** | Detect async library | anyio |
| **idna** | Async DNS handling | anyio |

---

## UTILITIES

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **python-dateutil** | Advanced date parsing | Notebooks - timestamp features |
| **tzdata** | Timezone data | pandas datetime operations |
| **packaging** | Version parsing | Various dependencies |
| **attrs** | Classes without boilerplate | Internal dependency |
| **setuptools** | Python package distribution | Building packages |
| **zipp** | Backport of zipfile.Path | importlib-metadata |
| **importlib-metadata** | Read package metadata | Various |
| **importlib-resources** | Access resources in packages | Various |
| **rpds-py** | Rust-backed data structures | referencing (JSON schema) |
| **referencing** | JSON Schema referencing | jsonschema |
| **jsonschema** | JSON Schema validation | ChromaDB, OpenAI |
| **jsonschema-specifications** | JSON Schema specs | jsonschema |
| **sympy** | Symbolic mathematics | transformers? |
| **mpmath** | Arbitrary precision math | sympy |
| **networkx** | Graph algorithms | transformers? |
| **flatbuffers** | Serialization format | ChromaDB, ONNX |
| **orjson** | Fast JSON (optional) | Some libraries |
| **websockets** | WebSocket server/client | uvicorn internal |
| **markupsafe** | HTML escaping | Jinja2 |
| **jinja2** | Templating engine | FastAPI error pages |

---

## DEVELOPMENT TOOLS

| Dependency | Why | Where Used |
|------------|-----|-------------|
| **black** | Code formatter | `pyproject.toml` - format Python code |
| **isort** | Import sorter | `pyproject.toml` - organize imports |
| **ruff** | Fast linter | `pyproject.toml` - code quality |

---

## Summary by Purpose

| Purpose | Main Dependencies |
|---------|-------------------|
| **API Server** | fastapi, uvicorn, starlette |
| **Data Validation** | pydantic, pydantic-settings |
| **Logging** | loguru |
| **ML Models** | scikit-learn, xgboost, numpy, pandas |
| **Text Processing** | nltk, textblob, regex |
| **Embeddings** | sentence-transformers, torch |
| **Vector DB** | chromadb |
| **LLM Calls** | openai, httpx, tenacity, tiktoken |
| **Cost Tracking** | tiktoken (tokens), tenacity (retries) |
| **Async** | anyio, httpx |
| **Development** | black, isort, ruff |

---

## Why Are All These in Backend?

**Because the backend is where ALL Python code runs:**

| Component | Location | Why |
|-----------|----------|-----|
| FastAPI server | Backend | Handles HTTP requests |
| ML model loading | Backend | Python ML libraries |
| Feature extraction | Backend | Python data processing |
| RAG retrieval | Backend | ChromaDB, embeddings |
| LLM calls | Backend | OpenAI SDK |
| Logging | Backend | Python logging |
| Data processing | Backend/Notebooks | pandas, numpy |

**The frontend (React) has NO Python dependencies.** It only needs:
- `react` - UI framework
- `axios` or `fetch` - API calls
- `vite` - build tool

**This separation is why `uv` is only in `backend/` folder.**

