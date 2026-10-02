# 🤖 Agentic YOLO — AI-Powered Object Detection Agent

An experimental **Agentic AI + Computer Vision** project that combines **YOLOv8**, **AutoGen AgentChat**, and a local **Ollama LLM** to create an intelligent image-analysis agent.

## 🚀 Project Overview

Traditional object detection systems return detected objects and visual detection results. This project adds an **agentic layer** on top of YOLOv8, allowing an AutoGen agent to access YOLO as a tool.

### Architecture

```text
User
  ↓
AutoGen YOLO Agent
  ↓
Qwen 2.5:3B via Ollama
  ↓
YOLO Tool
  ↓
YOLOv8n
  ↓
Image Object Detection
  ↓
Detection Results
```

> **LLM Agent + Tool + Computer Vision Model = Agentic Computer Vision Workflow**

## ✨ Features

- 🤖 AutoGen `AssistantAgent`
- 👁️ YOLOv8 object detection
- 🧠 Local LLM inference with Ollama
- ⚡ Qwen 2.5 3B as the agent model
- 🛠️ Custom YOLO detection tool
- 🖼️ Multimodal image input with AutoGen
- 🔄 `RoundRobinGroupChat` workflow
- 🛑 `TextMentionTermination`
- 🔐 `.env` support with `python-dotenv`
- 📓 Jupyter Notebook experimentation
- 🐍 Python 3.12+ support

## 🧰 Technologies

| Technology | Purpose |
|---|---|
| Python | Core programming |
| YOLOv8 | Object detection |
| Ultralytics | YOLO implementation |
| AutoGen AgentChat | Agent creation and orchestration |
| Ollama | Local LLM runtime |
| Qwen 2.5 3B | Agent language model |
| Pillow | Image processing |
| python-dotenv | Environment variables |
| Jupyter Notebook | Development |

## 📁 Project Structure

```text
agentic-yolo/
│
├── images/
│   ├── img-1.webp
│   └── img-2.webp
│
├── src/
│   └── agentic_yolo/
│       └── __init__.py
│
├── agent.ipynb
├── yolo.ipynb
├── doc_string.py
├── yolov8n.pt
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

### Important Files

- **`agent.ipynb`** — Main AutoGen agent and team workflow.
- **`yolo.ipynb`** — Basic YOLOv8 image-detection experiments.
- **`doc_string.py`** — Description used for the YOLO analysis tool.
- **`yolov8n.pt`** — Pretrained YOLOv8 nano model.

## ⚙️ How It Works

### 1. Load YOLOv8

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
```

### 2. Create the YOLO Tool

```python
def yolo_tool(path: str) -> str:
    """
    Analyze an image using YOLOv8 object detection.

    This tool takes the path of an image, runs YOLOv8 object detection,
    identifies objects present in the image, and returns a completion
    message after the detection process.

    Args:
        path (str): Path to the image.

    Returns:
        str: Status message indicating that YOLO analysis is complete.
    """
    model = YOLO("yolov8n.pt")
    result = model(source=path, show=True)
    print(result[0].show())
    return "YOLO analysis complete"
```

### 3. Create the AI Agent

```python
from autogen_agentchat.agents import AssistantAgent

yolo_agent = AssistantAgent(
    name="YOLO_Agent",
    model_client=ollama_model_client,
    system_message=(
        "You are a helpful assistant that can analyze images "
        "using YOLOv8 and provide insights based on detected objects."
    ),
    tools=[yolo_tool],
    reflect_on_tool_use=True,
)
```

The important concept is:

```text
LLM
 ↓
Tool Selection
 ↓
YOLO Tool
 ↓
Object Detection
 ↓
Agent Response
```

## 🧠 Why Use an Agent?

A normal YOLO application follows:

```text
Image → YOLO → Detection
```

This project explores:

```text
User Request
     ↓
AI Agent
     ↓
Available Tool
     ↓
YOLO
     ↓
Object Detection
     ↓
Agent Response
```

This provides practical experience with **LLMs interacting with external tools**.

## 🖼️ Multimodal Image Input

The project also experiments with AutoGen's `MultiModalMessage`:

```python
from autogen_agentchat.messages import MultiModalMessage
from autogen_core import Image
from PIL import Image as PILImage

pil_image = PILImage.open("images/img-2.webp")
img = Image(pil_image)

multi_modal_message = MultiModalMessage(
    content=[
        "Here is an image for analysis.",
        img
    ],
    source="user"
)
```

## 🔄 Agent Team

```python
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

team = RoundRobinGroupChat(
    participants=[yolo_agent],
    max_turns=3,
    termination_condition=TextMentionTermination("goodbye")
)
```

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/jillasrivardhan/agentic-yolo.git
cd agentic-yolo
```

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

The project requires **Python 3.12 or later**.

## 🦙 Ollama Setup

Install Ollama and verify the installation:

```bash
ollama --version
```

Pull the Qwen model:

```bash
ollama pull qwen2.5:3b
```

Verify:

```bash
ollama list
```

Make sure the model name in your Python code exactly matches the name shown by Ollama.

## 🔐 Environment Variables

If an API key is required for an experiment, store it in `.env`:

```env
GOOGLE_API_KEY=your_api_key_here
```

Load it with:

```python
from dotenv import load_dotenv

load_dotenv()
```

### Security

Never commit real API keys to GitHub.

Add this to `.gitignore`:

```gitignore
.env
```

For sharing the project, use `.env.example`:

```env
GOOGLE_API_KEY=your_api_key_here
```

## ▶️ Running the Project

Open:

```text
agent.ipynb
```

and run the cells in order.

For basic YOLO experiments, open:

```text
yolo.ipynb
```

### Test the YOLO Tool

```python
response = yolo_tool("images/img-1.webp")
print(response)
```

### Test the Agent

```python
result = await yolo_agent.run(task="Who are you?")
print(result.messages[-1].content)
```

### Run the Team

```python
from autogen_agentchat.ui import Console

result = await Console(
    team.run_stream(task=multi_modal_message)
)

print(result.messages[-1].content)
```

## 📊 System Architecture

```text
┌───────────────────────────┐
│           User            │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│     AutoGen Assistant     │
│          Agent            │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│      Qwen 2.5:3B          │
│        via Ollama         │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│        YOLO Tool          │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│        YOLOv8n            │
│    Object Detection       │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│       Image Results       │
└───────────────────────────┘
```

## 🎯 Learning Objectives

This project was built to strengthen practical understanding of:

- Agentic AI
- AutoGen AgentChat
- LLM tool calling
- Local LLMs with Ollama
- YOLOv8 object detection
- Computer vision
- Multimodal messages
- Agent workflows
- Tool descriptions and docstrings
- Environment-variable management
- AI + computer vision integration

## 🚀 Future Improvements

- [ ] Return structured YOLO detection results to the agent
- [ ] Include object names and confidence scores in the final response
- [ ] Add object counting
- [ ] Add image classification
- [ ] Add image comparison
- [ ] Build a Streamlit interface
- [ ] Add multiple specialized agents
- [ ] Add persistent memory
- [ ] Add image upload support
- [ ] Save detection results
- [ ] Generate detection reports
- [ ] Add confidence-threshold controls

## ⚠️ Project Note

This repository is primarily a **learning and experimentation project** demonstrating how an LLM agent can interact with a YOLOv8 computer-vision tool.

The current YOLO tool displays YOLO detection output and returns a simple completion message. Structured detection information can be added as a future enhancement.

## 👨‍💻 Author

**Srivardhan Jilla**

AI / Machine Learning / Generative AI Enthusiast

- GitHub: https://github.com/jillasrivardhan
- LinkedIn: https://www.linkedin.com/in/jilla-srivardhan/

## ⭐ Support

If this project helped you understand the connection between **Agentic AI, LLMs, tools, and computer vision**, consider giving the repository a ⭐.

## 📜 License

This project is intended for educational and experimental purposes.
