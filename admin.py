import streamlit as st
import yaml
import subprocess
import os
import signal
import time
import requests
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi

st.set_page_config(page_title="MLX Model Admin", layout="wide", page_icon="🤖")

MODELS_FILE = "models.yaml"
LOGS_DIR = "logs"

# --- Helper Functions ---
def load_config():
    if not os.path.exists(MODELS_FILE):
        return {"models": []}
    with open(MODELS_FILE, "r") as f:
        return yaml.safe_load(f)

def get_process_pid(port):
    """Find PID using lsof on the port"""
    try:
        result = subprocess.check_output(f"lsof -t -i:{port}", shell=True)
        return int(result.strip())
    except:
        return None

def is_running(port):
    return get_process_pid(port) is not None

def start_server(model):
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, f"{model['id']}.log")
    
    cmd = [
        "mlx_lm.server",
        "--model", model["repo_id"],
        "--port", str(model["port"])
    ]
    if "draft_model" in model:
        cmd.extend(["--draft-model", model["draft_model"]])
        if "num_draft_tokens" in model:
             cmd.extend(["--num-draft-tokens", str(model["num_draft_tokens"])])

    # Run in background, detached
    with open(log_file, "w") as out:
        subprocess.Popen(cmd, stdout=out, stderr=out, preexec_fn=os.setsid)

def stop_server(port):
    pid = get_process_pid(port)
    if pid:
        os.kill(pid, signal.SIGTERM)

# --- UI Layout ---

st.title("🤖 MLX Local Model Admin")

config = load_config()

# Sidebar: Global Controls
with st.sidebar:
    st.header("System Status")
    if st.button("🛑 Stop All Servers", type="primary"):
        for model in config["models"]:
            stop_server(model["port"])
        st.success("Stopped all servers")
        time.sleep(1)
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔌 Connect Clients")
    st.markdown("**1. VS Code (Continue)**")
    st.code(f"""
"apiBase": "http://localhost:9000/v1",
"model": "{config['models'][0]['repo_id']}"
    """, language="json")
    
    st.markdown("**2. Desktop App**")
    st.markdown("[Download Chatbox AI](https://chatboxai.app)")
    st.caption("Set API Host to `http://localhost:9000`")

# Tabs for Admin vs Chat
tab_admin, tab_chat, tab_add = st.tabs(["⚙️ Server Management", "💬 Web Chat", "➕ Add Model"])

with tab_admin:
    st.subheader("Managed Models")

    cols = st.columns(len(config["models"]))

    for idx, model in enumerate(config["models"]):
        with cols[idx]:
            running = is_running(model["port"])
            status_color = "green" if running else "red"
            status_text = "ONLINE" if running else "OFFLINE"
            
            st.markdown(f"### {model['name']}")
            st.caption(f"Port: `{model['port']}`")
            st.markdown(f"Status: :{status_color}[**{status_text}**]")
            
            if running:
                if st.button(f"Stop", key=f"stop_{model['id']}"):
                    stop_server(model["port"])
                    st.rerun()
                
                # Logs Viewer
                with st.expander("📄 Logs"):
                    log_path = os.path.join(LOGS_DIR, f"{model['id']}.log")
                    if os.path.exists(log_path):
                        with open(log_path, "r") as f:
                            lines = f.readlines()[-20:]
                            st.code("".join(lines), language="text")
                    else:
                        st.write("No logs yet.")
            else:
                if st.button(f"Start Server", key=f"start_{model['id']}"):
                    start_server(model)
                    time.sleep(2) # Give it a moment
                    st.rerun()

    st.markdown("---")
    st.caption("Edit `models.yaml` to add more models.")

with tab_chat:
    # Find running models
    running_models = [m for m in config["models"] if is_running(m["port"])]
    
    if not running_models:
        st.warning("⚠️ No models are currently running. Please start a server in the 'Server Management' tab.")
    else:
        # Model Selector
        selected_model_name = st.selectbox(
            "Select Model", 
            options=[m["name"] for m in running_models],
            index=0
        )
        selected_model = next(m for m in running_models if m["name"] == selected_model_name)
        
        # Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display Chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
        # Chat Input
        if prompt := st.chat_input("Say something..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Stream response
                    with requests.post(
                        f"http://localhost:{selected_model['port']}/v1/chat/completions",
                        json={
                            "model": selected_model["repo_id"],
                            "messages": st.session_state.messages,
                            "stream": True
                        },
                        stream=True,
                        timeout=60
                    ) as r:
                        for line in r.iter_lines():
                            if line:
                                line = line.decode('utf-8')
                                if line.startswith("data: ") and line != "data: [DONE]":
                                    import json
                                    try:
                                        json_str = line[6:] # Skip "data: "
                                        data = json.loads(json_str)
                                        if "content" in data['choices'][0]['delta']:
                                            content = data['choices'][0]['delta']['content']
                                            full_response += content
                                            message_placeholder.markdown(full_response + "▌")
                                    except:
                                        pass
                    
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error: {e}")

with tab_add:
    st.header("Add New MLX Model")
    
    # Search HF
    search_query = st.text_input("Search Hugging Face (tag: mlx)", value="mlx-community")
    
    @st.cache_data(ttl=300)
    def search_models(query):
        api = HfApi()
        models = api.list_models(search=query, filter="mlx", sort="downloads", direction=-1, limit=50)
        return [m.modelId for m in models]

    if search_query:
        try:
            model_options = search_models(search_query)
            selected_repo = st.selectbox("Select Repository", model_options)
            
            if selected_repo:
                st.markdown("---")
                st.subheader("Configuration")
                
                with st.form("add_model_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        # Default ID: clean up the repo name
                        default_id = selected_repo.split("/")[-1].lower().replace(".", "-")
                        new_id = st.text_input("ID (slug)", value=default_id)
                        new_name = st.text_input("Display Name", value=selected_repo.split("/")[-1])
                    with col2:
                        # Suggest next available port
                        used_ports = [m["port"] for m in config["models"]]
                        next_port = max(used_ports) + 1 if used_ports else 9000
                        new_port = st.number_input("Port", value=next_port, step=1)
                        new_desc = st.text_input("Description", value="Imported from Hugging Face")
                    
                    submitted = st.form_submit_button("Add Model")
                    
                    if submitted:
                        new_model = {
                            "id": new_id,
                            "name": new_name,
                            "repo_id": selected_repo,
                            "port": int(new_port),
                            "description": new_desc
                        }
                        
                        # Append and Save
                        config["models"].append(new_model)
                        with open(MODELS_FILE, "w") as f:
                            yaml.dump(config, f, sort_keys=False)
                        
                        st.success(f"Added {new_name}! Refreshing...")
                        time.sleep(1)
                        st.rerun()
        except Exception as e:
            st.error(f"Failed to search Hugging Face: {e}")

