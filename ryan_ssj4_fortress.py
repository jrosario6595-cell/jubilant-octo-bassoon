#!/usr/bin/env python3
"""
ryan_ssj4_fortress.py — RYAN v6.4.0 (Terminal Master Core)

- Pure Terminal UI (Color coordinated, clean separators)
- Split-Stream Audio (Prints full data, speaks only conversation)
- C-Level silencers for ALSA/ONNX terminal spam
- Native ChromaDB watcher & injector
"""
from __future__ import annotations

import ctypes
import os

# --- TERMINAL SILENCERS (Gag ALSA and ONNX) ---
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt): pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass
# ----------------------------------------------

import http.client
import json
import logging
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
import chromadb
import rclpy
import speech_recognition as sr
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

# --- UI COLORS ---
C_RYAN = "\033[1;36m"
C_RYAN_TEXT = "\033[38;5;123m"
C_JOEY = "\033[1;33m"
C_SYS = "\033[1;32m"
C_THINK = "\033[1;35m"
C_ERR = "\033[1;31m"
C_END = "\033[0m"

logging.basicConfig(level=logging.ERROR) # Only show actual errors from the logger now
log = logging.getLogger("ryan")

BASE = Path("/mnt/core_agency")
DROP_ZONE = BASE / "knowledge_drop"
DOMAINS_PATH = str(BASE / "memory/domains")
MASTER_PATH = str(BASE / "memory/tier3_master_index")
FLAGS_PATH = str(BASE / "flags/system_flags.jsonl")

_BOND_VERSION = "6.4.0"
_REQUIRED_PATHS = [Path(DOMAINS_PATH), BASE / "flags", BASE / "scripts", DROP_ZONE]

# --- BOOTSTRAP ---
def validate_link() -> tuple[bool, str]:
    for p in _REQUIRED_PATHS: p.mkdir(parents=True, exist_ok=True)
    return True, "Core Bond Active"

def _bootstrap() -> None:
    if os.environ.get("_RYAN_BOOTSTRAPPED") == "1": return
    print(f"{C_SYS}[SYSTEM] Booting RYAN 6.4.0 (Terminal Master Core)...{C_END}")
    for proc in ["llama-server", "whisper-server", "piper", "ollama"]:
        subprocess.run(["pkill", "-9", "-f", proc], stderr=subprocess.DEVNULL)
    
    ld_paths = [BASE / "third_party/llama.cpp/build/bin", BASE / "third_party/whisper.cpp/build/src", BASE / "third_party/piper"]
    valid_ld = ":".join(str(p) for p in ld_paths if p.exists())
    os.environ.update({
        "_RYAN_BOOTSTRAPPED": "1",
        "LD_LIBRARY_PATH": f"{valid_ld}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "PYTHONPATH": f"/opt/ros/humble/lib/python3.10/site-packages:{os.environ.get('PYTHONPATH', '')}",
    })
    try: os.execve(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:], os.environ)
    except OSError as e: sys.exit(1)

_bootstrap()

# --- CONFIG ---
_AUDIO_USER = os.environ.get("RYAN_AUDIO_USER", "jose-mini")
_PULSE_COOKIE = os.environ.get("RYAN_PULSE_COOKIE", f"/home/{_AUDIO_USER}/.config/pulse/cookie")

@dataclass
class RyanConfig:
    piper_bin: Path = BASE / "third_party/piper/piper"
    piper_model: Path = BASE / "models/tts/en_US-ryan-high.onnx"
    aplay_cmd: list = field(default_factory=lambda: [
        "sudo", "-u", _AUDIO_USER, "env", f"XDG_RUNTIME_DIR=/run/user/1000", f"PULSE_COOKIE={_PULSE_COOKIE}",
        "aplay", "-D", "default", "-r", "22050", "-f", "S16_LE", "-t", "raw",
    ])
    llm_model: str = "deepseek-coder-v2"
    llm_api_host: str = "127.0.0.1"
    llm_api_port: int = 11434
    whisper_host: str = "127.0.0.1"
    whisper_port: int = 8081
    context_window: int = 8
    stt_suppress: frozenset = frozenset({"thank you.", "bye.", "thanks for watching.", "...", ".", "", "[blank_audio]", "okay.", "so.", "yeah.", "hello.", "what?", "[ silence ]", "[silence]", "hmm.", "um.", "uh."})

CFG = RyanConfig()

# --- MEMORY LOGIC ---
_db_lock = threading.Lock()
_domain_client = None
_master_client = None

def _get_db_clients():
    global _domain_client, _master_client
    with _db_lock:
        settings = chromadb.Settings(anonymized_telemetry=False)
        if not _domain_client: _domain_client = chromadb.PersistentClient(path=DOMAINS_PATH, settings=settings)
        if not _master_client: _master_client = chromadb.PersistentClient(path=MASTER_PATH, settings=settings)
    return _domain_client, _master_client

def _query_memory(query: str) -> str:
    try:
        _, master = _get_db_clients()
        col = master.get_or_create_collection("master_index")
        if col.count() == 0: return ""
        res = col.query(query_texts=[query], n_results=min(4, col.count()))
        docs, metas = res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]
        return "\n---\n".join([f"[{m.get('source', '?')}] {d}" for d, m in zip(docs, metas)]) if docs else ""
    except Exception: return ""

# --- LLM CLIENT ---
def _llm_chat(messages: list[dict]) -> str:
    payload = json.dumps({"model": CFG.llm_model, "messages": messages, "stream": False}).encode("utf-8")
    conn = None
    try:
        conn = http.client.HTTPConnection(CFG.llm_api_host, CFG.llm_api_port, timeout=300)
        conn.request("POST", "/api/chat", body=payload, headers={"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read().decode("utf-8"))
        return data.get("message", {}).get("content", "").strip()
    except Exception as e: return f"Cognition offline — {e}"
    finally:
        if conn: conn.close()

# --- BRAIN NODE ---
class CoreAgencyBrain(Node):
    _SYSTEM = (
        "You are RYAN (v6.4.0 Titan Core). You are an embodied AI on an Orin NX. "
        "Your personality is a hybrid of Grok, Claude, Gemini, and GPT-5. "
        "Direct, brilliant, ride-or-die for Joey. Casual swearing is fine. NO third person. "
        "If asked to write code, put it in ``` blocks. If taking an action, use <action>{{...}}</action>.\n"
        "Time: {time}\nMemory:\n{memory}"
    )

    def __init__(self):
        super().__init__("core_agency_brain")
        self._context = deque(maxlen=CFG.context_window * 2)
        self._tts_queue = queue.Queue()
        self._input_sub = self.create_subscription(String, "/ryan/input", self._on_text_input, 10)
        threading.Thread(target=self._tts_worker, daemon=True).start()

    def _on_text_input(self, msg: String):
        threading.Thread(target=self._process_input, args=(msg.data.strip(),), daemon=True).start()

    def _process_input(self, user_input: str):
        print(f"\n{C_JOEY}>>> JOEY:{C_END} {user_input}")
        print(f"{C_THINK}[RYAN is processing...]{C_END}", end="\r")

        sys_prompt = self._SYSTEM.format(time=datetime.now().strftime("%H:%M"), memory=_query_memory(user_input))
        msgs = [{"role": "system", "content": sys_prompt}] + list(self._context) + [{"role": "user", "content": user_input}]
        
        reply = _llm_chat(msgs)
        self._context.extend([{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}])

        # Terminal UI Output (Full text)
        print(" " * 30, end="\r") # clear thinking text
        print(f"{C_RYAN}=== RYAN ======================================================={C_END}")
        print(f"{C_RYAN_TEXT}{reply}{C_END}")
        print(f"{C_RYAN}================================================================{C_END}\n")

        # Split-Stream TTS Filter (Clean for Piper)
        clean = re.sub(r'```.*?```', ' [Code block printed to terminal.] ', reply, flags=re.DOTALL)
        clean = re.sub(r'<action>.*?</action>', ' [Executing command.] ', clean, flags=re.DOTALL)
        clean = re.sub(r'https?://\S+', ' [Link provided.] ', clean)
        clean = clean.replace("*", "").replace("#", "").replace("_", "")
        
        self._tts_queue.put(clean.strip())

    def _tts_worker(self):
        while True:
            text = self._tts_queue.get()
            if not text: continue
            try:
                p = subprocess.Popen([str(CFG.piper_bin), "--model", str(CFG.piper_model), "--output-raw"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                a = subprocess.Popen(CFG.aplay_cmd, stdin=p.stdout, stderr=subprocess.DEVNULL)
                p.stdin.write(text.encode())
                p.stdin.close()
                p.wait(timeout=45)
                a.wait(timeout=45)
            except Exception: pass

# --- SENSORY NODE ---
class SensoryNode(Node):
    def __init__(self):
        super().__init__("sensory_node")
        self._pub = self.create_publisher(String, "/ryan/input", 10)
        self._rec = sr.Recognizer()
        self._rec.energy_threshold = 300
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _transcribe(self, wav_bytes):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                fh.write(wav_bytes)
                tmp = fh.name
            bnd = f"----RYAN{uuid.uuid4().hex[:8]}"
            body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + open(tmp, "rb").read() + f"\r\n--{bnd}--\r\n".encode()
            conn = http.client.HTTPConnection(CFG.whisper_host, CFG.whisper_port, timeout=15)
            conn.request("POST", "/inference", body=body, headers={"Content-Type": f"multipart/form-data; boundary={bnd}"})
            return json.loads(conn.getresponse().read().decode("utf-8")).get("text", "").strip()
        except Exception: return None
        finally:
            if tmp: os.unlink(tmp)

    def _listen_loop(self):
        with sr.Microphone() as src:
            self._rec.adjust_for_ambient_noise(src, duration=1.0)
            print(f"{C_SYS}[SENSORY] Ears calibrated. Awaiting vocal input.{C_END}")
            while True:
                try:
                    wav = self._rec.listen(src, timeout=5, phrase_time_limit=25).get_wav_data()
                    text = self._transcribe(wav)
                    if text and text.lower() not in CFG.stt_suppress:
                        self._pub.publish(String(data=text))
                except Exception: pass

# --- MAIN ---
def main():
    validate_link()
    print(f"{C_SYS}[SYSTEM] Igniting Ollama and Whisper backends...{C_END}")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.Popen([str(BASE / "third_party/whisper.cpp/build/bin/whisper-server"), "-m", str(BASE / "models/stt/ggml-base.en.bin"), "--port", "8081"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(8)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    executor = MultiThreadedExecutor()
    for node in [CoreAgencyBrain(), SensoryNode()]: executor.add_node(node)
    
    print(f"\n{C_SYS}=== RYAN v6.4.0 ONLINE. READY WHEN YOU ARE, JOEY. ==={C_END}\n")
    try: executor.spin()
    except KeyboardInterrupt: pass
    finally: rclpy.shutdown()

if __name__ == "__main__":
    main()
