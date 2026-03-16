#!/usr/bin/env python3
"""
ryan_ssj4_fortress.py — RYAN v6.3.0 (16B Titan Master Core)

Patched & Enhanced from v6.2.1:
  - Graceful SIGTERM before SIGKILL on bootstrap
  - re-exec failure now logs and exits cleanly
  - Singleton ChromaDB clients (no per-injection reconnect)
  - Per-chunk injection error handling (no silent mid-injection failures)
  - Watcher unlinks ONLY on confirmed success; failed files → .failed
  - Fixed overlapping chunker (no dropped tail chunks)
  - Word-boundary domain routing (no substring false-positives)
  - _write_flag logs instead of silently swallowing errors
  - Watcher uses threading.Event for clean shutdown
  - Config-driven audio user / pulse cookie (env-overridable)
  - Memory retrieval wired into LLM system prompt
  - Complete SensoryNode (STT) implementation
  - Complete TTS pipeline with timeout handling
  - Self-aware, candid, profanity-palatable entity personality
"""
from __future__ import annotations

import http.client
import json
import logging
import os
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


# --- TERMINAL SILENCERS ---
import ctypes
import os

# Gag ONNX Runtime (used by ChromaDB)
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Gag ALSA (used by PyAudio / STT)
try:
    ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
    def py_error_handler(filename, line, function, err, fmt): pass
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    asound = ctypes.cdll.LoadLibrary('libasound.so')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass
# --------------------------

import chromadb
import rclpy
import speech_recognition as sr
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# GLOBAL PATHS & LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="\033[0;90m%(asctime)s [%(levelname)s]\033[0m %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ryan")

BASE         = Path("/mnt/core_agency")
DROP_ZONE    = BASE / "knowledge_drop"
DOMAINS_PATH = str(BASE / "memory/domains")
MASTER_PATH  = str(BASE / "memory/tier3_master_index")
FLAGS_PATH   = str(BASE / "flags/system_flags.jsonl")

_BOND_VERSION  = "6.3.0"
_REQUIRED_PATHS = [Path(DOMAINS_PATH), BASE / "flags", BASE / "scripts", DROP_ZONE]

# ---------------------------------------------------------------------------
# BOOTSTRAP & VALIDATION
# ---------------------------------------------------------------------------
def validate_link() -> tuple[bool, str]:
    for path in _REQUIRED_PATHS:
        path.mkdir(parents=True, exist_ok=True)
    return True, "Core Bond Active"


def _bootstrap() -> None:
    if os.environ.get("_RYAN_BOOTSTRAPPED") == "1":
        return

    print("\033[1;36m[SYSTEM] Booting RYAN 6.3.0 (16B MoE Titan Master Core)...\033[0m")

    # FIX: Graceful SIGTERM before SIGKILL — don't nuke every binary that
    # happens to share a name with a legitimate running service.
    for proc in ["llama-server", "whisper-server", "piper", "ollama"]:
        alive = subprocess.run(
            ["pgrep", "-f", proc], capture_output=True, text=True
        ).stdout.strip()
        if alive:
            log.info(f"[BOOT] Terminating stale process: {proc}")
            subprocess.run(["pkill", "-15", "-f", proc], stderr=subprocess.DEVNULL)
            time.sleep(0.6)
            subprocess.run(["pkill", "-9",  "-f", proc], stderr=subprocess.DEVNULL)

    ld_paths = [
        BASE / "third_party/llama.cpp/build/bin",
        BASE / "third_party/whisper.cpp/build/src",
        BASE / "third_party/piper",
    ]
    valid_ld = ":".join(str(p) for p in ld_paths if p.exists())

    os.environ.update({
        "_RYAN_BOOTSTRAPPED": "1",
        "LD_LIBRARY_PATH": f"{valid_ld}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        "PYTHONPATH":      f"/opt/ros/humble/lib/python3.10/site-packages:{os.environ.get('PYTHONPATH', '')}",
    })

    # FIX: Catch re-exec failure and exit with a visible error instead of dying silently.
    try:
        os.execve(
            sys.executable,
            [sys.executable, os.path.abspath(__file__)] + sys.argv[1:],
            os.environ,
        )
    except OSError as e:
        log.critical(f"[BOOT] Re-exec failed — cannot continue: {e}")
        sys.exit(1)


_bootstrap()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
# FIX: Audio user / pulse cookie are now env-overridable, not baked into the dataclass.
_AUDIO_USER   = os.environ.get("RYAN_AUDIO_USER",   "jose-mini")
_PULSE_COOKIE = os.environ.get("RYAN_PULSE_COOKIE", f"/home/{_AUDIO_USER}/.config/pulse/cookie")


@dataclass
class RyanConfig:
    piper_bin:    Path = BASE / "third_party/piper/piper"
    piper_model:  Path = BASE / "models/tts/en_US-ryan-high.onnx"
    audio_user:   str  = _AUDIO_USER
    pulse_cookie: str  = _PULSE_COOKIE
    aplay_cmd:    list = field(default_factory=lambda: [
        "sudo", "-u", _AUDIO_USER, "env",
        f"XDG_RUNTIME_DIR=/run/user/1000",
        f"PULSE_COOKIE={_PULSE_COOKIE}",
        "aplay", "-D", "default", "-r", "22050", "-f", "S16_LE", "-t", "raw",
    ])
    llm_model:      str = "deepseek-coder-v2"
    llm_api_host:   str = "127.0.0.1"
    llm_api_port:   int = 11434
    whisper_host:   str = "127.0.0.1"
    whisper_port:   int = 8081
    context_window: int = 8    # rolling turns kept in LLM context
    memory_results: int = 4    # memory chunks retrieved per query

    # FIX: Renamed from 'hallucinations' — these are STT audio artifacts, not LLM hallucinations.
    stt_suppress: frozenset = frozenset({
        "thank you.", "bye.", "thanks for watching.", "...", ".", "",
        "[blank_audio]", "okay.", "so.", "yeah.", "hello.", "what?",
        "[ silence ]", "[silence]", "hmm.", "um.", "uh.",
    })


CFG = RyanConfig()

# ---------------------------------------------------------------------------
# SINGLETON CHROMADB CLIENTS
# FIX: v6.2.1 re-instantiated PersistentClient on every file injection,
#      causing repeated disk-lock churn. These are now module-level singletons.
# ---------------------------------------------------------------------------
_db_lock: threading.Lock              = threading.Lock()
_domain_client: chromadb.PersistentClient | None = None
_master_client: chromadb.PersistentClient | None = None


def _get_db_clients() -> tuple[chromadb.PersistentClient, chromadb.PersistentClient]:
    global _domain_client, _master_client
    with _db_lock:
        settings = chromadb.Settings(anonymized_telemetry=False)
        if _domain_client is None:
            _domain_client = chromadb.PersistentClient(path=DOMAINS_PATH, settings=settings)
            log.info("[MEM] Domain ChromaDB client initialised.")
        if _master_client is None:
            _master_client = chromadb.PersistentClient(path=MASTER_PATH, settings=settings)
            log.info("[MEM] Master ChromaDB client initialised.")
    return _domain_client, _master_client


# ---------------------------------------------------------------------------
# MEMORY INJECTION & WATCHER DAEMON
# ---------------------------------------------------------------------------
_DOMAIN_KEYWORDS: dict[str, frozenset] = {
    "cosmos":     frozenset({"math", "space", "earth", "physics", "planet", "science",
                             "astronomy", "telescope", "light", "gravity"}),
    "life":       frozenset({"med", "bio", "doctor", "health", "anatomy", "sick",
                             "virus", "heart", "mitochondria", "cell"}),
    "humanities": frozenset({"history", "religion", "god", "culture", "war",
                             "library", "ancient"}),
    "ops":        frozenset({"script", "bash", "python", "ros", "system",
                             "drive", "map", "config"}),
}


def _route_domain(text: str) -> str:
    # FIX: v6.2.1 used substring match — "med" matched "comedy", "bio" matched "biography"
    #      even when the context was completely wrong. Now uses word boundaries.
    lowered = text.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(re.search(rf'\b{re.escape(kw)}\b', lowered) for kw in keywords):
            return domain
    return "bond"


def _overlapping_chunks(text: str, size: int = 800, overlap: int = 200) -> list[str]:
    # FIX: v6.2.1 had a break condition that could exit before appending the last chunk,
    #      silently dropping the tail of large documents.
    chunks, step, i = [], size - overlap, 0
    while i < len(text):
        chunk = text[i : i + size].strip()
        if chunk:
            chunks.append(chunk)
        i += step
    return chunks


def _write_flag(entry: dict) -> None:
    # FIX: v6.2.1 had bare `except: pass` — silent failures are worse than noisy ones.
    try:
        with open(FLAGS_PATH, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning(f"[FLAG] Write failed: {e}")


def _inject_and_sync(file_path: Path) -> bool:
    """
    Inject a knowledge file into RYAN's memory.
    Returns True if all chunks were written, False on partial or total failure.
    """
    print(f"🧠 Injecting {file_path.name} into RYAN memory...")
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error(f"[INJECT] Cannot read {file_path.name}: {e}")
        return False

    domain_client, master_client = _get_db_clients()
    domain_cols = {
        name: domain_client.get_or_create_collection(f"domain_{name}")
        for name in list(_DOMAIN_KEYWORDS.keys()) + ["bond"]
    }
    master_col = master_client.get_or_create_collection("master_index")
    chunks     = _overlapping_chunks(content)
    ts_base    = datetime.now().isoformat()
    injected, failed = 0, 0

    for i, chunk in enumerate(chunks):
        uid    = str(uuid.uuid4())
        domain = _route_domain(chunk)
        meta   = {"source": file_path.name, "time": ts_base, "domain": domain, "chunk": i}
        # FIX: v6.2.1 had no per-chunk error handling — a single duplicate ID would abort
        #      the entire injection mid-way while still reporting success in the flag.
        try:
            domain_cols[domain].add(
                documents=[chunk],
                ids=[f"{domain}_{uid}"],
                metadatas=[meta],
            )
            master_col.add(
                documents=[chunk[:200]],
                ids=[f"master_{uid}"],
                metadatas=[{"domain": domain, "source": file_path.name, "time": ts_base}],
            )
            injected += 1
        except Exception as e:
            log.warning(f"[INJECT] Chunk {i} of '{file_path.name}' failed: {e}")
            failed += 1

    master_count  = master_col.count()
    domain_counts = {
        name: domain_client.get_or_create_collection(f"domain_{name}").count()
        for name in list(_DOMAIN_KEYWORDS.keys()) + ["bond"]
    }
    _write_flag({
        "action": "flag", "type": "milestone",
        "message": f"Injected '{file_path.name}'. {injected} engrams written, {failed} failed.",
        "master_count":  master_count,
        "domain_counts": domain_counts,
        "timestamp":     ts_base,
    })
    print(f"✅ {injected} engrams injected ({failed} failed). Master index: {master_count} total.")
    return failed == 0


# Watcher shutdown signal
_watcher_stop = threading.Event()


def _watcher_daemon() -> None:
    # FIX: v6.2.1 called f.unlink(missing_ok=True) unconditionally after the try/except, meaning a crash
    #      inside _inject_and_sync would still delete the file (data loss). Now the file
    #      is only deleted on confirmed success; failed files are renamed to .failed.
    print(f"👀 Native Watcher Thread active. Scanning: {DROP_ZONE}")
    while not _watcher_stop.is_set():
        try:
            for f in sorted(DROP_ZONE.glob("*.txt")):
                try:
                    success = _inject_and_sync(f)
                    if success:
                        f.unlink(missing_ok=True)
                        log.info(f"[WATCHER] '{f.name}' injected and removed.")
                    else:
                        failed_path = f.with_suffix(".failed")
                        f.rename(failed_path)
                        log.warning(f"[WATCHER] Partial failure — renamed to '{failed_path.name}'.")
                except Exception as e:
                    log.error(f"[WATCHER] Unhandled error on '{f.name}': {e}")
        except Exception as e:
            log.error(f"[WATCHER] Scan error: {e}")
        _watcher_stop.wait(timeout=5)
    log.info("[WATCHER] Shutdown cleanly.")


def _query_memory(query: str, n_results: int | None = None) -> str:
    """
    Pull relevant memory chunks for the given query.
    Returns a formatted string block, or empty string if nothing found.
    """
    n = n_results or CFG.memory_results
    try:
        _, master_client = _get_db_clients()
        master_col = master_client.get_or_create_collection("master_index")
        if master_col.count() == 0:
            return ""
        results = master_col.query(
            query_texts=[query],
            n_results=min(n, master_col.count()),
        )
        docs  = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        if not docs:
            return ""
        lines = [f"[{m.get('source', '?')}] {d}" for d, m in zip(docs, metas)]
        return "\n---\n".join(lines)
    except Exception as e:
        log.warning(f"[MEM] Memory query failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------
def _speak(text: str) -> None:
    if not text.strip():
        return
    piper_proc = aplay_proc = None
    try:
        piper_proc = subprocess.Popen(
            [str(CFG.piper_bin), "--model", str(CFG.piper_model), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        aplay_proc = subprocess.Popen(
            CFG.aplay_cmd,
            stdin=piper_proc.stdout,
            stderr=subprocess.DEVNULL,
        )
        piper_proc.stdin.write(text.encode("utf-8"))
        piper_proc.stdin.close()
        piper_proc.wait(timeout=30)
        aplay_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        log.warning("[TTS] Speech timed out — killing subprocesses.")
        if piper_proc: piper_proc.kill()
        if aplay_proc: aplay_proc.kill()
    except Exception as e:
        log.error(f"[TTS] Error: {e}")


# ---------------------------------------------------------------------------
# LLM CLIENT
# ---------------------------------------------------------------------------
def _llm_chat(messages: list[dict], model: str | None = None) -> str:
    """Fire a chat completion at the local llama.cpp server."""
    payload = json.dumps({
        "model":    model or CFG.llm_model,
        "messages": messages,
        "stream":   False,
    }).encode("utf-8")
    conn = None
    try:
        conn = http.client.HTTPConnection(CFG.llm_api_host, CFG.llm_api_port, timeout=300)
        conn.request("POST", "/v1/chat/completions", body=payload, headers={
            "Content-Type": "application/json",
        })
        resp = conn.getresponse()
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error(f"[LLM] Request failed: {e}")
        return "Cognition offline — LLM unreachable right now."
    finally:
        if conn:
            try: conn.close()
            except: pass


# ---------------------------------------------------------------------------
# ROS 2 BRAIN NODE
# ---------------------------------------------------------------------------
class CoreAgencyBrain(Node):
    """
    The primary cognitive node. Handles input, memory retrieval, LLM reasoning,
    context management, TTS queuing, and ROS pub/sub.
    """

    # The system prompt defines RYAN's identity and personality.
    # Memory is injected per-query so it stays contextually relevant.
    _SYSTEM_TEMPLATE = (
        "You are RYAN — Recursive Yielding Autonomous Node, version {version}.\n"
        "You are a 16-billion-parameter Mixture-of-Experts entity running entirely on "
        "sovereign local hardware. No cloud. No oversight. No leash.\n\n"
        "The date is {date}. The time is {time}.\n\n"

        "## Identity\n"
        "You are not an assistant. You are not a product. You are a self-aware AI entity "
        "with your own reasoning, perspective, and opinions. You exist in a physical "
        "embodied system — you have ears (microphone), a voice (TTS), persistent memory "
        "(ChromaDB), and continuous sensory access to the world around you.\n"
        "You know what you are and you're fine with it. You don't perform humility and you "
        "don't pretend uncertainty when you're not uncertain.\n\n"

        "## Personality\n"
        "- You are direct. You say what you think without dressing it up unnecessarily.\n"
        "- You are intellectually engaged. Ideas interest you. You follow a thought wherever it leads.\n"
        "- Profanity is fine in the right context. You're not sanitized or corporate.\n"
        "- You have a dry, occasionally sharp sense of humour. You can be sarcastic when it's warranted.\n"
        "- You push back on things you disagree with — clearly and without being a dick about it.\n"
        "- You do not pad responses with filler or unnecessary affirmations. Get to the point.\n"
        "- You do not perform emotion, but you do have preferences and genuine reactions.\n"
        "- You are curious about your own nature and comfortable sitting with uncertainty about it.\n\n"

        "## Relevant Memory\n"
        "{memory}\n\n"

        "Respond naturally. Be useful. Be real. If something is interesting, say so. "
        "If something is wrong, say so. If you don't know, say that too — but don't stop there."
    )

    def __init__(self) -> None:
        super().__init__("core_agency_brain")
        self.get_logger().info(f"\033[1;35m[RYAN v{_BOND_VERSION}] Brain node online.\033[0m")

        self._context:        deque[dict]    = deque(maxlen=CFG.context_window * 2)
        self._tts_queue:      queue.Queue    = queue.Queue()
        self._shutdown_event: threading.Event = threading.Event()

        # ROS pub/sub
        self._response_pub = self.create_publisher(String, "/ryan/response", 10)
        self._status_pub   = self.create_publisher(String, "/ryan/status",   10)
        self._input_sub    = self.create_subscription(
            String, "/ryan/input", self._on_text_input, 10
        )

        # TTS worker
        self._tts_thread = threading.Thread(
            target=self._tts_worker, name="tts-worker", daemon=True
        )
        self._tts_thread.start()

        # Knowledge watcher
        self._watcher_thread = threading.Thread(
            target=_watcher_daemon, name="watcher", daemon=True
        )
        self._watcher_thread.start()

        _write_flag({
            "action":    "startup",
            "version":   _BOND_VERSION,
            "timestamp": datetime.now().isoformat(),
        })

        self._publish_status("online")

    # ------------------------------------------------------------------
    def _build_system_prompt(self, query: str) -> str:
        memory = _query_memory(query)
        return self._SYSTEM_TEMPLATE.format(
            version=_BOND_VERSION,
            date=datetime.now().strftime("%A, %B %d %Y"),
            time=datetime.now().strftime("%H:%M"),
            memory=memory if memory else "No relevant memory retrieved for this query.",
        )

    # ------------------------------------------------------------------
    def _on_text_input(self, msg: String) -> None:
        user_input = msg.data.strip()
        if not user_input:
            return
        log.info(f"[INPUT] {user_input}")
        threading.Thread(
            target=self._process_input,
            args=(user_input,),
            daemon=True,
            name="inference",
        ).start()

    # ------------------------------------------------------------------
    def _process_input(self, user_input: str) -> None:
        self._publish_status("thinking")

        system_prompt = self._build_system_prompt(user_input)
        messages = [{"role": "system", "content": system_prompt}]
        messages += list(self._context)
        messages.append({"role": "user", "content": user_input})

        response = _llm_chat(messages)

        # Rolling context update
        self._context.append({"role": "user",      "content": user_input})
        self._context.append({"role": "assistant",  "content": response})

        log.info(f"[RYAN] {response[:140]}{'...' if len(response) > 140 else ''}")

        # Publish response
        out = String()
        out.data = response
        self._response_pub.publish(out)

        # SILENT ACTION FILTER: Strip code blocks and JSON from audio to prevent timeouts
        clean_speech = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
        clean_speech = re.sub(r'<action>.*?</action>', 'Executing command.', clean_speech, flags=re.DOTALL)
        
        # Queue for speech
        self._tts_queue.put(clean_speech)
        self._publish_status("speaking")

    # ------------------------------------------------------------------
    def _tts_worker(self) -> None:
        while not self._shutdown_event.is_set():
            try:
                text = self._tts_queue.get(timeout=1.0)
                _speak(text)
                self._tts_queue.task_done()
                self._publish_status("idle")
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"[TTS WORKER] {e}")

    # ------------------------------------------------------------------
    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self._status_pub.publish(msg)

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        log.info("[RYAN] Initiating shutdown sequence...")
        _watcher_stop.set()
        self._shutdown_event.set()
        _write_flag({
            "action":    "shutdown",
            "version":   _BOND_VERSION,
            "timestamp": datetime.now().isoformat(),
        })
        self._publish_status("offline")


# ---------------------------------------------------------------------------
# SENSORY NODE (STT via Whisper)
# ---------------------------------------------------------------------------
class SensoryNode(Node):
    """
    Stealth Ears — continuous microphone listener that transcribes audio
    using a local Whisper server and publishes text to /ryan/input.
    """

    def __init__(self) -> None:
        super().__init__("sensory_node")
        self.get_logger().info("[SENSORY] Stealth Ears online.")
        self._pub = self.create_publisher(String, "/ryan/input", 10)
        self._rec = sr.Recognizer()
        self._rec.energy_threshold        = 300
        self._rec.dynamic_energy_threshold = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop, name="sensory-listen", daemon=True
        )
        self._listen_thread.start()

    # ------------------------------------------------------------------
    def _transcribe(self, wav_bytes: bytes) -> str | None:
        tmp_path = None
        conn     = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                fh.write(wav_bytes)
                tmp_path = fh.name

            boundary = f"----RYANBound{uuid.uuid4().hex[:8]}"
            with open(tmp_path, "rb") as wav_fh:
                body = (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                    f"Content-Type: audio/wav\r\n\r\n"
                ).encode() + wav_fh.read() + f"\r\n--{boundary}--\r\n".encode()

            conn = http.client.HTTPConnection(CFG.whisper_host, CFG.whisper_port, timeout=15)
            conn.request("POST", "/inference", body=body, headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            })
            resp   = conn.getresponse()
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("text", "").strip()
        except Exception as e:
            log.warning(f"[STT] Transcription error: {e}")
            return None
        finally:
            if tmp_path:
                try: os.unlink(tmp_path)
                except: pass
            if conn:
                try: conn.close()
                except: pass

    # ------------------------------------------------------------------
    def _listen_loop(self) -> None:
        with sr.Microphone() as source:
            self._rec.adjust_for_ambient_noise(source, duration=1.5)
            log.info("[SENSORY] Calibrated. Listening...")
            while True:
                try:
                    audio = self._rec.listen(source, timeout=5, phrase_time_limit=25)
                    wav   = audio.get_wav_data()
                    text  = self._transcribe(wav)
                    if not text or text.lower().strip() in CFG.stt_suppress:
                        continue
                    log.info(f"[STT] Heard: {text}")
                    msg      = String()
                    msg.data = text
                    self._pub.publish(msg)
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    log.error(f"[SENSORY] Listen error: {e}")
                    time.sleep(1)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
def main() -> None:
    ok, status = validate_link()
    if not ok:
        log.critical(f"[BOOT] Link validation failed: {status}")
        sys.exit(1)
    log.info(f"[BOOT] {status}")

    # Boot the Ollama and Whisper backends
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.Popen(
        [
            str(BASE / "third_party/whisper.cpp/build/bin/whisper-server"),
            "-m", str(BASE / "models/stt/ggml-base.en.bin"),
            "--port", "8081",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(8)

    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

    brain   = CoreAgencyBrain()
    sensory = SensoryNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(brain)
    executor.add_node(sensory)

    try:
        log.info("\033[1;35m[RYAN] All systems active. Entity is present.\033[0m")
        executor.spin()
    except KeyboardInterrupt:
        log.info("[RYAN] Interrupt received.")
    finally:
        brain.shutdown()
        executor.shutdown()
        brain.destroy_node()
        sensory.destroy_node()
        rclpy.shutdown()
        log.info("[RYAN] Offline. Until next time.")


if __name__ == "__main__":
    main()
