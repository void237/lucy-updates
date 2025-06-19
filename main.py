import os
import sys
import requests
import shutil
import threading
import time
import asyncio
import json
import re
import difflib
import uuid
import pygame
import edge_tts
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from gpt4all import GPT4All
import wikipedia

# ---------- Self-Update Config ----------
UPDATE_CHECK_URL = "https://your-update-server.com/lucy/latest_version.txt"
UPDATE_SCRIPT_URL = "https://your-update-server.com/lucy/main.py"
LOCAL_VERSION_FILE = "version.txt"

def get_local_version():
    if os.path.exists(LOCAL_VERSION_FILE):
        with open(LOCAL_VERSION_FILE, "r") as f:
            return f.read().strip()
    return None

def write_local_version(version):
    with open(LOCAL_VERSION_FILE, "w") as f:
        f.write(version)

def download_update():
    try:
        print("[Updater] Downloading update...")
        response = requests.get(UPDATE_SCRIPT_URL, timeout=10)
        if response.status_code == 200:
            with open("main_new.py", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("[Updater] Update downloaded successfully.")
            return True
        else:
            print(f"[Updater] Failed to download update, HTTP {response.status_code}")
    except Exception as e:
        print(f"[Updater] Exception during download: {e}")
    return False

def apply_update():
    try:
        print("[Updater] Applying update...")
        shutil.move("main_new.py", "main.py")
        print("[Updater] Update applied. Please restart Lucy.")
        return True
    except Exception as e:
        print(f"[Updater] Failed to apply update: {e}")
    return False

def check_for_updates():
    try:
        print("[Updater] Checking for updates...")
        response = requests.get(UPDATE_CHECK_URL, timeout=5)
        if response.status_code == 200:
            latest_version = response.text.strip()
            local_version = get_local_version()
            print(f"[Updater] Local version: {local_version}, Latest version: {latest_version}")
            if local_version != latest_version:
                print("[Updater] New version detected.")
                if download_update():
                    if apply_update():
                        write_local_version(latest_version)
                        # Exit for restart
                        print("[Updater] Exiting to complete update.")
                        os._exit(0)
            else:
                print("[Updater] Lucy is up to date.")
        else:
            print(f"[Updater] Update check failed, HTTP {response.status_code}")
    except Exception as e:
        print(f"[Updater] Exception during update check: {e}")

def start_update_thread(interval_hours=1):
    def updater_loop():
        while True:
            check_for_updates()
            time.sleep(interval_hours * 3600)

    t = threading.Thread(target=updater_loop, daemon=True)
    t.start()

# ---------- Configuration for Lucy AI ----------

PERSIST_DIR = "./memory/chroma_store"
MEMORY_FILE = "scenario_memory.json"
GPT4ALL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "qwen2-1_5b-instruct-q4_0.gguf")
INTENT_MODEL_NAME = "./models/distilbert-sst2"
SEMANTIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------- Initialize models and memory ----------

pygame.mixer.init()

chroma_client = chromadb.Client(chromadb.config.Settings(persist_directory=PERSIST_DIR))
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=SEMANTIC_EMBEDDING_MODEL)
memory_collection = chroma_client.get_or_create_collection(name="lucy_memory", embedding_function=embedding_func)

intent_tokenizer = AutoTokenizer.from_pretrained(INTENT_MODEL_NAME)
intent_model = AutoModelForSequenceClassification.from_pretrained(INTENT_MODEL_NAME)
intent_pipeline = pipeline("sentiment-analysis", model=intent_model, tokenizer=intent_tokenizer)

semantic_embedder = SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)

# GPT4All model
class ReasoningEngine:
    def __init__(self, model_path=GPT4ALL_MODEL_PATH, memory_file=MEMORY_FILE):
        self.memory_file = memory_file
        self.knowledge = []
        self.load_memory()
        self.model = GPT4All(model_path, allow_download=False)

    def load_memory(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                self.knowledge = json.load(f)
        else:
            self.knowledge = []

    def save_memory(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.knowledge, f, indent=2)

    def add_scenario(self, situation, reasoning, decision):
        self.knowledge.append({
            "situation": situation,
            "reasoning": reasoning,
            "decision": decision
        })
        self.save_memory()

    def similarity(self, a, b):
        return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def query_similar(self, query, threshold=0.4, top_k=3):
        scored = []
        for entry in self.knowledge:
            score = self.similarity(entry["situation"], query)
            if score >= threshold:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def reason(self, user_input):
        context_scenarios = self.query_similar(user_input)
        prompt = "You are an ethical reasoning AI. Given the following known scenarios and decisions:\n\n"
        for i, sc in enumerate(context_scenarios):
            prompt += f"{i+1}. Situation: {sc['situation']}\nDecision: {sc['decision']}\nReasoning: {sc['reasoning']}\n\n"
        prompt += f"User's query: {user_input}\nProvide a reasoned decision step-by-step.\nAnswer:"

        try:
            output = self.model.generate(prompt, n_predict=256)
        except Exception as e:
            output = f"Error generating reasoning: {e}"

        return output.strip()

reasoning_engine = ReasoningEngine()

# ---------- Semantic Memory ----------

class SemanticMemory:
    def __init__(self, collection):
        self.collection = collection

    def add(self, text, metadata=None):
        doc_id = uuid.uuid4().hex
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )

    def query(self, text, n_results=5):
        results = self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
        return results["documents"][0] if results["documents"] else []

semantic_memory = SemanticMemory(memory_collection)

# ---------- Intent & Sentiment Analysis ----------

def analyze_intent_sentiment(text):
    results = intent_pipeline(text)
    if not results:
        return "unknown", "neutral"
    label = results[0]["label"].lower()
    sentiment = "positive" if label == "positive" else "negative" if label == "negative" else "neutral"
    lowered = text.lower()
    if any(greet in lowered for greet in ["hello", "hi", "hey"]):
        intent = "greeting"
    elif any(bye in lowered for bye in ["bye", "goodbye", "farewell", "exit", "quit"]):
        intent = "farewell"
    elif any(word in lowered for word in ["code", "run this", "python"]):
        intent = "code"
    elif any(word in lowered for word in ["wiki", "wikipedia", "who is", "what is"]):
        intent = "wikipedia"
    elif any(word in lowered for word in ["remember", "recall", "what do you know about"]):
        intent = "memory_query"
    else:
        intent = "unknown"
    return intent, sentiment

# ---------- TTS & Audio Playback ----------

_audio_lock = threading.Lock()

async def generate_speech(text, filename):
    communicator = edge_tts.Communicate(text, voice="en-US-MichelleNeural")
    await communicator.save(filename)

def play_audio_blocking(filename):
    try:
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"Audio playback error: {e}")

def speak(text):
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    async def tts_and_play():
        await generate_speech(text, filename)
        with _audio_lock:
            play_audio_blocking(filename)
        try:
            os.remove(filename)
        except Exception:
            pass
    threading.Thread(target=lambda: asyncio.run(tts_and_play()), daemon=True).start()

# ---------- Wikipedia Search ----------

wikipedia.set_lang("en")

def search_wikipedia(query):
    try:
        return wikipedia.summary(query, sentences=2)
    except wikipedia.DisambiguationError as e:
        return f"Your question may refer to multiple topics. Did you mean: {e.options[0]}?"
    except wikipedia.PageError:
        return "I couldn't find anything about that."
    except Exception as e:
        return f"Error searching Wikipedia: {e}"

# ---------- Skills ----------

class BaseSkill:
    def match(self, text, intent):
        raise NotImplementedError
    def respond(self, text, context):
        raise NotImplementedError

class GreetingSkill(BaseSkill):
    def match(self, text, intent):
        return intent == "greeting"
    def respond(self, text, context):
        return "Hello! How can I assist you today?"

class FarewellSkill(BaseSkill):
    def match(self, text, intent):
        return intent == "farewell"
    def respond(self, text, context):
        return "Goodbye! Have a great day."

class WikipediaSkill(BaseSkill):
    def match(self, text, intent):
        return intent == "wikipedia"
    def respond(self, text, context):
        lowered = text.lower()
        for trig in ["wiki", "wikipedia", "who is", "what is"]:
            lowered = lowered.replace(trig, "")
        topic = lowered.strip()
        if not topic:
            return "Please specify a topic to search for on Wikipedia."
        return search_wikipedia(topic)

class MemoryQuerySkill(BaseSkill):
    def match(self, text, intent):
        return intent == "memory_query"
    def respond(self, text, context):
        lowered = text.lower()
        for trig in ["remember", "recall", "what do you know about"]:
            lowered = lowered.replace(trig, "")
        query = lowered.strip()
        if not query:
            return "Please specify what you want me to recall."
        results = semantic_memory.query(query, n_results=3)
        if not results:
            return "I don't have anything relevant stored about that yet."
        return "Here's what I remember:\n" + "\n".join(results)

class CodeSkill(BaseSkill):
    def match(self, text, intent):
        return intent == "code"
    def respond(self, text, context):
        code_block = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        if code_block:
            code = code_block.group(1)
            return run_code_safely(code)
        return "Please provide Python code in a code block to run."

def run_code_safely(code):
    import subprocess
    filename = f"temp_code_{uuid.uuid4().hex}.py"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(code)
    try:
        result = subprocess.run(["python", filename], capture_output=True, text=True, timeout=5)
        output = result.stdout.strip()
        if not output:
            output = "(No output)"
        return f"Code output:\n{output}"
    except subprocess.TimeoutExpired:
        return "Code execution timed out."
    except Exception as e:
        return f"Error running code: {e}"
    finally:
        try:
            os.remove(filename)
        except Exception:
            pass

class FallbackSkill(BaseSkill):
    def match(self, text, intent):
        return True
    def respond(self, text, context):
        reasoning_response = reasoning_engine.reason(text)
        if reasoning_response:
            return reasoning_response
        return "Sorry, I don't understand that yet."

class SkillManager:
    def __init__(self):
        self.skills = [
            GreetingSkill(),
            FarewellSkill(),
            WikipediaSkill(),
            MemoryQuerySkill(),
            CodeSkill(),
            FallbackSkill(),
        ]
    def handle(self, text, context):
        intent, sentiment = analyze_intent_sentiment(text)
        for skill in self.skills:
            if skill.match(text, intent):
                response = skill.respond(text, context)
                return response, intent, sentiment
        return "I don't understand.", "unknown", "neutral"

skill_manager = SkillManager()

# ---------- Main Async Loop ----------

async def main_loop():
    context = {"semantic_memory": semantic_memory, "last_question": None, "last_response": None}
    print("Lucy is online. Type 'exit' to quit.")
    speak("Hello! I am Lucy, your advanced AI assistant. How can I help you today?")

    while True:
        user_input = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
        user_input = user_input.strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            farewell_msg = "Goodbye! It was a pleasure assisting you."
            print("Lucy:", farewell_msg)
            speak(farewell_msg)
            await asyncio.sleep(3)
            break

        response, intent, sentiment = skill_manager.handle(user_input, context)
        print("Lucy:", response)
        speak(response)

        context["last_question"] = user_input
        context["last_response"] = response

        if intent == "memory_query":
            semantic_memory.add(user_input, metadata={"intent": intent, "sentiment": sentiment})

# ---------- Start Updater Thread and Run Main Loop ----------

if __name__ == "__main__":
    start_update_thread()  # Background thread checking for updates every hour
    asyncio.run(main_loop())
