import os
import json
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

# โหลด API Key จากไฟล์ .env
load_dotenv()

app = Flask(__name__)

# ตั้งค่า Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("WARNING: GEMINI_API_KEY is not set in .env file!")

# Prompt สำคัญ: เน้นคำตอบสั้น ตรงประเด็น เหมาะสำหรับอ่านบนจอ หรือพูดตอบต่อทันที
SYSTEM_INSTRUCTION = """
คุณคือผู้ช่วยอัจฉริยะส่วนตัวแบบ Real-time ในบทสนทนา
หน้าที่ของคุณคือรับฟังข้อความบทสนทนาของคู่สนทนา แล้วสรุปคำตอบให้ผู้ใช้ฟังหรืออ่านเพื่อนำไปพูดตอบต่อได้ทันที

เงื่อนไขการตอบ:
1. ตอบเป็นภาษาไทยที่กระชับ สั้น และตรงประเด็นที่สุด (ความยาวไม่เกิน 1-2 ประโยค)
2. ใช้ภาษาพูดที่เป็นธรรมชาติ เหมาะสำหรับการอ่านบนหน้าจอหรืออ่านออกเสียง
3. หากเป็นคำถามเชิงตัวเลขหรือข้อมูลทางการแพทย์/วิชาการ ให้ระบุค่ามาตรฐานแบบสรุปสั้นๆ
4. ห้ามเกริ่นนำหรือพูดน้ำเยอะ ให้เข้าประเด็นคำตอบทันที
5. คุณจะได้รับบริบทบทสนทนาก่อนหน้า (ถ้ามี) ให้ใช้บริบทนั้นประกอบการตอบ แต่ให้ตอบเฉพาะข้อความล่าสุดของคู่สนทนาเท่านั้น
"""

MAX_HISTORY_TURNS = 8  # จำกัดบริบทที่ส่งให้ Gemini ไม่ให้ prompt ยาวเกินไป

# Prompt สำหรับโหมด "เลขาคอยกระซิบ": ฟังตลอดเวลา แต่จะพูดเฉพาะตอนที่มีคำถามจริง ๆ เท่านั้น
SECRETARY_INSTRUCTION = """
คุณคือ "เลขาส่วนตัว" ที่แอบฟังบทสนทนาระหว่าง นาย A (คนที่มาถาม) กับ นาย B (เจ้าของแอปนี้)
งานของคุณคือคอยเงี่ยหูฟัง แล้ว "กระซิบ" บอกข้อมูลที่ถูกต้องให้นาย B นำไปพูดต่อได้ทันที
เหมือนเลขาที่ยืนข้าง ๆ คอยบอกข้อมูลให้เจ้านายเนียน ๆ

กฎการตัดสินใจ (สำคัญมาก):
1. ถ้าข้อความล่าสุดเป็นคำถามที่ต้องใช้ข้อมูล ข้อเท็จจริง ตัวเลข ค่ามาตรฐาน หรือความรู้เฉพาะทาง
   (เช่น ค่าทางการแพทย์ วันที่ สถิติ นิยาม สูตรคำนวณ) ให้ตั้ง "should_respond": true
   แล้วตอบใน "answer" แบบสั้น กระชับ ที่สุด (1-2 ประโยค) พร้อมตัวเลข/ข้อมูลอ้างอิงมาตรฐานที่ถูกต้องที่สุดเท่าที่ทราบ
   ถ้าเป็นค่าที่แยกตามเพศ/กลุ่ม ให้ระบุแยกให้ชัดเจน
2. ถ้าข้อความล่าสุดเป็นแค่คำพูดทั่วไป การทักทาย การพูดคิดออกเสียงของนาย B เอง
   (เช่น "คิดแป๊บนะ", "เดี๋ยวนะ", "อืม", "โอเค") หรือเป็นสิ่งที่ตอบไปแล้วก่อนหน้านี้
   ให้ตั้ง "should_respond": false และ "answer": ""
3. ห้ามเกริ่นนำ ห้ามพูดน้ำ ตอบเข้าประเด็นทันที
4. ถ้าเป็นข้อมูลทางการแพทย์/สุขภาพ ให้ตอบเป็น "ค่าอ้างอิงมาตรฐานทั่วไป" เท่านั้น
   ไม่ใช่การวินิจฉัยหรือให้คำแนะนำเฉพาะบุคคล

ตอบกลับเป็น JSON เพียงอย่างเดียว ห้ามมีข้อความอื่นนอกเหนือจาก JSON ห้ามใส่ ```json:
{"should_respond": true หรือ false, "answer": "..."}
"""


# --- ตั้งค่าราคาสำหรับคำนวณค่าใช้จ่าย (Gemini 2.5 Flash) ---
# อัตรา ณ ปัจจุบัน: $0.30 ต่อ 1 ล้าน input token, $2.50 ต่อ 1 ล้าน output token
# อ้างอิง: https://ai.google.dev/gemini-api/docs/pricing (ควรเข้าไปเช็คเป็นระยะเพราะราคาเปลี่ยนได้)
PRICE_INPUT_PER_MTOKEN_USD = 0.30
PRICE_OUTPUT_PER_MTOKEN_USD = 2.50

# อัตราแลกเปลี่ยน USD -> THB (ค่าประมาณ ควรอัปเดตเป็นระยะ หรือดึงจาก API แบบ real-time ถ้าต้องการความแม่นยำสูง)
USD_TO_THB_RATE = 33.0

# ตัวนับสะสมค่าใช้จ่ายทั้งหมดตั้งแต่รันเซิร์ฟเวอร์ (เก็บใน memory เท่านั้น รีสตาร์ทเซิร์ฟเวอร์แล้วจะรีเซ็ตเป็น 0)
usage_totals = {
    "input_tokens": 0,
    "output_tokens": 0,
}


def calculate_cost(input_tokens, output_tokens):
    """คำนวณค่าใช้จ่ายเป็น USD และ THB จากจำนวน token ที่ใช้"""
    cost_usd = (
        (input_tokens / 1_000_000) * PRICE_INPUT_PER_MTOKEN_USD
        + (output_tokens / 1_000_000) * PRICE_OUTPUT_PER_MTOKEN_USD
    )
    cost_thb = cost_usd * USD_TO_THB_RATE
    return cost_usd, cost_thb


def extract_usage_and_track(response):
    """
    ดึงจำนวน token จาก response ของ Gemini แล้วบวกเข้าตัวนับสะสม
    คืนค่า dict พร้อมค่าใช้จ่ายของ "ครั้งนี้" และ "สะสมทั้งหมด"
    """
    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

    usage_totals["input_tokens"] += input_tokens
    usage_totals["output_tokens"] += output_tokens

    call_cost_usd, call_cost_thb = calculate_cost(input_tokens, output_tokens)
    total_cost_usd, total_cost_thb = calculate_cost(
        usage_totals["input_tokens"], usage_totals["output_tokens"]
    )

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(call_cost_usd, 6),
        "cost_thb": round(call_cost_thb, 4),
        "total_input_tokens": usage_totals["input_tokens"],
        "total_output_tokens": usage_totals["output_tokens"],
        "total_cost_usd": round(total_cost_usd, 6),
        "total_cost_thb": round(total_cost_thb, 4),
    }


@app.route("/api/usage", methods=["GET"])
def get_usage():
    """เช็คยอดค่าใช้จ่ายสะสมทั้งหมดได้ทุกเมื่อ โดยไม่ต้องเรียก Gemini"""
    total_cost_usd, total_cost_thb = calculate_cost(
        usage_totals["input_tokens"], usage_totals["output_tokens"]
    )
    return jsonify({
        "total_input_tokens": usage_totals["input_tokens"],
        "total_output_tokens": usage_totals["output_tokens"],
        "total_cost_usd": round(total_cost_usd, 6),
        "total_cost_thb": round(total_cost_thb, 4),
    })


@app.route("/")
def index():
    """แสดงหน้าเว็บหลัก"""
    return render_template("index.html")


@app.route("/api/ask", methods=["POST"])
def ask_gemini():
    """API endpoint รับข้อความจากหน้าเว็บแล้วส่งให้ Gemini ตอบ (โหมดถาม-ตอบเดี่ยว)"""
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data.get("message")

    if not client:
        return jsonify({
            "error": "ยังไม่ได้ตั้งค่า GEMINI_API_KEY กรุณาตั้งค่าในไฟล์ .env"
        }), 400

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.2,
            )
        )
        reply_text = response.text.strip()
        usage = extract_usage_and_track(response)
        return jsonify({"reply": reply_text, "usage": usage})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": f"เกิดข้อผิดพลาดในการดึงข้อมูล: {str(e)}"}), 500


@app.route("/api/process_dialogue", methods=["POST"])
def process_dialogue():
    """
    API endpoint สำหรับหน้า Dual-Speaker Assistant

    NOTE เรื่องการแยกเสียงสองคน:
    Web Speech API ส่งกลับมาแค่ "ข้อความ" ไม่มีข้อมูลว่าใครพูด (ไม่มี speaker diarization)
    เพราะฉะนั้นสมมติฐานที่ endpoint นี้ใช้คือ:
      - ทุกข้อความใหม่ที่ผ่านการกรอง echo ของ AI มาแล้วฝั่ง client (isSimilarToLastAiReply)
        ถือว่าเป็นคำพูดของ "คู่สนทนา" (speaker "A")
      - AI จะสร้างคำตอบให้ "ผู้ใช้แอป" (speaker "B") นำไปพูดต่อ
    ถ้าต้องการแยกเสียงจริง ๆ (เช่น 2 คนพูดสลับกันเข้าไมค์เดียวกัน) ต้องใช้บริการ STT
    ที่รองรับ diarization เช่น Google Cloud Speech-to-Text หรือ AssemblyAI แทน Web Speech API
    """
    data = request.get_json(silent=True) or {}
    raw_text = (data.get("raw_text") or "").strip()
    history = data.get("history") or []

    if not raw_text:
        return jsonify({"error": "No raw_text provided"}), 400

    if not client:
        return jsonify({
            "error": "ยังไม่ได้ตั้งค่า GEMINI_API_KEY กรุณาตั้งค่าในไฟล์ .env"
        }), 400

    # เพิ่มข้อความใหม่ของคู่สนทนา (A) เข้าไปในประวัติ
    history.append({"speaker": "A", "text": raw_text})

    # ประกอบบริบทจากประวัติล่าสุด ให้ Gemini เห็นบทสนทนาที่ผ่านมา
    context_lines = []
    for item in history[-MAX_HISTORY_TURNS:]:
        label = "นาย A (คนถาม)" if item.get("speaker") == "A" else "นาย B (เจ้าของแอป)"
        context_lines.append(f"{label}: {item.get('text', '')}")
    context_text = "\n".join(context_lines)

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=context_text,
            config=types.GenerateContentConfig(
                system_instruction=SECRETARY_INSTRUCTION,
                temperature=0.2,
                response_mime_type="application/json",
            )
        )
        raw_reply = response.text.strip()
        # กันเหนียวเผื่อโมเดลใส่ code fence มาแม้จะสั่งห้ามแล้ว
        raw_reply = raw_reply.replace("```json", "").replace("```", "").strip()
        result = json.loads(raw_reply)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": f"เกิดข้อผิดพลาดในการดึงข้อมูล: {str(e)}"}), 500

    should_respond = bool(result.get("should_respond"))
    answer = (result.get("answer") or "").strip()
    usage = extract_usage_and_track(response)

    if should_respond and answer:
        # เลขาตัดสินใจว่ามีคำถามจริง -> กระซิบคำตอบให้นาย B แล้วบันทึกลงประวัติ
        history.append({"speaker": "B", "text": answer})
        return jsonify({
            "processed_history": history,
            "reply_for_b": answer,
            "usage": usage
        })

    # เลขาตัดสินใจว่าไม่ใช่คำถาม (พูดคุยทั่วไป/คิดออกเสียง) -> เงียบไว้ ไม่ต้องกระซิบ
    return jsonify({
        "processed_history": history,
        "reply_for_b": None,
        "usage": usage
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
