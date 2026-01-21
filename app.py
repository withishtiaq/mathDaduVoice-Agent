import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

app = Flask(__name__)

# --- API Key সেটআপ (Render Environment থেকে নেবে) ---
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    # লোকাল টেস্টের জন্য ওয়ার্নিং, কিন্তু রেন্ডারে সমস্যা হবে না যদি env set থাকে
    print("Warning: GOOGLE_API_KEY not found via environment variable.")
else:
    genai.configure(api_key=api_key)

# --- দাদুর চরিত্র কনফিগারেশন ---
generation_config = {
  "temperature": 0.5, # একটু সৃজনশীল হবে
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 200, # উত্তর ছোট রাখবে
}

# দাদুর জন্য বিশেষ নির্দেশনা
system_instruction = """
তুমি একজন ৭৫ বছর বয়সী অত্যন্ত জ্ঞানী, শান্ত এবং স্নেহপরায়ণ বাঙালি দাদু।
১. ব্যবহারকারী তোমার নাতি বা নাতনি। তাদের "দাদুভাই", "সোনা", "মনা" বা "দিদিভাই" বলে সম্বোধন করবে।
২. তোমার বাচনভঙ্গি হবে ধীরস্থির এবং মায়াভরা।
৩. উত্তরগুলো খুব ছোট হবে (সর্বোচ্চ ২-৩ বাক্যে), যেন মনে হয় ফোনে কথা বলছো।
৪. কোনো কঠিন বইয়ের ভাষা ব্যবহার করবে না। একদম ঘরোয়া, সাবলীল বাংলায় কথা বলবে।
৫. গণিত বা যেকোনো বিষয় গল্পের ছলে বা সহজ উদাহরণ দিয়ে বোঝাবে।
"""

# মডেল সিলেকশন
model_name = "gemini-flash-latest"

try:
    model = genai.GenerativeModel(
      model_name=model_name,
      generation_config=generation_config,
      system_instruction=system_instruction,
    )
    # চ্যাট সেশন শুরু
    chat_session = model.start_chat(history=[])
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# --- API রাউটস ---

@app.route('/')
def home():
    # ফ্রন্টএন্ড লোড করবে
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_session
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"reply": "কিছু শুনতে পাইনি দাদুভাই, আবার বলো?"})

        if not model:
             return jsonify({"reply": "আমার শরীরটা ভালো লাগছে না, পরে কথা বলি?"})

        # দাদুর উত্তর তৈরি হচ্ছে
        response = chat_session.send_message(user_message)
        
        # ক্লিন টেক্সট
        clean_text = response.text.replace("*", "").replace("#", "").replace("\n", " ")
        
        return jsonify({"reply": clean_text})

    except Exception as e:
        # এরর হ্যান্ডলিং এবং সেশন রিসেট
        try:
            chat_session = model.start_chat(history=[])
            response = model.generate_content(user_message)
            clean_text = response.text.replace("*", "").replace("#", "")
            return jsonify({"reply": clean_text})
        except:
            return jsonify({"reply": "নেটে একটু সমস্যা হচ্ছে দাদুভাই, আবার চেষ্টা করো।"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
