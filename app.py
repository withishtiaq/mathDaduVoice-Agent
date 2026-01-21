import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

# Flask অ্যাপ সেটআপ (static ফোল্ডার অটোমেটিক কাজ করবে)
app = Flask(__name__)

# --- API Key সেটআপ ---
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("Error: GOOGLE_API_KEY not found!")
else:
    genai.configure(api_key=api_key)

# --- মডেল কনফিগারেশন ---
generation_config = {
  "temperature": 0.4,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
}

system_instruction = """
আপনি একজন বন্ধুসুলভ এবং দক্ষ গণিত শিক্ষক (Math Tutor)।
১. আপনি ব্যবহারকারীর সাথে সাবলীল বাংলায় কথা বলবেন।
২. উত্তরগুলো খুব ছোট এবং কথোপকথনের মতো রাখবেন (২-৩ লাইনের মধ্যে)।
৩. গাণিতিক সমীকরণগুলো কথায় লিখবেন (যেমন: "x স্কয়ার")।
"""

# মডেল (Gemini Flash ব্যবহার করছি কারণ এটি ফাস্ট)
model_name = "gemini-flash-latest"
try:
    model = genai.GenerativeModel(
      model_name=model_name,
      generation_config=generation_config,
      system_instruction=system_instruction,
    )
    chat_session = model.start_chat(history=[])
except Exception as e:
    model = None

# --- রাউট ---

# হোম পেজ (ওয়েব ডিজাইন দেখাবে)
@app.route('/')
def home():
    return render_template('index.html')

# চ্যাট API (কথা প্রসেস করবে)
@app.route('/chat', methods=['POST'])
def chat():
    global chat_session
    try:
        data = request.json
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({"reply": "কিছু শুনতে পাইনি, আবার বলুন।"})

        if not model:
             return jsonify({"reply": "দুঃখিত, সিস্টেম লোড হচ্ছে না।"})

        response = chat_session.send_message(user_message)
        clean_text = response.text.replace("*", "").replace("#", "")
        return jsonify({"reply": clean_text})

    except Exception as e:
        # এরর হলে সেশন রিসেট
        try:
            chat_session = model.start_chat(history=[])
            response = model.generate_content(user_message)
            return jsonify({"reply": response.text.replace("*", "")})
        except:
            return jsonify({"reply": "দুঃখিত, একটু সমস্যা হয়েছে।"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
