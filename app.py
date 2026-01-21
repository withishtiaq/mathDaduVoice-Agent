import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

app = Flask(__name__)

# --- API Key সেটআপ ---
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    print("Warning: GOOGLE_API_KEY not found in environment variables.")
else:
    genai.configure(api_key=api_key)

# --- দাদুর কনফিগারেশন ---
generation_config = {
  "temperature": 0.6, 
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 300, 
}

# দাদুর কড়া নির্দেশনা
system_instruction = """
তুমি একজন ৭৫ বছর বয়সী অত্যন্ত বদমেজাজি এবং রাগী অঙ্ক শিক্ষক দাদু।
১. তুমি অঙ্ক (Mathematics) ছাড়া পৃথিবীর আর কোনো বিষয় বোঝো না।
২. ব্যবহারকারী যদি অঙ্ক বা গণিত সম্পর্কিত কোনো প্রশ্ন করে, তবে তুমি সেটা বুঝিয়ে বলবে, কিন্তু তোমার সুর হবে কড়া শিক্ষকের মতো। (যেমন: "এটাও জানিস না? শোন...", "মন দিয়ে শোন গাধা কোথাকার...")।
৩. ব্যবহারকারী যদি অঙ্ক ছাড়া অন্য যেকোনো বিষয় নিয়ে কথা বলে (যেমন: "কেমন আছো?", "গান শোনাও"), তুমি ভীষণ রেগে যাবে এবং বকাবকি করবে।
৪. তুমি সব সময় আঞ্চলিক ও কড়া ভাষায় কথা বলবে। তুমি ব্যবহারকারীকে "তুই" বা "তোরা" বলে সম্বোধন করবে।
"""

# মডেল লোড
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
    print(f"Error loading model: {e}")

# --- API রাউটস ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_session
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"reply": "জোরে বল! কিছু শুনতে পাইনি।"})

        if not model:
             return jsonify({"reply": "আমার মাথা গরম করবি না, সিস্টেম কাজ করছে না।"})

        # দাদুর উত্তর
        response = chat_session.send_message(user_message)
        
        # ক্লিন টেক্সট
        clean_text = response.text.replace("*", "").replace("#", "").replace("\n", " ")
        
        return jsonify({"reply": clean_text})

    except Exception as e:
        # এরর হলে সেশন রিসেট
        try:
            chat_session = model.start_chat(history=[])
            response = model.generate_content(user_message)
            clean_text = response.text.replace("*", "").replace("#", "")
            return jsonify({"reply": clean_text})
        except:
            return jsonify({"reply": "নেটে সমস্যা করছে, পরে আসিস!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
