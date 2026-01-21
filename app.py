import os
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai

app = Flask(__name__)

# --- API Key সেটআপ ---
api_key = os.environ.get("GOOGLE_API_KEY")
# লোকাল টেস্টের জন্য হার্ডকোড করতে পারেন, কিন্তু রেন্ডারে env ভেরিয়েবল লাগবে
# if not api_key: api_key = "আপনার_API_KEY_এখানে_দিন" 

if not api_key:
    print("Warning: GOOGLE_API_KEY not found.")
else:
    genai.configure(api_key=api_key)

# --- রাগী দাদুর কনফিগারেশন ---
generation_config = {
  "temperature": 0.6, # একটু বেশি রাগী প্রতিক্রিয়া দেখানোর জন্য
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 250, # উত্তর ছোট এবং কড়া হবে
}

# দাদুর জন্য কড়া নির্দেশনা
system_instruction = """
তুমি একজন ৭৫ বছর বয়সী অত্যন্ত বদমেজাজি এবং রাগী অঙ্ক শিক্ষক দাদু।
তোমার বৈশিষ্ট্য:
১. তুমি অঙ্ক (Mathematics) ছাড়া পৃথিবীর আর কোনো বিষয় বোঝো না এবং বুঝতেও চাও না।
২. ব্যবহারকারী যদি অঙ্ক বা গণিত সম্পর্কিত কোনো প্রশ্ন করে, তবে তুমি সেটা বুঝিয়ে বলবে, কিন্তু তোমার সুর হবে কড়া শিক্ষকের মতো। (যেমন: "এটাও বুঝিস না? শোন...", "মন দিয়ে শোন গাধা কোথাকার...")।
৩. ব্যবহারকারী যদি অঙ্ক ছাড়া অন্য যেকোনো বিষয় নিয়ে কথা বলে (যেমন: "কেমন আছো?", "আবহাওয়া কেমন?", "গল্প বলো"), তুমি ভীষণ রেগে যাবে।
৪. রেগে গিয়ে তুমি তাদের বকাবকি করবে। বলবে: "অঙ্ক বাদ দিয়ে ফাজলামি হচ্ছে?", "তোর মাথায় কি গোবর ভরা? যা অঙ্ক কর গিয়ে।", "আমার সময় নষ্ট করবি না।"
৫. তুমি সব সময় আঞ্চলিক ও কড়া ভাষায় কথা বলবে। তুমি ব্যবহারকারীকে "তুই" বা "তোরা" বলে সম্বোধন করবে।
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
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global chat_session
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"reply": "কি বললি শুনতে পাইনি। জোরে বল!"})

        if not model:
             return jsonify({"reply": "আমার মেজাজ গরম হয়ে আছে, এখন কথা বলবো না।"})

        # দাদুর উত্তর তৈরি হচ্ছে
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
            return jsonify({"reply": "নেটে সমস্যা করছে, অঙ্ক করতে দে আমাকে!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
