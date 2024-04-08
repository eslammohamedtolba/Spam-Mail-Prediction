from flask import Flask, render_template, request
import pickle as pk

# load model
ModelClassifier = pk.load(open('PrepareModel\LRModel.sav','rb'))
# Load feature vectorizer
with open('PrepareModel\\tfidf_vectorizer.pk', 'rb') as f:  # Open in binary read mode
    feature_extra = pk.load(f)


# Create app
app = Flask(__name__, template_folder="tmeplates",static_folder="static",static_url_path="/")
# Create routes and urls
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input_text = request.form.get('text')  # Access user input from the form
        input_Mail = [user_input_text]
        input_Mail = feature_extra.transform(input_Mail)
        prediction = ModelClassifier.predict(input_Mail[0])[0]
        return render_template('index.html', pred = prediction)
    else:
        return render_template('index.html',pred = -1)


# 07732584351 - Rodger Burns - MSG = We tried to call you re your reply to our sms for a free nokia mobile + free camcorder. Please call now 08000930705 for delivery tomorrow
# Didn't you get hep b immunisation in nigeria. -> ham

if __name__ == "__main__":
    app.run(debug=True)




