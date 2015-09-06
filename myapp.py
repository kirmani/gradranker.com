from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
import json
import os.path
from grad_nn import NN
from grad_nn import Parser
app = Flask(__name__)

#settings.py
import os
# __file__ refers to the file settings.py 
APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
APP_STATIC = os.path.join(APP_ROOT, 'static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/gradschool/<school>/<field>/<predict>')
def get_grad_school(school, field, predict):
    with open(os.path.join(APP_ROOT, "data/schools/" + predict+ "/" + school + '-' + field + '.json')) as infile:
        result = json.load(infile)
    return jsonify(result)

@app.route('/predict/<school>/<field>')
def hasPredict(school, field):
    if os.path.exists('data/schools/infered/' + school + '-' + field + '.json'):
        return jsonify({'response':200})
    return jsonify({'response':404})

@app.route('/available')
def get_available():
    with open(os.path.join(APP_ROOT, "data/available.json")) as infile:
        result = json.load(infile)
    return jsonify(result)

@app.route('/schools')
def get_schools():
    with open(os.path.join(APP_ROOT, "data/schools.json")) as infile:
        result = json.load(infile)
    return jsonify(result)

@app.route('/fields')
def get_fields():
    with open(os.path.join(APP_ROOT, "data/fields.json")) as infile:
        result = json.load(infile)
    return jsonify(result)

@app.route('/inputs')
def get_inputs():
    with open(os.path.join(APP_ROOT, "data/inputs.json")) as infile:
        result = json.load(infile)
    return jsonify(result)

@app.route('/outputs')
def get_outputs():
    with open(os.path.join(APP_ROOT, "data/outputs.json")) as infile:
        result = json.load(infile)
    return jsonify(result)

@app.route('/submit', methods=['POST'])
def submit_application():
    try: 
      f=open(os.path.join(APP_ROOT, 'data/submissions.json'),'r')
    except IOError:
      f=open(os.path.join(APP_ROOT, 'data/submissions.json'),'w')
      json.dump([], f, indent=4)

    with open(os.path.join(APP_ROOT, 'data/submissions.json'),'r') as f:
        data = json.load(f)
    with open(os.path.join(APP_ROOT, 'data/submissions.json'),'w') as f:
        submission = json.loads(request.data)
        # print(submission)
        data.append(submission)
        f.seek(0)
        p = Parser()
        p.addSubmission(submission)
        json.dump(data, f, indent=4)
        return jsonify({'response':200})
    return jsonify({'response':404})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 33507))	    
    app.debug = True;
    app.run(host='0.0.0.0', port=port)
    # app.run()
