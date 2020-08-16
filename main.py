import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle

app=Flask('__name__')
model=pickle.load(open('Approval.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction=model.predict_proba(final_features)
    output=prediction[0][1]*100
    print(str(prediction))
    return render_template('home.html',prediction_text="Approval chances are {}".format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction=model.predict_proba(final_features)
    #output=prediction[0][1]*100
    #print(str(prediction))
    #return render_template('home.html',prediction_text="Approval chances are {}".format(output))
    #for Direct API calls
    print("Received")
    data=request.get_json(force=True)
    prediction=model.predict_proba([np.array(list(data['val']))])
    output=prediction[0][1]*100
    print(output)
    return jsonify(output)
    #return str(data)

@app.route('/predict_get',methods=['GET'])
def predict_get():
    try:
        arr=request.args.get('arr')
        val=[np.array(list([int(i) for i in arr.split(',')]))]
        lst=model.predict_proba(val)
        output=lst[0][1]*100
        return jsonify(output)
    except Exception:
        return str('err:')+str(arr)


if __name__=='__main__':
    app.run(debug=True)