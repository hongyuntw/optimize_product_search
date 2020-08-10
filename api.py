from flask import Flask


app = Flask(__name__)



@app.route('/')
def hello_world():
    return 'Flask Dockerized'

@app.route('/testing')
def mytest():
    return "testing...."



if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8787)